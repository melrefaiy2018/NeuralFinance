import pandas as pd
import datetime as dt
from typing import Optional, Tuple, List
import tensorflow as tf

try:
    from ..data.fetchers import StockDataFetcher, SentimentAnalyzer
    from ..data.processors import TechnicalIndicatorGenerator
    from ..models.improved_model import ImprovedStockModel
    from ..config.model_config import ModelConfig
except ImportError:
    # Fallback for direct script execution
    from data.fetchers import StockDataFetcher, SentimentAnalyzer
    from data.processors import TechnicalIndicatorGenerator
    from models.improved_model import ImprovedStockModel
    from config.model_config import ModelConfig


class StockAnalyzer:
    """
    Provides a high-level interface to run stock analysis, including data fetching,
    preprocessing, model training, and prediction.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize StockAnalyzer with optional configuration.
        
        Args:
            config (Optional[ModelConfig]): Configuration object for model parameters.
                                          If None, default configuration will be used.
        """
        self.config = config if config is not None else ModelConfig.default()
        print(f"Initialized StockAnalyzer with model type: {self.config.model_type}")
        print(f"Sequence length: {self.config.sequence_length}, Prediction horizon: {self.config.prediction_horizon}")

    def run_analysis_for_stock(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> Tuple[
        Optional[ImprovedStockModel], Optional[pd.DataFrame], Optional[List[float]], Optional[List]
    ]:
        """
        Runs a complete analysis for a given stock ticker.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            period (str, optional): The period to fetch data for (e.g., '1y', '6mo'). Defaults to '1y'.
            interval (str, optional): The data interval (e.g., '1d', '1wk'). Defaults to '1d'.

        Returns:
            Tuple[Optional[StockSentimentModel], Optional[pd.DataFrame], Optional[List[float]], Optional[List]]:
                A tuple containing the trained model, the combined data, future price predictions, and future dates.
                Returns (None, None, None, None) if analysis fails.
        """
        try:
            print(f"\n=== Analyzing {ticker} ===")

            print("Fetching stock data...")
            stock_fetcher = StockDataFetcher(ticker, period, interval)
            stock_df = stock_fetcher.fetch_data()

            if stock_df is None or len(stock_df) < 30:
                print(f"Insufficient data for {ticker}")
                return None, None, None, None

            print("Generating sentiment data...")
            sentiment_analyzer = SentimentAnalyzer(ticker)
            sentiment_df = sentiment_analyzer.fetch_news_sentiment(
                start_date=stock_df["date"].min(), end_date=stock_df["date"].max()
            )

            if sentiment_df is None:
                print(f"Failed to generate sentiment data for {ticker}")
                return None, None, None, None

            print("Preparing data...")

            # Normalize datetime columns to avoid merge conflicts
            # Convert both to timezone-naive dates for consistent merging

            # For stock data - convert to timezone-naive date
            stock_df = stock_df.copy()
            if hasattr(stock_df["date"].dtype, "tz") and stock_df["date"].dtype.tz is not None:
                # Convert timezone-aware to timezone-naive
                stock_df["date"] = stock_df["date"].dt.tz_convert("UTC").dt.tz_localize(None)

            # Normalize to date only (remove time component)
            stock_df["date"] = pd.to_datetime(stock_df["date"].dt.date)

            # For sentiment data - ensure timezone-naive
            sentiment_df = sentiment_df.copy()
            if (
                hasattr(sentiment_df["date"].dtype, "tz")
                and sentiment_df["date"].dtype.tz is not None
            ):
                sentiment_df["date"] = (
                    sentiment_df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
                )

            # Normalize to date only (remove time component)
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"].dt.date)

            print(
                f"Stock data dates: {stock_df['date'].min()} to {stock_df['date'].max()} ({len(stock_df)} rows)"
            )
            print(
                f"Sentiment data dates: {sentiment_df['date'].min()} to {sentiment_df['date'].max()} ({len(sentiment_df)} rows)"
            )

            combined_df = pd.merge(stock_df, sentiment_df, on="date", how="inner")
            print(f"Combined data: {len(combined_df)} rows after merge")

            combined_df["price_change"] = combined_df["close"].pct_change()
            combined_df["volatility"] = (
                combined_df["close"].rolling(window=5).std() / combined_df["close"]
            )
            combined_df["momentum"] = combined_df["close"] - combined_df["close"].shift(5)

            combined_df = TechnicalIndicatorGenerator.add_technical_indicators(combined_df)
            combined_df = combined_df.dropna()

            if len(combined_df) < 50:
                print(f"Insufficient data after processing for {ticker}")
                return None, None, None, None

            print("Training model...")
            # Use configuration parameters for model creation
            if self.config.model_type == 'improved':
                model = ImprovedStockModel(
                    look_back=self.config.sequence_length, 
                    forecast_horizon=self.config.prediction_horizon
                )
            else:
                # Default to improved model if lstm_attention is not available
                model = ImprovedStockModel(
                    look_back=self.config.sequence_length, 
                    forecast_horizon=self.config.prediction_horizon
                )
                
            X_market, X_sentiment, y = model.prepare_data(combined_df, target_col="close")

            if len(X_market) < 100:
                print(f"Warning: Limited training data ({len(X_market)} samples)")

            # Use configuration for train/test split
            split_idx = int((1.0 - self.config.test_split) * len(X_market))
            X_market_train = X_market[:split_idx]
            X_sentiment_train = X_sentiment[:split_idx]
            y_train = y[:split_idx]

            print(f"Training on {len(X_market_train)} samples, validating on {len(X_market) - len(X_market_train)} samples")
            print(f"Using config - Sequence length: {self.config.sequence_length}, Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")

            model.build_model(X_market.shape[2], X_sentiment.shape[2])

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.config.early_stopping_patience, restore_best_weights=True, min_delta=0.001
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
            )

            model.fit(
                X_market_train,
                X_sentiment_train,
                y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=1,  # Show training progress
                validation_split=self.config.validation_split,
                callbacks=[early_stopping, reduce_lr],
            )

            print("Making future predictions...")
            future_prices = model.predict_next_days(X_market[-1], X_sentiment[-1], days=self.config.prediction_horizon)

            last_date = combined_df["date"].iloc[-1]
            future_dates = [last_date + dt.timedelta(days=i + 1) for i in range(self.config.prediction_horizon)]

            print(f"Analysis completed for {ticker}")
            return model, combined_df, future_prices, future_dates

        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, None, None, None

    def self_diagnostic(
        self, ticker: str = "NVDA", period: str = "1y"
    ) -> Tuple[Optional[ImprovedStockModel], Optional[pd.DataFrame], Optional[List[float]]]:
        """
        Runs a self-diagnostic test of the analysis pipeline.

        Args:
            ticker (str, optional): The stock ticker to use for the diagnostic. Defaults to 'NVDA'.
            period (str, optional): The data period to use. Defaults to '1y'.

        Returns:
            Tuple[Optional[ImprovedStockModel], Optional[pd.DataFrame], Optional[List[float]]]:
                A tuple containing the trained model, the combined data, and future price predictions.
                Returns (None, None, None) if the diagnostic fails.
        """
        print("\n=== Running Self-Diagnostic ===\n")

        try:
            model, combined_df, future_prices, future_dates = self.run_analysis_for_stock(
                ticker, period
            )

            if model is None:
                print("Diagnostic failed - could not create model")
                return None, None, None

            X_market, X_sentiment, y = model.prepare_data(combined_df, target_col="close")

            if len(X_market) > 10:
                test_X_market = X_market[-10:]
                test_X_sentiment = X_sentiment[-10:]
                test_y = y[-10:]

                predictions = model.predict(test_X_market, test_X_sentiment)

                metrics = model.evaluate(test_y, predictions)

                print("\nDiagnostic Results:")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"MAE: {metrics['mae']:.4f}")
                print(f"R²: {metrics['r2']:.4f}")
                print(f"MAPE: {metrics['mape']:.2f}%")

                if metrics["r2"] > 0.5:
                    print("✅ Model performance is acceptable")
                else:
                    print("⚠️ Model performance could be improved")

            print(f"\nFuture predictions for {ticker}:")
            for i, price in enumerate(future_prices):
                print(f"Day {i+1}: ${price:.2f}")

            print("\nDiagnostic completed successfully!")
            return model, combined_df, future_prices

        except Exception as e:
            print(f"Diagnostic failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, None, None
