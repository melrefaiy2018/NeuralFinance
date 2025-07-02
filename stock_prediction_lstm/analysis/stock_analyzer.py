import pandas as pd
import datetime as dt
from typing import Optional, Tuple, List
from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
from stock_prediction_lstm.data.processors import TechnicalIndicatorGenerator
from stock_prediction_lstm.models import StockSentimentModel
import tensorflow as tf

class StockAnalyzer:
    def run_analysis_for_stock(self, ticker: str, period: str = '1y', interval: str = '1d') -> Tuple[Optional[StockSentimentModel], Optional[pd.DataFrame], Optional[List[float]], Optional[List]]:
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
                start_date=stock_df['date'].min(),
                end_date=stock_df['date'].max()
            )
            
            if sentiment_df is None:
                print(f"Failed to generate sentiment data for {ticker}")
                return None, None, None, None
            
            print("Preparing data...")
            combined_df = pd.merge(stock_df, sentiment_df, on='date', how='inner')
            
            combined_df['price_change'] = combined_df['close'].pct_change()
            combined_df['volatility'] = combined_df['close'].rolling(window=5).std() / combined_df['close']
            combined_df['momentum'] = combined_df['close'] - combined_df['close'].shift(5)
            
            combined_df = TechnicalIndicatorGenerator.add_technical_indicators(combined_df)
            combined_df = combined_df.dropna()
            
            if len(combined_df) < 50:
                print(f"Insufficient data after processing for {ticker}")
                return None, None, None, None
            
            print("Training model...")
            model = StockSentimentModel(look_back=20, forecast_horizon=1)
            X_market, X_sentiment, y = model.prepare_data(combined_df, target_col='close')
            
            split_idx = int(0.8 * len(X_market))
            X_market_train = X_market[:split_idx]
            X_sentiment_train = X_sentiment[:split_idx]
            y_train = y[:split_idx]
            
            model.build_model(X_market.shape[2], X_sentiment.shape[2])
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            model.fit(
                X_market_train, X_sentiment_train, y_train,
                epochs=30,
                batch_size=16,
                verbose=0,
                validation_split=0.2,
                callbacks=[early_stopping]
            )
            
            print("Making future predictions...")
            future_prices = model.predict_next_days(
                X_market[-1], X_sentiment[-1], days=5
            )
            
            last_date = combined_df['date'].iloc[-1]
            future_dates = [last_date + dt.timedelta(days=i+1) for i in range(5)]
            
            print(f"Analysis completed for {ticker}")
            return model, combined_df, future_prices, future_dates
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

    def self_diagnostic(self, ticker: str = 'NVDA', period: str = '1y') -> Tuple[Optional[StockSentimentModel], Optional[pd.DataFrame], Optional[List[float]]]:
        print("\n=== Running Self-Diagnostic ===\n")
        
        try:
            model, combined_df, future_prices, future_dates = self.run_analysis_for_stock(ticker, period)
            
            if model is None:
                print("Diagnostic failed - could not create model")
                return None, None, None
            
            X_market, X_sentiment, y = model.prepare_data(combined_df, target_col='close')
            
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
                
                if metrics['r2'] > 0.5:
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
