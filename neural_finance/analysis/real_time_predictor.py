"""
Real-time stock price prediction with streaming capabilities.
"""

import asyncio
import datetime as dt
import time
from typing import List, Dict, Optional, AsyncGenerator
import pandas as pd
import numpy as np
from dataclasses import dataclass

try:
    from ..data.fetchers import StockDataFetcher, SentimentAnalyzer
    from ..data.processors import TechnicalIndicatorGenerator
    from ..models.improved_model import ImprovedStockModel
    from ..config.model_config import ModelConfig
    from .stock_analyzer import StockAnalyzer
except ImportError:
    # Fallback for direct script execution
    from data.fetchers import StockDataFetcher, SentimentAnalyzer
    from data.processors import TechnicalIndicatorGenerator
    from models.improved_model import ImprovedStockModel
    from config.model_config import ModelConfig
    from analysis.stock_analyzer import StockAnalyzer


@dataclass
class PredictionResult:
    """Container for prediction results."""
    ticker: str
    current_price: float
    pred_1h: float
    confidence: float
    timestamp: dt.datetime
    trend: str  # 'up', 'down', 'stable'
    volatility: float


class RealTimePredictor:
    """
    Real-time stock price predictor with streaming capabilities.
    
    This class provides continuous stock price predictions by:
    1. Fetching real-time data at specified intervals
    2. Using pre-trained or periodically refreshed models
    3. Streaming predictions as they become available
    """
    
    def __init__(
        self,
        tickers: List[str],
        update_frequency: str = '1min',
        model_refresh_interval: str = '1h',
        config: Optional[ModelConfig] = None
    ):
        """
        Initialize the real-time predictor.
        
        Args:
            tickers: List of stock ticker symbols to track
            update_frequency: How often to fetch new data ('1min', '5min', '15min', '1h')
            model_refresh_interval: How often to retrain models ('1h', '6h', '1d')
            config: Optional model configuration
        """
        self.tickers = [ticker.upper() for ticker in tickers]
        self.update_frequency = update_frequency
        self.model_refresh_interval = model_refresh_interval
        self.config = config if config is not None else ModelConfig.default()
        
        # Convert frequency strings to seconds
        self.update_interval = self._parse_frequency(update_frequency)
        self.model_refresh_interval_sec = self._parse_frequency(model_refresh_interval)
        
        # Storage for models and data
        self.models: Dict[str, ImprovedStockModel] = {}
        self.last_data: Dict[str, pd.DataFrame] = {}
        self.last_model_update: Dict[str, dt.datetime] = {}
        
        # Initialize analyzer
        self.analyzer = StockAnalyzer(config=self.config)
        
        print(f"RealTimePredictor initialized for {len(self.tickers)} tickers")
        print(f"Update frequency: {update_frequency} ({self.update_interval}s)")
        print(f"Model refresh: {model_refresh_interval} ({self.model_refresh_interval_sec}s)")
    
    def _parse_frequency(self, freq_str: str) -> int:
        """Convert frequency string to seconds."""
        freq_map = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '30min': 1800,
            '1h': 3600,
            '6h': 21600,
            '1d': 86400
        }
        return freq_map.get(freq_str, 60)  # Default to 1 minute
    
    async def _initialize_models(self):
        """Initialize or refresh models for all tickers."""
        print("Initializing models...")
        
        for ticker in self.tickers:
            try:
                print(f"Training model for {ticker}...")
                
                # Use a longer period for initial training
                model, combined_df, future_prices, future_dates = self.analyzer.run_analysis_for_stock(
                    ticker, period="6mo", interval="1d"
                )
                
                if model is not None and combined_df is not None:
                    self.models[ticker] = model
                    self.last_data[ticker] = combined_df
                    self.last_model_update[ticker] = dt.datetime.now()
                    print(f"✅ Model ready for {ticker}")
                else:
                    print(f"❌ Failed to initialize model for {ticker}")
                    
            except Exception as e:
                print(f"Error initializing model for {ticker}: {e}")
        
        print(f"Initialized {len(self.models)} models successfully")
    
    async def _should_refresh_model(self, ticker: str) -> bool:
        """Check if model needs refreshing."""
        if ticker not in self.last_model_update:
            return True
        
        time_since_update = dt.datetime.now() - self.last_model_update[ticker]
        return time_since_update.total_seconds() > self.model_refresh_interval_sec
    
    async def _get_current_price(self, ticker: str) -> Optional[float]:
        """Fetch current price for a ticker."""
        try:
            # For real-time data, use 1-minute interval and recent data
            fetcher = StockDataFetcher(ticker, period="1d", interval="1m")
            data = fetcher.fetch_data()
            
            if data is not None and len(data) > 0:
                return float(data['close'].iloc[-1])
            
            # Fallback to daily data if minute data unavailable
            fetcher = StockDataFetcher(ticker, period="5d", interval="1d")
            data = fetcher.fetch_data()
            
            if data is not None and len(data) > 0:
                return float(data['close'].iloc[-1])
                
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
        
        return None
    
    async def _make_prediction(self, ticker: str) -> Optional[PredictionResult]:
        """Make a prediction for a single ticker."""
        if ticker not in self.models:
            print(f"No model available for {ticker}")
            return None
        
        try:
            # Get current price
            current_price = await self._get_current_price(ticker)
            if current_price is None:
                return None
            
            model = self.models[ticker]
            last_data = self.last_data[ticker]
            
            # Prepare the latest data for prediction
            X_market, X_sentiment, y = model.prepare_data(last_data, target_col="close")
            
            if len(X_market) == 0:
                return None
            
            # Use the most recent sequence for prediction
            latest_market = X_market[-1:] 
            latest_sentiment = X_sentiment[-1:]
            
            # Make prediction (next hour approximation)
            prediction = model.predict(latest_market, latest_sentiment)
            pred_price = float(prediction[0])
            
            # Calculate confidence based on recent model performance
            # This is a simplified confidence measure
            recent_predictions = model.predict(X_market[-10:], X_sentiment[-10:])
            recent_actual = y[-10:]
            
            if len(recent_actual) > 0:
                errors = np.abs(recent_predictions.flatten() - recent_actual.flatten())
                mean_error = np.mean(errors)
                relative_error = mean_error / np.mean(recent_actual)
                confidence = max(0.0, min(1.0, 1.0 - relative_error))
            else:
                confidence = 0.5  # Default confidence
            
            # Determine trend
            price_change = pred_price - current_price
            if abs(price_change) < current_price * 0.005:  # Less than 0.5% change
                trend = 'stable'
            elif price_change > 0:
                trend = 'up'
            else:
                trend = 'down'
            
            # Calculate volatility (simplified)
            recent_prices = last_data['close'].tail(20)
            volatility = float(recent_prices.std() / recent_prices.mean()) if len(recent_prices) > 1 else 0.0
            
            return PredictionResult(
                ticker=ticker,
                current_price=current_price,
                pred_1h=pred_price,
                confidence=confidence,
                timestamp=dt.datetime.now(),
                trend=trend,
                volatility=volatility
            )
            
        except Exception as e:
            print(f"Error making prediction for {ticker}: {e}")
            return None
    
    async def _refresh_model_if_needed(self, ticker: str):
        """Refresh model if needed."""
        if await self._should_refresh_model(ticker):
            try:
                print(f"Refreshing model for {ticker}...")
                
                model, combined_df, _, _ = self.analyzer.run_analysis_for_stock(
                    ticker, period="3mo", interval="1d"
                )
                
                if model is not None and combined_df is not None:
                    self.models[ticker] = model
                    self.last_data[ticker] = combined_df
                    self.last_model_update[ticker] = dt.datetime.now()
                    print(f"✅ Model refreshed for {ticker}")
                
            except Exception as e:
                print(f"Error refreshing model for {ticker}: {e}")
    
    async def get_single_prediction(self, ticker: str) -> Optional[Dict]:
        """Get a single prediction for a ticker."""
        await self._refresh_model_if_needed(ticker)
        result = await self._make_prediction(ticker)
        
        if result:
            return {
                'ticker': result.ticker,
                'current_price': result.current_price,
                'pred_1h': result.pred_1h,
                'confidence': result.confidence,
                'timestamp': result.timestamp.isoformat(),
                'trend': result.trend,
                'volatility': result.volatility
            }
        return None
    
    async def stream_predictions(self) -> AsyncGenerator[Dict, None]:
        """
        Stream predictions for all configured tickers.
        
        Yields:
            Dict containing prediction data for each ticker
        """
        # Initialize models if not already done
        if not self.models:
            await self._initialize_models()
        
        print(f"Starting prediction stream for {self.tickers}")
        print(f"Update interval: {self.update_interval} seconds")
        
        while True:
            try:
                # Make predictions for all tickers
                for ticker in self.tickers:
                    if ticker in self.models:
                        # Refresh model if needed (in background)
                        asyncio.create_task(self._refresh_model_if_needed(ticker))
                        
                        # Make prediction
                        result = await self._make_prediction(ticker)
                        
                        if result:
                            yield {
                                'ticker': result.ticker,
                                'current_price': result.current_price,
                                'pred_1h': result.pred_1h,
                                'confidence': result.confidence,
                                'timestamp': result.timestamp.isoformat(),
                                'trend': result.trend,
                                'volatility': result.volatility
                            }
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in prediction stream: {e}")
                await asyncio.sleep(self.update_interval)
    
    def stop(self):
        """Stop the predictor (cleanup method)."""
        print("RealTimePredictor stopped")
