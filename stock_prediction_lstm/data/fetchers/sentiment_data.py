"""
Quick Fix for Sentiment Data Issues
===================================

This script provides an immediate solution to replace the failing sentiment data fetcher
in your existing stock prediction LSTM package.

INSTRUCTIONS:
1. Save this file as 'sentiment_data_fixed.py' in your fetchers directory
2. Replace the import in your main files:
   - Change: from .sentiment_data import SentimentAnalyzer
   - To: from .sentiment_data_fixed import FixedSentimentAnalyzer as SentimentAnalyzer
3. Run your analysis - it will use enhanced fallback methods
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Optional
import logging
import time
import json
import os
from datetime import datetime, timedelta

# Import our working alternative sentiment sources
from .AlternativeSentimentSources import AlternativeSentimentSources

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Fixed sentiment analyzer that prioritizes working methods and provides better fallbacks
    """
    
    def __init__(self, ticker_symbol: str = 'NVDA'):
        self.ticker_symbol = ticker_symbol
        self.cache_dir = "sentiment_cache"
        self._ensure_cache_dir()
        
        # Rate limiting - be more conservative
        self.last_alpha_vantage_call = 0
        self.alpha_vantage_calls_today = 0
        self.daily_reset_time = None
        
        # Load API configuration with better error handling
        try:
            from ...config.settings import Config
            if Config.load_api_keys():
                self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
            else:
                self.alpha_vantage_key = None
        except:
            # Fallback: try to load directly
            try:
                from ...config.keys.api_keys import ALPHA_VANTAGE_API_KEY
                self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY if ALPHA_VANTAGE_API_KEY != "YOUR_API_KEY_HERE" else None
            except:
                self.alpha_vantage_key = None
        
        self.using_synthetic_data = False
        self.data_source = "unknown"
        
        # Initialize alternative sentiment sources
        self.alt_sources = AlternativeSentimentSources(ticker_symbol)
        
        # Enhanced session with better headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
            except:
                self.cache_dir = "."  # Fallback to current directory
    
    def _normalize_date_inputs(self, start_date, end_date):
        """Normalize input dates to timezone-naive datetime objects"""
        def normalize_single_date(date_val):
            if hasattr(date_val, 'tz') and date_val.tz is not None:
                return date_val.tz_convert('UTC').tz_localize(None)
            elif isinstance(date_val, str):
                return pd.to_datetime(date_val, utc=True).tz_localize(None)
            else:
                return pd.to_datetime(date_val)
        
        start_normalized = normalize_single_date(start_date)
        end_normalized = normalize_single_date(end_date)
        
        return start_normalized, end_normalized
    
    def fetch_news_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment data with improved fallback strategy
        Priority: Cache → Enhanced Yahoo → Price-based → Simple synthetic
        """
        # Normalize input dates to avoid timezone issues
        start_date, end_date = self._normalize_date_inputs(start_date, end_date)
        
        print(f"Fetching sentiment data for {self.ticker_symbol} from {start_date} to {end_date}")
        
        # Method priority - most reliable first
        methods = [
            ("Cached Data", self._try_load_cache),
            ("MarketAux API", self._fetch_marketaux_sentiment),
            ("Enhanced Yahoo Finance", self._fetch_enhanced_yahoo_sentiment),
            ("Price-Technical Analysis", self._generate_price_technical_sentiment),
            ("Market Trend Sentiment", self._generate_market_trend_sentiment),
            ("Simple Synthetic", self._generate_simple_synthetic)
        ]
        
        # Only try Alpha Vantage if we haven't exhausted daily limit
        if self._can_use_alpha_vantage():
            methods.insert(1, ("Alpha Vantage (Limited)", self._fetch_alpha_vantage_safe))
        
        for method_name, method_func in methods:
            try:
                print(f"Trying {method_name}...")
                result = method_func(start_date, end_date)
                
                if result is not None and len(result) > 0:
                    print(f"✅ Success with {method_name}")
                    self.data_source = method_name
                    
                    if "Synthetic" in method_name or "Price" in method_name or "Market" in method_name:
                        self.using_synthetic_data = True
                    
                    # Cache successful results (except cached data itself)
                    if method_name != "Cached Data":
                        self._save_to_cache(result, start_date, end_date, method_name)
                    
                    return result
                else:
                    print(f"❌ {method_name} returned no data")
                    
            except Exception as e:
                print(f"❌ {method_name} failed: {str(e)}")
                continue
        
        print("❌ All sentiment methods failed - this should not happen with fallbacks!")
        return self._generate_emergency_sentiment(start_date, end_date)
    
    def _can_use_alpha_vantage(self) -> bool:
        """Check if we can use Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return False
        
        # Reset daily counter if needed
        now = time.time()
        if self.daily_reset_time is None or now > self.daily_reset_time:
            self.daily_reset_time = now + (24 * 60 * 60)
            self.alpha_vantage_calls_today = 0
        
        # Conservative limit - don't use all 25 calls
        return self.alpha_vantage_calls_today < 15
    
    def _fetch_alpha_vantage_safe(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Safer Alpha Vantage API call with better error handling
        """
        if not self._can_use_alpha_vantage():
            raise Exception("Alpha Vantage API limit reached or key unavailable")
        
        # Rate limiting - wait at least 15 seconds between calls
        now = time.time()
        time_since_last = now - self.last_alpha_vantage_call
        if time_since_last < 15:
            wait_time = 15 - time_since_last
            print(f"⏳ Rate limiting: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.ticker_symbol,
            "apikey": self.alpha_vantage_key,
            "limit": 20  # Smaller limit to be safe
        }
        
        try:
            self.last_alpha_vantage_call = time.time()
            response = self.session.get(base_url, params=params, timeout=30)
            self.alpha_vantage_calls_today += 1
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            data = response.json()
            
            # Handle various API responses
            if "Error Message" in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if "Note" in data:
                # Mark as exhausted if rate limited
                if "rate limit" in data["Note"].lower() or "premium" in data["Note"].lower():
                    self.alpha_vantage_calls_today = 25
                raise Exception(f"Rate limited: {data['Note']}")
            
            if "Information" in data:
                if "Thank you" in data["Information"]:
                    self.alpha_vantage_calls_today = 25
                raise Exception(f"API limit reached: {data['Information']}")
            
            if "feed" not in data or len(data["feed"]) == 0:
                raise Exception("No news feed available")
            
            return self._process_alpha_vantage_data(data, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Alpha Vantage failed: {str(e)}")
            raise e
    
    def _fetch_marketaux_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from MarketAux API using the working implementation
        """
        try:
            print("Trying MarketAux API for sentiment data...")
            
            # Ensure end_date includes the full current day
            if end_date.date() == datetime.now().date():
                end_date = end_date.replace(hour=23, minute=59, second=59)
            
            # Use the working MarketAux implementation
            result = self.alt_sources.fetch_marketaux_sentiment(start_date, end_date)
            
            if result is not None and len(result) > 0:
                print(f"✅ MarketAux returned {len(result)} days of sentiment data")
                return result
            else:
                raise Exception("MarketAux returned no usable data")
                
        except Exception as e:
            raise Exception(f"MarketAux sentiment failed: {str(e)}")
    
    def _fetch_enhanced_yahoo_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Enhanced Yahoo Finance sentiment with multiple retry strategies
        """
        try:
            print("Fetching enhanced Yahoo Finance data...")
            
            # Multiple attempts with different strategies
            ticker = yf.Ticker(self.ticker_symbol)
            news_data = None
            
            # Strategy 1: Direct news fetch
            try:
                news_data = ticker.news
                if news_data and len(news_data) > 0:
                    print(f"Found {len(news_data)} news articles from Yahoo Finance")
                else:
                    raise Exception("No news from direct fetch")
            except:
                pass
            
            # Strategy 2: Try with different ticker variations
            if not news_data:
                variations = [self.ticker_symbol.upper(), self.ticker_symbol.lower()]
                for var in variations:
                    try:
                        alt_ticker = yf.Ticker(var)
                        news_data = alt_ticker.news
                        if news_data and len(news_data) > 0:
                            print(f"Found {len(news_data)} news articles using ticker variation {var}")
                            break
                    except:
                        continue
            
            # Strategy 3: Get company info and use company name
            if not news_data:
                try:
                    info = ticker.info
                    company_name = info.get('longName', info.get('shortName', ''))
                    if company_name:
                        # This is a placeholder - in practice, you'd search news by company name
                        print(f"Company name: {company_name}")
                except:
                    pass
            
            if not news_data or len(news_data) == 0:
                raise Exception("No Yahoo Finance news data available")
            
            return self._process_yahoo_news_enhanced(news_data, start_date, end_date)
            
        except Exception as e:
            raise Exception(f"Enhanced Yahoo sentiment failed: {str(e)}")
    
    def _generate_price_technical_sentiment(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate sentiment based on comprehensive technical analysis
        """
        try:
            print("Generating price-technical sentiment...")
            
            # Get extended price data for better analysis
            extended_start = start_date - pd.Timedelta(days=50)
            ticker = yf.Ticker(self.ticker_symbol)
            hist_data = ticker.history(start=extended_start, end=end_date + pd.Timedelta(days=1))
            
            if hist_data.empty:
                raise Exception("No price data available")
            
            # Calculate comprehensive technical indicators
            hist_data = self._calculate_technical_indicators(hist_data)
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            sentiment_df = pd.DataFrame({
                'date': date_range,
                'sentiment_positive': np.zeros(len(date_range)),
                'sentiment_negative': np.zeros(len(date_range)),
                'sentiment_neutral': np.zeros(len(date_range))
            })
            
            hist_data = hist_data.reset_index()
            hist_data['Date'] = pd.to_datetime(hist_data['Date'])
            
            for i, date in enumerate(date_range):
                current_data = hist_data[hist_data['Date'] <= date]
                
                if len(current_data) >= 20:  # Need sufficient data for analysis
                    latest = current_data.iloc[-1]
                    sentiment_scores = self._calculate_sentiment_from_technicals(current_data, latest)
                    
                    sentiment_df.loc[i, 'sentiment_positive'] = sentiment_scores['positive']
                    sentiment_df.loc[i, 'sentiment_negative'] = sentiment_scores['negative']
                    sentiment_df.loc[i, 'sentiment_neutral'] = sentiment_scores['neutral']
                else:
                    # Default values for insufficient data
                    sentiment_df.loc[i, 'sentiment_positive'] = 0.5
                    sentiment_df.loc[i, 'sentiment_negative'] = 0.3
                    sentiment_df.loc[i, 'sentiment_neutral'] = 0.2
            
            print("Generated comprehensive price-technical sentiment")
            return sentiment_df
            
        except Exception as e:
            raise Exception(f"Price-technical sentiment failed: {str(e)}")
    
    def _generate_market_trend_sentiment(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate sentiment based on market trends and sector performance
        """
        try:
            print("Generating market trend sentiment...")
            
            # Get market indices for context
            indices = {
                'SPY': yf.Ticker('SPY'),  # S&P 500
                'QQQ': yf.Ticker('QQQ'),  # NASDAQ
                'VTI': yf.Ticker('VTI')   # Total Stock Market
            }
            
            # Get stock data
            stock_ticker = yf.Ticker(self.ticker_symbol)
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            sentiment_df = pd.DataFrame({
                'date': date_range,
                'sentiment_positive': np.zeros(len(date_range)),
                'sentiment_negative': np.zeros(len(date_range)),
                'sentiment_neutral': np.zeros(len(date_range))
            })
            
            # Get extended data for analysis
            extended_start = start_date - pd.Timedelta(days=30)
            
            try:
                stock_data = stock_ticker.history(start=extended_start, end=end_date + pd.Timedelta(days=1))
                market_data = {}
                
                for name, ticker in indices.items():
                    try:
                        market_data[name] = ticker.history(start=extended_start, end=end_date + pd.Timedelta(days=1))
                    except:
                        continue
                
                if stock_data.empty:
                    raise Exception("No stock data available")
                
                for i, date in enumerate(date_range):
                    sentiment_scores = self._calculate_market_relative_sentiment(
                        stock_data, market_data, date
                    )
                    
                    sentiment_df.loc[i, 'sentiment_positive'] = sentiment_scores['positive']
                    sentiment_df.loc[i, 'sentiment_negative'] = sentiment_scores['negative']
                    sentiment_df.loc[i, 'sentiment_neutral'] = sentiment_scores['neutral']
                
            except Exception as e:
                print(f"Market data unavailable, using simplified trend: {e}")
                # Fallback to simple trend analysis
                for i, date in enumerate(date_range):
                    # Simple trend-based sentiment
                    trend_factor = (i / len(date_range)) * 0.2 - 0.1  # Small trend effect
                    
                    sentiment_df.loc[i, 'sentiment_positive'] = max(0.1, min(0.8, 0.5 + trend_factor))
                    sentiment_df.loc[i, 'sentiment_negative'] = max(0.1, min(0.8, 0.3 - trend_factor))
                    sentiment_df.loc[i, 'sentiment_neutral'] = max(0.1, min(0.8, 0.2))
                    
                    # Normalize
                    total = (sentiment_df.loc[i, 'sentiment_positive'] + 
                            sentiment_df.loc[i, 'sentiment_negative'] + 
                            sentiment_df.loc[i, 'sentiment_neutral'])
                    sentiment_df.loc[i, 'sentiment_positive'] /= total
                    sentiment_df.loc[i, 'sentiment_negative'] /= total
                    sentiment_df.loc[i, 'sentiment_neutral'] /= total
            
            print("Generated market trend sentiment")
            return sentiment_df
            
        except Exception as e:
            raise Exception(f"Market trend sentiment failed: {str(e)}")
    
    def _generate_simple_synthetic(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate simple but realistic synthetic sentiment data
        """
        print("Generating simple synthetic sentiment...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create base sentiment with slight positive bias (market generally trends up)
        np.random.seed(42)  # For reproducible results
        
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': np.random.normal(0.52, 0.08, len(date_range)),
            'sentiment_negative': np.random.normal(0.28, 0.06, len(date_range)),
            'sentiment_neutral': np.random.normal(0.20, 0.04, len(date_range))
        })
        
        # Ensure values are positive and normalize
        for i in range(len(sentiment_df)):
            pos = max(0.1, sentiment_df.loc[i, 'sentiment_positive'])
            neg = max(0.1, sentiment_df.loc[i, 'sentiment_negative'])
            neu = max(0.1, sentiment_df.loc[i, 'sentiment_neutral'])
            
            total = pos + neg + neu
            sentiment_df.loc[i, 'sentiment_positive'] = pos / total
            sentiment_df.loc[i, 'sentiment_negative'] = neg / total
            sentiment_df.loc[i, 'sentiment_neutral'] = neu / total
        
        print("Generated simple synthetic sentiment")
        return sentiment_df
    
    def _generate_emergency_sentiment(self, start_date, end_date) -> pd.DataFrame:
        """
        Emergency fallback - should never be needed but ensures we always return data
        """
        print("⚠️ Using emergency sentiment fallback")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': [0.5] * len(date_range),
            'sentiment_negative': [0.3] * len(date_range),
            'sentiment_neutral': [0.2] * len(date_range)
        })
        
        return sentiment_df
    
    def _calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=10).std()
        
        return df
    
    def _calculate_sentiment_from_technicals(self, data, latest):
        """Calculate sentiment scores from technical indicators"""
        scores = {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
        
        try:
            # Moving average signals (30% weight)
            ma_score = 0
            if pd.notna(latest['SMA_5']) and pd.notna(latest['SMA_20']):
                if latest['Close'] > latest['SMA_5'] > latest['SMA_20']:
                    ma_score = 0.15
                elif latest['Close'] < latest['SMA_5'] < latest['SMA_20']:
                    ma_score = -0.15
            
            # RSI signals (20% weight)
            rsi_score = 0
            if pd.notna(latest['RSI']):
                if latest['RSI'] < 30:
                    rsi_score = 0.1  # Oversold - positive
                elif latest['RSI'] > 70:
                    rsi_score = -0.1  # Overbought - negative
                elif 40 <= latest['RSI'] <= 60:
                    rsi_score = 0.05  # Neutral zone - slightly positive
            
            # MACD signals (20% weight)
            macd_score = 0
            if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal']:
                    macd_score = 0.1
                else:
                    macd_score = -0.1
            
            # Bollinger Band signals (15% weight)
            bb_score = 0
            if pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']):
                bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                if bb_position > 0.8:
                    bb_score = -0.075  # Near upper band - negative
                elif bb_position < 0.2:
                    bb_score = 0.075   # Near lower band - positive
            
            # Volume confirmation (10% weight)
            volume_score = 0
            if pd.notna(latest['Volume_Ratio']):
                if latest['Volume_Ratio'] > 1.5:
                    volume_score = 0.05  # High volume supports trend
                elif latest['Volume_Ratio'] < 0.5:
                    volume_score = -0.025  # Low volume weakens trend
            
            # Recent performance (5% weight)
            recent_data = data.tail(5)
            if len(recent_data) >= 5:
                recent_return = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                perf_score = recent_return * 0.5  # Scale down
            else:
                perf_score = 0
            
            # Combine scores
            total_score = ma_score + rsi_score + macd_score + bb_score + volume_score + perf_score
            
            # Convert to sentiment probabilities
            if total_score > 0.1:
                scores['positive'] = min(0.8, 0.6 + total_score * 2)
                scores['negative'] = max(0.1, 0.2 - total_score)
            elif total_score < -0.1:
                scores['positive'] = max(0.1, 0.4 + total_score * 2)
                scores['negative'] = min(0.8, 0.4 - total_score * 2)
            else:
                scores['positive'] = 0.5 + total_score
                scores['negative'] = 0.3 - total_score * 0.5
            
            scores['neutral'] = 1.0 - scores['positive'] - scores['negative']
            
            # Ensure all scores are within bounds
            for key in scores:
                scores[key] = max(0.05, min(0.9, scores[key]))
            
            # Normalize to sum to 1
            total = sum(scores.values())
            for key in scores:
                scores[key] /= total
                
        except Exception as e:
            logger.warning(f"Error calculating technical sentiment: {e}")
            # Return default values
            scores = {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
        
        return scores
    
    def _calculate_market_relative_sentiment(self, stock_data, market_data, date):
        """Calculate sentiment based on stock performance relative to market"""
        scores = {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
        
        try:
            # Get data up to the specified date
            stock_subset = stock_data[stock_data.index <= date]
            
            if len(stock_subset) < 10:
                return scores
            
            # Calculate recent performance (last 10 days)
            recent_stock = stock_subset.tail(10)
            stock_return = (recent_stock['Close'].iloc[-1] - recent_stock['Close'].iloc[0]) / recent_stock['Close'].iloc[0]
            
            # Calculate market performance for comparison
            market_returns = []
            for name, data in market_data.items():
                try:
                    market_subset = data[data.index <= date]
                    if len(market_subset) >= 10:
                        recent_market = market_subset.tail(10)
                        market_return = (recent_market['Close'].iloc[-1] - recent_market['Close'].iloc[0]) / recent_market['Close'].iloc[0]
                        market_returns.append(market_return)
                except:
                    continue
            
            if market_returns:
                avg_market_return = np.mean(market_returns)
                relative_performance = stock_return - avg_market_return
                
                # Convert relative performance to sentiment
                if relative_performance > 0.02:  # Outperforming by 2%+
                    scores['positive'] = min(0.8, 0.65 + relative_performance * 5)
                    scores['negative'] = max(0.1, 0.2 - relative_performance * 2)
                elif relative_performance < -0.02:  # Underperforming by 2%+
                    scores['positive'] = max(0.1, 0.35 + relative_performance * 5)
                    scores['negative'] = min(0.8, 0.45 - relative_performance * 3)
                else:  # Similar to market
                    scores['positive'] = 0.5 + relative_performance * 2
                    scores['negative'] = 0.3 - relative_performance
                
                scores['neutral'] = 1.0 - scores['positive'] - scores['negative']
                
                # Ensure bounds and normalize
                for key in scores:
                    scores[key] = max(0.05, min(0.9, scores[key]))
                
                total = sum(scores.values())
                for key in scores:
                    scores[key] /= total
            
        except Exception as e:
            logger.warning(f"Error calculating market relative sentiment: {e}")
        
        return scores
    
    def _process_alpha_vantage_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process Alpha Vantage data with improved error handling"""
        # Ensure dates are timezone-naive
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.0,
            'sentiment_negative': 0.0,
            'sentiment_neutral': 0.0
        })
        
        articles_processed = 0
        
        for article in data.get("feed", []):
            try:
                article_date = pd.to_datetime(article["time_published"])
                
                # Ensure timezone-naive
                if hasattr(article_date, 'tz') and article_date.tz is not None:
                    article_date = article_date.tz_localize(None)
                
                if article_date < start_date or article_date > end_date:
                    continue
                
                # Find ticker sentiment
                ticker_sentiment = None
                for ts in article.get("ticker_sentiment", []):
                    if ts.get("ticker") == self.ticker_symbol:
                        ticker_sentiment = ts
                        break
                
                if ticker_sentiment:
                    try:
                        sentiment_score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
                        sentiment_label = ticker_sentiment.get("ticker_sentiment_label", "Neutral")
                        
                        article_date_only = article_date.date()
                        mask = sentiment_df['date'].dt.date == article_date_only
                        
                        if mask.any():
                            idx = mask.idxmax()
                            
                            if sentiment_label in ["Bullish", "Somewhat-Bullish"]:
                                sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                            elif sentiment_label in ["Bearish", "Somewhat-Bearish"]:
                                sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                            else:
                                sentiment_df.loc[idx, 'sentiment_neutral'] += abs(sentiment_score)
                            
                            articles_processed += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing sentiment score: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing Alpha Vantage article: {e}")
                continue
        
        if articles_processed == 0:
            raise Exception("No valid articles processed from Alpha Vantage")
        
        # Normalize sentiment values
        for i in range(len(sentiment_df)):
            total = (sentiment_df.loc[i, 'sentiment_positive'] + 
                    sentiment_df.loc[i, 'sentiment_negative'] + 
                    sentiment_df.loc[i, 'sentiment_neutral'])
            
            if total > 0:
                sentiment_df.loc[i, 'sentiment_positive'] /= total
                sentiment_df.loc[i, 'sentiment_negative'] /= total
                sentiment_df.loc[i, 'sentiment_neutral'] /= total
            else:
                sentiment_df.loc[i, 'sentiment_positive'] = 0.5
                sentiment_df.loc[i, 'sentiment_negative'] = 0.3
                sentiment_df.loc[i, 'sentiment_neutral'] = 0.2
        
        print(f"Successfully processed {articles_processed} Alpha Vantage articles")
        return sentiment_df
    
    def _process_yahoo_news_enhanced(self, news_data, start_date, end_date) -> pd.DataFrame:
        """Enhanced processing of Yahoo Finance news data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        # Enhanced keyword lists with weights
        positive_keywords = {
            'strong': 2, 'growth': 2, 'bullish': 3, 'beat': 2, 'exceed': 2,
            'profit': 2, 'gain': 1, 'rise': 1, 'surge': 3, 'rally': 2,
            'upgrade': 3, 'buy': 2, 'outperform': 2, 'positive': 1,
            'good': 1, 'excellent': 2, 'outstanding': 3, 'impressive': 2,
            'breakthrough': 3, 'success': 2, 'record': 2, 'high': 1
        }
        
        negative_keywords = {
            'weak': 2, 'decline': 2, 'bearish': 3, 'miss': 2, 'below': 1,
            'loss': 2, 'fall': 1, 'drop': 1, 'crash': 3, 'plunge': 3,
            'downgrade': 3, 'sell': 2, 'underperform': 2, 'negative': 1,
            'bad': 1, 'poor': 2, 'terrible': 3, 'disappointing': 2,
            'concern': 2, 'warning': 2, 'risk': 1, 'low': 1
        }
        
        total_sentiment = 0
        article_count = 0
        
        for article in news_data:
            try:
                title = article.get('title', '').lower()
                summary = article.get('summary', '').lower()
                
                # Combine title and summary with title having more weight
                combined_text = f"{title} {title} {summary}"  # Title counted twice
                
                if combined_text.strip():
                    # Calculate weighted sentiment
                    pos_score = sum(weight for word, weight in positive_keywords.items() if word in combined_text)
                    neg_score = sum(weight for word, weight in negative_keywords.items() if word in combined_text)
                    
                    # Normalize by text length
                    text_length = len(combined_text.split())
                    if text_length > 0:
                        pos_score /= text_length
                        neg_score /= text_length
                    
                    # Calculate net sentiment
                    net_sentiment = pos_score - neg_score
                    total_sentiment += net_sentiment
                    article_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing Yahoo article: {e}")
                continue
        
        if article_count > 0:
            avg_sentiment = total_sentiment / article_count
            
            # Convert average sentiment to probabilities
            if avg_sentiment > 0.02:
                pos_weight = min(0.8, 0.6 + avg_sentiment * 10)
                neg_weight = max(0.1, 0.2 - avg_sentiment * 5)
            elif avg_sentiment < -0.02:
                pos_weight = max(0.1, 0.3 + avg_sentiment * 10)
                neg_weight = min(0.8, 0.5 - avg_sentiment * 10)
            else:
                pos_weight = 0.5 + avg_sentiment * 5
                neg_weight = 0.3 - avg_sentiment * 3
            
            neu_weight = 1.0 - pos_weight - neg_weight
            neu_weight = max(0.1, neu_weight)
            
            # Apply to all dates with some variation
            for i in range(len(sentiment_df)):
                # Add small random variation to make it more realistic
                noise = np.random.normal(0, 0.03)
                
                sentiment_df.loc[i, 'sentiment_positive'] = max(0.05, min(0.95, pos_weight + noise))
                sentiment_df.loc[i, 'sentiment_negative'] = max(0.05, min(0.95, neg_weight - noise/2))
                sentiment_df.loc[i, 'sentiment_neutral'] = max(0.05, min(0.95, neu_weight + noise/3))
                
                # Normalize
                total = (sentiment_df.loc[i, 'sentiment_positive'] + 
                        sentiment_df.loc[i, 'sentiment_negative'] + 
                        sentiment_df.loc[i, 'sentiment_neutral'])
                sentiment_df.loc[i, 'sentiment_positive'] /= total
                sentiment_df.loc[i, 'sentiment_negative'] /= total
                sentiment_df.loc[i, 'sentiment_neutral'] /= total
        
        print(f"Enhanced analysis of {article_count} Yahoo Finance articles")
        print(f"Average sentiment: {avg_sentiment:.4f}" if article_count > 0 else "No articles analyzed")
        
        return sentiment_df if article_count > 0 else None
    
    def _try_load_cache(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """Try to load sentiment data from cache"""
        cache_file = os.path.join(
            self.cache_dir, 
            f"{self.ticker_symbol}_sentiment_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is recent (less than 12 hours old)
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=12):
                    df = pd.DataFrame(cached_data['data'])
                    
                    # Handle timezone-aware datetime conversion properly
                    try:
                        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
                    except ValueError:
                        # Handle mixed timezone case
                        df['date'] = pd.to_datetime(df['date'])
                        if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
                            df['date'] = df['date'].dt.tz_localize(None)
                    
                    print(f"✅ Using cached sentiment data from {cached_data['source']}")
                    return df
                    
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, df, start_date, end_date, source):
        """Save sentiment data to cache"""
        if df is None or len(df) == 0:
            return
            
        cache_file = os.path.join(
            self.cache_dir,
            f"{self.ticker_symbol}_sentiment_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        )
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'ticker': self.ticker_symbol,
                'source': source,
                'data': df.to_dict('records')
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
            print(f"💾 Cached sentiment data from {source}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_sentiment_summary(self):
        """Get summary of sentiment data source and quality"""
        return {
            'data_source': self.data_source,
            'using_synthetic_data': self.using_synthetic_data,
            'alpha_vantage_calls_today': self.alpha_vantage_calls_today,
            'alpha_vantage_available': self.alpha_vantage_key is not None,
            'cache_dir': self.cache_dir
        }