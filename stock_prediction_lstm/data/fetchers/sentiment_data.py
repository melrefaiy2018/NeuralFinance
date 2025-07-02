import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Enhanced sentiment analyzer with multiple data sources and better error handling
    """
    
    def __init__(self, ticker_symbol: str = 'NVDA'):
        self.ticker_symbol = ticker_symbol
        # Use relative import to avoid circular import issues
        try:
            from config.settings import Config
            self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
        except ImportError:
            # Fallback if config is not available
            self.alpha_vantage_key = None
        self.using_synthetic_data = False
        self.data_source = "unknown"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StockAnalyzer/1.0'
        })
    
    def fetch_news_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment data with multiple fallback options
        """
        print(f"Fetching sentiment data for {self.ticker_symbol} from {start_date} to {end_date}")
        
        methods = [
            ("Alpha Vantage API", self._fetch_alpha_vantage_sentiment),
            ("Yahoo Finance News", self._fetch_yahoo_news_sentiment),
            ("Price-based Synthetic", self._generate_price_based_sentiment),
            ("Random Synthetic", self._generate_random_sentiment)
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"Trying {method_name}...")
                result = method_func(start_date, end_date)
                
                if result is not None and len(result) > 0:
                    print(f"✅ Success with {method_name}")
                    self.data_source = method_name
                    if "Synthetic" in method_name:
                        self.using_synthetic_data = True
                    return result
                else:
                    print(f"❌ {method_name} returned no data")
                    
            except Exception as e:
                print(f"❌ {method_name} failed: {str(e)}")
                continue
        
        print("❌ All sentiment data methods failed")
        return None
    
    def _fetch_alpha_vantage_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from Alpha Vantage API with enhanced error handling
        """
        base_url = "https://www.alphavantage.co/query"
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.ticker_symbol,
            "apikey": self.alpha_vantage_key,
            "limit": 50
        }
        
        try:
            test_response = self._test_api_key()
            if not test_response:
                raise Exception("API key test failed")
            
            response = self.session.get(base_url, params=params, timeout=30)
            
            print(f"Alpha Vantage response status: {response.status_code}")
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            data = response.json()
            
            if "Error Message" in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if "Note" in data:
                raise Exception(f"Rate limited: {data['Note']}")
            
            if "Information" in data:
                raise Exception(f"API Info: {data['Information']}")
            
            if "feed" not in data:
                raise Exception(f"No 'feed' in response. Keys: {list(data.keys())}")
            
            if len(data["feed"]) == 0:
                raise Exception("Empty news feed")
            
            return self._process_alpha_vantage_data(data, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Alpha Vantage API failed: {str(e)}")
            raise e
    
    def _test_api_key(self) -> bool:
        """Test if the API key is valid"""
        try:
            test_url = "https://www.alphavantage.co/query"
            test_params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": "AAPL",
                "apikey": self.alpha_vantage_key
            }
            
            response = self.session.get(test_url, params=test_params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "Error Message" in data:
                    print(f"API key error: {data['Error Message']}")
                    return False
                elif "Note" in data:
                    print(f"API rate limited: {data['Note']}")
                    return False
                elif "Time Series (Daily)" in data:
                    print("API key is valid")
                    return True
                    
            return False
            
        except Exception as e:
            print(f"API key test failed: {str(e)}")
            return False
    
    def _process_alpha_vantage_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process Alpha Vantage sentiment data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.0,
            'sentiment_negative': 0.0,
            'sentiment_neutral': 0.0
        })
        
        articles_processed = 0
        
        for article in data["feed"]:
            try:
                article_date = pd.to_datetime(article["time_published"])
                
                if article_date < start_date or article_date > end_date:
                    continue
                
                ticker_sentiment = None
                for ts in article.get("ticker_sentiment", []):
                    if ts["ticker"] == self.ticker_symbol:
                        ticker_sentiment = ts
                        break
                
                if ticker_sentiment:
                    sentiment_score = float(ticker_sentiment["ticker_sentiment_score"])
                    sentiment_label = ticker_sentiment["ticker_sentiment_label"]
                    
                    article_date_only = article_date.date()
                    mask = sentiment_df['date'].dt.date == article_date_only
                    
                    if mask.any():
                        idx = mask.idxmax()
                        
                        if sentiment_label == "Bullish":
                            sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                        elif sentiment_label == "Bearish":
                            sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                        else:
                            sentiment_df.loc[idx, 'sentiment_neutral'] += abs(sentiment_score)
                        
                        articles_processed += 1
                        
            except Exception as e:
                logger.warning(f"Error processing article: {str(e)}")
                continue
        
        if articles_processed == 0:
            raise Exception("No valid articles processed")
        
        for i in range(len(sentiment_df)):
            total = (
                sentiment_df.loc[i, 'sentiment_positive'] + 
                sentiment_df.loc[i, 'sentiment_negative'] + 
                sentiment_df.loc[i, 'sentiment_neutral']
            )
            
            if total > 0:
                sentiment_df.loc[i, 'sentiment_positive'] /= total
                sentiment_df.loc[i, 'sentiment_negative'] /= total
                sentiment_df.loc[i, 'sentiment_neutral'] /= total
            else:
                sentiment_df.loc[i, 'sentiment_positive'] = 0.5
                sentiment_df.loc[i, 'sentiment_negative'] = 0.3
                sentiment_df.loc[i, 'sentiment_neutral'] = 0.2
        
        print(f"Processed {articles_processed} articles")
        return sentiment_df
    
    def _fetch_yahoo_news_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch news from Yahoo Finance and perform basic sentiment analysis
        """
        try:
            ticker = yf.Ticker(self.ticker_symbol)
            news = ticker.news
            
            if not news or len(news) == 0:
                raise Exception("No news articles found")
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            sentiment_df = pd.DataFrame({
                'date': date_range,
                'sentiment_positive': 0.5,
                'sentiment_negative': 0.3,
                'sentiment_neutral': 0.2
            })
            
            positive_words = ['bullish', 'positive', 'growth', 'gains', 'rising', 'strong', 'beat', 'exceed', 'profit', 'surge']
            negative_words = ['bearish', 'negative', 'decline', 'loss', 'falling', 'weak', 'miss', 'below', 'drop', 'crash']
            
            total_sentiment = 0
            article_count = 0
            
            for article in news:
                try:
                    title = article.get('title', '').lower()
                    if title:
                        pos_count = sum(1 for word in positive_words if word in title)
                        neg_count = sum(1 for word in negative_words if word in title)
                        
                        if pos_count > neg_count:
                            total_sentiment += 1
                        elif neg_count > pos_count:
                            total_sentiment -= 1
                        
                        article_count += 1
                        
                except Exception as e:
                    continue
            
            if article_count > 0:
                avg_sentiment = total_sentiment / article_count
                
                if avg_sentiment > 0.2:
                    pos_weight = 0.6 + min(0.2, avg_sentiment * 0.2)
                    neg_weight = 0.2
                elif avg_sentiment < -0.2:
                    pos_weight = 0.2
                    neg_weight = 0.6 + min(0.2, abs(avg_sentiment) * 0.2)
                else:
                    pos_weight = 0.5
                    neg_weight = 0.3
                
                neu_weight = 1.0 - pos_weight - neg_weight
                
                sentiment_df['sentiment_positive'] = pos_weight
                sentiment_df['sentiment_negative'] = neg_weight
                sentiment_df['sentiment_neutral'] = neu_weight
            
            print(f"Analyzed {article_count} Yahoo Finance news articles")
            return sentiment_df
            
        except Exception as e:
            raise Exception(f"Yahoo news sentiment failed: {str(e)}")
    
    def _generate_price_based_sentiment(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate sentiment based on price movements (more realistic than random)
        """
        try:
            ticker = yf.Ticker(self.ticker_symbol)
            hist_data = ticker.history(period="1y")
            
            if hist_data.empty:
                raise Exception("No price data available")
            
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
                recent_data = hist_data[hist_data['Date'] <= date].tail(10)
                
                if not recent_data.empty:
                    returns = recent_data['Close'].pct_change().dropna()
                    
                    if not returns.empty:
                        avg_return = returns.mean()
                        volatility = returns.std()
                        
                        if avg_return > 0.01:
                            pos_sentiment = min(0.8, 0.6 + avg_return * 5)
                            neg_sentiment = max(0.1, 0.3 - avg_return * 3)
                        elif avg_return < -0.01:
                            pos_sentiment = max(0.1, 0.3 + avg_return * 3)
                            neg_sentiment = min(0.8, 0.6 - avg_return * 5)
                        else:
                            pos_sentiment = 0.5
                            neg_sentiment = 0.3
                        
                        if volatility > 0.02:
                            neutral_sentiment = min(0.4, 0.2 + volatility * 5)
                        else:
                            neutral_sentiment = 0.2
                        
                        total = pos_sentiment + neg_sentiment + neutral_sentiment
                        sentiment_df.loc[i, 'sentiment_positive'] = pos_sentiment / total
                        sentiment_df.loc[i, 'sentiment_negative'] = neg_sentiment / total
                        sentiment_df.loc[i, 'sentiment_neutral'] = neutral_sentiment / total
                    else:
                        sentiment_df.loc[i, 'sentiment_positive'] = 0.5
                        sentiment_df.loc[i, 'sentiment_negative'] = 0.3
                        sentiment_df.loc[i, 'sentiment_neutral'] = 0.2
                else:
                    sentiment_df.loc[i, 'sentiment_positive'] = 0.5
                    sentiment_df.loc[i, 'sentiment_negative'] = 0.3
                    sentiment_df.loc[i, 'sentiment_neutral'] = 0.2
            
            print("Generated price-based synthetic sentiment")
            return sentiment_df
            
        except Exception as e:
            raise Exception(f"Price-based sentiment failed: {str(e)}")
    
    def _generate_random_sentiment(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate random sentiment as absolute last resort
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': np.random.normal(0.5, 0.1, len(date_range)),
            'sentiment_negative': np.random.normal(0.3, 0.08, len(date_range)),
            'sentiment_neutral': np.random.normal(0.2, 0.05, len(date_range))
        })
        
        for i in range(len(sentiment_df)):
            pos = max(0.05, sentiment_df.loc[i, 'sentiment_positive'])
            neg = max(0.05, sentiment_df.loc[i, 'sentiment_negative'])
            neu = max(0.05, sentiment_df.loc[i, 'sentiment_neutral'])
            
            total = pos + neg + neu
            sentiment_df.loc[i, 'sentiment_positive'] = pos / total
            sentiment_df.loc[i, 'sentiment_negative'] = neg / total
            sentiment_df.loc[i, 'sentiment_neutral'] = neu / total
        
        print("Generated random synthetic sentiment")
        return sentiment_df
