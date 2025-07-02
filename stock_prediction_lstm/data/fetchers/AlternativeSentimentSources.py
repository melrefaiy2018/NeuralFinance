import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Optional, Dict, List
import logging
import time
import json
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class AlternativeSentimentSources:
    """
    Collection of alternative sentiment data sources that are free or have generous free tiers
    """
    
    def __init__(self, ticker_symbol: str = 'NVDA'):
        self.ticker_symbol = ticker_symbol
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_marketaux_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from MarketAux API (Free tier: 100 requests/month)
        Register at: https://www.marketaux.com/
        """
        try:
            print("Fetching from MarketAux API...")
            
            # Note: Users need to register for their own API key
            # This is a demonstration of the API structure
            
            base_url = "https://api.marketaux.com/v1/news/all"
            
            # Example parameters (users need their own API key)
            params = {
                'symbols': self.ticker_symbol,
                'filter_entities': 'true',
                'language': 'en',
                'api_token': 'mvUFaafhrSygJ9rk5JIXG5R0eXUL9cJ1QVbIfr2h'  # Demo API key - may have limited usage
            }
            
            # Try to use the API key - let the API respond with errors if invalid
            # if params['api_token'] == 'mvUFaafhrSygJ9rk5JIXG5R0eXUL9cJ1QVbIfr2h':
            #     raise Exception("MarketAux API key required - register at https://www.marketaux.com/")
            
            response = self.session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_marketaux_data(data, start_date, end_date)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"MarketAux sentiment failed: {str(e)}")
    
    def fetch_finnhub_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from Finnhub API (Free tier: 60 calls/minute)
        Register at: https://finnhub.io/
        """
        try:
            print("Fetching from Finnhub API...")
            
            base_url = "https://finnhub.io/api/v1/company-news"
            
            # Convert dates to Unix timestamps
            start_timestamp = int(start_date.timestamp()) if hasattr(start_date, 'timestamp') else int(time.mktime(start_date.timetuple()))
            end_timestamp = int(end_date.timestamp()) if hasattr(end_date, 'timestamp') else int(time.mktime(end_date.timetuple()))
            
            params = {
                'symbol': self.ticker_symbol,
                'from': datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d'),
                'to': datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d'),
                'token': 'd0i1aehr01qji78r3rfgd0i1aehr01qji78r3rg0'  # Users must replace this
            }
            
            if params['token'] == 'd0i1aehr01qji78r3rfgd0i1aehr01qji78r3rg0':
                raise Exception("Finnhub API key required - register at https://finnhub.io/")
            
            response = self.session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_finnhub_data(data, start_date, end_date)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Finnhub sentiment failed: {str(e)}")
    
    def fetch_fmp_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from Financial Modeling Prep (Free tier: 250 requests/day)
        Register at: https://financialmodelingprep.com/
        """
        try:
            print("Fetching from Financial Modeling Prep API...")
            
            base_url = "https://financialmodelingprep.com/api/v3/stock_news"
            
            params = {
                'tickers': self.ticker_symbol,
                'limit': 50,
                'from': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                'to': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                'apikey': 'w1eG3QpcKtSdQLrxpNNgzdRaUfPreZ1L'  # Users must replace this
            }
            
            if params['apikey'] == 'w1eG3QpcKtSdQLrxpNNgzdRaUfPreZ1L':
                raise Exception("FMP API key required - register at https://financialmodelingprep.com/")
            
            response = self.session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_fmp_data(data, start_date, end_date)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"FMP sentiment failed: {str(e)}")
    
    def fetch_reddit_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from Reddit using PRAW (Reddit API)
        Requires Reddit API credentials
        """
        try:
            print("Fetching from Reddit API...")
            
            # Note: This requires praw library and Reddit API credentials
            # pip install praw
            
            try:
                import praw
            except ImportError:
                raise Exception("praw library required: pip install praw")
            
            # Users need to create Reddit app and get credentials
            reddit = praw.Reddit(
                client_id='YOUR_REDDIT_CLIENT_ID',
                client_secret='YOUR_REDDIT_CLIENT_SECRET',
                user_agent='StockSentimentAnalyzer/1.0'
            )
            
            # Search for posts about the stock
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting']
            posts_data = []
            
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Search for posts mentioning the ticker
                for post in subreddit.search(self.ticker_symbol, time_filter='month', limit=25):
                    post_date = datetime.fromtimestamp(post.created_utc)
                    
                    if self._is_date_in_range(post_date, start_date, end_date):
                        posts_data.append({
                            'date': post_date,
                            'title': post.title,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'text': post.selftext
                        })
            
            if not posts_data:
                raise Exception("No Reddit posts found")
            
            return self._process_reddit_data(posts_data, start_date, end_date)
            
        except Exception as e:
            raise Exception(f"Reddit sentiment failed: {str(e)}")
    
    def fetch_newsapi_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from NewsAPI (Free tier: 1000 requests/month)
        Register at: https://newsapi.org/
        """
        try:
            print("Fetching from NewsAPI...")
            
            base_url = "https://newsapi.org/v2/everything"
            
            params = {
                'q': f'{self.ticker_symbol} OR {self._get_company_name()}',
                'from': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                'to': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 50,
                'apiKey': 'd0e12a3161de46739c3c83274bb00e81'  # Users must replace this
            }
            
            if params['apiKey'] == 'd0e12a3161de46739c3c83274bb00e81':
                raise Exception("NewsAPI key required - register at https://newsapi.org/")
            
            response = self.session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_newsapi_data(data, start_date, end_date)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"NewsAPI sentiment failed: {str(e)}")
    
    def fetch_polygon_sentiment(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """
        Fetch sentiment from Polygon.io (Free tier: 5 calls/minute)
        Register at: https://polygon.io/
        """
        try:
            print("Fetching from Polygon.io API...")
            
            base_url = f"https://api.polygon.io/v2/reference/news"
            
            params = {
                'ticker': self.ticker_symbol,
                'published_utc.gte': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                'published_utc.lte': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                'order': 'desc',
                'limit': 50,
                'apikey': 'YOUR_POLYGON_API_KEY_HERE'  # Users must replace this
            }
            
            if params['apikey'] == 'YOUR_POLYGON_API_KEY_HERE':
                raise Exception("Polygon.io API key required - register at https://polygon.io/")
            
            response = self.session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_polygon_data(data, start_date, end_date)
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Polygon.io sentiment failed: {str(e)}")
    
    def _get_company_name(self) -> str:
        """Get company name for the ticker symbol"""
        company_names = {
            'NVDA': 'NVIDIA',
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NFLX': 'Netflix'
        }
        return company_names.get(self.ticker_symbol, self.ticker_symbol)
    
    def _normalize_datetime(self, dt):
        """Convert datetime to timezone-naive for consistent comparison"""
        if dt is None:
            return None
        
        if hasattr(dt, 'tz') and dt.tz is not None:
            return dt.tz_convert('UTC').tz_localize(None)
        return dt
    
    def _is_date_in_range(self, pub_date, start_date, end_date) -> bool:
        """Check if publication date is within the specified range, handling timezones"""
        try:
            pub_date_norm = self._normalize_datetime(pub_date)
            start_date_norm = self._normalize_datetime(start_date)
            end_date_norm = self._normalize_datetime(end_date)
            
            return start_date_norm <= pub_date_norm <= end_date_norm
        except Exception:
            return False
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Basic sentiment analysis using keyword matching
        Returns: float between -1 (very negative) and 1 (very positive)
        """
        if not text:
            return 0.0
        
        text = text.lower()
        
        positive_words = [
            'bullish', 'positive', 'growth', 'gains', 'rising', 'strong', 'beat', 
            'exceed', 'profit', 'surge', 'optimistic', 'upgrade', 'outperform',
            'buy', 'rally', 'boom', 'success', 'breakthrough', 'record', 'high',
            'excellent', 'great', 'good', 'impressive', 'outstanding', 'solid'
        ]
        
        negative_words = [
            'bearish', 'negative', 'decline', 'loss', 'falling', 'weak', 'miss', 
            'below', 'drop', 'crash', 'pessimistic', 'downgrade', 'underperform',
            'sell', 'plunge', 'slump', 'concern', 'warning', 'risk', 'low',
            'terrible', 'bad', 'poor', 'disappointing', 'concerning', 'weak'
        ]
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        # Calculate sentiment score
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (pos_count - neg_count) / max(total_words, 1)
        return max(-1.0, min(1.0, sentiment_score * 10))  # Scale and bound
    
    def _process_marketaux_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process MarketAux API response"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        if 'data' not in data:
            print("No 'data' field in MarketAux response")
            return sentiment_df
        
        articles = data['data']
        processed_count = 0
        
        for article in articles:
            try:
                # Extract sentiment from article
                title = article.get('title', '')
                description = article.get('description', '')
                
                combined_text = f"{title} {description}"
                sentiment_score = self._analyze_text_sentiment(combined_text)
                
                # Find entities related to our ticker
                entities = article.get('entities', [])
                relevant = any(entity.get('symbol') == self.ticker_symbol for entity in entities)
                
                if relevant and sentiment_score != 0:
                    # Get article date
                    pub_date = pd.to_datetime(article.get('published_at'))
                    
                    if self._is_date_in_range(pub_date, start_date, end_date):
                        # Update sentiment for this date
                        date_mask = sentiment_df['date'].dt.date == pub_date.date()
                        if date_mask.any():
                            idx = date_mask.idxmax()
                            
                            if sentiment_score > 0:
                                sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                            else:
                                sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                            
                            processed_count += 1
                            
            except Exception as e:
                logger.warning(f"Error processing MarketAux article: {e}")
                continue
        
        # Normalize sentiment values
        self._normalize_sentiment_dataframe(sentiment_df)
        
        print(f"Processed {processed_count} MarketAux articles")
        return sentiment_df
    
    def _process_finnhub_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process Finnhub API response"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        if not isinstance(data, list):
            return sentiment_df
        
        processed_count = 0
        
        for article in data:
            try:
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                
                combined_text = f"{headline} {summary}"
                sentiment_score = self._analyze_text_sentiment(combined_text)
                
                if sentiment_score != 0:
                    # Get article date
                    pub_timestamp = article.get('datetime')
                    if pub_timestamp:
                        pub_date = pd.to_datetime(pub_timestamp, unit='s')
                        
                        if self._is_date_in_range(pub_date, start_date, end_date):
                            date_mask = sentiment_df['date'].dt.date == pub_date.date()
                            if date_mask.any():
                                idx = date_mask.idxmax()
                                
                                if sentiment_score > 0:
                                    sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                                else:
                                    sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                                
                                processed_count += 1
                                
            except Exception as e:
                logger.warning(f"Error processing Finnhub article: {e}")
                continue
        
        self._normalize_sentiment_dataframe(sentiment_df)
        
        print(f"Processed {processed_count} Finnhub articles")
        return sentiment_df
    
    def _process_fmp_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process Financial Modeling Prep API response"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        if not isinstance(data, list):
            return sentiment_df
        
        processed_count = 0
        
        for article in data:
            try:
                title = article.get('title', '')
                text = article.get('text', '')
                
                combined_text = f"{title} {text}"
                sentiment_score = self._analyze_text_sentiment(combined_text)
                
                if sentiment_score != 0:
                    pub_date = pd.to_datetime(article.get('publishedDate'))
                    
                    if self._is_date_in_range(pub_date, start_date, end_date):
                        date_mask = sentiment_df['date'].dt.date == pub_date.date()
                        if date_mask.any():
                            idx = date_mask.idxmax()
                            
                            if sentiment_score > 0:
                                sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                            else:
                                sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                            
                            processed_count += 1
                            
            except Exception as e:
                logger.warning(f"Error processing FMP article: {e}")
                continue
        
        self._normalize_sentiment_dataframe(sentiment_df)
        
        print(f"Processed {processed_count} FMP articles")
        return sentiment_df
    
    def _process_reddit_data(self, posts_data, start_date, end_date) -> pd.DataFrame:
        """Process Reddit posts data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        processed_count = 0
        
        for post in posts_data:
            try:
                title = post.get('title', '')
                text = post.get('text', '')
                score = post.get('score', 0)
                
                combined_text = f"{title} {text}"
                sentiment_score = self._analyze_text_sentiment(combined_text)
                
                # Weight by Reddit score (upvotes - downvotes)
                if score > 0:
                    sentiment_score *= min(2.0, 1 + (score / 100))
                
                if sentiment_score != 0:
                    pub_date = post.get('date')
                    
                    if self._is_date_in_range(pub_date, start_date, end_date):
                        date_mask = sentiment_df['date'].dt.date == pub_date.date()
                        if date_mask.any():
                            idx = date_mask.idxmax()
                            
                            if sentiment_score > 0:
                                sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                            else:
                                sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                            
                            processed_count += 1
                            
            except Exception as e:
                logger.warning(f"Error processing Reddit post: {e}")
                continue
        
        self._normalize_sentiment_dataframe(sentiment_df)
        
        print(f"Processed {processed_count} Reddit posts")
        return sentiment_df
    
    def _process_newsapi_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process NewsAPI response"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        articles = data.get('articles', [])
        processed_count = 0
        
        for article in articles:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                
                combined_text = f"{title} {description}"
                sentiment_score = self._analyze_text_sentiment(combined_text)
                
                if sentiment_score != 0:
                    pub_date = pd.to_datetime(article.get('publishedAt'))
                    
                    if self._is_date_in_range(pub_date, start_date, end_date):
                        date_mask = sentiment_df['date'].dt.date == pub_date.date()
                        if date_mask.any():
                            idx = date_mask.idxmax()
                            
                            if sentiment_score > 0:
                                sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                            else:
                                sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                            
                            processed_count += 1
                            
            except Exception as e:
                logger.warning(f"Error processing NewsAPI article: {e}")
                continue
        
        self._normalize_sentiment_dataframe(sentiment_df)
        
        print(f"Processed {processed_count} NewsAPI articles")
        return sentiment_df
    
    def _process_polygon_data(self, data, start_date, end_date) -> pd.DataFrame:
        """Process Polygon.io API response"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame({
            'date': date_range,
            'sentiment_positive': 0.5,
            'sentiment_negative': 0.3,
            'sentiment_neutral': 0.2
        })
        
        results = data.get('results', [])
        processed_count = 0
        
        for article in results:
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                
                combined_text = f"{title} {description}"
                sentiment_score = self._analyze_text_sentiment(combined_text)
                
                if sentiment_score != 0:
                    pub_date = pd.to_datetime(article.get('published_utc'))
                    
                    if self._is_date_in_range(pub_date, start_date, end_date):
                        date_mask = sentiment_df['date'].dt.date == pub_date.date()
                        if date_mask.any():
                            idx = date_mask.idxmax()
                            
                            if sentiment_score > 0:
                                sentiment_df.loc[idx, 'sentiment_positive'] += abs(sentiment_score)
                            else:
                                sentiment_df.loc[idx, 'sentiment_negative'] += abs(sentiment_score)
                            
                            processed_count += 1
                            
            except Exception as e:
                logger.warning(f"Error processing Polygon article: {e}")
                continue
        
        self._normalize_sentiment_dataframe(sentiment_df)
        
        print(f"Processed {processed_count} Polygon articles")
        return sentiment_df
    
    def _normalize_sentiment_dataframe(self, sentiment_df):
        """Normalize sentiment values to sum to 1.0 for each day"""
        for i in range(len(sentiment_df)):
            total = (sentiment_df.loc[i, 'sentiment_positive'] + 
                    sentiment_df.loc[i, 'sentiment_negative'] + 
                    sentiment_df.loc[i, 'sentiment_neutral'])
            
            if total > 0:
                sentiment_df.loc[i, 'sentiment_positive'] /= total
                sentiment_df.loc[i, 'sentiment_negative'] /= total
                sentiment_df.loc[i, 'sentiment_neutral'] /= total
            else:
                # Default neutral sentiment
                sentiment_df.loc[i, 'sentiment_positive'] = 0.5
                sentiment_df.loc[i, 'sentiment_negative'] = 0.3
                sentiment_df.loc[i, 'sentiment_neutral'] = 0.2


def get_api_registration_guide():
    """
    Returns a guide for registering with alternative sentiment data APIs
    """
    guide = """
    🔑 FREE SENTIMENT DATA API REGISTRATION GUIDE
    =============================================
    
    1. 📰 MarketAux (100 requests/month free)
       - URL: https://www.marketaux.com/
       - Features: Financial news with sentiment scores
       - Setup: Register → Get API token
    
    2. 📊 Finnhub (60 calls/minute free)
       - URL: https://finnhub.io/
       - Features: Company news, earnings, insider trading
       - Setup: Register → Get API key
    
    3. 💰 Financial Modeling Prep (250 requests/day free)
       - URL: https://financialmodelingprep.com/
       - Features: Stock news, financial statements, ratings
       - Setup: Register → Get API key
    
    4. 📱 NewsAPI (1000 requests/month free)
       - URL: https://newsapi.org/
       - Features: Global news from 80,000+ sources
       - Setup: Register → Get API key
    
    5. 📈 Polygon.io (5 calls/minute free)
       - URL: https://polygon.io/
       - Features: Market data, news, real-time feeds
       - Setup: Register → Get API key
    
    6. 🔴 Reddit API (Free with rate limits)
       - URL: https://www.reddit.com/prefs/apps
       - Features: Social sentiment from Reddit communities
       - Setup: Create Reddit app → Get client credentials
    
    7. 📊 StockGeist (Free tier available)
       - URL: https://www.stockgeist.ai/
       - Features: Social media sentiment analysis
       - Setup: Register → Get API access
    
    📝 SETUP INSTRUCTIONS:
    =====================
    
    1. Choose 2-3 APIs from the list above
    2. Register for free accounts
    3. Get your API keys/tokens
    4. Update the API key placeholders in the code
    5. Test with small requests first
    
    ⚠️  IMPORTANT NOTES:
    ===================
    
    - Always respect rate limits
    - Cache data to minimize API calls
    - Have fallback methods ready
    - Monitor your usage quotas
    - Keep API keys secure (never commit to version control)
    
    🆓 NO-API ALTERNATIVES:
    ======================
    
    - Web scraping (respect robots.txt)
    - Price-based sentiment generation
    - Technical indicator-based sentiment
    - Cached/historical sentiment data
    """
    
    return guide


def create_api_key_template():
    """
    Creates a template file for users to add their API keys
    """
    template = """
# Alternative Sentiment API Keys
# Replace the placeholder values with your actual API keys

# MarketAux API
MARKETAUX_API_KEY = "mvUFaafhrSygJ9rk5JIXG5R0eXUL9cJ1QVbIfr2h"

# Finnhub API  
FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY_HERE"

# Financial Modeling Prep API
FMP_API_KEY = "YOUR_FMP_API_KEY_HERE"

# NewsAPI
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"

# Polygon.io API
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY_HERE"

# Reddit API
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"

# Instructions:
# 1. Get free API keys from the providers above
# 2. Replace the placeholder values with your actual keys
# 3. Save this file as 'alternative_api_keys.py'
# 4. Import the keys in your sentiment analyzer
"""
    
    return template


# Example usage and testing
def test_alternative_apis():
    """
    Test alternative sentiment API sources
    """
    from datetime import datetime, timedelta
    
    print("🔍 Testing Alternative Sentiment APIs")
    print("=" * 50)
    
    # Show registration guide
    print(get_api_registration_guide())
    
    # Initialize alternative sources
    alt_sources = AlternativeSentimentSources('NVDA')
    
    # Test date range
    end_date = datetime.now().replace(hour=23, minute=59, second=59)  # End of today
    start_date = end_date - timedelta(days=7)
    
    # List of methods to try
    methods = [
        ("MarketAux", alt_sources.fetch_marketaux_sentiment),
        # ("Finnhub", alt_sources.fetch_finnhub_sentiment),
        # ("Financial Modeling Prep", alt_sources.fetch_fmp_sentiment),
        # ("NewsAPI", alt_sources.fetch_newsapi_sentiment),
        # ("Polygon.io", alt_sources.fetch_polygon_sentiment),
    ]
    
    results = {}
    
    for name, method in methods:
        try:
            print(f"\n🧪 Testing {name}...")
            result = method(start_date, end_date)
            
            if result is not None and len(result) > 0:
                results[name] = result
                print(f"✅ {name}: {len(result)} data points")
                
                # Show sample sentiment
                avg_pos = result['sentiment_positive'].mean()
                avg_neg = result['sentiment_negative'].mean()
                avg_neu = result['sentiment_neutral'].mean()
                
                print(f"   📊 Avg sentiment - Pos: {avg_pos:.3f}, Neg: {avg_neg:.3f}, Neu: {avg_neu:.3f}")
            else:
                print(f"❌ {name}: No data returned")
                
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
    
    if results:
        print(f"\n✅ Successfully tested {len(results)} alternative APIs!")
        print("💡 Consider registering for API keys to enable these sources")
    else:
        print("\n⚠️  No alternative APIs returned data (API keys needed)")
        print("📝 Follow the registration guide above to set up free API access")
    
    return results


if __name__ == "__main__":
    test_alternative_apis()