import os
import json
import pickle
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, Optional
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedStockDataManager:
    """
    Enhanced data storage manager that saves data to disk and only fetches new data when needed.
    Supports both market data and sentiment data with intelligent caching.
    """
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_directory()
        
        self.market_data_pattern = "{ticker}_{period}_{interval}_market.pkl"
        self.sentiment_data_pattern = "{ticker}_{start_date}_{end_date}_sentiment.pkl"
        self.metadata_pattern = "{ticker}_{data_type}_metadata.json"
        
        self._memory_cache = {}
        
    def ensure_cache_directory(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        for subdir in ['market_data', 'sentiment_data', 'metadata']:
            os.makedirs(os.path.join(self.cache_dir, subdir), exist_ok=True)
    
    def _get_cache_path(self, filename: str, data_type: str = 'market_data') -> str:
        return os.path.join(self.cache_dir, data_type, filename)
    
    def _save_metadata(self, ticker: str, data_type: str, metadata: Dict[str, Any]):
        filename = self.metadata_pattern.format(ticker=ticker, data_type=data_type)
        filepath = self._get_cache_path(filename, 'metadata')
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_metadata(self, ticker: str, data_type: str) -> Optional[Dict[str, Any]]:
        filename = self.metadata_pattern.format(ticker=ticker, data_type=data_type)
        filepath = self._get_cache_path(filename, 'metadata')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def _is_market_data_fresh(self, ticker: str, period: str, interval: str = '1d') -> bool:
        metadata = self._load_metadata(ticker, f"{period}_{interval}_market")
        
        if not metadata:
            logger.info(f"No metadata found for {ticker} market data")
            return False
        
        last_updated = datetime.fromisoformat(metadata['last_updated']).date()
        today = date.today()
        
        if last_updated < today:
            logger.info(f"Market data for {ticker} is outdated (last updated: {last_updated})")
            return False
        
        if metadata.get('period') != period or metadata.get('interval') != interval:
            logger.info(f"Period/interval mismatch for {ticker}: stored={metadata.get('period')}, requested={period}")
            return False
        
        filename = self.market_data_pattern.format(ticker=ticker, period=period, interval=interval)
        filepath = self._get_cache_path(filename, 'market_data')
        
        if not os.path.exists(filepath):
            logger.info(f"Market data file not found: {filepath}")
            return False
        
        return True
    
    def _is_sentiment_data_fresh(self, ticker: str, start_date: date, end_date: date) -> bool:
        metadata = self._load_metadata(ticker, f"{start_date}_{end_date}_sentiment")
        
        if not metadata:
            logger.info(f"No metadata found for {ticker} sentiment data")
            return False
        
        last_updated = datetime.fromisoformat(metadata['last_updated']).date()
        today = date.today()
        
        if last_updated < today:
            logger.info(f"Sentiment data for {ticker} is outdated (last updated: {last_updated})")
            return False
        
        cached_start = datetime.fromisoformat(metadata['start_date']).date()
        cached_end = datetime.fromisoformat(metadata['end_date']).date()
        
        if cached_start > start_date or cached_end < end_date:
            logger.info(f"Date range mismatch for {ticker} sentiment data")
            return False
        
        filename = self.sentiment_data_pattern.format(ticker=ticker, start_date=start_date, end_date=end_date)
        filepath = self._get_cache_path(filename, 'sentiment_data')
        
        if not os.path.exists(filepath):
            logger.info(f"Sentiment data file not found: {filepath}")
            return False
        
        return True
    
    def get_market_data(self, ticker: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{period}_{interval}_market"
        if cache_key in self._memory_cache:
            logger.info(f"Returning {ticker} market data from memory cache")
            return self._memory_cache[cache_key].copy()
        
        if self._is_market_data_fresh(ticker, period, interval):
            logger.info(f"Loading {ticker} market data from cache")
            filename = self.market_data_pattern.format(ticker=ticker, period=period, interval=interval)
            filepath = self._get_cache_path(filename, 'market_data')
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self._memory_cache[cache_key] = data.copy()
                return data
            except Exception as e:
                logger.error(f"Error loading cached market data for {ticker}: {e}")
        
        logger.info(f"Fetching fresh market data for {ticker} (period={period}, interval={interval})")
        try:
            from stock_prediction_lstm.data.fetchers import StockDataFetcher
            fetcher = StockDataFetcher(ticker, period, interval)
            data = fetcher.fetch_data()
            
            if data is not None and not data.empty:
                self._save_market_data(ticker, period, interval, data)
                self._memory_cache[cache_key] = data.copy()
                return data
            else:
                logger.error(f"No data returned for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None
    
    def _save_market_data(self, ticker: str, period: str, interval: str, data: pd.DataFrame):
        filename = self.market_data_pattern.format(ticker=ticker, period=period, interval=interval)
        filepath = self._get_cache_path(filename, 'market_data')
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        metadata = {
            'ticker': ticker,
            'period': period,
            'interval': interval,
            'last_updated': datetime.now().isoformat(),
            'record_count': len(data),
            'date_range': {
                'start': data['date'].min().isoformat() if 'date' in data.columns else None,
                'end': data['date'].max().isoformat() if 'date' in data.columns else None
            },
            'checksum': self._calculate_checksum(data)
        }
        
        self._save_metadata(ticker, f"{period}_{interval}_market", metadata)
        logger.info(f"Saved market data for {ticker} with {len(data)} records")
    
    def get_sentiment_data(self, ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{start_date}_{end_date}_sentiment"
        if cache_key in self._memory_cache:
            logger.info(f"Returning {ticker} sentiment data from memory cache")
            return self._memory_cache[cache_key].copy()
        
        if self._is_sentiment_data_fresh(ticker, start_date, end_date):
            logger.info(f"Loading {ticker} sentiment data from cache")
            filename = self.sentiment_data_pattern.format(ticker=ticker, start_date=start_date, end_date=end_date)
            filepath = self._get_cache_path(filename, 'sentiment_data')
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self._memory_cache[cache_key] = data.copy()
                return data
            except Exception as e:
                logger.error(f"Error loading cached sentiment data for {ticker}: {e}")
        
        logger.info(f"Fetching fresh sentiment data for {ticker} ({start_date} to {end_date})")
        try:
            from stock_prediction_lstm.data.fetchers import SentimentAnalyzer
            
            analyzer = SentimentAnalyzer(ticker)
            data = analyzer.fetch_news_sentiment(start_date, end_date)
            
            if data is not None and not data.empty:
                self._save_sentiment_data(ticker, start_date, end_date, data)
                self._memory_cache[cache_key] = data.copy()
                return data
            else:
                logger.error(f"No sentiment data returned for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error fetching sentiment data for {ticker}: {e}")
            return None
    
    def _save_sentiment_data(self, ticker: str, start_date: date, end_date: date, data: pd.DataFrame):
        filename = self.sentiment_data_pattern.format(ticker=ticker, start_date=start_date, end_date=end_date)
        filepath = self._get_cache_path(filename, 'sentiment_data')
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        metadata = {
            'ticker': ticker,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'last_updated': datetime.now().isoformat(),
            'record_count': len(data),
            'checksum': self._calculate_checksum(data)
        }
        
        self._save_metadata(ticker, f"{start_date}_{end_date}_sentiment", metadata)
        logger.info(f"Saved sentiment data for {ticker} with {len(data)} records")
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def clear_cache(self, ticker: Optional[str] = None, data_type: Optional[str] = None):
        if ticker is None:
            self._memory_cache.clear()
        else:
            keys_to_remove = [key for key in self._memory_cache.keys() if key.startswith(ticker)]
            for key in keys_to_remove:
                del self._memory_cache[key]
        
        data_types = [data_type] if data_type else ['market_data', 'sentiment_data', 'metadata']
        
        for dt in data_types:
            cache_path = os.path.join(self.cache_dir, dt)
            if os.path.exists(cache_path):
                for filename in os.listdir(cache_path):
                    if ticker is None or filename.startswith(ticker):
                        filepath = os.path.join(cache_path, filename)
                        try:
                            os.remove(filepath)
                            logger.info(f"Removed cached file: {filepath}")
                        except Exception as e:
                            logger.error(f"Error removing {filepath}: {e}")
        
        logger.info(f"Cache cleared for ticker: {ticker}, data_type: {data_type}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        info = {
            'cache_directory': self.cache_dir,
            'memory_cache_size': len(self._memory_cache),
            'disk_cache': {}
        }
        
        for data_type in ['market_data', 'sentiment_data', 'metadata']:
            cache_path = os.path.join(self.cache_dir, data_type)
            if os.path.exists(cache_path):
                files = os.listdir(cache_path)
                info['disk_cache'][data_type] = {
                    'file_count': len(files),
                    'files': files[:10]
                }
        
        return info
    
    def validate_cache_integrity(self) -> Dict[str, list[str]]:
        results = {'valid': [], 'invalid': [], 'missing_metadata': []}
        
        market_path = os.path.join(self.cache_dir, 'market_data')
        if os.path.exists(market_path):
            for filename in os.listdir(market_path):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(market_path, filename)
                    
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        
                        parts = filename.replace('_market.pkl', '').split('_')
                        if len(parts) >= 2:
                            ticker = parts[0]
                            period = parts[1] if len(parts) > 1 else 'unknown'
                            interval = parts[2] if len(parts) > 2 else 'unknown'
                            
                            metadata = self._load_metadata(ticker, f"{period}_{interval}_market")
                            
                            if metadata and 'checksum' in metadata:
                                current_checksum = self._calculate_checksum(data)
                                if current_checksum == metadata['checksum']:
                                    results['valid'].append(filepath)
                                else:
                                    results['invalid'].append(filepath)
                                    logger.warning(f"Checksum mismatch for {filepath}")
                            else:
                                results['missing_metadata'].append(filepath)
                    except Exception as e:
                        results['invalid'].append(filepath)
                        logger.error(f"Error validating {filepath}: {e}")
        
        return results

stock_data_manager = EnhancedStockDataManager()