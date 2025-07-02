import pandas as pd
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib

class DataPersistence:
    """
    Handles persistent storage of stock data and related information
    """
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_directory()
        
        self.STOCK_DATA_EXPIRY = 1
        self.COMPANY_INFO_EXPIRY = 24
        self.SENTIMENT_DATA_EXPIRY = 6
        self.TECHNICAL_INDICATORS_EXPIRY = 1
        
    def ensure_cache_directory(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        subdirs = ['stock_data', 'company_info', 'sentiment_data', 'technical_indicators', 'models']
        for subdir in subdirs:
            path = os.path.join(self.cache_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def _generate_cache_key(self, ticker: str, period: str, interval: str = None) -> str:
        if interval:
            key_string = f"{ticker}_{period}_{interval}"
        else:
            key_string = f"{ticker}_{period}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_filepath(self, cache_type: str, cache_key: str, extension: str = ".pkl") -> str:
        return os.path.join(self.cache_dir, cache_type, f"{cache_key}{extension}")
    
    def _is_cache_valid(self, filepath: str, expiry_hours: int) -> bool:
        if not os.path.exists(filepath):
            return False
            
        file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        expiry_time = datetime.now() - timedelta(hours=expiry_hours)
        
        return file_modified > expiry_time
    
    def _save_with_metadata(self, data: Any, filepath: str, metadata: Dict = None):
        save_data = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def _load_with_metadata(self, filepath: str) -> tuple:
        try:
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
            
            data = saved_data.get('data')
            timestamp = saved_data.get('timestamp')
            metadata = saved_data.get('metadata', {})
            
            return data, timestamp, metadata
        except Exception as e:
            print(f"Error loading cached data from {filepath}: {e}")
            return None, None, None
    
    def save_stock_data(self, ticker: str, period: str, interval: str, df: pd.DataFrame):
        cache_key = self._generate_cache_key(ticker, period, interval)
        filepath = self._get_cache_filepath('stock_data', cache_key)
        
        metadata = {
            'ticker': ticker,
            'period': period,
            'interval': interval,
            'rows': len(df),
            'columns': list(df.columns)
        }
        
        self._save_with_metadata(df, filepath, metadata)
        print(f"Saved stock data for {ticker} ({period}, {interval}) to cache")
    
    def load_stock_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        cache_key = self._generate_cache_key(ticker, period, interval)
        filepath = self._get_cache_filepath('stock_data', cache_key)
        
        if self._is_cache_valid(filepath, self.STOCK_DATA_EXPIRY):
            data, timestamp, metadata = self._load_with_metadata(filepath)
            if data is not None:
                print(f"Loaded cached stock data for {ticker} (cached at {timestamp})")
                return data
        
        return None
    
    def save_company_info(self, ticker: str, info: Dict):
        cache_key = self._generate_cache_key(ticker, 'info')
        filepath = self._get_cache_filepath('company_info', cache_key, '.json')
        
        metadata = {
            'ticker': ticker,
            'keys': list(info.keys())
        }
        
        serializable_info = {}
        for key, value in info.items():
            try:
                json.dumps(value)
                serializable_info[key] = value
            except TypeError:
                serializable_info[key] = str(value)
        
        save_data = {
            'data': serializable_info,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Saved company info for {ticker} to cache")
    
    def load_company_info(self, ticker: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key(ticker, 'info')
        filepath = self._get_cache_filepath('company_info', cache_key, '.json')
        
        if self._is_cache_valid(filepath, self.COMPANY_INFO_EXPIRY):
            try:
                with open(filepath, 'r') as f:
                    saved_data = json.load(f)
                
                timestamp = saved_data.get('timestamp')
                data = saved_data.get('data', {})
                
                print(f"Loaded cached company info for {ticker} (cached at {timestamp})")
                return data
            except Exception as e:
                print(f"Error loading company info: {e}")
        
        return None
    
    def save_sentiment_data(self, ticker: str, period: str, df: pd.DataFrame, is_synthetic: bool = False):
        cache_key = self._generate_cache_key(ticker, period, 'sentiment')
        filepath = self._get_cache_filepath('sentiment_data', cache_key)
        
        metadata = {
            'ticker': ticker,
            'period': period,
            'is_synthetic': is_synthetic,
            'rows': len(df),
            'columns': list(df.columns)
        }
        
        self._save_with_metadata(df, filepath, metadata)
        print(f"Saved sentiment data for {ticker} ({period}) to cache (synthetic: {is_synthetic})")
    
    def load_sentiment_data(self, ticker: str, period: str) -> tuple:
        cache_key = self._generate_cache_key(ticker, period, 'sentiment')
        filepath = self._get_cache_filepath('sentiment_data', cache_key)
        
        if self._is_cache_valid(filepath, self.SENTIMENT_DATA_EXPIRY):
            data, timestamp, metadata = self._load_with_metadata(filepath)
            if data is not None:
                is_synthetic = metadata.get('is_synthetic', False)
                print(f"Loaded cached sentiment data for {ticker} (cached at {timestamp}, synthetic: {is_synthetic})")
                return data, is_synthetic
        
        return None, False
    
    def save_technical_indicators(self, ticker: str, period: str, interval: str, df: pd.DataFrame):
        cache_key = self._generate_cache_key(ticker, period, f"{interval}_tech")
        filepath = self._get_cache_filepath('technical_indicators', cache_key)
        
        metadata = {
            'ticker': ticker,
            'period': period,
            'interval': interval,
            'rows': len(df),
            'columns': list(df.columns)
        }
        
        self._save_with_metadata(df, filepath, metadata)
        print(f"Saved technical indicators for {ticker} to cache")
    
    def load_technical_indicators(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        cache_key = self._generate_cache_key(ticker, period, f"{interval}_tech")
        filepath = self._get_cache_filepath('technical_indicators', cache_key)
        
        if self._is_cache_valid(filepath, self.TECHNICAL_INDICATORS_EXPIRY):
            data, timestamp, metadata = self._load_with_metadata(filepath)
            if data is not None:
                print(f"Loaded cached technical indicators for {ticker} (cached at {timestamp})")
                return data
        
        return None
    
    def save_model(self, model, ticker: str, period: str, model_type: str = "lstm"):
        cache_key = self._generate_cache_key(ticker, period, model_type)
        
        model_dir = os.path.join(self.cache_dir, 'models', cache_key)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, 'model.h5')
        model.model.save(model_path)
        
        scalers = {
            'price_scaler': model.price_scaler,
            'market_scaler': model.market_scaler,
            'sentiment_scaler': model.sentiment_scaler
        }
        
        scalers_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        metadata = {
            'ticker': ticker,
            'period': period,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'look_back': model.look_back,
            'forecast_horizon': model.forecast_horizon
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved model for {ticker} ({period}) to cache")
    
    def load_model(self, model_class, ticker: str, period: str, model_type: str = "lstm"):
        cache_key = self._generate_cache_key(ticker, period, model_type)
        model_dir = os.path.join(self.cache_dir, 'models', cache_key)
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        model_path = os.path.join(model_dir, 'model.h5')
        scalers_path = os.path.join(model_dir, 'scalers.pkl')
        
        if (os.path.exists(model_path) and os.path.exists(scalers_path) and 
            self._is_cache_valid(metadata_path, 24)):
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model = model_class(
                    look_back=metadata.get('look_back', 20),
                    forecast_horizon=metadata.get('forecast_horizon', 1)
                )
                
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                
                model.price_scaler = scalers['price_scaler']
                model.market_scaler = scalers['market_scaler']
                model.sentiment_scaler = scalers['sentiment_scaler']
                
                from tensorflow.keras.models import load_model
                model.model = load_model(model_path)
                
                print(f"Loaded cached model for {ticker} (cached at {metadata['timestamp']})")
                return model
                
            except Exception as e:
                print(f"Error loading cached model: {e}")
        
        return None
    
    def get_cache_info(self) -> Dict:
        cache_info = {
            'stock_data': {},
            'company_info': {},
            'sentiment_data': {},
            'technical_indicators': {},
            'models': {}
        }
        
        for cache_type in cache_info.keys():
            cache_dir = os.path.join(self.cache_dir, cache_type)
            if os.path.exists(cache_dir):
                files = os.listdir(cache_dir)
                cache_info[cache_type] = {
                    'count': len(files),
                    'files': files
                }
        
        return cache_info
    
    def clear_cache(self, cache_type: str = None):
        if cache_type:
            cache_dir = os.path.join(self.cache_dir, cache_type)
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)
                print(f"Cleared {cache_type} cache")
        else:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            self.ensure_cache_directory()
            print("Cleared all cache")
    
    def clear_expired_cache(self):
        cache_types = [
            ('stock_data', self.STOCK_DATA_EXPIRY),
            ('company_info', self.COMPANY_INFO_EXPIRY),
            ('sentiment_data', self.SENTIMENT_DATA_EXPIRY),
            ('technical_indicators', self.TECHNICAL_INDICATORS_EXPIRY),
            ('models', 24)
        ]
        
        removed_count = 0
        for cache_type, expiry_hours in cache_types:
            cache_dir = os.path.join(self.cache_dir, cache_type)
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    filepath = os.path.join(cache_dir, filename)
                    if os.path.isfile(filepath) and not self._is_cache_valid(filepath, expiry_hours):
                        os.remove(filepath)
                        removed_count += 1
        
        print(f"Removed {removed_count} expired cache files")
        return removed_count
