import pandas as pd
import yfinance as yf
from typing import Optional
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class StockDataFetcher:
    """
    Connection-safe StockDataFetcher that prevents 'Connection already opened' errors
    """
    
    _yfinance_lock = threading.Lock()
    
    def __init__(self, ticker_symbol: str = 'NVDA', period: str = '1y', interval: str = '1d'):
        self.ticker_symbol = ticker_symbol.upper()
        self.period = period
        self.interval = interval
        
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.storage_manager = None
        try:
            from stock_prediction_lstm.data.storage.cache_manager import stock_data_manager
            self.storage_manager = stock_data_manager
        except:
            pass
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data with connection safety
        
        Returns:
        --------
        pd.DataFrame or None
            Stock data with columns: date, open, high, low, close, volume, price
        """
        if self.storage_manager:
            try:
                cached_data = self.storage_manager.get_market_data(
                    self.ticker_symbol, self.period, self.interval
                )
                if cached_data is not None:
                    print(f"Using cached data for {self.ticker_symbol}")
                    return cached_data
            except Exception as e:
                print(f"Cache unavailable: {e}")
        
        print(f"Fetching fresh data for {self.ticker_symbol}...")
        
        with self._yfinance_lock:
            try:
                df = self._fetch_with_retry()
                
                if df is not None and not df.empty:
                    df = self._process_dataframe(df)
                    
                    print(f"Successfully fetched {len(df)} data points for {self.ticker_symbol}")
                    
                    if self.storage_manager:
                        try:
                            self.storage_manager._save_market_data(
                                self.ticker_symbol, self.period, self.interval, df
                            )
                        except:
                            pass
                    
                    return df
                else:
                    print(f"No data found for {self.ticker_symbol}")
                    return None
                    
            except Exception as e:
                print(f"Error fetching stock data for {self.ticker_symbol}: {str(e)}")
                return None
    
    def _fetch_with_retry(self, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch data with retry logic and connection safety
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = attempt * 2
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                
                ticker = yf.Ticker(self.ticker_symbol, session=self.session)
                
                df = ticker.history(
                    period=self.period, 
                    interval=self.interval,
                    auto_adjust=True,
                    prepost=True,
                    threads=False
                )
                
                if not df.empty:
                    return df
                else:
                    print(f"Empty dataframe returned for {self.ticker_symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {self.ticker_symbol}: {str(e)}")
                
                if "connection" in str(e).lower() or "already opened" in str(e).lower():
                    time.sleep(5)
                
                if attempt == max_retries - 1:
                    try:
                        print("Trying fallback simple fetch...")
                        simple_ticker = yf.Ticker(self.ticker_symbol)
                        df = simple_ticker.history(period=self.period)
                        if not df.empty:
                            return df
                    except:
                        pass
                    raise e
        
        return None
    
    def close_session(self):
        """Close the requests session"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.close_session()
        except:
            pass
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the raw dataframe from yfinance"""
        df = df.reset_index()
        
        df.columns = [col.lower() for col in df.columns]
        
        if 'adj close' in df.columns:
            df['close'] = df['adj close']
        
        if 'close' in df.columns:
            df['price'] = df['close']
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        if 'date' in df.columns:
            df = df.drop_duplicates(subset=['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
