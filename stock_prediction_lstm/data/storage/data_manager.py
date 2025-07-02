import os
import json
import pandas as pd
from datetime import datetime, timedelta
import pickle
import sqlite3
from typing import Optional, Dict, Tuple
import logging

from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
from stock_prediction_lstm.data.processors import TechnicalIndicatorGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages persistent storage of stock data, sentiment data, and model results
    """
    
    def __init__(self, data_dir: str = "data_cache"):
        logger.info("Initializing DataManager...")
        self.data_dir = data_dir
        self.stock_data_dir = os.path.join(data_dir, "stock_data")
        self.sentiment_data_dir = os.path.join(data_dir, "sentiment_data")
        self.models_dir = os.path.join(data_dir, "models")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        self.db_file = os.path.join(data_dir, "stock_data.db")
        
        for directory in [self.data_dir, self.stock_data_dir, self.sentiment_data_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
        logger.debug("Data directories created successfully")
        
        self.default_refresh_intervals = {
            "stock_data": 1,
            "sentiment_data": 6,
            "company_info": 24,
            "model": 24
        }
        logger.debug("Default refresh intervals configured")
        
        self.metadata = self._load_metadata()
        logger.debug("Metadata loaded successfully")
        
        self._init_database()
        logger.debug("DataManager initialization complete")
    
    def _load_metadata(self) -> Dict:
        logger.debug(f"Loading metadata from {self.metadata_file}")
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    logger.debug("Metadata loaded successfully")
                    return metadata
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding metadata JSON: {e}. Creating new metadata.")
            except IOError as e:
                logger.warning(f"IO error reading metadata file: {e}. Creating new metadata.")
            except Exception as e:
                logger.warning(f"Unexpected error loading metadata: {e}. Creating new metadata.")
        
        logger.info("Creating new metadata file")
        new_metadata = {
            "last_updated": {},
            "data_info": {},
            "refresh_intervals": self.default_refresh_intervals.copy()
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(new_metadata, f, indent=2, default=str)
                logger.debug("New metadata file created successfully")
        except Exception as e:
            logger.error(f"Error creating new metadata file: {e}")
        
        return new_metadata
    
    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _get_file_path(self, data_type: str, identifier: str, extension: str = "csv") -> str:
        if data_type == "stock_data":
            return os.path.join(self.stock_data_dir, f"{identifier}.{extension}")
        elif data_type == "sentiment_data":
            return os.path.join(self.sentiment_data_dir, f"{identifier}_sentiment.{extension}")
        elif data_type == "model":
            return os.path.join(self.models_dir, f"{identifier}_model.pkl")
        elif data_type == "company_info":
            return os.path.join(self.stock_data_dir, f"{identifier}_info.json")
        else:
            return os.path.join(self.data_dir, f"{identifier}.{extension}")
    
    def _is_data_fresh(self, data_type: str, identifier: str) -> bool:
        key = f"{data_type}_{identifier}"
        
        if key not in self.metadata["last_updated"]:
            return False
        
        last_updated = datetime.fromisoformat(self.metadata["last_updated"][key])
        refresh_interval = self.metadata["refresh_intervals"].get(data_type, 24)
        
        if data_type == "stock_data":
            refresh_interval = self._get_dynamic_refresh_interval()
        
        time_diff = datetime.now() - last_updated
        return time_diff.total_seconds() < (refresh_interval * 3600)
    
    def _get_dynamic_refresh_interval(self) -> float:
        now = datetime.now()
        
        if now.weekday() < 5:
            if 9 <= now.hour < 16:
                return 0.25
            elif 16 <= now.hour < 20:
                return 1
        
        return 4
    
    def _init_database(self):
        logger.info("Initializing SQLite database...")
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            logger.debug("Creating tables if they don't exist...")
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    period TEXT,
                    interval TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, period, interval, date)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    date TEXT,
                    positive_score REAL,
                    negative_score REAL,
                    neutral_score REAL,
                    compound_score REAL,
                    headline TEXT,
                    is_synthetic INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date, headline)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    period TEXT,
                    interval TEXT,
                    date TEXT,
                    rsi14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    ma7 REAL,
                    ma14 REAL,
                    ma30 REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, period, interval, date)
                )
            ''')
            
            logger.debug("Creating indexes for faster queries...")
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_data (ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data (date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_data (ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_data (date)')
            
            conn.commit()
            logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_stock_data(self, ticker: str, period: str, interval: str, df: pd.DataFrame):
        try:
            identifier = f"{ticker}_{period}_{interval}"
            file_path = self._get_file_path("stock_data", identifier)
            
            df.to_csv(file_path, index=False)
            
            key = f"stock_data_{identifier}"
            self.metadata["last_updated"][key] = datetime.now().isoformat()
            self.metadata["data_info"][key] = {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "rows": len(df),
                "date_range": {
                    "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                    "end": df['date'].max().isoformat() if 'date' in df.columns else None
                }
            }
            
            self._save_metadata()
            logger.info(f"Saved stock data for {ticker} ({len(df)} rows)")
            
        except Exception as e:
            logger.error(f"Error saving stock data for {ticker}: {e}")
    
    def load_stock_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        identifier = f"{ticker}_{period}_{interval}"
        
        if not self._is_data_fresh("stock_data", identifier):
            return None
        
        file_path = self._get_file_path("stock_data", identifier)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded cached stock data for {ticker} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error loading stock data for {ticker}: {e}")
            return None
    
    def save_stock_data_to_db(self, ticker: str, period: str, interval: str, df: pd.DataFrame):
        try:
            if 'date' not in df.columns:
                logger.error(f"DataFrame for {ticker} doesn't have a date column")
                return False
            
            if isinstance(df['date'].iloc[0], pd.Timestamp):
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(self.db_file)
            
            sql_df = df.copy()
            if 'date' in sql_df.columns:
                if isinstance(sql_df['date'].iloc[0], pd.Timestamp):
                    sql_df['date'] = sql_df['date'].dt.strftime('%Y-%m-%d')
            
            if 'ticker' not in sql_df.columns:
                sql_df['ticker'] = ticker
            if 'period' not in sql_df.columns:
                sql_df['period'] = period
            if 'interval' not in sql_df.columns:
                sql_df['interval'] = interval
            
            required_columns = ['ticker', 'period', 'interval', 'date', 'open', 'high', 'low', 'close', 'volume']
            sql_df = sql_df[required_columns]
            
            sql_df.to_sql('stock_data', conn, if_exists='append', index=False, 
                          index_label='id', method='multi', chunksize=1000)
            
            key = f"stock_data_{ticker}_{period}_{interval}"
            self.metadata["last_updated"][key] = datetime.now().isoformat()
            self.metadata["data_info"][key] = {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "rows": len(df),
                "date_range": {
                    "start": df['date'].min() if 'date' in df.columns else None,
                    "end": df['date'].max() if 'date' in df.columns else None
                },
                "storage": "sqlite"
            }
            
            self._save_metadata()
            conn.commit()
            conn.close()
            
            logger.info(f"Saved stock data to SQLite for {ticker} ({len(df)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving stock data to SQLite for {ticker}: {e}")
            return False
    
    def load_stock_data_from_db(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        identifier = f"{ticker}_{period}_{interval}"
        
        if not self._is_data_fresh("stock_data", identifier):
            logger.info(f"Stock data for {ticker} is not fresh, needs to be fetched")
            return None
        
        try:
            conn = sqlite3.connect(self.db_file)
            
            query = f"""
                SELECT date, open, high, low, close, volume
                FROM stock_data
                WHERE ticker = ? AND period = ? AND interval = ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=(ticker, period, interval))
            conn.close()
            
            if len(df) == 0:
                logger.info(f"No data found in SQLite for {ticker}")
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded stock data from SQLite for {ticker} ({len(df)} rows)")
            return df
            
        except Exception as e:
            logger.error(f"Error loading stock data from SQLite for {ticker}: {e}")
            return None
    
    def save_sentiment_data(self, ticker: str, df: pd.DataFrame, using_synthetic: bool = False):
        try:
            identifier = f"{ticker}"
            file_path = self._get_file_path("sentiment_data", identifier)
            
            df.to_csv(file_path, index=False)
            
            key = f"sentiment_data_{identifier}"
            self.metadata["last_updated"][key] = datetime.now().isoformat()
            self.metadata["data_info"][key] = {
                "ticker": ticker,
                "rows": len(df),
                "using_synthetic": using_synthetic,
                "date_range": {
                    "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                    "end": df['date'].max().isoformat() if 'date' in df.columns else None
                }
            }
            
            self._save_metadata()
            logger.info(f"Saved sentiment data for {ticker} ({len(df)} rows)")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data for {ticker}: {e}")
    
    def load_sentiment_data(self, ticker: str) -> Optional[Tuple[pd.DataFrame, bool]]:
        if not self._is_data_fresh("sentiment_data", ticker):
            return None
        
        file_path = self._get_file_path("sentiment_data", ticker)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            key = f"sentiment_data_{ticker}"
            using_synthetic = self.metadata["data_info"].get(key, {}).get("using_synthetic", False)
            
            logger.info(f"Loaded cached sentiment data for {ticker} ({len(df)} rows)")
            return df, using_synthetic
            
        except Exception as e:
            logger.error(f"Error loading sentiment data for {ticker}: {e}")
            return None
    
    def save_sentiment_data_to_db(self, ticker: str, df: pd.DataFrame, using_synthetic: bool = False):
        try:
            if 'date' not in df.columns:
                logger.error(f"DataFrame for {ticker} doesn't have a date column")
                return False
            
            conn = sqlite3.connect(self.db_file)
            
            sql_df = df.copy()
            if 'date' in sql_df.columns:
                if isinstance(sql_df['date'].iloc[0], pd.Timestamp):
                    sql_df['date'] = sql_df['date'].dt.strftime('%Y-%m-%d')
            
            if 'ticker' not in sql_df.columns:
                sql_df['ticker'] = ticker
            
            sql_df['is_synthetic'] = 1 if using_synthetic else 0
            
            required_columns = ['ticker', 'date', 'positive_score', 'negative_score', 'neutral_score', 'compound_score', 'headline', 'is_synthetic']
            for col in required_columns:
                if col not in sql_df.columns:
                    if col in ['positive_score', 'negative_score', 'neutral_score', 'compound_score']:
                        sql_df[col] = 0.0
                    elif col == 'headline':
                        sql_df[col] = ''
            
            sql_df = sql_df[required_columns]
            
            sql_df.to_sql('sentiment_data', conn, if_exists='append', index=False, 
                          index_label='id', method='multi', chunksize=1000)
            
            key = f"sentiment_data_{ticker}"
            self.metadata["last_updated"][key] = datetime.now().isoformat()
            self.metadata["data_info"][key] = {
                "ticker": ticker,
                "rows": len(df),
                "using_synthetic": using_synthetic,
                "date_range": {
                    "start": df['date'].min() if 'date' in df.columns else None,
                    "end": df['date'].max() if 'date' in df.columns else None
                },
                "storage": "sqlite"
            }
            
            self._save_metadata()
            conn.commit()
            conn.close()
            
            logger.info(f"Saved sentiment data to SQLite for {ticker} ({len(df)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving sentiment data to SQLite for {ticker}: {e}")
            return False
    
    def load_sentiment_data_from_db(self, ticker: str) -> Optional[Tuple[pd.DataFrame, bool]]:
        if not self._is_data_fresh("sentiment_data", ticker):
            logger.info(f"Sentiment data for {ticker} is not fresh, needs to be fetched")
            return None
        
        try:
            conn = sqlite3.connect(self.db_file)
            
            query = f"""
                SELECT date, positive_score, negative_score, neutral_score, compound_score, headline, is_synthetic
                FROM sentiment_data
                WHERE ticker = ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            
            if len(df) == 0:
                logger.info(f"No sentiment data found in SQLite for {ticker}")
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            
            using_synthetic = bool(df['is_synthetic'].max())
            
            logger.info(f"Loaded sentiment data from SQLite for {ticker} ({len(df)} rows)")
            return df, using_synthetic
            
        except Exception as e:
            logger.error(f"Error loading sentiment data from SQLite for {ticker}: {e}")
            return None
    
    def save_company_info(self, ticker: str, info: Dict):
        try:
            file_path = self._get_file_path("company_info", ticker, "json")
            
            with open(file_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            
            key = f"company_info_{ticker}"
            self.metadata["last_updated"][key] = datetime.now().isoformat()
            
            self._save_metadata()
            logger.info(f"Saved company info for {ticker}")
            
        except Exception as e:
            logger.error(f"Error saving company info for {ticker}: {e}")
    
    def load_company_info(self, ticker: str) -> Optional[Dict]:
        if not self._is_data_fresh("company_info", ticker):
            return None
        
        file_path = self._get_file_path("company_info", ticker, "json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                info = json.load(f)
            
            logger.info(f"Loaded cached company info for {ticker}")
            return info
            
        except Exception as e:
            logger.error(f"Error loading company info for {ticker}: {e}")
            return None
    
    def save_model(self, identifier: str, model_data: Dict):
        try:
            file_path = self._get_file_path("model", identifier, "pkl")
            
            model_data['saved_at'] = datetime.now().isoformat()
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            key = f"model_{identifier}"
            self.metadata["last_updated"][key] = datetime.now().isoformat()
            
            self._save_metadata()
            logger.info(f"Saved model for {identifier}")
            
        except Exception as e:
            logger.error(f"Error saving model for {identifier}: {e}")
    
    def load_model(self, identifier: str) -> Optional[Dict]:
        if not self._is_data_fresh("model", identifier):
            return None
        
        file_path = self._get_file_path("model", identifier, "pkl")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info(f"Loaded cached model for {identifier}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model for {identifier}: {e}")
            return None
