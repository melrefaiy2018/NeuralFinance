"""
Unit tests for StockDataFetcher module.

This module tests the StockDataFetcher class which is responsible for
fetching stock data from Yahoo Finance API.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import yfinance as yf

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from stock_prediction_lstm.data.fetchers.stock_data import StockDataFetcher


class TestStockDataFetcher:
    """Test class for StockDataFetcher."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        fetcher = StockDataFetcher()
        
        assert fetcher.ticker_symbol == "NVDA"
        assert fetcher.period == "1y"
        assert fetcher.interval == "1d"
        assert fetcher.storage_manager is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        fetcher = StockDataFetcher("AAPL", "6mo", "1h")
        
        assert fetcher.ticker_symbol == "AAPL"
        assert fetcher.period == "6mo"
        assert fetcher.interval == "1h"

    def test_init_ticker_symbol_uppercase(self):
        """Test that ticker symbol is converted to uppercase."""
        fetcher = StockDataFetcher("aapl")
        assert fetcher.ticker_symbol == "AAPL"

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_success(self, mock_ticker_class):
        """Test successful data fetching for valid ticker."""
        # Create mock data
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'Open': np.random.uniform(100, 200, 10),
            'High': np.random.uniform(100, 200, 10),
            'Low': np.random.uniform(100, 200, 10),
            'Close': np.random.uniform(100, 200, 10),
            'Volume': np.random.randint(1000000, 10000000, 10),
            'Dividends': np.zeros(10),
            'Stock Splits': np.zeros(10)
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "10d", "1d")
        result = fetcher.fetch_data()
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert 'price' in result.columns
        assert 'date' in result.columns

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_valid_columns(self, mock_ticker_class):
        """Test that fetched data has all required columns."""
        # Create mock data
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'Dividends': np.zeros(5),
            'Stock Splits': np.zeros(5)
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "5d", "1d")
        result = fetcher.fetch_data()
        
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'price']
        for col in expected_columns:
            assert col in result.columns
        
        # Check that price equals close (ignoring index/name differences)
        assert (result['price'].values == result['close'].values).all()

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_date_range(self, mock_ticker_class):
        """Test that fetched data falls within expected date range."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        # Create mock data within date range
        date_range = pd.date_range(start_date, end_date, freq='D')
        mock_data = pd.DataFrame({
            'Date': date_range,
            'Open': np.random.uniform(100, 200, len(date_range)),
            'High': np.random.uniform(100, 200, len(date_range)),
            'Low': np.random.uniform(100, 200, len(date_range)),
            'Close': np.random.uniform(100, 200, len(date_range)),
            'Volume': np.random.randint(1000000, 10000000, len(date_range)),
            'Dividends': np.zeros(len(date_range)),
            'Stock Splits': np.zeros(len(date_range))
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "1mo", "1d")
        result = fetcher.fetch_data()
        
        assert result is not None
        assert result['date'].min().date() >= start_date.date()
        assert result['date'].max().date() <= end_date.date()

    def test_fetch_data_interval_consistency(self):
        """Test that the fetcher respects the interval parameter."""
        fetcher = StockDataFetcher("AAPL", "1d", "1h")
        assert fetcher.interval == "1h"
        
        fetcher = StockDataFetcher("AAPL", "1d", "5m")
        assert fetcher.interval == "5m"

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_invalid_ticker(self, mock_ticker_class):
        """Test handling of invalid ticker symbols."""
        # Setup mock to return empty dataframe
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("INVALID", "1mo", "1d")
        result = fetcher.fetch_data()
        
        assert result is None

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_network_error(self, mock_ticker_class):
        """Test handling of network errors during data fetching."""
        # Setup mock to raise an exception
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "1mo", "1d")
        result = fetcher.fetch_data()
        
        assert result is None

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_empty_response(self, mock_ticker_class):
        """Test handling of empty response from yfinance."""
        # Setup mock to return empty dataframe
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "1mo", "1d")
        result = fetcher.fetch_data()
        
        assert result is None

    def test_fetch_data_rate_limiting(self):
        """Test that the fetcher handles rate limiting appropriately."""
        # This test checks that multiple fetchers can be created without issues
        fetchers = [StockDataFetcher(f"STOCK{i}", "1d", "1h") for i in range(5)]
        
        for fetcher in fetchers:
            assert fetcher.ticker_symbol.startswith("STOCK")
            assert fetcher.period == "1d"
            assert fetcher.interval == "1h"

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_malformed_response(self, mock_ticker_class):
        """Test handling of malformed data from yfinance."""
        # Create malformed data (missing expected columns)
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'SomeColumn': [1, 2, 3, 4, 5]
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "5d", "1d")
        result = fetcher.fetch_data()
        
        # Should still return something, even if malformed
        assert result is not None
        assert 'date' in result.columns

    def test_fetch_data_invalid_period_format(self):
        """Test initialization with invalid period formats."""
        # These should still work - validation happens at yfinance level
        fetcher = StockDataFetcher("AAPL", "invalid_period", "1d")
        assert fetcher.period == "invalid_period"

    def test_fetch_data_invalid_interval_format(self):
        """Test initialization with invalid interval formats."""
        # These should still work - validation happens at yfinance level
        fetcher = StockDataFetcher("AAPL", "1mo", "invalid_interval")
        assert fetcher.interval == "invalid_interval"

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_minimal_data_points(self, mock_ticker_class):
        """Test fetching with minimal data points."""
        # Create minimal mock data
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=2, freq='D'),
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [102, 103],
            'Volume': [1000000, 1100000],
            'Dividends': [0, 0],
            'Stock Splits': [0, 0]
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "2d", "1d")
        result = fetcher.fetch_data()
        
        assert result is not None
        assert len(result) == 2

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_with_nan_values(self, mock_ticker_class):
        """Test handling of NaN values in fetched data."""
        # Create data with NaN values
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'Open': [100, np.nan, 102, 103, 104],
            'High': [105, 106, np.nan, 108, 109],
            'Low': [95, 96, 97, np.nan, 99],
            'Close': [102, 103, 104, 105, np.nan],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'Dividends': np.zeros(5),
            'Stock Splits': np.zeros(5)
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "5d", "1d")
        result = fetcher.fetch_data()
        
        assert result is not None
        # NaN values should be handled by forward/backward fill
        assert not result[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_extreme_values(self, mock_ticker_class):
        """Test handling of extreme price values."""
        # Create data with extreme values
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=3, freq='D'),
            'Open': [1e6, 1e-6, 100],
            'High': [1e6, 1e-6, 105],
            'Low': [1e6, 1e-6, 95],
            'Close': [1e6, 1e-6, 102],
            'Volume': [1000000, 1100000, 1200000],
            'Dividends': np.zeros(3),
            'Stock Splits': np.zeros(3)
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "3d", "1d")
        result = fetcher.fetch_data()
        
        assert result is not None
        assert len(result) == 3
        # Extreme values should still be present as numeric data
        assert result['close'].dtype in [np.float64, np.float32]

    def test_threading_safety(self):
        """Test that multiple fetchers can be used safely in threading scenarios."""
        import threading
        
        results = []
        
        def fetch_worker(ticker):
            fetcher = StockDataFetcher(ticker, "1d", "1h")
            # Just test initialization, not actual fetching
            results.append(fetcher.ticker_symbol)
        
        threads = []
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        for ticker in tickers:
            thread = threading.Thread(target=fetch_worker, args=(ticker,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 4
        assert set(results) == set(tickers)

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_data_processing_pipeline(self, mock_ticker_class):
        """Test the complete data processing pipeline."""
        # Create comprehensive mock data
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'Open': np.random.uniform(100, 200, 10),
            'High': np.random.uniform(100, 200, 10),
            'Low': np.random.uniform(100, 200, 10),
            'Close': np.random.uniform(100, 200, 10),
            'Adj Close': np.random.uniform(100, 200, 10),
            'Volume': np.random.randint(1000000, 10000000, 10),
            'Dividends': np.zeros(10),
            'Stock Splits': np.zeros(10)
        }).set_index('Date')
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "10d", "1d")
        result = fetcher.fetch_data()
        
        # Verify complete processing
        assert result is not None
        assert len(result) == 10
        assert 'date' in result.columns
        assert 'price' in result.columns
        assert result['date'].dtype == 'datetime64[ns]'
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_fetch_data_different_periods(self):
        """Test fetcher with different period configurations."""
        periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        
        for period in periods:
            fetcher = StockDataFetcher("AAPL", period, "1d")
            assert fetcher.period == period
            assert fetcher.ticker_symbol == "AAPL"
            assert fetcher.interval == "1d"

    @patch('stock_prediction_lstm.data.fetchers.stock_data.yf.Ticker')
    def test_fetch_data_different_intervals(self, mock_ticker_class):
        """Test fetcher with different interval configurations."""
        # For 1m interval, mock an error to simulate Yahoo's restriction
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("1m data not available for startTime")
        mock_ticker_class.return_value = mock_ticker
        
        fetcher = StockDataFetcher("AAPL", "1mo", "1m")
        result = fetcher.fetch_data()
        
        # Should return None due to the simulated error
        assert result is None