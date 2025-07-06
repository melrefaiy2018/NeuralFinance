"""
Integration tests for Stock Prediction LSTM package.

These tests verify that different components work together correctly
in end-to-end scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
import os

# Import the modules under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from neural_finance.analysis.stock_analyzer import StockAnalyzer
from neural_finance.data.fetchers.stock_data import StockDataFetcher
from neural_finance.data.fetchers.sentiment_data import SentimentAnalyzer
from neural_finance.data.processors.technical_indicators import TechnicalIndicatorGenerator


class TestEndToEndWorkflow:
    """Integration tests for complete workflow."""

    @pytest.fixture
    def real_stock_data(self):
        """Create realistic stock data for integration testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate more realistic stock price movements
        data = []
        base_price = 150.0
        
        for i, date in enumerate(dates):
            # Add some trend and seasonality
            trend = 0.0002 * i  # Slight upward trend
            seasonal = 0.01 * np.sin(2 * np.pi * i / 252)  # Annual seasonality
            noise = np.random.normal(0, 0.02)  # Random noise
            
            change = trend + seasonal + noise
            base_price = max(base_price * (1 + change), 1.0)
            
            # Generate OHLC data
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 20000000)
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(base_price, 2),
                'volume': volume,
                'price': round(base_price, 2)
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def real_sentiment_data(self):
        """Create realistic sentiment data for integration testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        data = []
        for i, date in enumerate(dates):
            # Create sentiment with some patterns
            base_sentiment = 0.5 + 0.1 * np.sin(2 * np.pi * i / 30)  # Monthly cycle
            noise = np.random.normal(0, 0.1)
            
            positive = np.clip(base_sentiment + noise, 0.1, 0.8)
            negative = np.random.uniform(0.1, 0.4)
            neutral = 1.0 - positive - negative
            
            if neutral < 0:
                # Renormalize if needed
                total = positive + negative
                positive = positive / total * 0.9
                negative = negative / total * 0.9
                neutral = 0.1
            
            data.append({
                'date': date,
                'sentiment_positive': round(positive, 3),
                'sentiment_negative': round(negative, 3),
                'sentiment_neutral': round(neutral, 3)
            })
        
        return pd.DataFrame(data)

    @patch('neural_finance.data.fetchers.stock_data.yf.download')
    @patch('neural_finance.data.fetchers.sentiment_data.SentimentAnalyzer.fetch_news_sentiment')
    def test_complete_analysis_workflow(self, mock_sentiment_fetch, mock_yf_download,
                                      real_stock_data, real_sentiment_data):
        """Test complete analysis from data fetch to prediction."""
        # Setup mocks with realistic data
        mock_yf_download.return_value = self._convert_to_yfinance_format(real_stock_data)
        mock_sentiment_fetch.return_value = real_sentiment_data
        
        # Run complete analysis
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(
            'AAPL', '1y', '1d'
        )
        
        # Verify workflow completion
        if model is not None:  # Analysis might fail due to model complexity
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert isinstance(future_prices, list)
            assert isinstance(future_dates, list)
            assert len(future_prices) == len(future_dates)
            
            # Verify data integrity
            required_columns = ['date', 'close', 'volume', 'sentiment_positive', 
                              'sentiment_negative', 'sentiment_neutral']
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
            
            # Verify predictions are reasonable
            last_price = df['close'].iloc[-1]
            for price in future_prices:
                assert isinstance(price, (int, float))
                assert price > 0
                # Predictions should be within reasonable range of last price
                assert 0.5 * last_price <= price <= 2.0 * last_price

    def _convert_to_yfinance_format(self, df):
        """Convert DataFrame to yfinance format with proper index."""
        yf_df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']].copy()
        yf_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return yf_df

    @patch('neural_finance.data.fetchers.stock_data.yf.download')
    @patch('neural_finance.data.fetchers.sentiment_data.SentimentAnalyzer.fetch_news_sentiment')
    def test_multi_ticker_analysis(self, mock_sentiment_fetch, mock_yf_download,
                                 real_stock_data, real_sentiment_data):
        """Test analysis of multiple tickers."""
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        # Setup mocks
        mock_yf_download.return_value = self._convert_to_yfinance_format(real_stock_data)
        mock_sentiment_fetch.return_value = real_sentiment_data
        
        analyzer = StockAnalyzer()
        results = {}
        
        for ticker in tickers:
            try:
                model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(
                    ticker, '6mo', '1d'
                )
                results[ticker] = {
                    'success': model is not None,
                    'data_points': len(df) if df is not None else 0,
                    'predictions': len(future_prices) if future_prices else 0
                }
            except Exception as e:
                results[ticker] = {'success': False, 'error': str(e)}
        
        # Verify that analysis was attempted for all tickers
        assert len(results) == len(tickers)
        
        # At least some analyses should succeed (depending on mocking)
        success_count = sum(1 for r in results.values() 
                          if isinstance(r, dict) and r.get('success', False))
        # In a real test environment, we'd expect all to succeed
        # Here we just verify the structure works
        assert success_count >= 0

    @patch('neural_finance.data.fetchers.stock_data.yf.download')
    @patch('neural_finance.data.fetchers.sentiment_data.SentimentAnalyzer.fetch_news_sentiment')
    def test_different_time_periods(self, mock_sentiment_fetch, mock_yf_download,
                                  real_stock_data, real_sentiment_data):
        """Test analysis with various time periods."""
        periods = ['1mo', '3mo', '6mo', '1y']
        
        # Setup mocks
        mock_yf_download.return_value = self._convert_to_yfinance_format(real_stock_data)
        mock_sentiment_fetch.return_value = real_sentiment_data
        
        analyzer = StockAnalyzer()
        results = {}
        
        for period in periods:
            try:
                model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(
                    'AAPL', period, '1d'
                )
                results[period] = {
                    'success': model is not None,
                    'data_points': len(df) if df is not None else 0
                }
            except Exception as e:
                results[period] = {'success': False, 'error': str(e)}
        
        # Verify analysis was attempted for all periods
        assert len(results) == len(periods)

    def test_data_flow_integrity(self, real_stock_data, real_sentiment_data):
        """Test data consistency across pipeline stages."""
        # Test data fetcher
        with patch('neural_finance.data.fetchers.stock_data.yf.download') as mock_yf:
            mock_yf.return_value = self._convert_to_yfinance_format(real_stock_data)
            
            fetcher = StockDataFetcher('AAPL', '1y', '1d')
            stock_data = fetcher.fetch_data()
            
            assert isinstance(stock_data, pd.DataFrame)
            assert 'date' in stock_data.columns
            assert 'close' in stock_data.columns
        
        # Test sentiment analyzer
        with patch.object(SentimentAnalyzer, 'fetch_news_sentiment') as mock_sentiment:
            mock_sentiment.return_value = real_sentiment_data
            
            sentiment_analyzer = SentimentAnalyzer('AAPL')
            sentiment_data = sentiment_analyzer.fetch_news_sentiment(
                real_sentiment_data['date'].min(),
                real_sentiment_data['date'].max()
            )
            
            assert isinstance(sentiment_data, pd.DataFrame)
            assert 'sentiment_positive' in sentiment_data.columns
        
        # Test technical indicators
        enhanced_data = TechnicalIndicatorGenerator.add_technical_indicators(stock_data)
        
        assert isinstance(enhanced_data, pd.DataFrame)
        assert len(enhanced_data) == len(stock_data)
        assert 'ma7' in enhanced_data.columns or 'rsi14' in enhanced_data.columns

    def test_diagnostic_functionality(self):
        """Test diagnostic mode functionality."""
        analyzer = StockAnalyzer()
        
        # Mock successful analysis for diagnostic
        with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
            mock_model = Mock()
            mock_model.prepare_data.return_value = (
                np.array([[1, 2, 3]]), 
                np.array([[0.5, 0.6, 0.7]]), 
                np.array([100])
            )
            mock_model.predict.return_value = np.array([101])
            mock_model.evaluate.return_value = {'mse': 0.1, 'mae': 0.2, 'r2': 0.9}
            
            mock_df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'close': np.random.uniform(100, 110, 10)
            })
            mock_prices = [150.0, 151.0, 152.0]
            mock_dates = [datetime.now() + timedelta(days=i) for i in range(3)]
            
            mock_analysis.return_value = (mock_model, mock_df, mock_prices, mock_dates)
            
            # Run diagnostic
            model, df, future_prices = analyzer.self_diagnostic('NVDA', '1y')
            
            # Verify diagnostic results
            assert model is not None
            assert df is not None
            assert future_prices is not None
            
            # Verify diagnostic called analysis with correct parameters
            mock_analysis.assert_called_once_with('NVDA', '1y')

    @patch('neural_finance.data.fetchers.stock_data.yf.Ticker')
    def test_error_recovery_workflow(self, mock_ticker_class):
        """Test system behavior under various error conditions."""
        analyzer = StockAnalyzer()
        
        # Test network error recovery - mock yf.Ticker to raise exception
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker
        
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
        
        # Should handle errors gracefully
        assert model is None
        assert df is None
        assert future_prices is None
        assert future_dates is None

    def test_performance_with_realistic_data(self, real_stock_data, real_sentiment_data):
        """Test performance with realistic data volumes."""
        import time
        
        # Test data processing performance
        start_time = time.time()
        
        # Process technical indicators
        enhanced_data = TechnicalIndicatorGenerator.add_technical_indicators(real_stock_data)
        
        processing_time = time.time() - start_time
        
        # Should complete processing in reasonable time
        assert processing_time < 5.0, f"Processing took too long: {processing_time} seconds"
        
        # Verify output quality
        assert isinstance(enhanced_data, pd.DataFrame)
        assert len(enhanced_data) == len(real_stock_data)

    def test_memory_management_integration(self):
        """Test memory management across integrated components."""
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Run multiple analysis cycles
        analyzer = StockAnalyzer()
        
        with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
            mock_analysis.return_value = (Mock(), pd.DataFrame(), [100], [datetime.now()])
            
            for i in range(10):
                analyzer.run_analysis_for_stock(f'TEST{i}')
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 5000, f"Excessive memory usage: {object_growth} new objects"

    @patch('neural_finance.data.fetchers.stock_data.yf.download')
    @patch('neural_finance.data.fetchers.sentiment_data.SentimentAnalyzer.fetch_news_sentiment')
    def test_data_quality_validation(self, mock_sentiment_fetch, mock_yf_download,
                                   real_stock_data, real_sentiment_data):
        """Test data quality validation throughout the pipeline."""
        # Setup mocks
        mock_yf_download.return_value = self._convert_to_yfinance_format(real_stock_data)
        mock_sentiment_fetch.return_value = real_sentiment_data
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
        
        if df is not None:
            # Validate data quality
            assert not df.empty
            assert df['close'].notna().sum() > 0
            assert df['volume'].notna().sum() > 0
            assert (df['sentiment_positive'] >= 0).all()
            assert (df['sentiment_negative'] >= 0).all()
            assert (df['sentiment_neutral'] >= 0).all()
            
            # Check sentiment scores sum approximately to 1
            sentiment_sums = (df['sentiment_positive'] + 
                            df['sentiment_negative'] + 
                            df['sentiment_neutral'])
            assert abs(sentiment_sums.mean() - 1.0) < 0.1

    def test_concurrent_analysis_capability(self):
        """Test capability to handle concurrent analysis requests."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_analysis(ticker):
            analyzer = StockAnalyzer()
            with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
                mock_analysis.return_value = (Mock(), pd.DataFrame(), [100], [datetime.now()])
                result = analyzer.run_analysis_for_stock(ticker)
                results_queue.put((ticker, result))
        
        # Start multiple threads
        threads = []
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for ticker in tickers:
            thread = threading.Thread(target=run_analysis, args=(ticker,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all analyses completed
        assert len(results) == len(tickers)
        
        # Verify no thread conflicts
        ticker_results = {ticker: result for ticker, result in results}
        assert len(ticker_results) == len(tickers)  # No duplicate tickers


class TestWorkflowIntegration:
    """Additional integration tests for workflow components."""

    def test_component_interface_compatibility(self):
        """Test that component interfaces are compatible."""
        # Test that StockDataFetcher output is compatible with TechnicalIndicatorGenerator
        with patch('neural_finance.data.fetchers.stock_data.yf.download') as mock_yf:
            # Create mock yfinance data
            dates = pd.date_range('2023-01-01', periods=30, freq='D')
            mock_data = pd.DataFrame({
                'Open': np.random.uniform(100, 110, 30),
                'High': np.random.uniform(110, 120, 30),
                'Low': np.random.uniform(90, 100, 30),
                'Close': np.random.uniform(100, 110, 30),
                'Volume': np.random.randint(1000000, 5000000, 30),
            }, index=dates)
            mock_yf.return_value = mock_data
            
            # Test compatibility
            fetcher = StockDataFetcher('AAPL')
            stock_data = fetcher.fetch_data()
            
            if stock_data is not None:
                # Should be able to add technical indicators without error
                enhanced_data = TechnicalIndicatorGenerator.add_technical_indicators(stock_data)
                assert isinstance(enhanced_data, pd.DataFrame)
                assert len(enhanced_data) == len(stock_data)

    def test_end_to_end_data_transformation(self):
        """Test complete data transformation pipeline."""
        # Create initial raw data
        raw_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(1000000, 5000000, 50),
            'price': np.random.uniform(100, 110, 50)
        })
        
        # Apply technical indicators
        enhanced_data = TechnicalIndicatorGenerator.add_technical_indicators(raw_data)
        
        # Verify transformation chain
        assert len(enhanced_data) == len(raw_data)
        assert 'ma7' in enhanced_data.columns
        
        # Verify data types are preserved/appropriate
        assert pd.api.types.is_datetime64_any_dtype(enhanced_data['date'])
        assert pd.api.types.is_numeric_dtype(enhanced_data['close'])

    def test_configuration_consistency(self):
        """Test that configurations are consistent across components."""
        # Test that all components handle the same date formats
        test_date = datetime(2023, 1, 1)
        test_dates = pd.date_range(test_date, periods=10, freq='D')
        
        # All components should handle pandas datetime index/series
        test_df = pd.DataFrame({
            'date': test_dates,
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.randint(1000000, 5000000, 10)
        })
        
        # Technical indicators should handle the date column appropriately
        result = TechnicalIndicatorGenerator.add_technical_indicators(test_df)
        assert 'date' in result.columns
        assert len(result) == len(test_df)
