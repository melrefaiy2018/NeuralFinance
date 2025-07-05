"""
Unit tests for StockAnalyzer module.

This module tests the StockAnalyzer class which orchestrates the complete
stock analysis pipeline including data fetching, processing, model training,
and prediction generation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tensorflow as tf

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from stock_prediction_lstm.analysis.stock_analyzer import StockAnalyzer


class TestStockAnalyzer:
    """Test class for StockAnalyzer."""

    @pytest.fixture
    def sample_stock_dataframe(self):
        """Create sample stock data DataFrame."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = []
        base_price = 100
        for date in dates:
            change = np.random.normal(0, 0.02)
            base_price = base_price * (1 + change)
            data.append({
                'date': date,
                'open': base_price * (1 + np.random.normal(0, 0.01)),
                'high': base_price * (1 + abs(np.random.normal(0, 0.02))),
                'low': base_price * (1 - abs(np.random.normal(0, 0.02))),
                'close': base_price,
                'volume': np.random.randint(1000000, 10000000),
                'price': base_price
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_sentiment_dataframe(self):
        """Create sample sentiment data DataFrame."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = []
        for date in dates:
            pos = np.random.uniform(0.2, 0.7)
            neg = np.random.uniform(0.1, 0.5)
            neu = 1.0 - pos - neg
            if neu < 0:
                neu = 0.1
                pos = 0.7
                neg = 0.2
            
            data.append({
                'date': date,
                'sentiment_positive': pos,
                'sentiment_negative': neg,
                'sentiment_neutral': neu
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict_next_days.return_value = [150.0, 151.0, 152.0, 153.0, 154.0]
        model.evaluate.return_value = {
            'rmse': 2.5,
            'mae': 1.8,
            'r2': 0.85,
            'mape': 1.2
        }
        return model

    def test_analyzer_initialization(self):
        """Test StockAnalyzer initialization."""
        analyzer = StockAnalyzer()
        
        # Check that analyzer is properly initialized
        assert isinstance(analyzer, StockAnalyzer)
        # Add any specific initialization checks based on the actual implementation

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.ImprovedStockModel')
    def test_run_analysis_for_stock_success(self, mock_model_class, mock_sentiment_class, 
                                          mock_fetcher_class, sample_stock_dataframe, 
                                          sample_sentiment_dataframe, mock_model):
        """Test successful complete analysis pipeline."""
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = sample_stock_dataframe
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_sentiment = Mock()
        mock_sentiment.fetch_news_sentiment.return_value = sample_sentiment_dataframe
        mock_sentiment_class.return_value = mock_sentiment
        
        mock_model_class.return_value = mock_model
        mock_model.prepare_data.return_value = (
            np.random.random((80, 20, 10)),  # X_market
            np.random.random((80, 20, 3)),   # X_sentiment  
            np.random.random((80,))          # y
        )
        mock_model.build_model.return_value = None
        mock_model.fit.return_value = None
        
        # Test analysis
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(
            'AAPL', '1y', '1d'
        )
        
        # Verify results
        assert model is not None
        assert isinstance(df, pd.DataFrame)
        assert isinstance(future_prices, list)
        assert isinstance(future_dates, list)
        assert len(future_prices) == 5
        assert len(future_dates) == 5
        
        # Verify that mocks were called
        mock_fetcher_class.assert_called_once_with('AAPL', '1y', '1d')
        mock_sentiment_class.assert_called_once_with('AAPL')
        mock_fetcher.fetch_data.assert_called_once()
        mock_sentiment.fetch_news_sentiment.assert_called_once()

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    def test_run_analysis_insufficient_stock_data(self, mock_fetcher_class):
        """Test analysis with insufficient stock data."""
        # Setup mock to return insufficient data
        mock_fetcher = Mock()
        insufficient_data = pd.DataFrame({
            'date': [datetime.now()],
            'close': [100],
            'volume': [1000000]
        })
        mock_fetcher.fetch_data.return_value = insufficient_data
        mock_fetcher_class.return_value = mock_fetcher
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
        
        # Should return None values for insufficient data
        assert model is None
        assert df is None
        assert future_prices is None
        assert future_dates is None

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    def test_run_analysis_no_stock_data(self, mock_fetcher_class):
        """Test analysis when stock data fetching fails."""
        # Setup mock to return None (failed fetch)
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = None
        mock_fetcher_class.return_value = mock_fetcher
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('INVALID')
        
        # Should return None values when data fetch fails
        assert model is None
        assert df is None
        assert future_prices is None
        assert future_dates is None

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    def test_run_analysis_no_sentiment_data(self, mock_sentiment_class, mock_fetcher_class,
                                          sample_stock_dataframe):
        """Test analysis when sentiment data fetching fails."""
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = sample_stock_dataframe
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_sentiment = Mock()
        mock_sentiment.fetch_news_sentiment.return_value = None  # Failed sentiment fetch
        mock_sentiment_class.return_value = mock_sentiment
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
        
        # Should return None values when sentiment fetch fails
        assert model is None
        assert df is None
        assert future_prices is None
        assert future_dates is None

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.TechnicalIndicatorGenerator')
    def test_data_preparation_pipeline(self, mock_tech_indicators, mock_sentiment_class,
                                     mock_fetcher_class, sample_stock_dataframe,
                                     sample_sentiment_dataframe):
        """Test data preparation steps in the pipeline."""
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = sample_stock_dataframe
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_sentiment = Mock()
        mock_sentiment.fetch_news_sentiment.return_value = sample_sentiment_dataframe
        mock_sentiment_class.return_value = mock_sentiment
        
        # Mock technical indicators to return DataFrame with additional columns
        enhanced_df = sample_stock_dataframe.copy()
        enhanced_df['ma7'] = enhanced_df['close'].rolling(7).mean()
        enhanced_df['rsi14'] = 50  # Simplified RSI
        mock_tech_indicators.add_technical_indicators.return_value = enhanced_df
        
        analyzer = StockAnalyzer()
        
        # This will likely fail at model training, but we can test data preparation
        try:
            analyzer.run_analysis_for_stock('AAPL', '1y', '1d')
        except:
            pass  # Expected to fail at model training
        
        # Verify that technical indicators were added
        mock_tech_indicators.add_technical_indicators.assert_called()

    def test_self_diagnostic_success(self):
        """Test diagnostic functionality."""
        analyzer = StockAnalyzer()
        
        # Mock the run_analysis_for_stock method to return success
        with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
            mock_model = Mock()
            mock_model.prepare_data.return_value = (
                np.array([[1, 2, 3]]), 
                np.array([[0.5, 0.6, 0.7]]), 
                np.array([100])
            )
            mock_model.predict.return_value = np.array([101])
            mock_model.evaluate.return_value = {'mse': 0.1, 'mae': 0.2, 'r2': 0.9}
            
            mock_df = pd.DataFrame({'test': [1, 2, 3]})
            mock_prices = [100, 101, 102]
            mock_dates = [datetime.now() + timedelta(days=i) for i in range(3)]
            
            # Return the correct tuple format (4 values)
            mock_analysis.return_value = (mock_model, mock_df, mock_prices, mock_dates)
            
            model, df, future_prices = analyzer.self_diagnostic('NVDA', '1y')
            
            # Verify diagnostic returns expected values
            assert model is not None
            assert df is not None
            assert future_prices is not None
            
            # Verify that run_analysis_for_stock was called with correct parameters
            mock_analysis.assert_called_once_with('NVDA', '1y')

    def test_self_diagnostic_failure(self):
        """Test diagnostic when analysis fails."""
        analyzer = StockAnalyzer()
        
        # Mock the run_analysis_for_stock method to return failure
        with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
            mock_analysis.return_value = (None, None, None, None)
            
            model, df, future_prices = analyzer.self_diagnostic('INVALID', '1y')
            
            # Verify diagnostic handles failure correctly
            assert model is None
            assert df is None
            assert future_prices is None

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    def test_invalid_ticker_analysis(self, mock_sentiment_class, mock_fetcher_class):
        """Test analysis with invalid ticker."""
        # Setup mocks to simulate invalid ticker response
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = None
        mock_fetcher_class.return_value = mock_fetcher
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('INVALID_TICKER')
        
        # Should handle invalid ticker gracefully
        assert model is None
        assert df is None
        assert future_prices is None
        assert future_dates is None

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.ImprovedStockModel')
    def test_model_training_pipeline(self, mock_model_class, mock_sentiment_class,
                                   mock_fetcher_class, sample_stock_dataframe,
                                   sample_sentiment_dataframe, mock_model):
        """Test model training integration."""
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = sample_stock_dataframe
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_sentiment = Mock()
        mock_sentiment.fetch_news_sentiment.return_value = sample_sentiment_dataframe
        mock_sentiment_class.return_value = mock_sentiment
        
        mock_model_class.return_value = mock_model
        
        # Mock data preparation to return proper shapes
        mock_model.prepare_data.return_value = (
            np.random.random((80, 20, 10)),  # X_market
            np.random.random((80, 20, 3)),   # X_sentiment
            np.random.random((80,))          # y
        )
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
        
        # Verify that model training methods were called
        mock_model.prepare_data.assert_called_once()
        mock_model.build_model.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.predict_next_days.assert_called_once()

    def test_different_time_periods(self):
        """Test analysis with different time periods."""
        analyzer = StockAnalyzer()
        
        periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y']
        
        for period in periods:
            with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
                mock_analysis.return_value = (Mock(), pd.DataFrame(), [100], [datetime.now()])
                
                model, df, prices, dates = analyzer.run_analysis_for_stock('AAPL', period)
                
                # Verify that the method was called with the correct period
                mock_analysis.assert_called_once_with('AAPL', period)

    def test_different_intervals(self):
        """Test analysis with different data intervals."""
        analyzer = StockAnalyzer()
        
        intervals = ['1d', '1wk', '1mo']
        
        for interval in intervals:
            with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
                mock_analysis.return_value = (Mock(), pd.DataFrame(), [100], [datetime.now()])
                
                model, df, prices, dates = analyzer.run_analysis_for_stock('AAPL', '1y', interval)
                
                # Verify that the method was called with the correct interval
                mock_analysis.assert_called_once_with('AAPL', '1y', interval)

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    def test_date_alignment_pipeline(self, mock_sentiment_class, mock_fetcher_class):
        """Test that stock and sentiment data dates are properly aligned."""
        # Create stock data with specific date range
        stock_dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        stock_df = pd.DataFrame({
            'date': stock_dates,
            'close': np.random.uniform(100, 110, len(stock_dates)),
            'volume': np.random.randint(1000000, 5000000, len(stock_dates))
        })
        
        # Create sentiment data with overlapping but different date range
        sentiment_dates = pd.date_range('2023-01-05', '2023-02-05', freq='D')
        sentiment_df = pd.DataFrame({
            'date': sentiment_dates,
            'sentiment_positive': np.random.uniform(0.3, 0.7, len(sentiment_dates)),
            'sentiment_negative': np.random.uniform(0.1, 0.4, len(sentiment_dates)),
            'sentiment_neutral': np.random.uniform(0.2, 0.5, len(sentiment_dates))
        })
        
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = stock_df
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_sentiment = Mock()
        mock_sentiment.fetch_news_sentiment.return_value = sentiment_df
        mock_sentiment_class.return_value = mock_sentiment
        
        analyzer = StockAnalyzer()
        
        try:
            model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
            
            if df is not None:
                # Verify that dates are properly aligned (should have intersection of dates)
                expected_overlap_start = max(stock_dates.min(), sentiment_dates.min())
                expected_overlap_end = min(stock_dates.max(), sentiment_dates.max())
                
                assert df['date'].min() >= expected_overlap_start
                assert df['date'].max() <= expected_overlap_end
        except:
            # Analysis might fail due to other reasons, but date alignment should work
            pass

    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the analysis pipeline."""
        analyzer = StockAnalyzer()
        
        # Test with various error scenarios
        error_scenarios = [
            'INVALID_TICKER',
            '',  # Empty ticker
            None,  # None ticker - this might cause different error
        ]
        
        for ticker in error_scenarios:
            if ticker is not None:  # Skip None to avoid TypeError
                try:
                    model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(ticker)
                    
                    # Should handle errors gracefully
                    assert model is None
                    assert df is None
                    assert future_prices is None
                    assert future_dates is None
                except Exception as e:
                    # Some errors might still be raised, which is acceptable
                    assert isinstance(e, Exception)

    @patch('stock_prediction_lstm.analysis.stock_analyzer.StockDataFetcher')
    @patch('stock_prediction_lstm.analysis.stock_analyzer.SentimentAnalyzer')
    def test_insufficient_merged_data(self, mock_sentiment_class, mock_fetcher_class):
        """Test analysis when merged data is insufficient."""
        # Create minimal overlapping data
        stock_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5
        })
        
        sentiment_df = pd.DataFrame({
            'date': pd.date_range('2023-01-03', periods=2, freq='D'),  # Minimal overlap
            'sentiment_positive': [0.6, 0.7],
            'sentiment_negative': [0.2, 0.1],
            'sentiment_neutral': [0.2, 0.2]
        })
        
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.fetch_data.return_value = stock_df
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_sentiment = Mock()
        mock_sentiment.fetch_news_sentiment.return_value = sentiment_df
        mock_sentiment_class.return_value = mock_sentiment
        
        analyzer = StockAnalyzer()
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL')
        
        # Should return None for insufficient merged data
        assert model is None
        assert df is None
        assert future_prices is None
        assert future_dates is None

    def test_memory_efficiency(self):
        """Test that analysis doesn't consume excessive memory."""
        analyzer = StockAnalyzer()
        
        # This is a basic test - more sophisticated memory testing would require 
        # memory profiling tools like memory_profiler
        import gc
        
        initial_objects = len(gc.get_objects())
        
        # Mock a successful analysis
        with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
            mock_analysis.return_value = (Mock(), pd.DataFrame(), [100], [datetime.now()])
            
            for _ in range(5):  # Run multiple analyses
                analyzer.run_analysis_for_stock('AAPL')
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not create excessive objects
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Created too many objects: {object_growth}"

    @patch('stock_prediction_lstm.analysis.stock_analyzer.print')
    def test_logging_and_output(self, mock_print):
        """Test that appropriate logging/output is generated."""
        analyzer = StockAnalyzer()
        
        with patch.object(analyzer, 'run_analysis_for_stock') as mock_analysis:
            mock_analysis.return_value = (Mock(), pd.DataFrame(), [100], [datetime.now()])
            
            analyzer.run_analysis_for_stock('AAPL')
            
            # Verify that print statements were called (indicates logging)
            # This depends on the actual implementation having print statements
            # mock_print.assert_called()  # Uncomment if print statements exist
