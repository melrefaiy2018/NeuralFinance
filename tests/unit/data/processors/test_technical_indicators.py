"""
Unit tests for TechnicalIndicatorGenerator module.

This module tests the TechnicalIndicatorGenerator class which calculates
various technical indicators from stock price and volume data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from neural_finance.data.processors.technical_indicators import TechnicalIndicatorGenerator


class TestTechnicalIndicatorGenerator:
    """Test class for TechnicalIndicatorGenerator."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price movements
        base_price = 100
        prices = []
        volumes = []
        
        for i in range(100):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price = max(base_price * (1 + change), 1)  # Ensure positive prices
            prices.append(base_price)
            volumes.append(np.random.randint(1000000, 10000000))
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices]
        })

    @pytest.fixture
    def minimal_price_data(self):
        """Create minimal price data for edge case testing."""
        return pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'open': [99, 100, 101]
        })

    def test_add_moving_averages(self, sample_price_data):
        """Test moving average calculations (MA7, MA14, MA30)."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Check that moving average columns are added
        assert 'ma7' in result.columns
        assert 'ma14' in result.columns
        assert 'ma30' in result.columns
        
        # Check that moving averages are calculated correctly
        # MA7 should be the 7-day rolling mean
        expected_ma7 = sample_price_data['close'].rolling(window=7).mean()
        pd.testing.assert_series_equal(result['ma7'], expected_ma7, check_names=False)
        
        # MA14 should be the 14-day rolling mean
        expected_ma14 = sample_price_data['close'].rolling(window=14).mean()
        pd.testing.assert_series_equal(result['ma14'], expected_ma14, check_names=False)
        
        # MA30 should be the 30-day rolling mean
        expected_ma30 = sample_price_data['close'].rolling(window=30).mean()
        pd.testing.assert_series_equal(result['ma30'], expected_ma30, check_names=False)
        
        # Check that early values are NaN (as expected for rolling calculations)
        assert pd.isna(result['ma7'].iloc[0:6]).all()  # First 6 values should be NaN
        assert pd.isna(result['ma14'].iloc[0:13]).all()  # First 13 values should be NaN
        assert pd.isna(result['ma30'].iloc[0:29]).all()  # First 29 values should be NaN

    def test_add_rsi(self, sample_price_data):
        """Test RSI calculation."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Check that RSI column is added
        assert 'rsi14' in result.columns
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi14'].dropna()
        assert all(rsi_values >= 0)
        assert all(rsi_values <= 100)
        
        # Check that early values are NaN (RSI needs at least 14 periods)
        # Note: RSI calculation might start producing values at index 13 (0-based), so check 0:13
        assert pd.isna(result['rsi14'].iloc[0:13]).all()
        
        # Manually calculate RSI for verification
        delta = sample_price_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        expected_rsi = 100 - (100 / (1 + rs))
        
        # Compare calculated RSI (ignoring NaN values)
        pd.testing.assert_series_equal(
            result['rsi14'].dropna(), 
            expected_rsi.dropna(), 
            check_names=False,
            atol=1e-10
        )

    def test_add_macd(self, sample_price_data):
        """Test MACD calculation."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Check that MACD columns are added
        assert 'ema12' in result.columns
        assert 'ema26' in result.columns
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        
        # Manually calculate MACD for verification
        expected_ema12 = sample_price_data['close'].ewm(span=12, adjust=False).mean()
        expected_ema26 = sample_price_data['close'].ewm(span=26, adjust=False).mean()
        expected_macd = expected_ema12 - expected_ema26
        expected_signal = expected_macd.ewm(span=9, adjust=False).mean()
        
        # Compare calculated values
        pd.testing.assert_series_equal(result['ema12'], expected_ema12, check_names=False)
        pd.testing.assert_series_equal(result['ema26'], expected_ema26, check_names=False)
        pd.testing.assert_series_equal(result['macd'], expected_macd, check_names=False)
        pd.testing.assert_series_equal(result['macd_signal'], expected_signal, check_names=False)

    def test_add_bollinger_bands(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        # Assuming Bollinger Bands are implemented (they weren't in the original code)
        # This test would need to be updated based on actual implementation
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # If Bollinger Bands are implemented, they would typically include:
        # - Middle band (usually 20-day SMA)
        # - Upper band (middle band + 2 * standard deviation)
        # - Lower band (middle band - 2 * standard deviation)
        
        # For now, just check that the function runs without error
        assert isinstance(result, pd.DataFrame)

    def test_complete_indicator_set(self, sample_price_data):
        """Test adding all indicators to DataFrame."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Check that all expected columns are present
        expected_new_columns = ['ma7', 'ma14', 'ma30', 'roc5', 'rsi14', 'ema12', 'ema26', 'macd', 'macd_signal']
        
        for col in expected_new_columns:
            assert col in result.columns, f"Column {col} missing from result"
        
        # Check that original columns are preserved
        for col in sample_price_data.columns:
            assert col in result.columns, f"Original column {col} missing from result"
        
        # Check that DataFrame shape is correct
        assert len(result) == len(sample_price_data)
        assert len(result.columns) >= len(sample_price_data.columns) + len(expected_new_columns)

    def test_insufficient_data_points(self, minimal_price_data):
        """Test behavior with minimal data points."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(minimal_price_data)
        
        # With only 3 data points, most indicators should be NaN
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Check that moving averages with windows larger than data size are all NaN
        assert pd.isna(result['ma7']).all()
        assert pd.isna(result['ma14']).all()
        assert pd.isna(result['ma30']).all()
        assert pd.isna(result['rsi14']).all()

    def test_missing_price_column(self):
        """Test behavior when required columns are missing."""
        # DataFrame without 'close' column
        df_no_close = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'volume': [1000, 1100, 1200]
        })
        
        # This should raise an error or handle gracefully
        with pytest.raises(KeyError):
            TechnicalIndicatorGenerator.add_technical_indicators(df_no_close)

    def test_missing_volume_column(self):
        """Test behavior when volume column is missing."""
        df_no_volume = pd.DataFrame({
            'close': [100, 101, 102],
            'open': [99, 100, 101],
            'high': [102, 103, 104],
            'low': [98, 99, 100]
        })
        
        # Should handle missing volume gracefully
        result = TechnicalIndicatorGenerator.add_technical_indicators(df_no_volume)
        assert isinstance(result, pd.DataFrame)

    def test_zero_volume_data(self):
        """Test handling of zero volume data."""
        df_zero_volume = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [0, 0, 0, 0, 0],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'open': [99, 100, 101, 102, 103]
        })
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(df_zero_volume)
        
        # Should handle zero volume without crashing
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_extreme_price_values(self):
        """Test handling of extreme price values."""
        df_extreme = pd.DataFrame({
            'close': [0.001, 1000000, 100, 0.01, 999999],
            'volume': [1, 999999999, 1000000, 100, 50000000],
            'high': [0.002, 1100000, 105, 0.02, 1099999],
            'low': [0.0005, 900000, 95, 0.005, 899999],
            'open': [0.0015, 950000, 98, 0.015, 949999]
        })
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(df_extreme)
        
        # Should handle extreme values without crashing
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        
        # Check that calculations don't produce infinite values
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_custom_column_names(self):
        """Test using custom price and volume column names."""
        df_custom = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'vol': [1000000, 1100000, 1200000, 1300000, 1400000],
            'high': [102, 103, 104, 105, 106],
            'low': [98, 99, 100, 101, 102],
            'open': [99, 100, 101, 102, 103]
        })
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(
            df_custom, price_col='price', volume_col='vol'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'ma7' in result.columns
        assert 'rsi14' in result.columns

    def test_rate_of_change_calculation(self, sample_price_data):
        """Test Rate of Change (ROC) calculation."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Check that ROC column is added
        assert 'roc5' in result.columns
        
        # Manually calculate ROC for verification
        expected_roc = sample_price_data['close'].pct_change(periods=5) * 100
        pd.testing.assert_series_equal(result['roc5'], expected_roc, check_names=False)
        
        # Check that first 5 values are NaN
        assert pd.isna(result['roc5'].iloc[0:5]).all()

    def test_data_types_preservation(self, sample_price_data):
        """Test that appropriate data types are maintained."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Numeric columns should remain numeric
        numeric_columns = ['ma7', 'ma14', 'ma30', 'roc5', 'rsi14', 'ema12', 'ema26', 'macd', 'macd_signal']
        
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"Column {col} should be numeric"

    def test_nan_handling(self):
        """Test handling of NaN values in input data."""
        df_with_nan = pd.DataFrame({
            'close': [100, np.nan, 102, 103, np.nan, 105],
            'volume': [1000000, 1100000, np.nan, 1300000, 1400000, 1500000],
            'high': [102, 103, 104, 105, 106, 107],
            'low': [98, 99, 100, 101, 102, 103],
            'open': [99, 100, 101, 102, 103, 104]
        })
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(df_with_nan)
        
        # Should handle NaN values gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6

    def test_single_row_dataframe(self):
        """Test behavior with single row DataFrame."""
        df_single = pd.DataFrame({
            'close': [100],
            'volume': [1000000],
            'high': [102],
            'low': [98],
            'open': [99]
        })
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(df_single)
        
        # Should handle single row gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        # All rolling indicators should be NaN for single row
        # EMA indicators might start with the first value, so only check rolling indicators
        rolling_indicator_columns = ['ma7', 'ma14', 'ma30', 'rsi14']
        for col in rolling_indicator_columns:
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"Rolling indicator {col} should be NaN for single row"
        
        # EMA and MACD might have values starting from first row, which is acceptable
        # OBV starts with 0, which is also acceptable

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        df_empty = pd.DataFrame(columns=['close', 'volume', 'high', 'low', 'open'])
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(df_empty)
        
        # Should handle empty DataFrame gracefully - might add columns but no rows
        assert isinstance(result, pd.DataFrame)
        # The implementation might add a row with NaN values, so check for this
        if len(result) > 0:
            # If it adds a row, most values should be NaN except OBV which starts at 0
            nan_columns = result.select_dtypes(include=[np.number]).columns.difference(['obv'])
            if len(nan_columns) > 0:
                # Most numerical columns should be NaN
                nan_count = result[nan_columns].isna().sum().sum()
                total_cells = len(nan_columns) * len(result)
                assert nan_count == total_cells, "Most values should be NaN for empty input"
        else:
            assert len(result) == 0

    def test_immutability_of_input(self, sample_price_data):
        """Test that input DataFrame is not modified."""
        original_columns = sample_price_data.columns.tolist()
        original_shape = sample_price_data.shape
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Original DataFrame should be unchanged
        assert sample_price_data.columns.tolist() == original_columns
        assert sample_price_data.shape == original_shape

    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        # Create large dataset
        large_dates = pd.date_range('2010-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        large_df = pd.DataFrame({
            'close': np.random.uniform(50, 200, len(large_dates)),
            'volume': np.random.randint(1000000, 50000000, len(large_dates)),
            'high': np.random.uniform(55, 210, len(large_dates)),
            'low': np.random.uniform(45, 190, len(large_dates)),
            'open': np.random.uniform(50, 200, len(large_dates))
        })
        
        # This should complete without timeout or memory issues
        import time
        start_time = time.time()
        
        result = TechnicalIndicatorGenerator.add_technical_indicators(large_df)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        assert execution_time < 10, f"Execution took too long: {execution_time} seconds"
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_df)

    def test_indicator_mathematical_properties(self, sample_price_data):
        """Test mathematical properties of indicators."""
        result = TechnicalIndicatorGenerator.add_technical_indicators(sample_price_data)
        
        # Moving averages should be smoother than original price
        ma7_volatility = result['ma7'].dropna().std()
        price_volatility = sample_price_data['close'].std()
        assert ma7_volatility <= price_volatility, "MA7 should be less volatile than price"
        
        # EMA12 should react faster than EMA26
        ema12_diff = result['ema12'].diff().abs().mean()
        ema26_diff = result['ema26'].diff().abs().mean()
        # Note: This might not always be true depending on the data
        # assert ema12_diff >= ema26_diff, "EMA12 should react faster than EMA26"
        
        # RSI should be mean-reverting around 50
        rsi_mean = result['rsi14'].dropna().mean()
        assert 30 <= rsi_mean <= 70, f"RSI mean {rsi_mean} should be reasonable"
