# neural_finance - Comprehensive Unit Testing Agent Instructions

## Overview

This document provides detailed instructions for implementing comprehensive unit tests for the Neural Finance package. The package consists of multiple modules for stock data fetching, sentiment analysis, LSTM model training, visualization, and command-line interfaces.

## Package Structure Analysis

Based on the codebase analysis, the following key modules require testing:

### Core Modules
1. **Data Fetchers** (`neural_finance/data/fetchers/`)
   - `StockDataFetcher` - Fetches stock price data from Yahoo Finance
   - `SentimentAnalyzer` - Fetches and processes sentiment data
   - `AlternativeSentimentSources` - Alternative sentiment data sources

2. **Data Processors** (`neural_finance/data/processors/`)
   - `TechnicalIndicatorGenerator` - Calculates technical indicators

3. **Models** (`neural_finance/models/`)
   - `ImprovedStockModel` - Core LSTM model for predictions
   - `StockSentimentModel` - LSTM with sentiment integration

4. **Analysis** (`neural_finance/analysis/`)
   - `StockAnalyzer` - Main analysis orchestrator

5. **Visualization** (`neural_finance/visualization/`)
   - `plotters` - Plotting functions for visualization

6. **CLI** (`neural_finance/cli/`)
   - `main` - Command-line interface

7. **Storage & Utilities** (`neural_finance/data/storage/`, `neural_finance/core/`)
   - Cache managers, data managers, utilities

## Testing Framework Setup

### Prerequisites
```bash
# Install testing dependencies
pip install pytest>=7.0.0 pytest-cov>=4.0.0 pytest-mock>=3.10.0
```

### Directory Structure
Create the following test directory structure:
```
tests/
├── __init__.py
├── conftest.py
├── test_data/
│   ├── sample_stock_data.csv
│   ├── sample_sentiment_data.json
│   └── mock_api_responses.json
├── unit/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetchers/
│   │   │   ├── __init__.py
│   │   │   ├── test_stock_data.py
│   │   │   ├── test_sentiment_data.py
│   │   │   └── test_alternative_sentiment_sources.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   └── test_technical_indicators.py
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── test_cache_manager.py
│   │       └── test_data_manager.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_improved_model.py
│   │   └── test_lstm_attention.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── test_stock_analyzer.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── test_plotters.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── test_main.py
│   └── core/
│       ├── __init__.py
│       ├── test_utils.py
│       └── test_exceptions.py
├── integration/
│   ├── __init__.py
│   ├── test_end_to_end.py
│   └── test_workflow.py
└── performance/
    ├── __init__.py
    └── test_benchmarks.py
```

## Detailed Testing Requirements

### 1. Data Fetchers Testing (`tests/unit/data/fetchers/`)

#### `test_stock_data.py`
**Test Class:** `TestStockDataFetcher`

**Normal Use Cases:**
- ✅ **test_init_with_default_params** - Test initialization with default parameters
- ✅ **test_init_with_custom_params** - Test initialization with custom ticker, period, interval
- ✅ **test_fetch_data_success** - Test successful data fetching for valid ticker
- ✅ **test_fetch_data_valid_columns** - Verify returned DataFrame has required columns (date, open, high, low, close, volume)
- ✅ **test_fetch_data_date_range** - Verify data covers requested period
- ✅ **test_fetch_data_interval_consistency** - Verify data interval matches request

**Edge Cases:**
- ❌ **test_fetch_data_invalid_ticker** - Test behavior with invalid ticker symbol
- ❌ **test_fetch_data_invalid_period** - Test behavior with invalid period format
- ❌ **test_fetch_data_network_error** - Test behavior when network is unavailable
- ❌ **test_fetch_data_empty_response** - Test behavior when API returns empty data
- ❌ **test_fetch_data_rate_limiting** - Test behavior under API rate limits
- ❌ **test_fetch_data_malformed_response** - Test behavior with malformed API response

**Expected Outputs/Assertions:**
- DataFrame structure validation (columns, data types)
- Date range validation
- Non-null value checks for critical columns
- Exception handling verification

#### `test_sentiment_data.py`
**Test Class:** `TestSentimentAnalyzer`

**Normal Use Cases:**
- ✅ **test_init_with_ticker** - Test initialization with ticker symbol
- ✅ **test_fetch_news_sentiment_success** - Test successful sentiment data fetching
- ✅ **test_sentiment_data_structure** - Verify sentiment DataFrame structure
- ✅ **test_sentiment_score_ranges** - Verify sentiment scores are within expected ranges (0-1)
- ✅ **test_date_filtering** - Test date range filtering functionality

**Edge Cases:**
- ❌ **test_no_news_available** - Test behavior when no news is available for ticker
- ❌ **test_api_key_missing** - Test fallback behavior when API keys are not configured
- ❌ **test_api_quota_exceeded** - Test behavior when API quotas are exceeded
- ❌ **test_invalid_date_range** - Test behavior with invalid date ranges
- ❌ **test_future_dates** - Test behavior when requesting future dates

**Expected Outputs/Assertions:**
- Sentiment DataFrame with columns: date, sentiment_positive, sentiment_negative, sentiment_neutral
- Proper fallback to synthetic data when APIs fail
- Caching mechanism verification

#### `test_alternative_sentiment_sources.py`
**Test Class:** `TestAlternativeSentimentSources`

**Normal Use Cases:**
- ✅ **test_marketaux_sentiment_fetch** - Test MarketAux API integration
- ✅ **test_synthetic_sentiment_generation** - Test synthetic sentiment data generation
- ✅ **test_sentiment_aggregation** - Test combining multiple sentiment sources

**Edge Cases:**
- ❌ **test_api_source_failures** - Test behavior when all external APIs fail
- ❌ **test_partial_api_failures** - Test behavior when some APIs fail
- ❌ **test_malformed_api_responses** - Test handling of malformed API responses

### 2. Data Processors Testing (`tests/unit/data/processors/`)

#### `test_technical_indicators.py`
**Test Class:** `TestTechnicalIndicatorGenerator`

**Normal Use Cases:**
- ✅ **test_add_moving_averages** - Test moving average calculations (MA7, MA14, MA30)
- ✅ **test_add_rsi** - Test RSI calculation
- ✅ **test_add_macd** - Test MACD calculation
- ✅ **test_add_bollinger_bands** - Test Bollinger Bands calculation
- ✅ **test_complete_indicator_set** - Test adding all indicators to DataFrame

**Edge Cases:**
- ❌ **test_insufficient_data_points** - Test behavior with minimal data points
- ❌ **test_missing_price_column** - Test behavior when required columns are missing
- ❌ **test_zero_volume_data** - Test handling of zero volume data
- ❌ **test_extreme_price_values** - Test handling of extreme price values

**Expected Outputs/Assertions:**
- Verify technical indicator calculations match expected formulas
- Check for NaN handling in rolling calculations
- Validate indicator value ranges

### 3. Models Testing (`tests/unit/models/`)

#### `test_improved_model.py`
**Test Class:** `TestImprovedStockModel`

**Normal Use Cases:**
- ✅ **test_model_initialization** - Test model initialization with default/custom parameters
- ✅ **test_prepare_data** - Test data preparation and scaling
- ✅ **test_build_model** - Test LSTM model architecture creation
- ✅ **test_model_training** - Test model training with sample data
- ✅ **test_prediction_generation** - Test generating predictions
- ✅ **test_future_price_prediction** - Test future price prediction functionality
- ✅ **test_model_evaluation** - Test model evaluation metrics

**Edge Cases:**
- ❌ **test_insufficient_training_data** - Test behavior with minimal training data
- ❌ **test_single_feature_training** - Test training with single feature
- ❌ **test_prediction_with_untrained_model** - Test prediction before training
- ❌ **test_extreme_input_values** - Test handling of extreme input values
- ❌ **test_memory_constraints** - Test behavior under memory constraints

**Expected Outputs/Assertions:**
- Model architecture validation
- Training convergence verification
- Prediction accuracy metrics (RMSE, MAE, R², MAPE)
- Proper scaling and inverse scaling

#### `test_lstm_attention.py`
**Test Class:** `TestStockSentimentModel`

**Normal Use Cases:**
- ✅ **test_sentiment_integration** - Test integration of sentiment data with market data
- ✅ **test_attention_mechanism** - Test attention mechanism functionality
- ✅ **test_multi_input_training** - Test training with both market and sentiment inputs

**Edge Cases:**
- ❌ **test_missing_sentiment_data** - Test behavior when sentiment data is missing
- ❌ **test_sentiment_market_mismatch** - Test handling of mismatched sentiment/market data

### 4. Analysis Testing (`tests/unit/analysis/`)

#### `test_stock_analyzer.py`
**Test Class:** `TestStockAnalyzer`

**Normal Use Cases:**
- ✅ **test_analyzer_initialization** - Test StockAnalyzer initialization
- ✅ **test_run_analysis_for_stock** - Test complete analysis pipeline
- ✅ **test_self_diagnostic** - Test diagnostic functionality
- ✅ **test_data_preparation_pipeline** - Test data preparation steps
- ✅ **test_model_training_pipeline** - Test model training integration

**Edge Cases:**
- ❌ **test_invalid_ticker_analysis** - Test analysis with invalid ticker
- ❌ **test_insufficient_data_analysis** - Test analysis with insufficient historical data
- ❌ **test_api_failure_analysis** - Test analysis when external APIs fail
- ❌ **test_model_training_failure** - Test analysis when model training fails

**Expected Outputs/Assertions:**
- Return tuple validation (model, dataframe, predictions, dates)
- Pipeline success/failure handling
- Proper error propagation

### 5. Visualization Testing (`tests/unit/visualization/`)

#### `test_plotters.py`
**Test Class:** `TestVisualizationPlotters`

**Normal Use Cases:**
- ✅ **test_visualize_stock_data** - Test stock data visualization
- ✅ **test_visualize_prediction_comparison** - Test prediction vs actual comparison plots
- ✅ **test_visualize_future_predictions** - Test future prediction visualization
- ✅ **test_visualize_feature_importance** - Test feature importance plots
- ✅ **test_plot_saving** - Test plot saving functionality

**Edge Cases:**
- ❌ **test_empty_data_visualization** - Test visualization with empty datasets
- ❌ **test_single_data_point_visualization** - Test visualization with single data point
- ❌ **test_invalid_output_directory** - Test behavior with invalid output directories
- ❌ **test_display_environment_handling** - Test headless environment handling

**Expected Outputs/Assertions:**
- Plot generation without errors
- File saving verification
- Plot content validation

### 6. CLI Testing (`tests/unit/cli/`)

#### `test_main.py`
**Test Class:** `TestCLIMain`

**Normal Use Cases:**
- ✅ **test_cli_analyze_command** - Test analyze command execution
- ✅ **test_cli_diagnostic_command** - Test diagnostic command execution
- ✅ **test_cli_help_display** - Test help command display

**Edge Cases:**
- ❌ **test_cli_invalid_arguments** - Test CLI with invalid arguments
- ❌ **test_cli_missing_required_args** - Test CLI with missing required arguments

### 7. Core Utilities Testing (`tests/unit/core/`)

#### `test_utils.py`
**Test Class:** `TestCoreUtils`

**Normal Use Cases:**
- ✅ **test_create_output_directory** - Test output directory creation
- ✅ **test_save_plot** - Test plot saving utility
- ✅ **test_date_utilities** - Test date manipulation utilities

**Edge Cases:**
- ❌ **test_directory_creation_permissions** - Test directory creation with insufficient permissions
- ❌ **test_file_saving_disk_full** - Test file saving when disk is full

### 8. Storage Testing (`tests/unit/data/storage/`)

#### `test_cache_manager.py`
**Test Class:** `TestCacheManager`

**Normal Use Cases:**
- ✅ **test_cache_storage** - Test data caching functionality
- ✅ **test_cache_retrieval** - Test cached data retrieval
- ✅ **test_cache_invalidation** - Test cache invalidation

**Edge Cases:**
- ❌ **test_cache_corruption** - Test handling of corrupted cache files
- ❌ **test_cache_size_limits** - Test behavior when cache size limits are exceeded

## Integration Testing (`tests/integration/`)

### `test_end_to_end.py`
**Test Class:** `TestEndToEndWorkflow`

- ✅ **test_complete_analysis_workflow** - Test full analysis from data fetch to prediction
- ✅ **test_multi_ticker_analysis** - Test analysis of multiple tickers
- ✅ **test_different_time_periods** - Test analysis with various time periods

### `test_workflow.py`
**Test Class:** `TestWorkflowIntegration`

- ✅ **test_data_flow_integrity** - Test data consistency across pipeline stages
- ✅ **test_model_prediction_accuracy** - Test prediction accuracy on known datasets

## Performance Testing (`tests/performance/`)

### `test_benchmarks.py`
**Test Class:** `TestPerformanceBenchmarks`

- ✅ **test_data_fetching_performance** - Benchmark data fetching speeds
- ✅ **test_model_training_performance** - Benchmark model training times
- ✅ **test_memory_usage** - Monitor memory usage patterns

## Test Configuration (`conftest.py`)

### Fixtures Required:
```python
@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for testing"""
    pass

@pytest.fixture
def sample_sentiment_data():
    """Provide sample sentiment data for testing"""
    pass

@pytest.fixture
def mock_api_responses():
    """Mock external API responses"""
    pass

@pytest.fixture
def temp_output_dir():
    """Provide temporary output directory for tests"""
    pass

@pytest.fixture
def trained_model():
    """Provide a pre-trained model for testing"""
    pass
```

### Mock Requirements:
- Mock external API calls (Yahoo Finance, Alpha Vantage, news APIs)
- Mock file system operations for edge cases
- Mock TensorFlow/Keras operations for unit tests
- Mock plotting libraries for headless testing

## Test Data Requirements (`tests/test_data/`)

### Required Test Data Files:
1. **sample_stock_data.csv** - Sample stock price data with multiple tickers
2. **sample_sentiment_data.json** - Sample sentiment data for various dates
3. **mock_api_responses.json** - Mocked API responses for different scenarios
4. **corrupted_data.csv** - Intentionally corrupted data for edge case testing

## Testing Best Practices

### Code Coverage Goals:
- **Target**: 90%+ overall code coverage
- **Critical modules**: 95%+ coverage (data fetchers, models, analysis)
- **Utility modules**: 85%+ coverage

### Test Naming Convention:
- Use descriptive test method names: `test_<functionality>_<scenario>_<expected_outcome>`
- Example: `test_fetch_data_invalid_ticker_raises_exception`

### Assertion Patterns:
```python
# Data structure validation
assert isinstance(result, pd.DataFrame)
assert list(result.columns) == expected_columns
assert len(result) > 0

# Value range validation
assert 0 <= sentiment_score <= 1
assert result['close'].min() > 0

# Exception testing
with pytest.raises(ValueError, match="Invalid ticker"):
    fetcher.fetch_data("INVALID")

# Approximate equality for float comparisons
assert pytest.approx(calculated_value, rel=1e-3) == expected_value
```

### Mock Patterns:
```python
# API mocking
@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = expected_response
    result = api_function()
    assert result == expected_result

# File system mocking
@patch('os.path.exists')
@patch('builtins.open', mock_open(read_data='test data'))
def test_file_operation(mock_exists):
    mock_exists.return_value = True
    result = file_function()
    assert result == expected_result
```

## Continuous Integration Configuration

### pytest.ini Configuration:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
addopts = 
    --strict-markers
    --strict-config
    --cov=neural_finance
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=85
```

### GitHub Actions Workflow (`.github/workflows/test.yml`):
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest tests/unit tests/integration -v
        pytest tests/performance --maxfail=1
```

## Implementation Priority

### Phase 1 (High Priority):
1. Core data fetchers (`StockDataFetcher`, `SentimentAnalyzer`)
2. Data processors (`TechnicalIndicatorGenerator`)
3. Main analysis module (`StockAnalyzer`)

### Phase 2 (Medium Priority):
1. Model testing (`ImprovedStockModel`, `StockSentimentModel`)
2. Integration tests
3. CLI testing

### Phase 3 (Low Priority):
1. Visualization testing
2. Performance benchmarks
3. Storage and cache testing

## Expected Timeline

- **Phase 1**: 1-2 weeks
- **Phase 2**: 1 week  
- **Phase 3**: 1 week
- **Total**: 3-4 weeks

## Success Criteria

1. ✅ All critical paths covered with unit tests
2. ✅ 90%+ code coverage achieved
3. ✅ All edge cases identified and tested
4. ✅ CI/CD pipeline passing consistently
5. ✅ Performance benchmarks established
6. ✅ Test documentation complete and maintainable

## Maintenance Guidelines

1. **Test Update Policy**: Update tests when adding new features or modifying existing functionality
2. **Coverage Monitoring**: Monitor coverage reports and address any drops below threshold
3. **Performance Regression**: Run performance tests regularly to catch regressions
4. **Mock Maintenance**: Keep mocks updated with actual API changes
5. **Test Data Refresh**: Periodically update test data to reflect current market conditions

---

This comprehensive testing plan ensures robust, reliable, and maintainable code quality for the Neural Finance package. Follow these instructions to implement a complete test suite that covers normal use cases, edge cases, and provides confidence in the system's reliability.
