# Stock Prediction LSTM

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A comprehensive stock prediction system using LSTM neural networks with sentiment analysis and technical indicators. This package provides a complete solution for stock price prediction, combining deep learning models with real-time sentiment analysis and technical analysis indicators.

## Features

- **LSTM Neural Networks**: Advanced deep learning models for time series prediction
- **Sentiment Analysis**: Real-time sentiment analysis using various APIs and fallback methods
- **Technical Indicators**: Comprehensive technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- **Visualization**: Rich visualizations for stock data, predictions, and analysis
- **Web Interface**: Flask-based web interface for interactive analysis
- **CLI Tool**: Command-line interface for batch processing and automation
- **Flexible API**: Easy-to-use Python API for integration into existing workflows

## Installation

### From PyPI (Recommended)

```bash
pip install stock-prediction-lstm
```

### From Source

```bash
git clone https://github.com/melrefaiy2018/stock_prediction_lstm.git
cd stock_prediction_lstm
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/melrefaiy2018/stock_prediction_lstm.git
cd stock_prediction_lstm
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from stock_prediction_lstm import StockAnalyzer

# Create analyzer instance
analyzer = StockAnalyzer()

# Run analysis for a stock
model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL', '1y', '1d')

# Print future predictions
if future_prices is not None:
    print("Future price predictions:")
    for i, price in enumerate(future_prices):
        print(f"Day {i+1}: ${price:.2f}")
```

### CLI Usage

```bash
# Run analysis for a stock
stock-predict analyze --ticker AAPL --period 1y --interval 1d

# Run diagnostic
stock-predict diagnostic --ticker NVDA --period 6mo
```

### Web Interface

```bash
# Launch Flask web interface
python stock_prediction_lstm/web/flask_app.py
```

## API Key Setup (Optional but Recommended)

The system uses various APIs for real-time sentiment analysis. While the model works without them using synthetic sentiment data, configuring real API keys provides better predictions.

### Manual Setup

1. **Get free API keys** from providers like Alpha Vantage, MarketAux, Finnhub, NewsAPI, Polygon.io, and Reddit API.
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key

2. **Configure the keys**:
   - Navigate to: `stock_prediction_lstm/config/keys/`
   - Edit `api_keys.py` (or `alternative_api_keys.py` for other sources)
   - Replace `"YOUR_API_KEY_HERE"` with your actual API key for each service.

3. **Test your setup**:
   ```bash
   python examples/demo/real_demo.py --ticker AAPL
   ```

## Configuration

The package supports various configuration options:

### Model Configuration

```python
from stock_prediction_lstm import StockAnalyzer

analyzer = StockAnalyzer(
    lstm_units=50,
    dropout_rate=0.2,
    epochs=50,
    batch_size=32,
    sequence_length=60
)
```

### Data Configuration

```python
# Configure data fetching
analyzer.configure_data(
    technical_indicators=['RSI', 'MACD', 'BB', 'SMA', 'EMA'],
    sentiment_analysis=True,
    volume_analysis=True
)
```

## Advanced Usage

### Custom Model Training

```python
from stock_prediction_lstm import StockSentimentModel
from stock_prediction_lstm.data import StockDataFetcher

# Fetch data
fetcher = StockDataFetcher()
data = fetcher.fetch_data('AAPL', '2y', '1d')

# Create and train model
model = StockSentimentModel()
# Assuming train_model and predict_future_prices are methods of StockSentimentModel
# You might need to adapt this based on the actual implementation
# trained_model = model.train_model(data) 
# predictions = model.predict_future_prices(data, days=30)
```

### Batch Processing

```python
from stock_prediction_lstm import StockAnalyzer

analyzer = StockAnalyzer()
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

results = {}
for stock in stocks:
    model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(stock, '1y', '1d')
    results[stock] = {
        'model': model,
        'predictions': future_prices,
        'dates': future_dates
    }
```

### Visualization

```python
from stock_prediction_lstm.visualization import (
    visualize_stock_data,
    visualize_prediction_comparison,
    visualize_future_predictions,
    visualize_feature_importance,
    visualize_sentiment_impact
)

# Assuming 'df' and 'model' are available from a StockAnalyzer run
# visualize_stock_data(df, 'AAPL')

# Assuming 'actual_prices' and 'predicted_prices' are available
# visualize_prediction_comparison(model, X_market_test, X_sentiment_test, y_test, 'AAPL')

# Assuming 'future_prices', 'future_dates', and 'df' are available
# visualize_future_predictions(future_prices, future_dates, df, 'AAPL')
```

## API Reference

### Core Classes

#### StockAnalyzer
Main class for stock analysis and prediction.

**Methods:**
- `run_analysis_for_stock(ticker, period, interval)`: Complete analysis pipeline
- `self_diagnostic(ticker, period)`: Run diagnostic checks

#### StockDataFetcher
Handles data fetching from various sources.

**Methods:**
- `fetch_data(ticker, period, interval)`: Fetch stock price data

#### SentimentAnalyzer
Handles sentiment data fetching and processing.

**Methods:**
- `fetch_news_sentiment(start_date, end_date)`: Fetch news sentiment data

#### StockSentimentModel
LSTM model with sentiment analysis integration.

**Methods:**
- `prepare_data(df, target_col)`: Prepares data for the model
- `build_model(market_input_dim, sentiment_input_dim)`: Builds the LSTM model
- `fit(X_market, X_sentiment, y, ...)`: Fits the model to data
- `predict(X_market, X_sentiment)`: Makes predictions
- `predict_next_days(latest_market_data, latest_sentiment_data, days)`: Predicts future prices
- `evaluate(y_true, y_pred)`: Evaluates model performance

### Utility Functions

#### Technical Indicators (via `TechnicalIndicatorGenerator` class)
- `add_technical_indicators(df, price_col, volume_col)`: Adds common technical indicators to a DataFrame

#### Plotting Functions (via `stock_prediction_lstm.visualization` module)
- `visualize_stock_data(df, ticker_symbol, output_dir)`: Visualizes stock data
- `visualize_prediction_comparison(model, X_market_test, X_sentiment_test, y_test, ticker_symbol, output_dir)`: Visualizes prediction vs actual
- `visualize_future_predictions(future_prices, future_dates, df, ticker_symbol, output_dir)`: Visualizes future predictions
- `visualize_feature_importance(df, target_col, output_dir)`: Visualizes feature importance
- `visualize_sentiment_impact(df, window, output_dir)`: Visualizes sentiment impact

## Examples

The `examples/` directory contains various usage examples:

- **basic_usage.py**: Basic usage examples
- **advanced_analysis.py**: Advanced analysis with custom parameters
- **batch_processing.py**: Process multiple stocks
- **demo/real_demo.py**: Complete demonstration script

## Testing

The package includes a comprehensive test suite covering normal use cases, edge cases, and integration scenarios.

### Running Tests

#### Quick Start
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
python run_tests.py --all

# Run specific test types
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests only
python run_tests.py --quick         # Quick tests (exclude slow ones)
```

#### Detailed Test Commands
```bash
# Run tests with coverage report
python run_tests.py --coverage

# Run tests in parallel (faster)
python run_tests.py --parallel

# Run specific test file
python run_tests.py --file tests/unit/data/fetchers/test_stock_data.py

# Check test environment
python run_tests.py --check-env
```

#### Manual pytest Commands
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=stock_prediction_lstm --cov-report=html

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests
pytest -m "not slow"                  # Skip slow tests
pytest -k "test_fetch_data"          # Run tests matching pattern
```

### Test Structure

The test suite is organized as follows:

```
tests/
├── conftest.py                      # Test configuration and fixtures
├── pytest.ini                      # Pytest configuration
├── test_data/                       # Sample test data
│   ├── sample_stock_data.csv
│   └── sample_sentiment_data.json
├── unit/                            # Unit tests
│   ├── data/
│   │   ├── fetchers/               # Data fetching tests
│   │   ├── processors/             # Data processing tests
│   │   └── storage/                # Data storage tests
│   ├── models/                     # Model tests
│   ├── analysis/                   # Analysis pipeline tests
│   └── visualization/              # Plotting tests
└── integration/                     # End-to-end tests
    └── test_end_to_end.py
```

### Test Coverage

The test suite aims for:
- **90%+ overall code coverage**
- **95%+ coverage for critical modules** (data fetchers, models, analysis)
- **Comprehensive edge case testing**
- **Integration testing for complete workflows**

### Writing New Tests

When adding new features, include tests that cover:

1. **Normal Use Cases**
   ```python
   def test_feature_normal_operation():
       # Test typical usage scenarios
       pass
   ```

2. **Edge Cases**
   ```python
   def test_feature_with_invalid_input():
       # Test error handling and boundary conditions
       pass
   ```

3. **Integration**
   ```python
   def test_feature_integration_with_other_components():
       # Test how feature works with other parts
       pass
   ```

### Test Fixtures and Mocking

Common test fixtures are available in `conftest.py`:
- `sample_stock_data` - Realistic stock price data
- `sample_sentiment_data` - Sample sentiment analysis data
- `mock_api_responses` - Mocked external API responses
- `temp_output_dir` - Temporary directory for test outputs

Example usage:
```python
def test_my_feature(sample_stock_data, mock_api_responses):
    # Use fixtures in your tests
    result = my_function(sample_stock_data)
    assert result is not None
```

### Continuous Integration

Tests are automatically run on:
- **GitHub Actions** for pull requests and commits
- **Multiple Python versions** (3.8, 3.9, 3.10, 3.11)
- **Different operating systems** (Ubuntu, macOS, Windows)

### Test Performance

- **Unit tests**: Should complete in < 30 seconds
- **Integration tests**: Should complete in < 2 minutes  
- **Full test suite**: Should complete in < 5 minutes

For detailed testing instructions and guidelines, see [agent.md](agent.md).

## Performance Considerations

- **Memory Usage**: LSTM models can be memory-intensive. Consider reducing batch size or sequence length for large datasets.
- **Training Time**: Model training time depends on data size and complexity. Use GPU acceleration when available.
- **API Limits**: External APIs have rate limits. Consider caching results for frequently accessed data.

## Troubleshooting

### Common Issues

1. **TensorFlow Installation**: Ensure TensorFlow is properly installed for your system
2. **API Key Issues**: Verify your API keys are correctly configured
3. **Data Fetching**: Check internet connection and ticker symbol validity
4. **Memory Errors**: Reduce batch size or sequence length for large datasets

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# For StockAnalyzer, you might need to pass a debug flag or configure logging directly
# analyzer = StockAnalyzer(debug=True)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/melrefaiy2018/stock_prediction_lstm.git
cd stock_prediction_lstm
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest
mypy stock_prediction_lstm/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 1.0.0 (Alpha Release)
- Initial alpha release
- Implemented Flask-based web interface
- Enhanced sentiment data fetching with multiple fallbacks
- Removed hardcoded API keys and improved API key management
- Improved docstring coverage across the codebase
- Cleaned up unused and temporary files
- Updated documentation (README, PRD)

## Support

- **Documentation**: [Full documentation](https://melrefaiy2018.github.io/stock_prediction_lstm/) (Coming Soon)
- **Issues**: [GitHub Issues](https://github.com/melrefaiy2018/stock_prediction_lstm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/melrefaiy2018/stock_prediction_lstm/discussions)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{stock_prediction_lstm,
  title={Stock Prediction LSTM: A Comprehensive Stock Prediction System},
  author={Mohamed A.A. Elrefaiy},
  year={2025},
  url={https://github.com/melrefaiy2018/stock_prediction_lstm}
}
```

## Disclaimer

This software is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment decisions.
