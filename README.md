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

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analysis.py
```

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
