# Stock Prediction LSTM

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A comprehensive stock prediction system using LSTM neural networks with sentiment analysis and technical indicators. This package provides a complete solution for stock price prediction, combining deep learning models with real-time sentiment analysis and technical analysis indicators.

## Features

- **LSTM Neural Networks**: Advanced deep learning models for time series prediction
- **Sentiment Analysis**: Real-time sentiment analysis using Alpha Vantage API
- **Technical Indicators**: Comprehensive technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- **Visualization**: Rich visualizations for stock data, predictions, and analysis
- **Web Interface**: Streamlit-based web interface for interactive analysis
- **CLI Tool**: Command-line interface for batch processing and automation
- **Flexible API**: Easy-to-use Python API for integration into existing workflows

## Installation

### From PyPI (Recommended)

```bash
pip install stock-prediction-lstm
```

### From Source

```bash
git clone https://github.com/yourusername/stock_prediction_lstm.git
cd stock_prediction_lstm
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/stock_prediction_lstm.git
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
# Launch Streamlit web interface
streamlit run stock_prediction_lstm/web/app.py
```

## API Key Setup (Optional but Recommended)

The system uses Alpha Vantage API for real-time sentiment analysis. While the model works without it using synthetic sentiment data, configuring a real API key provides better predictions.

### Quick Setup

```bash
# Interactive setup
python setup_api_key.py

# Or use the shell script
./setup_api_key.sh
```

### Manual Setup

1. **Get a free Alpha Vantage API key** (takes < 20 seconds):
   - Visit: https://www.alphavantage.co/support/#api-key
   - Sign up and copy your API key

2. **Configure the key**:
   - Navigate to: `config/keys/`
   - Edit `api_keys.py`
   - Replace `"YOUR_API_KEY_HERE"` with your actual API key

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
data = fetcher.fetch_stock_data('AAPL', '2y', '1d')

# Create and train model
model = StockSentimentModel()
trained_model = model.train_model(data)

# Make predictions
predictions = model.predict_future_prices(data, days=30)
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

# Visualize stock data with technical indicators
visualize_stock_data(df, 'AAPL')

# Visualize prediction vs actual
visualize_prediction_comparison(actual_prices, predicted_prices, 'AAPL')

# Visualize future predictions
visualize_future_predictions(future_prices, future_dates, 'AAPL')
```

## API Reference

### Core Classes

#### StockAnalyzer
Main class for stock analysis and prediction.

**Methods:**
- `run_analysis_for_stock(ticker, period, interval)`: Complete analysis pipeline
- `self_diagnostic(ticker, period)`: Run diagnostic checks
- `configure_data(**kwargs)`: Configure data fetching parameters
- `configure_model(**kwargs)`: Configure model parameters

#### StockDataFetcher
Handles data fetching from various sources.

**Methods:**
- `fetch_stock_data(ticker, period, interval)`: Fetch stock price data
- `fetch_sentiment_data(ticker)`: Fetch sentiment analysis data
- `calculate_technical_indicators(df)`: Calculate technical indicators

#### StockSentimentModel
LSTM model with sentiment analysis integration.

**Methods:**
- `train_model(data)`: Train the LSTM model
- `predict_future_prices(data, days)`: Predict future prices
- `evaluate_model(test_data)`: Evaluate model performance

### Utility Functions

#### Technical Indicators
- `calculate_rsi(df, period=14)`: Relative Strength Index
- `calculate_macd(df)`: MACD indicator
- `calculate_bollinger_bands(df, period=20)`: Bollinger Bands
- `calculate_moving_averages(df, periods)`: Moving averages

#### Sentiment Analysis
- `analyze_sentiment(ticker)`: Analyze stock sentiment
- `fetch_news_sentiment(ticker)`: Fetch news sentiment
- `calculate_sentiment_score(text)`: Calculate sentiment score

## Examples

The `examples/` directory contains various usage examples:

- **basic_usage.py**: Basic usage examples
- **advanced_analysis.py**: Advanced analysis with custom parameters
- **batch_processing.py**: Process multiple stocks
- **demo/real_demo.py**: Complete demonstration script
- **web_demo.py**: Web interface demonstration

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stock_prediction_lstm

# Run specific test file
pytest tests/test_analysis.py
```

## Performance Considerations

- **Memory Usage**: LSTM models can be memory-intensive. Consider reducing batch size or sequence length for large datasets.
- **Training Time**: Model training time depends on data size and complexity. Use GPU acceleration when available.
- **API Limits**: Alpha Vantage API has rate limits. Consider caching results for frequently accessed data.

## Troubleshooting

### Common Issues

1. **TensorFlow Installation**: Ensure TensorFlow is properly installed for your system
2. **API Key Issues**: Verify your Alpha Vantage API key is correctly configured
3. **Data Fetching**: Check internet connection and ticker symbol validity
4. **Memory Errors**: Reduce batch size or sequence length for large datasets

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = StockAnalyzer(debug=True)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/stock_prediction_lstm.git
cd stock_prediction_lstm
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest
black .
flake8 .
mypy stock_prediction_lstm/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 1.0.0 (Current)
- Initial release
- LSTM model implementation
- Sentiment analysis integration
- Technical indicators
- Web interface
- CLI tool
- Comprehensive documentation

## Support

- **Documentation**: [Full documentation](https://yourusername.github.io/stock_prediction_lstm/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/stock_prediction_lstm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/stock_prediction_lstm/discussions)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{stock_prediction_lstm,
  title={Stock Prediction LSTM: A Comprehensive Stock Prediction System},
  author={Mohamed A.A. Elrefaiy},
  year={2025},
  url={https://github.com/yourusername/stock_prediction_lstm}
}
```

## Disclaimer

This software is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment decisions.
