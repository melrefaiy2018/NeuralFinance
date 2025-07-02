# Stock Prediction LSTM

A comprehensive stock prediction system using LSTM neural networks with sentiment analysis.

## Installation

### Development Installation

To install the package in development mode (recommended for development):

```bash
cd stock_prediction_lstm
pip install -e .
```

This allows you to make changes to the code and have them immediately reflected without reinstalling.

### Regular Installation

```bash
cd stock_prediction_lstm
pip install .
```

## Usage

After installation, you can import and use the package from anywhere:

```python
from stock_prediction_lstm.analysis import StockAnalyzer
from stock_prediction_lstm.visualization import visualize_stock_data

# Create analyzer instance
analyzer = StockAnalyzer()

# Run analysis
model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL', '1y', '1d')
```

## Examples

See the `examples/` directory for usage examples:
- `basic_usage.py` - Basic usage examples
- `advanced_analysis.py` - Advanced analysis with visualization
- `demo/real_demo.py` - Complete demo script

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies
