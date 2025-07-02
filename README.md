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

## API Key Setup (Required for Sentiment Analysis)

The system uses Alpha Vantage API for real-time sentiment analysis. While the model will work without it (using synthetic sentiment data), configuring a real API key provides better predictions.

### Quick Setup

Run the setup script:
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

### Without API Key

The system will automatically fall back to synthetic sentiment data if no API key is configured. You'll see a warning message, but the model will still work.

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
