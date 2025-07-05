# Troubleshooting Guide

## Quick Fixes for Common Issues

### 🔧 Issue 1: CLI Command Not Found
**Problem**: `stock-predict: command not found` or `AttributeError: module 'stock_prediction_lstm.cli.main' has no attribute 'main'`

**Solution**:
```bash
# Navigate to the package directory
cd stock_prediction_lstm/

# Reinstall the package
pip uninstall stock-prediction-lstm -y
pip install -e .

# Test the CLI
stock-predict --help
```

**Root Cause**: Entry point configuration in setup.py was incorrect.

### 🔧 Issue 2: Examples Not Found
**Problem**: `python: can't open file 'examples/basic_usage.py': [Errno 2] No such file or directory`

**Solution**:
```bash
# Make sure you're in the right directory
cd /Users/mohamed/Documents/Personal/extra/stocks/predict_stocks_LSTM/stock_prediction_lstm/

# Run from correct location
python examples/basic_usage.py
```

**Root Cause**: Running from wrong directory - examples are at package root level.

### 🔧 Issue 3: Configuration Import Errors
**Problem**: Import errors when trying to use the configuration system

**Solution**:
```bash
# Test configuration
python -c "from stock_prediction_lstm.config import Config, load_config; print('Config OK')"

# If it fails, reinstall
pip install -e . --force-reinstall
```

## Fixed Issues

### ✅ CLI Entry Point Fixed
- **Before**: `"stock-predict=stock_prediction_lstm.cli.main:cli"`
- **After**: `"stock-predict=stock_prediction_lstm.cli.main:main"`
- **Fix**: Added proper `main()` function in CLI module

### ✅ Configuration Security Fixed
- **Before**: Real API key exposed in code
- **After**: Secure placeholder with environment variable support

### ✅ Path Issues Fixed
- **Before**: Incorrect path references in setup scripts
- **After**: Proper relative path handling

## Verification Commands

### Test CLI
```bash
stock-predict --help
stock-predict analyze --ticker AAPL --period 1mo --interval 1d
stock-predict diagnostic --ticker NVDA --period 1mo
```

### Test Examples
```bash
python examples/basic_usage.py
python examples/advanced_analysis.py
```

### Test Configuration
```bash
python config/setup_api_key.py
```

### Test Package Import
```python
from stock_prediction_lstm import StockAnalyzer
from stock_prediction_lstm.config import Config
```

## Complete Reinstallation

If you're still having issues, try a complete reinstallation:

```bash
# Navigate to package directory
cd /Users/mohamed/Documents/Personal/extra/stocks/predict_stocks_LSTM/stock_prediction_lstm/

# Complete cleanup
pip uninstall stock-prediction-lstm -y
rm -rf stock_prediction_lstm.egg-info/
rm -rf build/ dist/

# Fresh installation
pip install -e .

# Verify installation
stock-predict --help
python examples/basic_usage.py
```

## API Key Configuration

If you need to set up your API key:

```bash
# Method 1: Environment variable
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# Method 2: Configuration script
python config/setup_api_key.py

# Method 3: Manual edit
nano config/keys/api_keys.py
```

Remember: Always run commands from the `stock_prediction_lstm/` directory (the one containing `setup.py`).
