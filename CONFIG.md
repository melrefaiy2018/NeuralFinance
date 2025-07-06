# Configuration Guide

This guide explains how to configure the Neural Finance package for optimal performance.

## Overview

The configuration system supports multiple methods for setting up API keys and customizing system behavior:

1. **Environment Variables** (Recommended for production)
2. **Configuration Files** (Good for development)
3. **Runtime Configuration** (Programmatic setup)

## API Key Setup

### Method 1: Environment Variables (Recommended)

```bash
# Set environment variable (Linux/Mac)
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# Set environment variable (Windows)
set ALPHA_VANTAGE_API_KEY=your_api_key_here

# Or add to your shell profile (.bashrc, .zshrc, etc.)
echo 'export ALPHA_VANTAGE_API_KEY="your_api_key_here"' >> ~/.bashrc
```

### Method 2: Configuration File

```bash
# Option A: Use the setup script
python config/setup_api_key.py

# Option B: Use the shell script
./config/setup_api_key.sh

# Option C: Manual setup
# 1. Copy the template
cp config/keys/api_keys.example config/keys/api_keys.py

# 2. Edit the file and replace YOUR_API_KEY_HERE with your actual key
nano config/keys/api_keys.py
```

### Method 3: Runtime Configuration

```python
from neural_finance.config import Config

# Set API key programmatically
Config.ALPHA_VANTAGE_API_KEY = "your_api_key_here"
```

## Getting an API Key

1. **Visit Alpha Vantage**: https://www.alphavantage.co/support/#api-key
2. **Sign up** for a free account (takes less than 20 seconds)
3. **Copy your API key** - it will look like: `ABCD1234EFGH5678`
4. **Configure using one of the methods above**

## Configuration Options

### Model Configuration

```python
from neural_finance.config import Config

# LSTM Model Parameters
Config.DEFAULT_LSTM_UNITS = 50          # Number of LSTM units
Config.DEFAULT_DROPOUT_RATE = 0.2       # Dropout rate (0.0-1.0)
Config.DEFAULT_EPOCHS = 50              # Training epochs
Config.DEFAULT_BATCH_SIZE = 32          # Batch size
Config.DEFAULT_SEQUENCE_LENGTH = 60     # Input sequence length

# Prediction Parameters
Config.DEFAULT_LOOKBACK = 20            # Days to look back
Config.DEFAULT_PREDICTION_DAYS = 5      # Days to predict forward
```

### Data Configuration

```python
# Data Fetching
Config.DEFAULT_PERIOD = "1y"            # Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
Config.DEFAULT_INTERVAL = "1d"          # Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
Config.DEFAULT_CACHE_DURATION = 3600    # Cache duration in seconds

# Feature Flags
Config.CACHE_ENABLED = True             # Enable data caching
Config.SENTIMENT_ANALYSIS_ENABLED = True    # Enable sentiment analysis
Config.FALLBACK_TO_SYNTHETIC_SENTIMENT = True  # Use synthetic data if API unavailable
```

### Technical Indicators

```python
# RSI (Relative Strength Index)
Config.DEFAULT_RSI_PERIOD = 14

# MACD (Moving Average Convergence Divergence)
Config.DEFAULT_MACD_FAST = 12
Config.DEFAULT_MACD_SLOW = 26
Config.DEFAULT_MACD_SIGNAL = 9

# Bollinger Bands
Config.DEFAULT_BOLLINGER_PERIOD = 20
Config.DEFAULT_BOLLINGER_STD = 2
```

### System Settings

```python
# Debugging and Logging
Config.DEBUG = False                    # Enable debug mode
Config.VERBOSE_LOGGING = False          # Enable verbose logging

# API Rate Limiting
Config.ALPHA_VANTAGE_CALLS_PER_MINUTE = 5
Config.ALPHA_VANTAGE_CALLS_PER_DAY = 500
```

## Environment Variables

All configuration options can be overridden with environment variables:

```bash
# Model configuration
export DEFAULT_LSTM_UNITS=100
export DEFAULT_EPOCHS=100
export DEFAULT_BATCH_SIZE=64

# Data configuration
export DEFAULT_PERIOD="2y"
export DEFAULT_INTERVAL="1d"

# System configuration
export DEBUG=true
export VERBOSE_LOGGING=true
export CACHE_ENABLED=false
```

## Security Best Practices

1. **Never commit API keys** - Use `.gitignore` to exclude sensitive files
2. **Use environment variables** for production deployments
3. **Rotate API keys** regularly
4. **Use different keys** for development and production
5. **Monitor API usage** to detect unauthorized access

This configuration system provides flexibility while maintaining security and ease of use.
