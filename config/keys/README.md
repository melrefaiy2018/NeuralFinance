# API Keys Directory

This directory contains sensitive API key configurations for the Stock Prediction LSTM system.

## ⚠️ IMPORTANT SECURITY NOTICE

**NEVER commit API keys to version control!**

## Quick Setup

### 1. Get Your Alpha Vantage API Key
- Visit: https://www.alphavantage.co/support/#api-key
- Sign up for a free account (takes less than 20 seconds)
- Copy your API key

### 2. Configure Your API Key
- Open `api_keys.py` in this directory
- Replace `"YOUR_API_KEY_HERE"` with your actual API key
- Save the file

### 3. Verify Setup
Run the demo script to test your configuration:
```bash
cd /Users/mohamed/Documents/Personal/extra/stocks/predict_stocks_LSTM/stock_prediction_lstm/examples/demo
python real_demo.py --ticker AAPL
```

## File Structure
```
config/keys/
├── README.md          # This file
├── api_keys.py        # Your API keys (edit this!)
├── .gitignore         # Prevents accidental commits
└── api_keys.example   # Example template
```

## Troubleshooting

### Error: "API key is invalid or missing"
1. Check that you've replaced `"YOUR_API_KEY_HERE"` with your actual key
2. Verify your key is valid at: https://www.alphavantage.co/
3. Make sure there are no extra spaces or quotes around your key

### Error: "Module not found"
Make sure you're running from the correct directory and the package is installed:
```bash
cd /Users/mohamed/Documents/Personal/extra/stocks/predict_stocks_LSTM/stock_prediction_lstm
pip install -e .
```

## Getting Help
- Alpha Vantage API Documentation: https://www.alphavantage.co/documentation/
- Stock Prediction LSTM Issues: Create an issue in the project repository
