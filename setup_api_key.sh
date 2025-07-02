#!/bin/bash

# Stock Prediction LSTM - Quick API Key Setup
# This script helps configure your Alpha Vantage API key

echo "🔑 Stock Prediction LSTM - Quick Setup"
echo "=================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
KEYS_DIR="$SCRIPT_DIR/stock_prediction_lstm/config/keys"
API_KEYS_FILE="$KEYS_DIR/api_keys.py"

echo "📁 Keys directory: $KEYS_DIR"

# Check if keys directory exists
if [ ! -d "$KEYS_DIR" ]; then
    echo "❌ Keys directory not found at: $KEYS_DIR"
    exit 1
fi

# Check if api_keys.py exists
if [ ! -f "$API_KEYS_FILE" ]; then
    echo "📋 Creating api_keys.py from template..."
    cp "$KEYS_DIR/api_keys.example" "$API_KEYS_FILE"
    echo "✅ Created api_keys.py"
fi

echo ""
echo "🚀 To configure your Alpha Vantage API key:"
echo "1. Get a free API key at: https://www.alphavantage.co/support/#api-key"
echo "2. Edit the file: $API_KEYS_FILE"
echo "3. Replace 'YOUR_API_KEY_HERE' with your actual API key"
echo ""

# Ask if user wants to open the file
read -p "Would you like to open the API keys file now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Try different editors
    if command -v code &> /dev/null; then
        echo "🔧 Opening in VS Code..."
        code "$API_KEYS_FILE"
    elif command -v nano &> /dev/null; then
        echo "🔧 Opening in nano..."
        nano "$API_KEYS_FILE"
    elif command -v vim &> /dev/null; then
        echo "🔧 Opening in vim..."
        vim "$API_KEYS_FILE"
    else
        echo "📝 Please edit this file manually: $API_KEYS_FILE"
    fi
else
    echo "📝 Remember to edit: $API_KEYS_FILE"
fi

echo ""
echo "💡 After updating your API key, you can test the setup by running:"
echo "   python examples/demo/real_demo.py --ticker AAPL"
echo ""
echo "✅ Setup complete!"
