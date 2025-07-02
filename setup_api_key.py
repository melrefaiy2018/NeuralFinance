#!/usr/bin/env python3
"""
API Key Configuration Helper for Stock Prediction LSTM

This script helps users set up their Alpha Vantage API key for sentiment analysis.
"""

import os
import sys
from pathlib import Path

def main():
    print("🔑 Stock Prediction LSTM - API Key Setup")
    print("=" * 50)
    
    # Find the keys directory
    current_dir = Path(__file__).parent
    keys_dir = current_dir / "config" / "keys"
    api_keys_file = keys_dir / "api_keys.py"
    example_file = keys_dir / "api_keys.example"
    
    print(f"API keys directory: {keys_dir}")
    
    # Check if keys directory exists
    if not keys_dir.exists():
        print("❌ Keys directory not found!")
        print(f"Expected location: {keys_dir}")
        return False
    
    # Check if api_keys.py exists
    if not api_keys_file.exists():
        if example_file.exists():
            print("📋 Creating api_keys.py from template...")
            # Copy the example file
            with open(example_file, 'r') as src:
                content = src.read()
            with open(api_keys_file, 'w') as dst:
                dst.write(content)
            print(f"✅ Created {api_keys_file}")
        else:
            print("❌ Template file not found!")
            return False
    
    # Check current API key
    print("\n🔍 Checking current API key configuration...")
    
    with open(api_keys_file, 'r') as f:
        content = f.read()
    
    if 'YOUR_API_KEY_HERE' in content:
        print("⚠️  API key not configured (still using placeholder)")
        
        # Prompt user for API key
        print("\n🚀 Let's configure your Alpha Vantage API key!")
        print("1. Visit: https://www.alphavantage.co/support/#api-key")
        print("2. Sign up for a free account (takes less than 20 seconds)")
        print("3. Copy your API key")
        print("")
        
        api_key = input("Enter your Alpha Vantage API key (or press Enter to skip): ").strip()
        
        if api_key:
            # Update the file
            updated_content = content.replace('YOUR_API_KEY_HERE', api_key)
            with open(api_keys_file, 'w') as f:
                f.write(updated_content)
            print(f"✅ API key updated in {api_keys_file}")
            
            # Test the API key
            print("\n🧪 Testing API key...")
            test_result = test_api_key(api_key)
            if test_result:
                print("✅ API key is working correctly!")
                print("\n🎉 Setup complete! You can now run the stock prediction model.")
            else:
                print("❌ API key test failed. Please check your key and try again.")
                
        else:
            print("⏭️  Skipped API key configuration.")
            print(f"You can manually edit {api_keys_file} later.")
    else:
        print("✅ API key appears to be configured")
        
        # Extract the API key for testing
        lines = content.split('\n')
        api_key = None
        for line in lines:
            if line.strip().startswith('ALPHA_VANTAGE_API_KEY') and '=' in line:
                api_key = line.split('=')[1].strip().strip('"\'')
                break
        
        if api_key:
            print("\n🧪 Testing API key...")
            test_result = test_api_key(api_key)
            if test_result:
                print("✅ API key is working correctly!")
            else:
                print("❌ API key test failed. Please check your configuration.")
        else:
            print("⚠️  Could not extract API key for testing")
    
    print(f"\n📁 Configuration file location: {api_keys_file}")
    print("💡 You can edit this file directly to update your API key anytime.")
    
    return True

def test_api_key(api_key: str) -> bool:
    """Test if the provided API key works"""
    try:
        import requests
        
        test_url = "https://www.alphavantage.co/query"
        test_params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "AAPL",
            "apikey": api_key
        }
        
        response = requests.get(test_url, params=test_params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if "Error Message" in data:
                print(f"❌ API Error: {data['Error Message']}")
                return False
            elif "Note" in data:
                print(f"⚠️  Rate Limited: {data['Note']}")
                return False
            elif "Time Series (Daily)" in data:
                return True
                
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)
