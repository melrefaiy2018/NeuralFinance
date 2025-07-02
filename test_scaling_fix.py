"""
Quick test to verify the scaling fix is working
"""

import sys
import os
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append('.')

def test_scaling_fix():
    """Test that the scaling fix prevents extreme predictions"""
    
    try:
        from stock_prediction_lstm.models.improved_model import ImprovedStockModel
        print("✅ Successfully imported ImprovedStockModel")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create test data with NVDA-like prices
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # Realistic NVDA price range: $120-$180
    base_price = 150
    prices = base_price + np.random.normal(0, 10, n_days)
    prices = np.clip(prices, 120, 180)  # Keep in realistic range
    
    # Add some technical indicators
    ma7 = pd.Series(prices).rolling(7).mean().fillna(prices[0])
    rsi = np.random.uniform(30, 70, n_days)
    
    # Add sentiment data
    sentiment_pos = np.random.uniform(0.3, 0.7, n_days)
    sentiment_neg = np.random.uniform(0.1, 0.3, n_days)
    sentiment_neu = 1 - sentiment_pos - sentiment_neg
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'ma7': ma7,
        'rsi14': rsi,
        'sentiment_positive': sentiment_pos,
        'sentiment_negative': sentiment_neg,
        'sentiment_neutral': sentiment_neu,
        'volume': np.random.uniform(1000000, 5000000, n_days)
    })
    
    print(f"✅ Created test data with price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Initialize model
    model = ImprovedStockModel(look_back=10)
    
    # Prepare data
    print("Preparing data...")
    X_market, X_sentiment, y = model.prepare_data(df, target_col='close')
    
    # Build a simple model for testing
    print("Building model...")
    model.build_model(X_market.shape[2], X_sentiment.shape[2])
    
    # Make a quick prediction to test scaling
    print("Testing prediction scaling...")
    test_pred = model.predict(X_market[:1], X_sentiment[:1])
    
    print(f"Raw prediction: {test_pred}")
    print(f"Prediction range: ${test_pred.min():.2f} - ${test_pred.max():.2f}")
    
    # Check if prediction is reasonable
    original_price_range = [df['close'].min(), df['close'].max()]
    is_reasonable = (test_pred.min() >= original_price_range[0] * 0.5 and 
                    test_pred.max() <= original_price_range[1] * 2.0)
    
    if is_reasonable:
        print("✅ Predictions are in reasonable range!")
        return True
    else:
        print(f"❌ Predictions are unreasonable!")
        print(f"Expected range: ${original_price_range[0]*0.5:.2f} - ${original_price_range[1]*2:.2f}")
        return False
        
if __name__ == "__main__":
    print("=== TESTING SCALING FIX ===")
    success = test_scaling_fix()
    if success:
        print("\n🎉 Scaling fix test PASSED!")
    else:
        print("\n💥 Scaling fix test FAILED!")
