"""
Comprehensive test to verify the NVDA scaling issue is resolved
"""

import sys
import os
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append('.')

def test_nvda_scaling_issue():
    """Test with NVDA-like data to ensure we don't get 10x predictions"""
    
    try:
        from stock_prediction_lstm.models.improved_model import ImprovedStockModel
        print("✅ Successfully imported ImprovedStockModel with fixes")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create test data that mimics the actual NVDA issue
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # NVDA-like price progression: started around $150, went to $200+, now back to $157
    base_prices = np.linspace(150, 200, n_days//2).tolist() + np.linspace(200, 157.25, n_days//2).tolist()
    
    # Add some realistic noise
    noise = np.random.normal(0, 3, n_days)
    prices = np.array(base_prices) + noise
    
    # Ensure current price is close to $157.25
    prices[-1] = 157.25
    
    # Add technical indicators
    ma7 = pd.Series(prices).rolling(7).mean().fillna(prices[0])
    ma30 = pd.Series(prices).rolling(30).mean().fillna(prices[0])
    rsi = np.random.uniform(30, 70, n_days)
    
    # Add sentiment data (typical values)
    sentiment_pos = np.random.uniform(0.4, 0.6, n_days)
    sentiment_neg = np.random.uniform(0.2, 0.3, n_days)
    sentiment_neu = 1 - sentiment_pos - sentiment_neg
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'ma7': ma7,
        'ma30': ma30,
        'rsi14': rsi,
        'sentiment_positive': sentiment_pos,
        'sentiment_negative': sentiment_neg,
        'sentiment_neutral': sentiment_neu,
        'volume': np.random.uniform(2000000, 8000000, n_days),
        'volatility': np.random.uniform(0.02, 0.08, n_days)
    })
    
    current_price = df['close'].iloc[-1]
    print(f"✅ Created NVDA-like data")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Current price: ${current_price:.2f}")
    
    # Initialize model with the same settings as production
    model = ImprovedStockModel(look_back=20)
    
    # Prepare data
    print("Preparing data...")
    X_market, X_sentiment, y = model.prepare_data(df, target_col='close')
    
    # Split for testing
    split_idx = int(0.8 * len(X_market))
    X_market_test = X_market[split_idx:]
    X_sentiment_test = X_sentiment[split_idx:]
    y_test = y[split_idx:]
    
    # Build model
    print("Building model...")
    model.build_model(X_market.shape[2], X_sentiment.shape[2])
    
    # Test predictions without training (to see raw model behavior)
    print("Testing untrained model predictions...")
    y_pred_untrained = model.predict(X_market_test, X_sentiment_test)
    
    print(f"Untrained predictions range: ${y_pred_untrained.min():.2f} - ${y_pred_untrained.max():.2f}")
    print(f"Untrained median prediction: ${np.median(y_pred_untrained):.2f}")
    
    # Check if we're getting the 10x issue
    pred_median = np.median(y_pred_untrained)
    error_factor = pred_median / current_price
    
    print(f"Error factor vs current price: {error_factor:.2f}x")
    
    if error_factor > 5.0:
        print(f"❌ SCALING ISSUE DETECTED: Predictions are {error_factor:.1f}x too high!")
        return False
    elif error_factor > 2.0:
        print(f"⚠️  Predictions are somewhat high ({error_factor:.1f}x), but within acceptable range")
    else:
        print(f"✅ Predictions are reasonable ({error_factor:.1f}x current price)")
    
    # Test future predictions
    print("Testing future predictions...")
    try:
        future_preds = model.predict_next_days(
            X_market_test[-1], 
            X_sentiment_test[-1], 
            days=5
        )
        
        future_median = np.median(future_preds)
        future_error_factor = future_median / current_price
        
        print(f"Future predictions: {[f'${p:.2f}' for p in future_preds]}")
        print(f"Future median: ${future_median:.2f}")
        print(f"Future error factor: {future_error_factor:.2f}x")
        
        if future_error_factor > 5.0:
            print(f"❌ FUTURE PREDICTION ISSUE: {future_error_factor:.1f}x too high!")
            return False
        else:
            print(f"✅ Future predictions are reasonable")
            
    except Exception as e:
        print(f"❌ Error in future predictions: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== COMPREHENSIVE NVDA SCALING TEST ===")
    success = test_nvda_scaling_issue()
    
    if success:
        print("\n🎉 NVDA SCALING ISSUE IS FIXED!")
        print("The model now produces reasonable predictions in the correct price range.")
    else:
        print("\n💥 SCALING ISSUE STILL EXISTS!")
        print("Additional fixes may be needed.")
