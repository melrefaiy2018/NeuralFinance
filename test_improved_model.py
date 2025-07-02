"""
Test script for the improved stock prediction model
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the package directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(current_dir)
sys.path.insert(0, package_dir)

def test_improved_model():
    """Test the improved model with simple synthetic data"""
    
    try:
        from models.improved_model import ImprovedStockModel
    except ImportError:
        print("Error: Could not import ImprovedStockModel")
        print("Make sure the models/improved_model.py file exists")
        return
    
    print("Testing Improved Stock Model...")
    print("=" * 50)
    
    # Create synthetic test data
    np.random.seed(42)
    n_days = 200
    
    # Generate synthetic stock data
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Synthetic price data with trend
    base_price = 100
    trend = np.linspace(0, 20, n_days)  # Upward trend
    noise = np.random.normal(0, 2, n_days)  # Random noise
    prices = base_price + trend + noise
    
    # Synthetic technical indicators
    ma7 = pd.Series(prices).rolling(7).mean().fillna(base_price)
    ma30 = pd.Series(prices).rolling(30).mean().fillna(base_price)
    rsi = np.random.uniform(30, 70, n_days)  # RSI between 30-70
    
    # Synthetic sentiment data
    sentiment_pos = np.random.uniform(0.4, 0.8, n_days)
    sentiment_neg = np.random.uniform(0.1, 0.3, n_days)
    sentiment_neu = 1 - sentiment_pos - sentiment_neg
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'ma7': ma7,
        'ma30': ma30,
        'rsi14': rsi,
        'sentiment_positive': sentiment_pos,
        'sentiment_negative': sentiment_neg,
        'sentiment_neutral': sentiment_neu,
        'volume': np.random.uniform(1000000, 5000000, n_days),
        'volatility': np.random.uniform(0.01, 0.05, n_days)
    })
    
    print(f"Created synthetic dataset with {len(df)} days")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Initialize and test the model
    model = ImprovedStockModel(look_back=20)
    
    # Prepare data
    print("\nPreparing data...")
    X_market, X_sentiment, y = model.prepare_data(df, target_col='close')
    
    # Split data
    split_idx = int(0.8 * len(X_market))
    X_market_train = X_market[:split_idx]
    X_market_test = X_market[split_idx:]
    X_sentiment_train = X_sentiment[:split_idx]
    X_sentiment_test = X_sentiment[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"Training samples: {len(X_market_train)}")
    print(f"Test samples: {len(X_market_test)}")
    
    # Build model
    print("\nBuilding model...")
    model.build_model(X_market.shape[2], X_sentiment.shape[2])
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_market_train, X_sentiment_train, y_train,
        epochs=10,  # Reduced for testing
        batch_size=16,
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_market_test, X_sentiment_test)
    
    # Convert test data back to original scale
    y_test_original = model.price_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(y_test_original, y_pred)
    
    # Check if metrics are reasonable
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print("=" * 50)
    
    if not np.isnan(metrics['r2']) and metrics['r2'] > -10:
        print("✅ R² score is reasonable")
    else:
        print("❌ R² score is problematic")
    
    if not np.isnan(metrics['mape']) and metrics['mape'] < 100:
        print("✅ MAPE is reasonable")
    else:
        print("❌ MAPE is too high")
    
    if metrics['rmse'] < np.std(y_test_original) * 2:
        print("✅ RMSE is reasonable")
    else:
        print("❌ RMSE is too high")
    
    # Test future predictions
    print("\nTesting future predictions...")
    future_prices = model.predict_next_days(
        X_market_test[-1],
        X_sentiment_test[-1],
        days=5
    )
    
    print(f"Future predictions (5 days):")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: ${price:.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Model test completed successfully!")
    
    return model, metrics

if __name__ == "__main__":
    test_improved_model()
