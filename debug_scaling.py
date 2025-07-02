"""
Debug script to identify the exact scaling issue
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def debug_scaling_issue():
    """Debug the scaling issue with a simple test"""
    
    print("=== SCALING DEBUG TEST ===")
    
    # Create test data similar to NVDA prices
    test_prices = np.array([120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 157.25])
    print(f"Original prices: {test_prices}")
    print(f"Price range: ${test_prices.min():.2f} - ${test_prices.max():.2f}")
    
    # Test the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform
    prices_2d = test_prices.reshape(-1, 1)
    scaled_prices = scaler.fit_transform(prices_2d)
    
    print(f"Scaled prices: {scaled_prices.flatten()}")
    print(f"Scaled range: [{scaled_prices.min():.6f}, {scaled_prices.max():.6f}]")
    
    # Test inverse transform
    reconstructed = scaler.inverse_transform(scaled_prices)
    
    print(f"Reconstructed prices: {reconstructed.flatten()}")
    print(f"Reconstruction error: {np.abs(test_prices - reconstructed.flatten()).max():.6f}")
    
    # Test what happens if we have values outside [0,1]
    print("\n=== TESTING OUT-OF-RANGE VALUES ===")
    
    # Simulate model output that might be outside [0,1]
    bad_outputs = np.array([[0.5], [1.2], [-0.1], [0.8], [10.0]])  # Some values outside [0,1]
    print(f"Bad model outputs: {bad_outputs.flatten()}")
    
    # What happens if we inverse transform directly?
    try:
        bad_reconstruction = scaler.inverse_transform(bad_outputs)
        print(f"Bad reconstruction: {bad_reconstruction.flatten()}")
        print(f"This could explain the 10x error!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with clipping
    clipped_outputs = np.clip(bad_outputs, 0, 1)
    print(f"Clipped outputs: {clipped_outputs.flatten()}")
    
    clipped_reconstruction = scaler.inverse_transform(clipped_outputs)
    print(f"Clipped reconstruction: {clipped_reconstruction.flatten()}")
    
    # Check scaler properties
    print(f"\n=== SCALER PROPERTIES ===")
    print(f"data_min_: {scaler.data_min_}")
    print(f"data_max_: {scaler.data_max_}")
    print(f"scale_: {scaler.scale_}")
    print(f"feature_range: {scaler.feature_range}")
    
    return True

def simulate_model_issue():
    """Simulate the exact issue we're seeing"""
    
    print("\n=== SIMULATING THE ACTUAL ISSUE ===")
    
    # Current NVDA price and prediction
    current_price = 157.25
    predicted_median = 1529.19
    error_factor = predicted_median / current_price
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Predicted median: ${predicted_median:.2f}")
    print(f"Error factor: {error_factor:.2f}x")
    
    # This suggests the model output might be around 1.0 or higher
    # when it should be around 0.1 for the current price range
    
    # Let's see what happens with different scaler ranges
    test_data = np.array([100, 120, 140, 160, 180, 200]).reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(test_data)
    
    # What if model outputs 1.0 (max) when it should output something for 157.25?
    target_price = 157.25
    expected_scaled = scaler.transform([[target_price]])[0][0]
    print(f"Expected scaled value for ${target_price}: {expected_scaled:.6f}")
    
    # What if model outputs 1.0 instead?
    wrong_output = 1.0
    wrong_reconstruction = scaler.inverse_transform([[wrong_output]])[0][0]
    print(f"If model outputs {wrong_output}, we get: ${wrong_reconstruction:.2f}")
    
    # What about if model outputs > 1.0?
    very_wrong_output = 10.0  # Way outside [0,1]
    very_wrong_reconstruction = scaler.inverse_transform([[very_wrong_output]])[0][0]
    print(f"If model outputs {very_wrong_output}, we get: ${very_wrong_reconstruction:.2f}")
    print("^ This could explain the extreme predictions!")

if __name__ == "__main__":
    debug_scaling_issue()
    simulate_model_issue()
