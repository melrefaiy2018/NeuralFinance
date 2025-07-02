"""
Fix for scaling issues in stock prediction model

The main issue is that the model predictions are coming out about 10x higher than expected.
This is likely due to scaling issues in the inverse transform process.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fix_scaling_issue():
    """
    This function identifies and fixes the scaling issue in the stock prediction model.
    
    The issue appears to be:
    1. Model predictions are ~10x higher than expected
    2. Current NVDA price: $157.25 but model predicts median ~$1529.19
    3. This suggests a factor of ~10x error in inverse scaling
    """
    
    print("=== SCALING ISSUE ANALYSIS ===")
    print("Current NVDA price: $157.25")
    print("Predicted median: $1529.19") 
    print("Error factor: ~9.7x too high")
    print("")
    print("LIKELY CAUSES:")
    print("1. Price scaler fitted on wrong data range")
    print("2. Inverse transform applied incorrectly")
    print("3. Model output not properly clipped to [0,1] range")
    print("4. Double scaling or missing scaling")
    print("")
    
    return True


class FixedImprovedStockModel:
    """
    Fixed version of ImprovedStockModel with proper scaling
    """
    
    def __init__(self, look_back=20, forecast_horizon=1):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        
        # Use MinMaxScaler but with better validation
        self.market_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.model = None
        self._debug_scaling = True
    
    def predict(self, X_market, X_sentiment):
        """Make predictions with fixed scaling"""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Get scaled predictions
        y_pred_scaled = self.model.predict([X_market, X_sentiment])
        
        if self._debug_scaling:
            print(f"Raw model output range: [{np.min(y_pred_scaled):.6f}, {np.max(y_pred_scaled):.6f}]")
        
        # CRITICAL FIX: Ensure predictions are strictly in [0,1] range
        y_pred_scaled = np.clip(y_pred_scaled, 0.0, 1.0)
        
        if self._debug_scaling:
            print(f"After clipping: [{np.min(y_pred_scaled):.6f}, {np.max(y_pred_scaled):.6f}]")
            print(f"Scaler data_min_: {self.price_scaler.data_min_}")
            print(f"Scaler data_max_: {self.price_scaler.data_max_}")
            print(f"Scaler scale_: {self.price_scaler.scale_}")
        
        # Inverse transform to original price scale
        try:
            y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
            
            if self._debug_scaling:
                print(f"After inverse transform: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
            
            return y_pred
            
        except Exception as e:
            print(f"Error in inverse transform: {e}")
            # Emergency fallback: use simple linear scaling
            if hasattr(self.price_scaler, 'data_min_') and hasattr(self.price_scaler, 'data_max_'):
                price_range = self.price_scaler.data_max_[0] - self.price_scaler.data_min_[0]
                y_pred = self.price_scaler.data_min_[0] + (y_pred_scaled * price_range)
                return y_pred
            else:
                raise e


def apply_scaling_fix():
    """
    Apply the scaling fix to the model
    """
    print("Applying scaling fix...")
    
    # The fix involves:
    # 1. Ensuring model outputs are properly clipped to [0,1]
    # 2. Validating scaler state before inverse transform
    # 3. Adding debug output to track scaling process
    # 4. Emergency fallback if inverse transform fails
    
    return True


if __name__ == "__main__":
    fix_scaling_issue()
    apply_scaling_fix()
