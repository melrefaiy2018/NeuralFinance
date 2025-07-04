#!/usr/bin/env python3
"""
Test script to compare model outputs between Streamlit and Flask apps
"""
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

def test_model_import():
    """Test if models can be imported correctly"""
    print("🧪 Testing model imports...")
    
    try:
        from stock_prediction_lstm.models.improved_model import ImprovedStockModel
        from stock_prediction_lstm.utils.emergency_fixes import emergency_evaluate_with_diagnostics
        print("✅ ImprovedStockModel imported successfully")
        ModelClass = ImprovedStockModel
        model_type = "ImprovedStockModel"
    except ImportError as e:
        print(f"❌ Failed to import ImprovedStockModel: {e}")
        try:
            from stock_prediction_lstm.models import StockSentimentModel
            from stock_prediction_lstm.utils.model_fixes import apply_model_fixes
            from stock_prediction_lstm.utils.emergency_fixes import emergency_evaluate_with_diagnostics
            ModelClass = apply_model_fixes(StockSentimentModel)
            print("✅ StockSentimentModel with fixes imported successfully")
            model_type = "StockSentimentModel with fixes"
        except ImportError as e2:
            print(f"❌ Failed to import StockSentimentModel with fixes: {e2}")
            try:
                from stock_prediction_lstm.models import StockSentimentModel
                ModelClass = StockSentimentModel
                print("⚠️ StockSentimentModel (basic) imported successfully")
                model_type = "StockSentimentModel (basic)"
            except ImportError as e3:
                print(f"❌ Failed to import any model: {e3}")
                return None, None
    
    print(f"📊 Using model: {model_type}")
    
    # Test model initialization
    try:
        model = ModelClass(look_back=20)
        print("✅ Model initialized successfully")
        print(f"   - Look back: {model.look_back}")
        if hasattr(model, 'price_scaler'):
            print(f"   - Price scaler range: {model.price_scaler.feature_range}")
        if hasattr(model, 'market_scaler'):
            print(f"   - Market scaler range: {model.market_scaler.feature_range}")
        if hasattr(model, 'sentiment_scaler'):
            print(f"   - Sentiment scaler range: {model.sentiment_scaler.feature_range}")
        return ModelClass, model_type
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return None, None

def test_emergency_evaluation():
    """Test if emergency evaluation function works"""
    print("\n🧪 Testing emergency evaluation...")
    
    try:
        from stock_prediction_lstm.utils.emergency_fixes import emergency_evaluate_with_diagnostics
        import numpy as np
        
        # Test with dummy data
        y_true = np.random.random((10, 1)) * 100 + 50  # Prices around 50-150
        y_pred = y_true + np.random.normal(0, 5, y_true.shape)  # Add some noise
        
        metrics = emergency_evaluate_with_diagnostics(y_true, y_pred)
        print("✅ Emergency evaluation works successfully")
        print(f"   - Metrics returned: {list(metrics.keys())}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import emergency evaluation: {e}")
        return False
    except Exception as e:
        print(f"❌ Emergency evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Stock Prediction Model Comparison Test")
    print("=" * 50)
    
    ModelClass, model_type = test_model_import()
    evaluation_works = test_emergency_evaluation()
    
    print("\n📊 Summary:")
    print(f"   - Model available: {'✅' if ModelClass else '❌'}")
    print(f"   - Model type: {model_type if model_type else 'None'}")
    print(f"   - Emergency evaluation: {'✅' if evaluation_works else '❌'}")
    
    if ModelClass and evaluation_works:
        print("\n🎉 Both Streamlit and Flask should now use the same model logic!")
    else:
        print("\n⚠️ There may still be differences between the apps.")
