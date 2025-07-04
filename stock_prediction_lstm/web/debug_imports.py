#!/usr/bin/env python3
"""
Test imports for Flask app debugging
"""
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

print("🔍 Testing Flask app imports...")
print(f"Current dir: {current_dir}")
print(f"Parent dir: {parent_dir}")
print(f"Project root: {project_root}")

try:
    print("Testing basic data imports...")
    from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
    from stock_prediction_lstm.data.processors import TechnicalIndicatorGenerator
    print("✅ Data modules imported successfully")
    
    print("Testing improved model import...")
    try:
        from stock_prediction_lstm.models.improved_model import ImprovedStockModel
        from stock_prediction_lstm.utils.emergency_fixes import emergency_evaluate_with_diagnostics
        ModelClass = ImprovedStockModel
        print("✅ Improved model imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import improved model: {e}")
        print("Testing fallback model...")
        try:
            from stock_prediction_lstm.models import StockSentimentModel
            from stock_prediction_lstm.utils.model_fixes import apply_model_fixes
            from stock_prediction_lstm.utils.emergency_fixes import emergency_evaluate_with_diagnostics
            ModelClass = apply_model_fixes(StockSentimentModel)
            print("✅ Fallback model with fixes imported successfully")
        except ImportError as e2:
            print(f"❌ Failed to import fallback model with fixes: {e2}")
            try:
                from stock_prediction_lstm.models import StockSentimentModel
                ModelClass = StockSentimentModel
                print("✅ Basic model imported successfully")
            except ImportError as e3:
                print(f"❌ Failed to import any model: {e3}")
                
except ImportError as e:
    print(f"❌ Failed to import basic data modules: {e}")
    print("This suggests a path or package structure issue")
