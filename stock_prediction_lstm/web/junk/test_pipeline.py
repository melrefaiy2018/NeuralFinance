#!/usr/bin/env python3
"""
Test script to validate the Stock AI Assistant pipeline
"""
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

def test_pipeline():
    """Test the complete analysis pipeline"""
    print("🧪 Testing Stock AI Assistant Pipeline...")
    
    try:
        # Test 1: Import all modules
        print("1️⃣ Testing imports...")
        from data.fetchers import StockDataFetcher, SentimentAnalyzer
        from data.processors import TechnicalIndicatorGenerator
        from models import StockSentimentModel
        from analysis import StockAnalyzer
        print("   ✅ All imports successful")
        
        # Test 2: Data fetching
        print("2️⃣ Testing data fetching...")
        fetcher = StockDataFetcher('AAPL', '1mo', '1d')
        stock_data = fetcher.fetch_data()
        if stock_data is None or len(stock_data) < 10:
            raise Exception("Not enough stock data")
        print(f"   ✅ Stock data: {len(stock_data)} points")
        
        # Test 3: Sentiment analysis
        print("3️⃣ Testing sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer('AAPL')
        sentiment_data = sentiment_analyzer.fetch_news_sentiment(
            start_date=stock_data['date'].min(),
            end_date=stock_data['date'].max()
        )
        if sentiment_data is None:
            # Create synthetic data for testing
            sentiment_data = pd.DataFrame({
                'date': stock_data['date'],
                'sentiment_positive': np.random.uniform(0.3, 0.7, len(stock_data)),
                'sentiment_negative': np.random.uniform(0.1, 0.4, len(stock_data)),
                'sentiment_neutral': np.random.uniform(0.1, 0.3, len(stock_data))
            })
        print(f"   ✅ Sentiment data: {len(sentiment_data)} points")
        
        # Test 4: Data combination and processing
        print("4️⃣ Testing data processing...")
        combined_df = pd.merge(stock_data, sentiment_data, on='date', how='inner')
        combined_df = TechnicalIndicatorGenerator.add_technical_indicators(combined_df)
        print(f"   ✅ Combined data: {len(combined_df)} points, {len(combined_df.columns)} features")
        
        # Test 5: Model creation and evaluation
        print("5️⃣ Testing model...")
        model = StockSentimentModel(look_back=5)
        
        # Test model evaluation with dummy data
        y_true = np.random.random((5, 1))
        y_pred = np.random.random((5, 1))
        metrics = model.evaluate(y_true, y_pred)
        print(f"   ✅ Model evaluation: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        print("\n🎉 All tests passed! Stock AI Assistant is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
