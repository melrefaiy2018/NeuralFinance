#!/usr/bin/env python3
"""
Debug script to check date formats from different data sources
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
import pandas as pd

# Test stock data dates
print("=== Testing Stock Data Date Format ===")
stock_fetcher = StockDataFetcher('NVDA', '1mo', '1d')
stock_df = stock_fetcher.fetch_data()

if stock_df is not None:
    print(f"Stock DataFrame shape: {stock_df.shape}")
    print(f"Stock date column dtype: {stock_df['date'].dtype}")
    print(f"Stock date sample: {stock_df['date'].iloc[0]}")
    print(f"Stock date type: {type(stock_df['date'].iloc[0])}")
    
    # Check timezone info
    if hasattr(stock_df['date'].dtype, 'tz'):
        print(f"Stock date timezone: {stock_df['date'].dtype.tz}")
    else:
        print("Stock date has no timezone attribute")
else:
    print("❌ Failed to fetch stock data")

print("\n=== Testing Sentiment Data Date Format ===")
sentiment_analyzer = SentimentAnalyzer('NVDA')
sentiment_df = sentiment_analyzer.fetch_news_sentiment(
    start_date=stock_df['date'].min(),
    end_date=stock_df['date'].max()
)

if sentiment_df is not None:
    print(f"Sentiment DataFrame shape: {sentiment_df.shape}")
    print(f"Sentiment date column dtype: {sentiment_df['date'].dtype}")
    print(f"Sentiment date sample: {sentiment_df['date'].iloc[0]}")
    print(f"Sentiment date type: {type(sentiment_df['date'].iloc[0])}")
    
    # Check timezone info
    if hasattr(sentiment_df['date'].dtype, 'tz'):
        print(f"Sentiment date timezone: {sentiment_df['date'].dtype.tz}")
    else:
        print("Sentiment date has no timezone attribute")
        
    print(f"Using synthetic data: {sentiment_analyzer.using_synthetic_data}")
else:
    print("❌ Failed to fetch sentiment data")

print("\n=== Testing Direct Merge ===")
try:
    if stock_df is not None and sentiment_df is not None:
        # Try direct merge
        combined_df = pd.merge(stock_df, sentiment_df, on='date', how='inner')
        print(f"✅ Direct merge successful: {combined_df.shape[0]} rows")
    else:
        print("❌ Cannot test merge - missing data")
except Exception as e:
    print(f"❌ Direct merge failed: {e}")
    
    # Try with date normalization
    print("\n=== Testing Normalized Merge ===")
    try:
        stock_normalized = stock_df.copy()
        sentiment_normalized = sentiment_df.copy()
        
        # Normalize dates
        if pd.api.types.is_datetime64_any_dtype(stock_normalized['date']):
            if hasattr(stock_normalized['date'].dtype, 'tz') and stock_normalized['date'].dtype.tz is not None:
                stock_normalized['date'] = stock_normalized['date'].dt.tz_localize(None)
            stock_normalized['date'] = stock_normalized['date'].dt.date
            
        if pd.api.types.is_datetime64_any_dtype(sentiment_normalized['date']):
            if hasattr(sentiment_normalized['date'].dtype, 'tz') and sentiment_normalized['date'].dtype.tz is not None:
                sentiment_normalized['date'] = sentiment_normalized['date'].dt.tz_localize(None)
            sentiment_normalized['date'] = sentiment_normalized['date'].dt.date
        
        combined_normalized = pd.merge(stock_normalized, sentiment_normalized, on='date', how='inner')
        print(f"✅ Normalized merge successful: {combined_normalized.shape[0]} rows")
        
    except Exception as e2:
        print(f"❌ Normalized merge also failed: {e2}")
