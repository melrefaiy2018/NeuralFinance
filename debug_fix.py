#!/usr/bin/env python3
"""
Test the timezone fix
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
import pandas as pd

# Test updated stock data dates
print("=== Testing Updated Stock Data Date Format ===")
stock_fetcher = StockDataFetcher('NVDA', '1mo', '1d')
stock_df = stock_fetcher.fetch_data()

if stock_df is not None:
    print(f"Stock date column dtype: {stock_df['date'].dtype}")
    print(f"Stock date sample: {stock_df['date'].iloc[0]}")
    
    # Check timezone info
    if hasattr(stock_df['date'].dtype, 'tz'):
        print(f"Stock date timezone: {stock_df['date'].dtype.tz}")
    else:
        print("Stock date has no timezone attribute")

print("\n=== Testing Direct Merge After Fix ===")
sentiment_analyzer = SentimentAnalyzer('NVDA')
sentiment_df = sentiment_analyzer.fetch_news_sentiment(
    start_date=stock_df['date'].min(),
    end_date=stock_df['date'].max()
)

if sentiment_df is not None:
    print(f"Sentiment date column dtype: {sentiment_df['date'].dtype}")
    
    try:
        combined_df = pd.merge(stock_df, sentiment_df, on='date', how='inner')
        print(f"✅ Direct merge successful: {combined_df.shape[0]} rows")
        print(f"✅ Using real sentiment data: {not sentiment_analyzer.using_synthetic_data}")
    except Exception as e:
        print(f"❌ Direct merge failed: {e}")
else:
    print("❌ Failed to fetch sentiment data")
