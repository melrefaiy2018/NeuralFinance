#!/usr/bin/env python3
"""
Example script showing how to use the stock_prediction_lstm package for advanced analysis.
"""

from stock_prediction_lstm.analysis import StockAnalyzer
from stock_prediction_lstm.visualization import visualize_stock_data, visualize_future_predictions

def main():
    analyzer = StockAnalyzer()

    # Run full analysis for a single stock
    ticker = 'GOOGL'
    period = '2y'
    interval = '1d'
    print(f"\nRunning full analysis for {ticker}...")
    model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(ticker, period, interval)

    if df is not None and future_prices is not None:
        # Visualize the results
        print("\nVisualizing stock data...")
        visualize_stock_data(df, ticker)

        print("\nVisualizing future predictions...")
        visualize_future_predictions(future_prices, future_dates, df, ticker)

if __name__ == "__main__":
    main()

