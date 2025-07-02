#!/usr/bin/env python3
"""
Example script showing how to use the stock_prediction_lstm package programmatically
"""

from stock_prediction_lstm.analysis import StockAnalyzer

def main():
    analyzer = StockAnalyzer()

    # Example 1: Run diagnostic for NVDA
    print("\nExample 1: Running diagnostic for NVDA")
    analyzer.self_diagnostic('NVDA', '1y')
    
    # Example 2: Run full analysis for AAPL
    print("\nExample 2: Running full analysis for AAPL")
    model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL', '6mo', '1d')
    
    if future_prices is not None:
        print("\nFuture price predictions for AAPL:")
        for i, price in enumerate(future_prices):
            print(f"Day {i+1}: ${price:.2f}")

if __name__ == "__main__":
    main()

