"""
Backward compatibility layer - imports from new structure
"""
from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
from stock_prediction_lstm.data.processors import TechnicalIndicatorGenerator
from stock_prediction_lstm.models import StockSentimentModel
from stock_prediction_lstm.analysis import StockAnalyzer

# Preserve existing API
# For run_analysis_for_stock and self_diagnostic, we'll create instances of StockAnalyzer
# and call their methods.

def run_analysis_for_stock(ticker: str, period: str = '1y', interval: str = '1d'):
    analyzer = StockAnalyzer()
    return analyzer.run_analysis_for_stock(ticker, period, interval)

def self_diagnostic(ticker: str = 'NVDA', period: str = '1y'):
    analyzer = StockAnalyzer()
    return analyzer.self_diagnostic(ticker, period)

__all__ = [
    "StockDataFetcher",
    "SentimentAnalyzer", 
    "TechnicalIndicatorGenerator",
    "StockSentimentModel",
    "run_analysis_for_stock",
    "self_diagnostic"
]
