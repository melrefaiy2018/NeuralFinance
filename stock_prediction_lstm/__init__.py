"""
Stock Prediction LSTM Package

This package provides tools for fetching stock data, analyzing sentiment,
building and training LSTM models for stock price prediction, and visualizing results.
"""

from .data.fetchers import StockDataFetcher, SentimentAnalyzer
from .data.processors import TechnicalIndicatorGenerator
from .models import StockSentimentModel
from .analysis import StockAnalyzer, PortfolioAnalyzer
from .visualization import (
    visualize_stock_data,
    visualize_prediction_comparison,
    visualize_future_predictions,
    visualize_feature_importance,
    visualize_sentiment_impact,
)

__version__ = "1.0.0"
__all__ = [
    "StockDataFetcher",
    "SentimentAnalyzer",
    "TechnicalIndicatorGenerator",
    "StockSentimentModel",
    "StockAnalyzer",
    "PortfolioAnalyzer",
    "visualize_stock_data",
    "visualize_prediction_comparison",
    "visualize_future_predictions",
    "visualize_feature_importance",
    "visualize_sentiment_impact",
]
