# Test configuration and fixtures for Stock Prediction LSTM package

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json


@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible test data
    
    base_price = 100
    data = []
    
    for i, date in enumerate(dates):
        # Generate realistic stock price movements
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price = base_price * (1 + change)
        
        high = base_price * (1 + np.random.uniform(0, 0.03))
        low = base_price * (1 - np.random.uniform(0, 0.03))
        open_price = base_price + np.random.normal(0, base_price * 0.01)
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(base_price, 2),
            'volume': volume,
            'price': round(base_price, 2)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_sentiment_data():
    """Provide sample sentiment data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    data = []
    for date in dates:
        # Generate realistic sentiment scores that sum to 1
        pos = np.random.uniform(0.1, 0.8)
        neg = np.random.uniform(0.1, 0.8 - pos)
        neu = 1.0 - pos - neg
        
        data.append({
            'date': date,
            'sentiment_positive': round(pos, 3),
            'sentiment_negative': round(neg, 3),
            'sentiment_neutral': round(neu, 3)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_api_responses():
    """Mock external API responses."""
    return {
        'yahoo_finance_success': {
            'AAPL': {
                'Open': [150.0, 151.0, 152.0],
                'High': [155.0, 156.0, 157.0],
                'Low': [149.0, 150.0, 151.0],
                'Close': [154.0, 155.0, 156.0],
                'Volume': [1000000, 1100000, 1200000]
            }
        },
        'alpha_vantage_success': {
            'Time Series (Daily)': {
                '2023-01-01': {
                    '1. open': '150.0',
                    '2. high': '155.0',
                    '3. low': '149.0',
                    '4. close': '154.0',
                    '5. volume': '1000000'
                }
            }
        },
        'alpha_vantage_error': {
            'Error Message': 'Invalid API call'
        },
        'alpha_vantage_rate_limit': {
            'Note': 'Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute'
        },
        'news_api_success': {
            'articles': [
                {
                    'title': 'Stock rises',
                    'description': 'Positive news about the stock',
                    'publishedAt': '2023-01-01T10:00:00Z',
                    'content': 'Very positive outlook for the company'
                }
            ]
        }
    }


@pytest.fixture
def temp_output_dir():
    """Provide temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def minimal_stock_data():
    """Provide minimal stock data for edge case testing."""
    data = {
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [99.0, 100.0],
        'close': [101.0, 102.0],
        'volume': [1000, 1100],
        'price': [101.0, 102.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def corrupted_data():
    """Provide intentionally corrupted data for testing."""
    data = {
        'date': [datetime(2023, 1, 1), None, 'invalid_date'],
        'open': [100.0, 'invalid', np.inf],
        'high': [102.0, -999, None],
        'low': [99.0, np.nan, 'text'],
        'close': [101.0, None, -1],
        'volume': [1000, 'abc', np.nan],
        'price': [101.0, None, 'invalid']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([[150.0], [151.0], [152.0]])
    model.evaluate.return_value = {
        'rmse': 2.5,
        'mae': 1.8,
        'r2': 0.85,
        'mape': 1.2,
        'mse': 6.25
    }
    return model


@pytest.fixture
def mock_yfinance():
    """Mock yfinance responses."""
    with patch('yfinance.download') as mock_download:
        # Create sample DataFrame with MultiIndex columns (as yfinance returns)
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            ('Open', 'AAPL'): np.random.uniform(150, 160, 10),
            ('High', 'AAPL'): np.random.uniform(155, 165, 10),
            ('Low', 'AAPL'): np.random.uniform(145, 155, 10),
            ('Close', 'AAPL'): np.random.uniform(150, 160, 10),
            ('Volume', 'AAPL'): np.random.randint(1000000, 5000000, 10),
        }, index=dates)
        
        mock_download.return_value = data
        yield mock_download


@pytest.fixture
def mock_requests():
    """Mock requests for API calls."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return os.path.join(os.path.dirname(__file__), 'test_data')


# Test data creation functions
def create_test_data_files(test_data_dir):
    """Create test data files if they don't exist."""
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create sample stock data CSV
    stock_data_path = os.path.join(test_data_dir, 'sample_stock_data.csv')
    if not os.path.exists(stock_data_path):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        base_price = 100
        data = []
        
        for date in dates:
            change = np.random.normal(0, 0.02)
            base_price = base_price * (1 + change)
            data.append({
                'date': date,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': np.random.randint(1000000, 5000000),
                'price': base_price
            })
        
        pd.DataFrame(data).to_csv(stock_data_path, index=False)
    
    # Create sample sentiment data JSON
    sentiment_data_path = os.path.join(test_data_dir, 'sample_sentiment_data.json')
    if not os.path.exists(sentiment_data_path):
        sentiment_data = {
            'AAPL': [
                {
                    'date': '2023-01-01',
                    'sentiment_positive': 0.6,
                    'sentiment_negative': 0.2,
                    'sentiment_neutral': 0.2
                },
                {
                    'date': '2023-01-02',
                    'sentiment_positive': 0.7,
                    'sentiment_negative': 0.1,
                    'sentiment_neutral': 0.2
                }
            ]
        }
        
        with open(sentiment_data_path, 'w') as f:
            json.dump(sentiment_data, f, indent=2)


# Pytest configuration
def pytest_configure():
    """Configure pytest."""
    # Create test data files
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    create_test_data_files(test_data_dir)


# Custom markers
pytest.main.pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]
