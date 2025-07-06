"""
Model Configuration Classes for Stock Prediction LSTM

This module provides advanced configuration objects for customizing model behavior,
training parameters, and feature engineering options.
"""

from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Advanced configuration class for stock prediction models.
    
    This class allows fine-tuning of LSTM architecture, training parameters,
    data processing options, and prediction settings.
    """
    
    # LSTM Architecture
    lstm_units: List[int] = field(default_factory=lambda: [50, 25])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.2])
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    
    # Training Parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Data Parameters
    sequence_length: int = 20  # lookback window
    test_split: float = 0.2
    prediction_horizon: int = 1  # forecast horizon
    
    # Feature Engineering
    technical_indicators: List[str] = field(default_factory=lambda: ['RSI', 'MACD'])
    sentiment_sources: List[str] = field(default_factory=lambda: ['news'])
    normalize_features: bool = True
    
    # Prediction Parameters
    confidence_intervals: List[float] = field(default_factory=lambda: [0.95])
    monte_carlo_samples: int = 100
    
    # Model Type Selection
    model_type: str = 'lstm_attention'  # 'lstm_attention' or 'improved'
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if len(self.lstm_units) != len(self.dropout_rates):
            raise ValueError("lstm_units and dropout_rates must have the same length")
        
        if not (0.0 <= self.validation_split <= 1.0):
            raise ValueError("validation_split must be between 0.0 and 1.0")
            
        if not (0.0 <= self.test_split <= 1.0):
            raise ValueError("test_split must be between 0.0 and 1.0")
            
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
            
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
            
        if self.model_type not in ['lstm_attention', 'improved']:
            raise ValueError("model_type must be 'lstm_attention' or 'improved'")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary format."""
        return {
            'lstm_units': self.lstm_units,
            'dropout_rates': self.dropout_rates,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split,
            'sequence_length': self.sequence_length,
            'test_split': self.test_split,
            'prediction_horizon': self.prediction_horizon,
            'technical_indicators': self.technical_indicators,
            'sentiment_sources': self.sentiment_sources,
            'normalize_features': self.normalize_features,
            'confidence_intervals': self.confidence_intervals,
            'monte_carlo_samples': self.monte_carlo_samples,
            'model_type': self.model_type
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def default(cls) -> 'ModelConfig':
        """Get default configuration."""
        return cls()
    
    @classmethod
    def high_performance(cls) -> 'ModelConfig':
        """Get high-performance configuration with more complex architecture."""
        return cls(
            lstm_units=[128, 64, 32],
            dropout_rates=[0.2, 0.3, 0.2],
            epochs=200,
            batch_size=64,
            learning_rate=0.0005,
            early_stopping_patience=20,
            sequence_length=60,
            technical_indicators=['RSI', 'MACD', 'BB', 'STOCH', 'WILLIAMS'],
            sentiment_sources=['news', 'social', 'analyst'],
            prediction_horizon=5,
            monte_carlo_samples=1000
        )
    
    @classmethod
    def fast_training(cls) -> 'ModelConfig':
        """Get configuration optimized for fast training."""
        return cls(
            lstm_units=[25],
            dropout_rates=[0.1],
            epochs=20,
            batch_size=64,
            learning_rate=0.01,
            early_stopping_patience=5,
            sequence_length=10,
            technical_indicators=['RSI'],
            sentiment_sources=['news'],
            prediction_horizon=1,
            monte_carlo_samples=50
        )
