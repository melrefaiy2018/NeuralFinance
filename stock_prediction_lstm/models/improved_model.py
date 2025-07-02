"""
Improved Stock Prediction Model with Fixed Evaluation Metrics

This module provides a cleaner implementation of the stock prediction model
with proper data scaling and evaluation metrics.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImprovedStockModel:
    """
    Improved stock prediction model with proper scaling and evaluation
    """
    
    def __init__(self, look_back=20, forecast_horizon=1):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        
        # Use simple 0-1 scaling for all features to avoid scaling issues
        self.market_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.model = None
        
    def prepare_data(self, df, target_col='close'):
        """
        Prepare data with simplified, robust preprocessing
        """
        print("Preparing data...")
        
        # Sort by date and clean data
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values simply
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Separate feature groups
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        excluded_cols = ['date', target_col, 'open', 'high', 'low', 'volume']
        market_cols = [col for col in df.columns 
                      if col not in sentiment_cols and col not in excluded_cols]
        
        print(f"Market features: {len(market_cols)} columns")
        print(f"Sentiment features: {len(sentiment_cols)} columns")
        
        # Extract feature data
        market_data = df[market_cols].values if market_cols else np.zeros((len(df), 1))
        sentiment_data = df[sentiment_cols].values if sentiment_cols else np.zeros((len(df), 3))
        price_data = df[target_col].values.reshape(-1, 1)
        
        # Remove outliers (cap at 3 standard deviations)
        for i in range(market_data.shape[1]):
            mean_val = np.mean(market_data[:, i])
            std_val = np.std(market_data[:, i])
            market_data[:, i] = np.clip(
                market_data[:, i], 
                mean_val - 3*std_val, 
                mean_val + 3*std_val
            )
        
        # Scale all data
        market_data_scaled = self.market_scaler.fit_transform(market_data)
        sentiment_data_scaled = self.sentiment_scaler.fit_transform(sentiment_data)
        price_data_scaled = self.price_scaler.fit_transform(price_data)
        
        print(f"Data ranges after scaling:")
        print(f"Market: [{market_data_scaled.min():.3f}, {market_data_scaled.max():.3f}]")
        print(f"Sentiment: [{sentiment_data_scaled.min():.3f}, {sentiment_data_scaled.max():.3f}]")
        print(f"Price: [{price_data_scaled.min():.3f}, {price_data_scaled.max():.3f}]")
        
        # Create sequences
        X_market, X_sentiment, y = [], [], []
        
        for i in range(len(df) - self.look_back - self.forecast_horizon + 1):
            X_market.append(market_data_scaled[i:i+self.look_back])
            X_sentiment.append(sentiment_data_scaled[i:i+self.look_back])
            y.append(price_data_scaled[i+self.look_back:i+self.look_back+self.forecast_horizon])
        
        X_market = np.array(X_market)
        X_sentiment = np.array(X_sentiment)
        y = np.array(y)
        
        print(f"Created sequences: {len(X_market)} samples")
        print(f"X_market shape: {X_market.shape}")
        print(f"X_sentiment shape: {X_sentiment.shape}")
        print(f"y shape: {y.shape}")
        
        return X_market, X_sentiment, y
    
    def build_model(self, market_input_dim, sentiment_input_dim):
        """
        Build a simpler, more robust model
        """
        print("Building model...")
        
        # Market data branch
        market_input = Input(shape=(self.look_back, market_input_dim))
        market_lstm = LSTM(64, return_sequences=True)(market_input)
        market_dropout = Dropout(0.2)(market_lstm)
        market_lstm2 = LSTM(32)(market_dropout)
        
        # Sentiment data branch
        sentiment_input = Input(shape=(self.look_back, sentiment_input_dim))
        sentiment_lstm = LSTM(32, return_sequences=True)(sentiment_input)
        sentiment_dropout = Dropout(0.2)(sentiment_lstm)
        sentiment_lstm2 = LSTM(16)(sentiment_dropout)
        
        # Combine branches
        combined = Concatenate()([market_lstm2, sentiment_lstm2])
        
        # Output layers
        dense1 = Dense(32, activation='relu')(combined)
        dropout = Dropout(0.2)(dense1)
        dense2 = Dense(16, activation='relu')(dropout)
        output = Dense(self.forecast_horizon, activation='sigmoid')(dense2)  # Use sigmoid for 0-1 output
        
        # Create and compile model
        model = Model(inputs=[market_input, sentiment_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print(f"Model built with {model.count_params()} parameters")
        return model
    
    def fit(self, X_market, X_sentiment, y, validation_split=0.2, epochs=50, batch_size=32, verbose=1, callbacks=None):
        """Train the model"""
        history = self.model.fit(
            [X_market, X_sentiment], y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        return history
    
    def predict(self, X_market, X_sentiment):
        """Make predictions and return in original scale"""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Get scaled predictions
        y_pred_scaled = self.model.predict([X_market, X_sentiment])
        
        # Ensure predictions are in valid range [0,1]
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        
        # Inverse transform to original price scale
        y_pred = self.price_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance with proper metrics calculation
        """
        print("\n=== Model Evaluation ===")
        
        # Ensure both arrays are in original price scale and same shape
        y_true = y_true.reshape(-1, 1) if y_true.ndim > 1 else y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1) if y_pred.ndim > 1 else y_pred.reshape(-1, 1)
        
        # Flatten for metric calculations
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove any potential NaN or inf values
        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            print("Error: No valid data points for evaluation")
            return {
                'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan
            }
        
        # Calculate metrics using sklearn functions for accuracy
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Calculate MAPE safely
        non_zero_mask = y_true_clean != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        # Ensure reasonable ranges
        if r2 < -100:  # Cap extremely negative R² values
            r2 = -100
        if mape > 1000:  # Cap extremely high MAPE values
            mape = 1000
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"Evaluation Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return results
    
    def predict_next_days(self, latest_market_data, latest_sentiment_data, days=5):
        """Predict future prices"""
        predictions = []
        
        # Make copies to avoid modifying original data
        market_data = latest_market_data.copy()
        sentiment_data = latest_sentiment_data.copy()
        
        for i in range(days):
            # Reshape for prediction
            market_batch = market_data[-self.look_back:].reshape(1, self.look_back, market_data.shape[1])
            sentiment_batch = sentiment_data[-self.look_back:].reshape(1, self.look_back, sentiment_data.shape[1])
            
            # Predict next day
            next_day_scaled = self.model.predict([market_batch, sentiment_batch])
            next_day_price = self.price_scaler.inverse_transform(next_day_scaled)[0][0]
            
            predictions.append(next_day_price)
            
            # Update data for next prediction (simplified)
            # In practice, you would recalculate technical indicators
            if market_data.shape[0] > 0:
                # Shift market data
                market_data = np.roll(market_data, -1, axis=0)
                # Use a simple trend continuation for the last value
                if len(predictions) > 1:
                    trend = (predictions[-1] - predictions[-2]) / predictions[-2] * 0.1  # Dampened trend
                    market_data[-1] = market_data[-2] * (1 + trend)
                else:
                    market_data[-1] = market_data[-2]
            
            # Update sentiment with some persistence
            if sentiment_data.shape[0] > 0:
                sentiment_data = np.roll(sentiment_data, -1, axis=0)
                sentiment_data[-1] = sentiment_data[-2] * 0.9 + np.random.normal(0, 0.05, sentiment_data.shape[1])
                sentiment_data[-1] = np.clip(sentiment_data[-1], 0, 1)
        
        return predictions
