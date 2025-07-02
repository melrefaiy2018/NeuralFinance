import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, MultiHeadAttention, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import pandas as pd
from typing import Tuple, List, Dict

class StockSentimentModel:
    """
    The main model class implementing sentiment-based prediction with multi-head attention
    """
    
    def __init__(self, look_back: int = 20, forecast_horizon: int = 1):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.market_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_importance = {}
        
        self.original_price_data = None
        self.transformed_price_data = None
        self.last_actual_price = None
        
    def prepare_data(self, df: np.ndarray, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            
        df = self._handle_missing_values(df)
        
        price_data = df[target_col].values.reshape(-1, 1)
        self.original_price_data = price_data.copy()
        
        if np.all(price_data > 0):
            price_data_transformed = np.log1p(price_data)
            self.transformed_price_data = price_data_transformed.copy()
            print("Applied log transform to price data")
        else:
            price_data_transformed = price_data
            self.transformed_price_data = price_data_transformed.copy()
            
        self.price_scaler.fit(price_data_transformed)
        
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        excluded_cols = ['date', target_col, 'open', 'high', 'low', 'price', 'volume']
        market_cols = [col for col in df.columns if col not in sentiment_cols and col not in excluded_cols]
        
        print(f"Market features ({len(market_cols)}): {market_cols[:5]}...")
        print(f"Sentiment features ({len(sentiment_cols)}): {sentiment_cols}")
        
        market_data = df[market_cols].values
        
        # Handle case when no sentiment data is available
        if len(sentiment_cols) == 0:
            print("No sentiment features found - using market data only")
            sentiment_data = np.zeros((len(market_data), 1))  # Create dummy sentiment data
            self.has_sentiment = False
        else:
            sentiment_data = df[sentiment_cols].values
            self.has_sentiment = True
        
        self.market_scaler.fit(market_data)
        self.sentiment_scaler.fit(sentiment_data)
        
        market_data_scaled = self.market_scaler.transform(market_data)
        sentiment_data_scaled = self.sentiment_scaler.transform(sentiment_data)
        target_data_scaled = self.price_scaler.transform(price_data_transformed)
        
        X_market, X_sentiment, y = self._create_sequences(
            market_data_scaled, sentiment_data_scaled, target_data_scaled
        )
        
        self.last_actual_price = price_data[-1][0]
        
        return X_market, X_sentiment, y
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_percent = df.isnull().sum() / len(df) * 100
        cols_to_drop = missing_percent[missing_percent > 10].index.tolist()
        if cols_to_drop:
            print(f"Dropping columns with >10% missing values: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        df = df.fillna(df.median())
        
        return df
    
    def _create_sequences(self, market_data: np.ndarray, sentiment_data: np.ndarray, 
                         target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_market, X_sentiment, y = [], [], []
        
        for i in range(len(market_data) - self.look_back - self.forecast_horizon + 1):
            X_market.append(market_data[i:i+self.look_back])
            X_sentiment.append(sentiment_data[i:i+self.look_back])
            y.append(target_data[i+self.look_back:i+self.look_back+self.forecast_horizon])
            
        return np.array(X_market), np.array(X_sentiment), np.array(y)
    
    def build_model(self, market_input_dim: int, sentiment_input_dim: int) -> Model:
        market_input = Input(shape=(self.look_back, market_input_dim))
        market_lstm1 = LSTM(64, return_sequences=True)(market_input)
        market_dropout1 = Dropout(0.2)(market_lstm1)
        market_lstm2 = LSTM(32, return_sequences=True)(market_dropout1)
        market_dropout2 = Dropout(0.2)(market_lstm2)
        
        market_attention = MultiHeadAttention(num_heads=4, key_dim=16)(market_dropout2, market_dropout2)
        market_residual = Concatenate()([market_dropout2, market_attention])
        
        # Only add sentiment branch if sentiment data is available
        if hasattr(self, 'has_sentiment') and self.has_sentiment:
            sentiment_input = Input(shape=(self.look_back, sentiment_input_dim))
            sentiment_lstm1 = LSTM(32, return_sequences=True)(sentiment_input)
            sentiment_dropout1 = Dropout(0.2)(sentiment_lstm1)
            sentiment_lstm2 = LSTM(16, return_sequences=True)(sentiment_dropout1)
            sentiment_dropout2 = Dropout(0.2)(sentiment_lstm2)
            
            sentiment_attention = MultiHeadAttention(num_heads=2, key_dim=8)(sentiment_dropout2, sentiment_dropout2)
            sentiment_residual = Concatenate()([sentiment_dropout2, sentiment_attention])
            
            combined = Concatenate()([market_residual, sentiment_residual])
            
            cross_attention = MultiHeadAttention(num_heads=4, key_dim=16)(combined, combined)
            combined_residual = Concatenate()([combined, cross_attention])
            
            model_inputs = [market_input, sentiment_input]
        else:
            # Market data only
            combined_residual = market_residual
            model_inputs = [market_input]
        
        flat = tf.keras.layers.Flatten()(combined_residual)
        dense1 = Dense(64, activation='relu')(flat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        output = Dense(self.forecast_horizon, activation='linear')(dropout2)
        
        model = Model(inputs=model_inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def fit(self, X_market: np.ndarray, X_sentiment: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2, epochs: int = 50, batch_size: int = 32, 
            verbose: int = 1, callbacks=None):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Use appropriate inputs based on whether sentiment data is available
        if hasattr(self, 'has_sentiment') and self.has_sentiment:
            model_inputs = [X_market, X_sentiment]
        else:
            model_inputs = [X_market]
            
        history = self.model.fit(
            model_inputs, y, 
            validation_split=validation_split,
            epochs=epochs, 
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        return history
    
    def predict(self, X_market: np.ndarray, X_sentiment: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Use appropriate inputs based on whether sentiment data is available
        if hasattr(self, 'has_sentiment') and self.has_sentiment:
            model_inputs = [X_market, X_sentiment]
        else:
            model_inputs = [X_market]
            
        y_pred_scaled = self.model.predict(model_inputs)
        
        y_pred_scaled = np.clip(y_pred_scaled, -1, 1)
        
        if y_pred_scaled.ndim > 2:
            y_pred_scaled = y_pred_scaled.reshape(y_pred_scaled.shape[0], -1)
        elif y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            
        y_pred_transformed = self.price_scaler.inverse_transform(y_pred_scaled)
        
        if hasattr(self, 'transformed_price_data') and hasattr(self, 'original_price_data'):
            if np.all(self.original_price_data > 0):
                y_pred = np.expm1(y_pred_transformed)
            else:
                y_pred = y_pred_transformed
        else:
            y_pred = y_pred_transformed
            
        return y_pred
    
    def predict_next_days(self, latest_market_data: np.ndarray, latest_sentiment_data: np.ndarray, 
                         days: int = 5) -> List[float]:
        predictions = []
        
        market_data = latest_market_data.copy()
        sentiment_data = latest_sentiment_data.copy()
        
        if hasattr(self, 'last_actual_price') and self.last_actual_price is not None:
            current_price = self.last_actual_price
        else:
            current_price = 100.0
        
        for i in range(days):
            try:
                market_batch = market_data.reshape(1, self.look_back, market_data.shape[1])
                sentiment_batch = sentiment_data.reshape(1, self.look_back, sentiment_data.shape[1])
                
                # Use appropriate inputs based on whether sentiment data is available
                if hasattr(self, 'has_sentiment') and self.has_sentiment:
                    model_inputs = [market_batch, sentiment_batch]
                else:
                    model_inputs = [market_batch]
                
                next_day_scaled = self.model.predict(model_inputs, verbose=0)
                next_day_scaled = np.clip(next_day_scaled, -1, 1)
                
                next_day_transformed = self.price_scaler.inverse_transform(next_day_scaled.reshape(-1, 1))[0][0]
                
                if hasattr(self, 'transformed_price_data') and hasattr(self, 'original_price_data'):
                    if np.all(self.original_price_data > 0):
                        next_day_price = np.expm1(next_day_transformed)
                    else:
                        next_day_price = next_day_transformed
                else:
                    next_day_price = next_day_transformed
                
                if i == 0:
                    if next_day_price < current_price * 0.5 or next_day_price > current_price * 2.0:
                        scale_factor = current_price / next_day_price if next_day_price != 0 else 1.0
                        next_day_price = next_day_price * scale_factor
                else:
                    if next_day_price < current_price * 0.8 or next_day_price > current_price * 1.2:
                        direction = 1 if next_day_price > current_price else -1
                        max_change = current_price * 0.05
                        next_day_price = current_price + (max_change * direction)
                
                predictions.append(float(next_day_price))
                current_price = next_day_price
                
                if i < days - 1:
                    if market_data.shape[1] > 0:
                        market_data[-1, 0] = self.price_scaler.transform([[next_day_price]])[0][0]
                    
                    sentiment_data[:-1] = sentiment_data[1:]
                        
            except Exception as e:
                print(f"Error predicting day {i+1}: {str(e)}")
                if i > 0:
                    change = np.random.uniform(-0.01, 0.01) * current_price
                    next_day_price = current_price + change
                else:
                    next_day_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
                
                predictions.append(float(next_day_price))
                current_price = next_day_price
        
        return predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        if y_true.ndim == 3:
            y_true = y_true.reshape(y_true.shape[0], -1)
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
            
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
