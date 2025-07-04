"""
Stock Price Prediction Web App using Flask
"""

from flask import Flask, render_template, request, jsonify, url_for, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
import time
import os
import json
import traceback
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.io as pio
import uuid
from werkzeug.utils import secure_filename
from stock_prediction_lstm.core.utils import create_output_directory

# Set fixed random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Enable deterministic operations for TensorFlow
tf.config.experimental.enable_op_determinism()

# Import your modules (make sure these are in the same directory)
try:
    from stock_prediction_lstm.data.fetchers import StockDataFetcher, SentimentAnalyzer
    from stock_prediction_lstm.data.processors import TechnicalIndicatorGenerator
    from stock_prediction_lstm.models import StockSentimentModel
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required module files are in the same directory as app.py")
    exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/generated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def get_market_state():
    """Return the current market state (open/closed)"""
    now = datetime.now()
    # US Market hours are 9:30 AM to 4:00 PM Eastern Time
    # This is a simplified check
    if now.weekday() < 5:  # Monday to Friday
        if 9 <= now.hour < 16:  # 9 AM to 4 PM (simplified)
            return "open", "The US stock market is currently open."
        else:
            return "closed", "The US stock market is currently closed."
    else:
        return "closed", "The US stock market is closed for the weekend."

def create_candlestick_chart(stock_df, ticker_symbol, output_path=None):
    """Create a candlestick chart and return the HTML or save to file"""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock_df['date'],
        open=stock_df['open'],
        high=stock_df['high'],
        low=stock_df['low'],
        close=stock_df['close'],
        name="Price"
    ))
    
    fig.update_layout(
        title=f"{ticker_symbol} Price Movement",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_white"
    )
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
        return output_path
    else:
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_price_trends_chart(stock_df, ticker_symbol, output_path=None):
    """Create a price trends chart with moving averages"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_df['date'],
        y=stock_df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add moving averages if available
    if len(stock_df) >= 50:
        stock_df['MA50'] = stock_df['close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=stock_df['date'],
            y=stock_df['MA50'],
            mode='lines',
            name='50-day MA',
            line=dict(color='orange', width=1.5, dash='dash')
        ))
    
    if len(stock_df) >= 200:
        stock_df['MA200'] = stock_df['close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(
            x=stock_df['date'],
            y=stock_df['MA200'],
            mode='lines',
            name='200-day MA',
            line=dict(color='green', width=1.5, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{ticker_symbol} Price Trends",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
        return output_path
    else:
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_volume_chart(stock_df, ticker_symbol, output_path=None):
    """Create a volume chart"""
    fig = go.Figure()
    # Color the volume bars based on price change
    colors = ['green' if close >= open else 'red' for close, open in zip(stock_df['close'], stock_df['open'])]
    fig.add_trace(go.Bar(
        x=stock_df['date'],
        y=stock_df['volume'],
        marker_color=colors,
        name="Volume"
    ))
    
    # Add volume moving average
    stock_df['volume_ma20'] = stock_df['volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=stock_df['date'],
        y=stock_df['volume_ma20'],
        mode='lines',
        name='20-day Volume MA',
        line=dict(color='blue', width=1.5)
    ))
    
    fig.update_layout(
        title=f"{ticker_symbol} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=500,
        template="plotly_white"
    )
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
        return output_path
    else:
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_technical_indicators_chart(combined_df, output_path=None):
    """Create technical indicators chart with RSI, MACD, etc."""
    fig = make_subplots(rows=3, cols=1, 
                      shared_xaxes=True, 
                      vertical_spacing=0.05,
                      row_heights=[0.5, 0.25, 0.25])
    
    # Price with moving averages
    fig.add_trace(
        go.Scatter(
            x=combined_df['date'], 
            y=combined_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    if 'ma7' in combined_df.columns:
        fig.add_trace(
            go.Scatter(
                x=combined_df['date'], 
                y=combined_df['ma7'],
                mode='lines',
                name='7-day MA',
                line=dict(color='orange', width=1.5, dash='dash')
            ),
            row=1, col=1
        )
        
    if 'ma30' in combined_df.columns:
        fig.add_trace(
            go.Scatter(
                x=combined_df['date'], 
                y=combined_df['ma30'],
                mode='lines',
                name='30-day MA',
                line=dict(color='green', width=1.5, dash='dash')
            ),
            row=1, col=1
        )
    
    # RSI
    if 'rsi14' in combined_df.columns:
        fig.add_trace(
            go.Scatter(
                x=combined_df['date'], 
                y=combined_df['rsi14'],
                mode='lines',
                name='RSI (14)',
                line=dict(color='purple', width=1.5)
            ),
            row=2, col=1
        )
        
        # Add RSI bands
        fig.add_shape(
            type="line", line_color="red", line_dash="dash",
            x0=combined_df['date'].iloc[0], y0=70, 
            x1=combined_df['date'].iloc[-1], y1=70,
            row=2, col=1
        )
        
        fig.add_shape(
            type="line", line_color="green", line_dash="dash",
            x0=combined_df['date'].iloc[0], y0=30, 
            x1=combined_df['date'].iloc[-1], y1=30,
            row=2, col=1
        )
    
    # MACD
    if 'macd' in combined_df.columns and 'macd_signal' in combined_df.columns:
        fig.add_trace(
            go.Scatter(
                x=combined_df['date'], 
                y=combined_df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=combined_df['date'], 
                y=combined_df['macd_signal'],
                mode='lines',
                name='Signal Line',
                line=dict(color='red', width=1.5)
            ),
            row=3, col=1
        )
        
        # Calculate MACD histogram
        combined_df['macd_hist'] = combined_df['macd'] - combined_df['macd_signal']
        colors = ['green' if x > 0 else 'red' for x in combined_df['macd_hist']]
        
        fig.add_trace(
            go.Bar(
                x=combined_df['date'],
                y=combined_df['macd_hist'],
                marker_color=colors,
                name='MACD Histogram'
            ),
            row=3, col=1
        )
    
    fig.update_layout(
        title="Technical Indicators",
        height=800,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
        return output_path
    else:
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def create_future_predictions_chart(hist_dates, hist_prices, future_dates, future_prices, ticker_symbol, output_path=None):
    """Create future predictions chart"""
    fig = go.Figure()
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_prices,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8, symbol='circle', line=dict(width=2, color='DarkSlateGrey'))
    ))
    
    # Add confidence interval (simplified approach)
    upper_bound = [p * (1 + 0.02 * (i+1)) for i, p in enumerate(future_prices)]
    lower_bound = [p * (1 - 0.02 * (i+1)) for i, p in enumerate(future_prices)]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(width=0),
        name='Confidence Interval'
    ))
    
    # Add annotations for the last historical and predicted prices
    fig.add_annotation(
        x=hist_dates.iloc[-1],
        y=hist_prices.iloc[-1],
        text=f"${hist_prices.iloc[-1]:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0, ay=-40
    )
    
    fig.add_annotation(
        x=future_dates[-1],
        y=future_prices[-1],
        text=f"${future_prices[-1]:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0, ay=-40
    )
    
    fig.update_layout(
        title=f"{ticker_symbol} Future Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if output_path:
        pio.write_html(fig, file=output_path, auto_open=False)
        return output_path
    else:
        return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

# Routes
@app.route('/')
def index():
    """Render the home page"""
    market_state, market_message = get_market_state()
    
    # Popular stocks for display
    popular_stocks = [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "icon": "🍎"},
        {"ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "icon": "🪟"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "icon": "🔍"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "icon": "📦"},
        {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "icon": "🚗"},
        {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "icon": "👥"},
        {"ticker": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology", "icon": "🎮"},
        {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services", "icon": "🏦"}
    ]
    
    return render_template('index.html', 
                          market_state=market_state, 
                          market_message=market_message,
                          popular_stocks=popular_stocks)

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    """Handle the analysis request"""
    if request.method != 'POST':
        return redirect(url_for('index'))
    
    # Get form data
    ticker_symbol = request.form.get('ticker_symbol', 'NVDA')
    period = request.form.get('period', '1y')
    interval = request.form.get('interval', '1d')
    prediction_days = int(request.form.get('prediction_days', 5))
    
    try:
        # Generate a unique ID for this analysis
        analysis_id = f"{ticker_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Fetch stock data
        stock_fetcher = StockDataFetcher(ticker_symbol, period, interval)
        stock_df = stock_fetcher.fetch_data()
        
        if stock_df is None or len(stock_df) < 30:
            return render_template('error.html', 
                                message="Not enough data to analyze this stock. Please try a different stock or time period.")
        
        # Get company info
        try:
            stock_info = yf.Ticker(ticker_symbol).info
        except:
            stock_info = {'longName': ticker_symbol}
        
        # Create charts (return HTML directly instead of saving to files)
        candlestick_chart = create_candlestick_chart(stock_df, ticker_symbol)
        price_trends_chart = create_price_trends_chart(stock_df, ticker_symbol)
        volume_chart = create_volume_chart(stock_df, ticker_symbol)
        
        # Calculate key metrics
        current_price = stock_df['close'].iloc[-1]
        prev_price = stock_df['close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        period_high = stock_df['high'].max()
        period_low = stock_df['low'].min()
        period_volatility = stock_df['close'].pct_change().std() * 100 * (252 ** 0.5)  # Annualized
        
        # Step 2: Generate sentiment data
        sentiment_analyzer = SentimentAnalyzer(ticker_symbol)
        sentiment_df = sentiment_analyzer.fetch_news_sentiment(
            start_date=stock_df['date'].min(),
            end_date=stock_df['date'].max()
        )
        
        # Track if synthetic data was used for the UI
        using_synthetic_data = sentiment_analyzer.using_synthetic_data
        
        if sentiment_df is None:
            # Create a dummy sentiment dataframe if we couldn't get real data
            sentiment_df = pd.DataFrame({
                'date': stock_df['date'],
                'sentiment_positive': np.random.normal(0.6, 0.15, len(stock_df)),
                'sentiment_negative': np.random.normal(0.3, 0.1, len(stock_df)),
                'sentiment_neutral': np.random.normal(0.1, 0.05, len(stock_df))
            })
            # Normalize sentiment to sum to 1
            for i in range(len(sentiment_df)):
                total = (
                    sentiment_df.loc[i, 'sentiment_positive'] + 
                    sentiment_df.loc[i, 'sentiment_negative'] + 
                    sentiment_df.loc[i, 'sentiment_neutral']
                )
                sentiment_df.loc[i, 'sentiment_positive'] /= total
                sentiment_df.loc[i, 'sentiment_negative'] /= total
                sentiment_df.loc[i, 'sentiment_neutral'] /= total
        
        # Calculate average sentiment
        avg_pos = sentiment_df['sentiment_positive'].mean() * 100
        avg_neg = sentiment_df['sentiment_negative'].mean() * 100
        avg_neu = sentiment_df['sentiment_neutral'].mean() * 100
        
        # Step 3: Add technical indicators
        combined_df = pd.merge(stock_df, sentiment_df, on='date', how='inner')
        combined_df = TechnicalIndicatorGenerator.add_technical_indicators(combined_df)
        
        # Add other features for prediction
        combined_df['price_change'] = combined_df['close'].pct_change()
        combined_df['price_change_3d'] = combined_df['close'].pct_change(periods=3)
        combined_df['price_change_5d'] = combined_df['close'].pct_change(periods=5)
        combined_df['volatility'] = combined_df['close'].rolling(window=5).std() / combined_df['close']
        combined_df['momentum'] = combined_df['close'] - combined_df['close'].shift(5)
        combined_df['sentiment_pos_ma5'] = combined_df['sentiment_positive'].rolling(window=5).mean()
        combined_df['sentiment_neg_ma5'] = combined_df['sentiment_negative'].rolling(window=5).mean()
        
        # Drop NaN values
        combined_df = combined_df.dropna()
        
        # Create technical indicators chart
        tech_indicators_chart = create_technical_indicators_chart(combined_df)
        
        # Step 4: Train model
        model = StockSentimentModel(look_back=20)
        X_market, X_sentiment, y = model.prepare_data(combined_df, target_col='close')
        
        # Split data
        split_idx = int(0.8 * len(X_market))
        X_market_train, X_market_test = X_market[:split_idx], X_market[split_idx:]
        X_sentiment_train, X_sentiment_test = X_sentiment[:split_idx], X_sentiment[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        model.build_model(X_market.shape[2], X_sentiment.shape[2])
        
        # Use early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        epochs = 20
        history = model.fit(
            X_market_train, X_sentiment_train, y_train,
            epochs=epochs,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Make test predictions
        y_pred = model.predict(X_market_test, X_sentiment_test)
        
        # Ensure correct shapes
        if y_test.ndim == 3:
            y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[2])
        else:
            y_test_reshaped = y_test
        
        # Calculate metrics
        metrics = model.evaluate(y_test_reshaped, y_pred)
        
        # Step 5: Future predictions
        model.last_actual_price = combined_df['close'].iloc[-1]
        model.original_price_data = np.array([[model.last_actual_price]])
        
        # Generate future predictions
        future_prices = []
        for i in range(prediction_days):
            if i == 0:
                # First day prediction uses the last market and sentiment data
                day_pred = model.predict_next_days(
                    X_market[-1],
                    X_sentiment[-1],
                    days=1
                )[0]
                future_prices.append(day_pred)
            else:
                # Subsequent predictions use previous predictions (simplification)
                day_pred = model.predict_next_days(
                    X_market[-1],
                    X_sentiment[-1],
                    days=1
                )[0]
                future_prices.append(day_pred)
        
        # Generate future dates
        last_date = combined_df['date'].iloc[-1]
        future_dates = [(last_date + timedelta(days=i+1)) for i in range(prediction_days)]
        
        # Create future predictions chart
        hist_dates = combined_df['date'].iloc[-30:]
        hist_prices = combined_df['close'].iloc[-30:]
        
        future_predictions_chart = create_future_predictions_chart(
            hist_dates,
            hist_prices,
            future_dates,
            future_prices,
            ticker_symbol
        )
        
        # Calculate predicted returns
        starting_price = hist_prices.iloc[-1]
        final_price = future_prices[-1]
        overall_return = ((final_price - starting_price) / starting_price) * 100
        
        # Prepare prediction data for table
        prediction_data = []
        for i, (date, price) in enumerate(zip(future_dates, future_prices)):
            if i == 0:
                prev_price = hist_prices.iloc[-1]
            else:
                prev_price = future_prices[i-1]
                
            # Calculate daily change
            change = ((price - prev_price) / prev_price) * 100
            
            prediction_data.append({
                "day": i+1,
                "date": date.strftime('%Y-%m-%d'),
                "price": f"${price:.2f}",
                "change": f"{change:+.2f}%",
                "change_class": "positive" if change >= 0 else "negative"
            })
        
        # Volatility calculation for risk assessment
        volatility = np.std([((future_prices[i] - future_prices[i-1]) / future_prices[i-1]) * 100 
                           for i in range(1, len(future_prices))])
        
        # Determine risk level
        if volatility < 1.0:
            risk_level = "Low"
            risk_color = "green"
        elif volatility < 2.0:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
            
        # Show prediction confidence
        r2_score = metrics['r2']
        if r2_score > 0.7:
            confidence = "High"
            conf_color = "green"
        elif r2_score > 0.5:
            confidence = "Moderate"
            conf_color = "orange"
        else:
            confidence = "Low"
            conf_color = "red"
            
        # Determine sentiment based on predicted return
        if overall_return > 5:
            sentiment = "Strongly Bullish"
            recommendation = "Consider buying for short-term growth potential"
            sentiment_icon = "📈"
            sentiment_color = "#43A047"
        elif overall_return > 2:
            sentiment = "Moderately Bullish"
            recommendation = "Potential for slight upward movement"
            sentiment_icon = "↗️"
            sentiment_color = "#7CB342"
        elif overall_return > -2:
            sentiment = "Neutral"
            recommendation = "Sideways trading expected"
            sentiment_icon = "↔️"
            sentiment_color = "#FFB300"
        elif overall_return > -5:
            sentiment = "Moderately Bearish"
            recommendation = "Potential for slight downward movement"
            sentiment_icon = "↘️"
            sentiment_color = "#EF6C00"
        else:
            sentiment = "Strongly Bearish"
            recommendation = "Consider avoiding or selling"
            sentiment_icon = "📉"
            sentiment_color = "#D32F2F"
        
        # Create a dictionary to hold all data for the dashboard
        dashboard_data = {
            'ticker': ticker_symbol,
            'last_price': current_price,
            'price_change': current_price - prev_price,
            'price_change_pct': price_change,
            'predictions': future_prices,
            'prediction_dates': [date.strftime('%Y-%m-%d') for date in future_dates],
            'predicted_return': overall_return,
            'using_synthetic_data': using_synthetic_data,
            'dashboard_data': {
                'price_prediction': {
                    'data': [
                        {
                            'x': [d.strftime('%Y-%m-%d') for d in hist_dates],
                            'y': hist_prices.tolist(),
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': 'Historical',
                            'line': {'color': 'blue', 'width': 2}
                        },
                        {
                            'x': [d.strftime('%Y-%m-%d') for d in future_dates],
                            'y': future_prices,
                            'type': 'scatter',
                            'mode': 'lines+markers',
                            'name': 'Predicted',
                            'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
                        }
                    ],
                    'layout': {
                        'title': f"{ticker_symbol} Price Prediction",
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Price ($)'},
                        'height': 500,
                        'template': 'plotly_white'
                    }
                },
                'sentiment_analysis': {
                    'data': [
                        {
                            'x': combined_df['date'].dt.strftime('%Y-%m-%d').tolist()[-30:],
                            'y': combined_df['close'].tolist()[-30:],
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': 'Price',
                            'yaxis': 'y'
                        },
                        {
                            'x': combined_df['date'].dt.strftime('%Y-%m-%d').tolist()[-30:],
                            'y': (combined_df['sentiment_positive'] * 100).tolist()[-30:],
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': 'Positive Sentiment',
                            'yaxis': 'y2'
                        }
                    ],
                    'layout': {
                        'title': 'Price vs Sentiment',
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Price ($)', 'side': 'left'},
                        'yaxis2': {
                            'title': 'Sentiment %',
                            'overlaying': 'y',
                            'side': 'right',
                            'range': [0, 100]
                        },
                        'height': 500,
                        'template': 'plotly_white'
                    }
                },
                'feature_importance': {
                    'data': [
                        {
                            'x': [0.8, 0.7, 0.6, 0.5, 0.4],
                            'y': ['Price', 'Volume', 'RSI', 'MACD', 'Sentiment'],
                            'type': 'bar',
                            'orientation': 'h',
                            'marker': {'color': 'rgba(58, 71, 80, 0.6)'}
                        }
                    ],
                    'layout': {
                        'title': 'Feature Importance',
                        'xaxis': {'title': 'Importance Score'},
                        'height': 500,
                        'template': 'plotly_white'
                    }
                },
                'technical_indicators': {
                    'data': [
                        {
                            'x': combined_df['date'].dt.strftime('%Y-%m-%d').tolist()[-60:],
                            'y': combined_df['close'].tolist()[-60:],
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': 'Price'
                        },
                        {
                            'x': combined_df['date'].dt.strftime('%Y-%m-%d').tolist()[-60:],
                            'y': combined_df['rsi14'].tolist()[-60:] if 'rsi14' in combined_df.columns else [],
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': 'RSI',
                            'yaxis': 'y2'
                        }
                    ],
                    'layout': {
                        'title': 'Technical Indicators',
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Price ($)'},
                        'yaxis2': {
                            'title': 'Indicator Value',
                            'overlaying': 'y',
                            'side': 'right'
                        },
                        'height': 800,
                        'template': 'plotly_white'
                    }
                }
            }
        }
        
        # Get market state
        market_state, market_message = get_market_state()
        
        # Prepare data for the new template structure
        template_data = {
            'ticker_symbol': ticker_symbol.upper(),
            'company_name': stock_info.get('longName', ticker_symbol),
            'period': period,
            'interval': interval,
            'prediction_days': prediction_days,
            'market_state': market_state,
            'current_price': current_price,
            'price_change': price_change,
            'predicted_return': overall_return,
            'analysis_conclusion': sentiment,
            'volatility': {
                'level': risk_level,
                'value': volatility
            },
            'model_performance': {
                'r2_score': metrics['r2'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics.get('mape', 0.0)
            },
            'sentiment_score': avg_pos / 100.0,
            'company_info': stock_info,
            'charts': {
                'price_history': {
                    'title': 'Price History',
                    'html': candlestick_chart,
                    'description': 'Historical price movements with candlestick representation'
                },
                'future_prediction': {
                    'title': 'Future Prediction',
                    'html': future_predictions_chart,
                    'description': 'Predicted price path with confidence intervals'
                },
                'technical_indicators': {
                    'title': 'Technical Analysis',
                    'html': tech_indicators_chart,
                    'description': 'Technical indicators and trading signals'
                },
                'volume_analysis': {
                    'title': 'Volume Analysis',
                    'html': volume_chart,
                    'description': 'Trading volume trends and patterns'
                }
            },
            'future_predictions': {
                str(i): {
                    'date': pred_data['date'],
                    'price': float(pred_data['price'].replace('$', '')),
                    'change': float(pred_data['change'].replace('%', '').replace('+', ''))
                }
                for i, pred_data in enumerate(prediction_data)
            },
            'technical_indicators': {
                'rsi': f"{combined_df['rsi14'].iloc[-1]:.2f}" if 'rsi14' in combined_df.columns else 'N/A',
                'macd': f"{combined_df['macd'].iloc[-1]:.4f}" if 'macd' in combined_df.columns else 'N/A',
                'bb_position': 'N/A'  # Add more indicators as needed
            }
        }
        
        # Render the dashboard template
        return render_template('dashboard.html', **template_data)
    
    except Exception as e:
        return render_template('error.html', 
                              message=f"An error occurred during the analysis: {str(e)}",
                              error_details=str(e),
                              traceback=traceback.format_exc())

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Compare multiple stocks"""
    if request.method != 'POST':
        return render_template('compare.html')
    
    ticker_symbols = request.form.get('ticker_symbols', '').split(',')
    ticker_symbols = [t.strip().upper() for t in ticker_symbols if t.strip()]
    
    if not ticker_symbols:
        return render_template('compare.html', error="Please enter at least one ticker symbol.")
    
    period = request.form.get('period', '1y')
    interval = request.form.get('interval', '1d')
    
    # Fetch data for each ticker
    stock_data = {}
    synthetic_data_flags = {}
    for ticker in ticker_symbols:
        try:
            # Fetch stock data
            stock_fetcher = StockDataFetcher(ticker, period, interval)
            df = stock_fetcher.fetch_data()
            if df is not None and len(df) > 0:
                stock_data[ticker] = df
                
                # Check if sentiment data would be synthetic
                sentiment_analyzer = SentimentAnalyzer(ticker)
                sentiment_analyzer.fetch_news_sentiment(
                    start_date=df['date'].min(),
                    end_date=df['date'].max()
                )
                synthetic_data_flags[ticker] = sentiment_analyzer.using_synthetic_data
        except Exception as e:
            return render_template('compare.html', 
                                  error=f"Error fetching data for {ticker}: {str(e)}",
                                  ticker_symbols=','.join(ticker_symbols))
    
    if not stock_data:
        return render_template('compare.html', 
                              error="Could not fetch data for any of the provided tickers.",
                              ticker_symbols=','.join(ticker_symbols))
    
    # Create comparison chart
    fig = go.Figure()
    
    # Normalize all prices to percentage change from first day
    for ticker, df in stock_data.items():
        if len(df) == 0:
            continue
        first_price = df['close'].iloc[0]
        normalized_prices = [(price / first_price - 1) * 100 for price in df['close']]
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=normalized_prices,
            mode='lines',
            name=ticker,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Stock Price Comparison (% Change)",
        xaxis_title="Date",
        yaxis_title="Price Change (%)",
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Generate a unique ID for this comparison
    comparison_id = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], comparison_id)
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_chart = os.path.join('generated', comparison_id, 'comparison.html')
    pio.write_html(fig, file=os.path.join(app.config['UPLOAD_FOLDER'], comparison_id, 'comparison.html'), auto_open=False)
    
    # Calculate performance metrics for each stock
    performance_data = []
    for ticker, df in stock_data.items():
        if len(df) < 2:
            continue
            
        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]
        total_return = ((last_price / first_price) - 1) * 100
        
        # Calculate volatility and other metrics
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
        
        # Calculate max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        performance_data.append({
            'ticker': ticker,
            'start_price': f"${first_price:.2f}",
            'end_price': f"${last_price:.2f}",
            'total_return': f"{total_return:.2f}%",
            'annualized_volatility': f"{volatility:.2f}%",
            'max_drawdown': f"{max_drawdown:.2f}%",
            'return_class': 'positive' if total_return >= 0 else 'negative'
        })
    
    # Sort by return (descending)
    performance_data.sort(key=lambda x: float(x['total_return'].replace('%', '')), reverse=True)
    
    return render_template('compare.html',
                          comparison_chart=comparison_chart,
                          performance_data=performance_data,
                          ticker_symbols=','.join(ticker_symbols),
                          period=period,
                          interval=interval,
                          synthetic_data_flags=synthetic_data_flags)

if __name__ == '__main__':
    app.run(debug=True, port=8081)

# Function to create the technical indicators visualization
def make_subplots(rows, cols, shared_xaxes=True, vertical_spacing=0.05, row_heights=None):
    """
    This function replicates the functionality of plotly's make_subplots.
    """
    fig = go.Figure()
    
    fig.update_layout(
        grid = {'rows': rows, 'columns': cols, 'pattern': 'independent'},
        plot_bgcolor='white',  # Sets background color to white
        paper_bgcolor='white'  # Sets paper background color to white
    )
    
    return fig
