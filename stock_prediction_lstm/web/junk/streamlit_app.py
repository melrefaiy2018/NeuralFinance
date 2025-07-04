"""
Stock Price Prediction Web App using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
import time
from PIL import Image
import os
import json
from plotly.subplots import make_subplots

# Set fixed random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Enable deterministic operations for TensorFlow
tf.config.experimental.enable_op_determinism()

# Set page config
st.set_page_config(
    page_title="Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more visually appealing
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px #00000030;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #43A047;
        margin-top: 2rem;
        padding-left: 0.5rem;
        border-left: 4px solid #43A047;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .prediction-table {
        font-size: 1.1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .ticker-info {
        background-color: #f0f8ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .market-state {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Add the parent directory to Python path for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import your modules
try:
    from data.fetchers import StockDataFetcher, SentimentAnalyzer
    from data.processors import TechnicalIndicatorGenerator
    from analysis import StockAnalyzer
    from visualization import (
        visualize_stock_data,
        visualize_prediction_comparison,
        visualize_future_predictions,
        visualize_feature_importance,
        visualize_sentiment_impact
    )
    
    # Try to import the improved model, fallback to original with fixes
    try:
        from models.improved_model import ImprovedStockModel
        from utils.emergency_fixes import emergency_evaluate_with_diagnostics
        ModelClass = ImprovedStockModel
        st.info("🚀 Using improved model with emergency evaluation fixes")
    except ImportError:
        try:
            from models import StockSentimentModel
            from utils.model_fixes import apply_model_fixes
            from utils.emergency_fixes import emergency_evaluate_with_diagnostics
            ModelClass = apply_model_fixes(StockSentimentModel)
            st.info("🔧 Using original model with evaluation fixes applied")
        except ImportError:
            st.error("Could not import any model class")
            st.stop()
            
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Make sure all required module files are in the correct package structure")
    st.stop()

def get_market_state():
    """Return the current market state (open/closed)"""
    now = datetime.now()
    # US Market hours are 9:30 AM to 4:00 PM Eastern Time
    # This is a simplified check
    if now.weekday() < 5:  # Monday to Friday
        if 9 <= now.hour < 16:  # 9 AM to 4 PM (simplified)
            return "Open 🟢", "The US stock market is currently open."
        else:
            return "Closed 🔴", "The US stock market is currently closed."
    else:
        return "Closed 🔴", "The US stock market is closed for the weekend."

def display_company_info(ticker_symbol):
    """Display company information card"""
    try:
        # Get company info
        stock_info = yf.Ticker(ticker_symbol).info
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Try to get company logo
            if 'logo_url' in stock_info and stock_info['logo_url']:
                st.image(stock_info['logo_url'], width=100)
            else:
                # Display a placeholder if no logo is available
                st.markdown("💹")
        
        with col2:
            if 'longName' in stock_info:
                st.markdown(f"### {stock_info['longName']}")
            else:
                st.markdown(f"### {ticker_symbol}")
                
            if 'sector' in stock_info and 'industry' in stock_info:
                st.markdown(f"**Sector:** {stock_info['sector']} | **Industry:** {stock_info['industry']}")
            
            if 'website' in stock_info and stock_info['website']:
                st.markdown(f"[Company Website]({stock_info['website']})")
        
        # Additional company info
        with st.expander("More Company Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                if 'marketCap' in stock_info:
                    st.metric("Market Cap", f"${stock_info['marketCap']:,}")
                if 'trailingPE' in stock_info:
                    st.metric("P/E Ratio", f"{stock_info['trailingPE']:.2f}")
                if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                    st.metric("Dividend Yield", f"{stock_info['dividendYield']*100:.2f}%")
                    
            with col2:
                if 'fiftyTwoWeekHigh' in stock_info:
                    st.metric("52 Week High", f"${stock_info['fiftyTwoWeekHigh']:.2f}")
                if 'fiftyTwoWeekLow' in stock_info:
                    st.metric("52 Week Low", f"${stock_info['fiftyTwoWeekLow']:.2f}")
                if 'beta' in stock_info:
                    st.metric("Beta", f"{stock_info['beta']:.2f}")
            
            if 'longBusinessSummary' in stock_info:
                st.markdown("### Business Summary")
                st.markdown(stock_info['longBusinessSummary'])
    
    except Exception as e:
        st.warning(f"Couldn't retrieve detailed company information. Error: {str(e)}")

# Title and description in the main area
st.markdown('<h1 class="main-header">Stock AI Assistant 📈</h1>', unsafe_allow_html=True)

# Create the card with proper markdown
st.markdown("""
<div class="card">
    <p>Welcome to Stock AI Assistant, an advanced machine learning platform for stock price prediction. 
    Our system leverages historical data, technical indicators, market trends, and sentiment analysis to 
    generate accurate forecasts of future stock prices.</p>
</div>
""", unsafe_allow_html=True)

# Add the features list using native Streamlit components instead of HTML
st.write("This AI-powered tool is built with:")

col1, col2 = st.columns(2)
with col1:
    st.markdown("- 🧠 Neural networks with multi-head attention mechanisms")
    st.markdown("- 📊 Technical indicator analysis")
with col2:
    st.markdown("- 📰 Financial news sentiment analysis")
    st.markdown("- 📉 Volatility and market trend modeling")

# Display market state
market_state, market_message = get_market_state()
st.markdown(f'<div class="market-state"><strong>Market Status:</strong> {market_state}</div>', unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.title("Settings")
    
    # Add a nice divider
    st.markdown("---")
    
    # Stock selection
    st.subheader("Stock Ticker")
    ticker_symbol = st.text_input("", "NVDA")
    
    # Add some suggested tickers
    st.caption("Popular tickers: AAPL, MSFT, GOOGL, AMZN, TSLA")
    
    st.markdown("---")
    
    # Time period selection with icons
    st.subheader("📆 Select Time Period")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "Year to Date": "ytd",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    selected_period = st.selectbox(
        "", 
        list(period_options.keys())
    )
    period = period_options[selected_period]
    
    # Interval selection
    st.subheader("⏱️ Select Interval")
    interval_options = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Week": "1wk",
        "1 Month": "1mo"
    }
    selected_interval = st.selectbox(
        "",
        list(interval_options.keys())
    )
    interval = interval_options[selected_interval]
    
    # Prediction days with a prettier slider
    st.subheader("🔮 Days to Predict")
    prediction_days = st.slider("", 1, 30, 5)
    st.caption(f"Forecasting {prediction_days} days into the future")
    
    # Add a divider before the run button
    st.markdown("---")
    
    # Make the run button more prominent
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True)

# Main content
if run_analysis:
    # Company information at the top
    display_company_info(ticker_symbol)
    
    # Set up a progress tracker
    progress = st.progress(0)
    status_text = st.empty()
    
    try:
        stock_analyzer = StockAnalyzer()
        # Step 1: Fetch stock data (10%)
        status_text.markdown("#### 🔍 Fetching historical stock data...")
        stock_fetcher = StockDataFetcher(ticker_symbol, period, interval)
        stock_df = stock_fetcher.fetch_data()
        
        if stock_df is None or len(stock_df) < 30:
            st.error("❌ Error: Not enough data to analyze this stock.")
            st.info("Please try a different stock or time period.")
            st.stop()
        
        progress.progress(10)
        
        # Step 2: Display stock data (20%)
        status_text.markdown("#### 📊 Analyzing price history...")
        
        st.markdown('<h2 class="sub-header">Stock Price History</h2>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        price_tabs = st.tabs(["Candlestick Chart", "Price Trends", "Volume Analysis"])
        
        with price_tabs[0]:
            # Candlestick chart
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
            st.plotly_chart(fig, use_container_width=True)
        
        with price_tabs[1]:
            # Line chart with additional metrics
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
            st.plotly_chart(fig, use_container_width=True)
        
        with price_tabs[2]:
            # Volume chart
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
            st.plotly_chart(fig, use_container_width=True)
        
        # Display key metrics in a nice UI
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        # Current price
        current_price = stock_df['close'].iloc[-1]
        prev_price = stock_df['close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Period stats
        period_high = stock_df['high'].max()
        period_low = stock_df['low'].min()
        period_volatility = stock_df['close'].pct_change().std() * 100 * (252 ** 0.5)  # Annualized
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
        with col2:
            st.metric("Period High", f"${period_high:.2f}")
        with col3:
            st.metric("Period Low", f"${period_low:.2f}")
        with col4:
            st.metric("Volatility (Ann.)", f"{period_volatility:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        progress.progress(20)
        
        # Step 3: Generate sentiment data (35%)
        status_text.markdown("#### 📰 Analyzing sentiment from financial news...")
        
        sentiment_analyzer = SentimentAnalyzer(ticker_symbol)
        sentiment_df = sentiment_analyzer.fetch_news_sentiment(
            start_date=stock_df['date'].min(),
            end_date=stock_df['date'].max()
        )
        
        # Check if real sentiment data was obtained
        using_synthetic_data = sentiment_analyzer.using_synthetic_data
        use_sentiment_analysis = False
        
        if sentiment_df is None or len(sentiment_df) == 0 or using_synthetic_data:
            st.markdown('<h2 class="sub-header">Market Sentiment Analysis</h2>', unsafe_allow_html=True)
            st.info("""
            📢 **Sentiment Analysis Unavailable**
            
            We were unable to fetch real sentiment data from financial news sources. This could be due to:
            - API rate limits
            - Network connectivity issues  
            - Limited news coverage for this time period
            
            **The analysis will continue using technical indicators only.** This purely technical approach can still provide valuable insights into price movements and trends.
            """)
            
            # Set flag to exclude sentiment from model training
            use_sentiment_analysis = False
            sentiment_df = None
            
            # Set default sentiment values when no data is available
            avg_pos = 50.0  # Neutral default
            avg_neg = 30.0  # Neutral default
            avg_neu = 20.0  # Neutral default
        else:
            use_sentiment_analysis = True
            
            st.markdown('<h2 class="sub-header">Market Sentiment Analysis</h2>', unsafe_allow_html=True)
            
            # Display success message for real data
            st.success("✅ Successfully obtained real sentiment data from financial news sources!")
            
            # Calculate average sentiment with safety checks
            avg_pos = sentiment_df['sentiment_positive'].mean() * 100 if 'sentiment_positive' in sentiment_df.columns else 50.0
            avg_neg = sentiment_df['sentiment_negative'].mean() * 100 if 'sentiment_negative' in sentiment_df.columns else 30.0
            avg_neu = sentiment_df['sentiment_neutral'].mean() * 100 if 'sentiment_neutral' in sentiment_df.columns else 20.0
            
            # Ensure values are valid numbers
            avg_pos = avg_pos if not np.isnan(avg_pos) else 50.0
            avg_neg = avg_neg if not np.isnan(avg_neg) else 30.0
            avg_neu = avg_neu if not np.isnan(avg_neu) else 20.0
            
            # Display sentiment distribution in a nice gauge chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Create a gauge chart for sentiment
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = avg_pos,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Positive Sentiment", 'font': {'size': 24}},
                    delta = {'reference': 50, 'increasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "green"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'lightgray'},
                            {'range': [30, 70], 'color': 'lightgreen'},
                            {'range': [70, 100], 'color': 'darkgreen'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sentiment metrics
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="margin-bottom: 8px; font-size: 0.9rem; color: green; font-weight: bold;">(Real data)</div>
                    <div style="display: inline-block; margin: 0 10px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: green;">
                            {avg_pos:.1f}%
                        </div>
                        <div>Positive</div>
                    </div>
                    <div style="display: inline-block; margin: 0 10px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: red;">
                            {avg_neg:.1f}%
                        </div>
                        <div>Negative</div>
                    </div>
                    <div style="display: inline-block; margin: 0 10px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: gray;">
                            {avg_neu:.1f}%
                        </div>
                        <div>Neutral</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # Create sentiment over time chart
                fig = go.Figure()
                
                # Check if required columns exist
                if 'sentiment_positive' in sentiment_df.columns and 'sentiment_negative' in sentiment_df.columns and 'sentiment_neutral' in sentiment_df.columns:
                    fig.add_trace(go.Scatter(
                        x=sentiment_df['date'],
                        y=sentiment_df['sentiment_positive'],
                        mode='lines',
                        name='Positive',
                        line=dict(color='green', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=sentiment_df['date'],
                        y=sentiment_df['sentiment_negative'],
                        mode='lines',
                        name='Negative',
                        line=dict(color='red', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=sentiment_df['date'],
                        y=sentiment_df['sentiment_neutral'],
                        mode='lines',
                        name='Neutral',
                        line=dict(color='gray', width=2)
                    ))
                
                fig.update_layout(
                    title="Sentiment Trend Over Time",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score",
                    height=300,
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        progress.progress(35)
        
        # Step 4: Technical analysis (50%)
        status_text.markdown("#### 📊 Performing technical analysis...")
        
        # Prepare data based on whether sentiment analysis is available
        if use_sentiment_analysis and sentiment_df is not None:
            # Merge stock data with sentiment data
            # Both datasets now use timezone-naive dates for consistent merging
            combined_df = pd.merge(stock_df, sentiment_df, on='date', how='inner')
            st.info("📊 **Analysis Mode**: Using both technical indicators AND sentiment analysis")
        else:
            # Use only stock data
            combined_df = stock_df.copy()
            st.info("📊 **Analysis Mode**: Using technical indicators only (no sentiment data)")
        
        # Add technical indicators
        combined_df = TechnicalIndicatorGenerator.add_technical_indicators(combined_df)
        
        # Add other features for prediction
        combined_df['price_change'] = combined_df['close'].pct_change()
        combined_df['price_change_3d'] = combined_df['close'].pct_change(periods=3)
        combined_df['price_change_5d'] = combined_df['close'].pct_change(periods=5)
        combined_df['volatility'] = combined_df['close'].rolling(window=5).std() / combined_df['close']
        combined_df['momentum'] = combined_df['close'] - combined_df['close'].shift(5)
        
        # Add sentiment features only if sentiment analysis is available
        if use_sentiment_analysis and sentiment_df is not None:
            combined_df['sentiment_pos_ma5'] = combined_df['sentiment_positive'].rolling(window=5).mean()
            combined_df['sentiment_neg_ma5'] = combined_df['sentiment_negative'].rolling(window=5).mean()
        
        # Drop NaN values
        combined_df = combined_df.dropna()
        
        # Calculate feature importance
        correlations = combined_df.corr()['close'].sort_values(ascending=False)
        
        st.markdown('<h2 class="sub-header">Technical Analysis</h2>', unsafe_allow_html=True)
        
        # Tabs for technical analysis visualizations
        tech_tabs = st.tabs(["Feature Importance", "Technical Indicators", "Correlation Matrix"])
        
        with tech_tabs[0]:
            # Drop the target column itself
            correlations = correlations.drop('close')
            
            # Get top positive and negative correlations
            top_pos = correlations.head(10)
            top_neg = correlations.tail(10).iloc[::-1]  # Reverse to show most negative first
            
            # Create feature importance visualization
            fig = go.Figure()
            
            # Add positive correlations
            fig.add_trace(go.Bar(
                y=top_pos.index,
                x=top_pos.values,
                orientation='h',
                name='Positive Correlation',
                marker=dict(color='green')
            ))
            
            # Add negative correlations on the same chart
            fig.add_trace(go.Bar(
                y=top_neg.index,
                x=top_neg.values,
                orientation='h',
                name='Negative Correlation',
                marker=dict(color='red')
            ))
            
            fig.update_layout(
                title="Feature Importance (Correlation with Price)",
                xaxis_title="Correlation Coefficient",
                yaxis_title="Features",
                height=600,
                template="plotly_white",
                barmode='relative',
                bargap=0.15,
                bargroupgap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tech_tabs[1]:
            # Create RSI, MACD, etc. visualizations
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
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tech_tabs[2]:
            # Create correlation matrix visualization
            # Select most important features
            selected_cols = ['close', 'volume', 'rsi14', 'macd', 'volatility', 'momentum',
                             'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
            selected_cols = [col for col in selected_cols if col in combined_df.columns]
            
            # Calculate correlation matrix
            corr_matrix = combined_df[selected_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect="auto"
            )
            
            fig.update_layout(
                title="Correlation Matrix of Key Features",
                height=600,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        progress.progress(50)
        
        # Step 5: Train model and make predictions (80%)
        status_text.markdown("#### 🧠 Training prediction model...")
        
        # Initialize model
        model = ModelClass(look_back=20)
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
        
        # Show training progress bar
        st.markdown('<h2 class="sub-header">AI Model Training</h2>', unsafe_allow_html=True)
        my_bar = st.progress(0)
        epoch_status = st.empty()
        
        # Custom callback to update progress bar
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.total_epochs
                my_bar.progress(progress)
                epoch_status.markdown(f"Training epoch: {epoch+1}/{self.total_epochs} (Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f})")
        
        # Train model with epochs
        epochs = 20
        history = model.fit(
            X_market_train, X_sentiment_train, y_train,
            epochs=epochs,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stopping, ProgressCallback(epochs)]
        )
        
        progress.progress(65)
        
        # Make test predictions
        status_text.markdown("#### 📈 Evaluating model and making predictions...")
        y_pred = model.predict(X_market_test, X_sentiment_test)
        
        # Ensure correct shapes for evaluation
        if y_test.ndim == 3:
            y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[2])
        else:
            y_test_reshaped = y_test
        
        # Convert both test data and predictions back to original scale for evaluation
        if hasattr(model, 'price_scaler'):
            # Ensure proper shape for inverse transform
            if y_test_reshaped.ndim == 1:
                y_test_reshaped = y_test_reshaped.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            try:
                y_test_original = model.price_scaler.inverse_transform(y_test_reshaped)
                y_pred_original = model.price_scaler.inverse_transform(y_pred)
            except Exception as e:
                st.warning(f"Issue with inverse scaling: {e}. Using emergency evaluation.")
                y_test_original = y_test_reshaped
                y_pred_original = y_pred
        else:
            y_test_original = y_test_reshaped
            y_pred_original = y_pred
        
        # Debug output to understand scaling
        with st.expander("🔧 Scaling Debug Info", expanded=False):
            st.write(f"**Before scaling:**")
            st.write(f"- y_test_reshaped range: [{np.min(y_test_reshaped):.6f}, {np.max(y_test_reshaped):.6f}]")
            st.write(f"- y_pred range: [{np.min(y_pred):.6f}, {np.max(y_pred):.6f}]")
            st.write(f"**After scaling:**")
            st.write(f"- y_test_original range: [{np.min(y_test_original):.6f}, {np.max(y_test_original):.6f}]")
            st.write(f"- y_pred_original range: [{np.min(y_pred_original):.6f}, {np.max(y_pred_original):.6f}]")
            st.write(f"**Current stock price:** ${stock_df['close'].iloc[-1]:.2f}")
            if hasattr(model, 'price_scaler'):
                st.write(f"**Scaler info:**")
                st.write(f"- data_min_: {model.price_scaler.data_min_}")
                st.write(f"- data_max_: {model.price_scaler.data_max_}")
                st.write(f"- scale_: {model.price_scaler.scale_}")
                st.write(f"- feature_range: {model.price_scaler.feature_range}")
        
        # Calculate metrics using the emergency evaluation method with diagnostics
        try:
            metrics = emergency_evaluate_with_diagnostics(y_test_original, y_pred_original)
            
            # Show diagnostic information if metrics are capped
            if 'diagnostics' in metrics:
                with st.expander("🔍 Diagnostic Information", expanded=False):
                    diag = metrics['diagnostics']
                    st.write(f"**Data Shape Check:**")
                    st.write(f"- y_true shape: {diag['y_true_shape']}")
                    st.write(f"- y_pred shape: {diag['y_pred_shape']}")
                    st.write(f"- Has NaN in y_true: {diag['has_nan_true']}")
                    st.write(f"- Has NaN in y_pred: {diag['has_nan_pred']}")
                    st.write(f"- Has Inf in y_true: {diag['has_inf_true']}")
                    st.write(f"- Has Inf in y_pred: {diag['has_inf_pred']}")
                    st.write(f"**Value Ranges:**")
                    st.write(f"- y_true range: [{diag['y_true_min']:.6f}, {diag['y_true_max']:.6f}]")
                    st.write(f"- y_pred range: [{diag['y_pred_min']:.6f}, {diag['y_pred_max']:.6f}]")
                    st.write(f"- y_true mean: {diag['y_true_mean']:.6f}")
                    st.write(f"- y_pred mean: {diag['y_pred_mean']:.6f}")
                    st.write(f"- y_true std: {diag['y_true_std']:.6f}")
                    st.write(f"- y_pred std: {diag['y_pred_std']:.6f}")
                    if diag.get('samples_removed', 0) > 0:
                        st.write(f"- Samples removed due to NaN/Inf: {diag['samples_removed']}")
                    if diag.get('scaling_correction_applied'):
                        st.info("✓ Scaling correction was applied to improve metrics.")
                    if diag.get('scaling_correction_failed'):
                        st.warning("⚠ Scaling correction failed, using raw metrics.")
                    if diag.get('fallback_used'):
                        st.warning("⚠️ Emergency fallback evaluation was used due to invalid metrics.")
                
        except Exception as e:
            st.error(f"Error in emergency evaluation: {e}")
            # Fallback to original evaluation
            metrics = model.evaluate(y_test_original, y_pred_original)
        
        # Add information about model improvements with more detailed analysis
        if metrics['r2'] > 0.5 and metrics['mape'] < 10:
            st.success("🎯 **Excellent Model Performance!** The model shows strong predictive capability.")
        elif metrics['r2'] > 0.0 and metrics['mape'] < 20:
            st.success("✅ **Good Model Performance!** The evaluation metrics show reasonable predictive ability.")
        elif metrics['r2'] > -1.0 and metrics['mape'] < 50:
            st.warning("⚠️ **Moderate Model Performance:** The model shows some predictive ability but may need optimization.")
        elif metrics['r2'] == -100.0 or metrics['mape'] >= 1000.0:
            st.error("🚨 **Emergency Fixes Applied:** Extreme metrics detected and capped. This indicates underlying data or scaling issues.")
            st.info("The emergency evaluation system has applied safety limits. Please check the diagnostic information below.")
        else:
            st.warning("⚠️ **Poor Model Performance:** The model may need significant improvements for reliable predictions.")
        
        # Show metric explanations
        with st.expander("📊 Understanding the Metrics"):
            st.markdown("""
            **RMSE (Root Mean Square Error)**: Measures the average magnitude of prediction errors.
            - Lower values are better
            - Good range: < 10% of stock price
            
            **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices.
            - Lower values are better
            - More interpretable than RMSE
            
            **R² (R-squared)**: Proportion of variance explained by the model.
            - Range: -∞ to 1, where 1 is perfect
            - Values > 0 mean the model is better than simply predicting the mean
            - Values < 0 mean the model performs worse than the mean
            
            **MAPE (Mean Absolute Percentage Error)**: Average percentage error.
            - Lower values are better
            - Good range: < 20% for stock predictions
            """)
        
        
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        # Display metrics in a nice card layout
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
            st.markdown("Root Mean Squared Error<br><small>Lower is better</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("MAE", f"{metrics['mae']:.4f}")
            st.markdown("Mean Absolute Error<br><small>Lower is better</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("R²", f"{metrics['r2']:.4f}")
            st.markdown("R-squared<br><small>Higher is better</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("MAPE", f"{metrics['mape']:.2f}%")
            st.markdown("Mean Absolute % Error<br><small>Lower is better</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot training history
        st.subheader("Training Progress")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Training history plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=history.history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="Model Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss (MSE)",
                height=300,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Show training statistics
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Training Statistics")
            
            # Calculate convergence speed
            initial_loss = history.history['loss'][0]
            final_loss = history.history['loss'][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            st.write(f"**Total Epochs:** {len(history.history['loss'])}")
            st.write(f"**Initial Loss:** {initial_loss:.4f}")
            st.write(f"**Final Loss:** {final_loss:.4f}")
            st.write(f"**Improvement:** {improvement:.2f}%")
            
            # Determine if early stopping kicked in
            if len(history.history['loss']) < epochs:
                st.markdown(f"""
                <div style="background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px; margin-top: 10px; margin-bottom: 10px;">
                  <p style="margin: 0;"><strong>Early stopping activated at epoch {len(history.history['loss'])}</strong> to prevent overfitting and ensure optimal model performance.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Plot test predictions vs actual using the properly scaled data
        st.subheader("Prediction Accuracy")
        
        # Use the already properly scaled data from evaluation
        y_true_plot = y_test_original
        y_pred_plot = y_pred_original
            
        # Simple validation - ensure predictions are reasonable
        current_price = stock_df['close'].iloc[-1]
        pred_median = np.median(y_pred_plot)
        true_median = np.median(y_true_plot)
        
        # Check if predictions are in a reasonable range
        if not np.isnan(pred_median) and not np.isnan(true_median):
            price_diff_pct = abs(pred_median - true_median) / true_median * 100
            if price_diff_pct > 50:
                st.info(f"""
                📊 **Prediction Analysis**  
                Current {ticker_symbol} price: ${current_price:.2f}  
                Predicted median: ${pred_median:.2f}  
                Actual test median: ${true_median:.2f}  
                Difference: {price_diff_pct:.1f}%
                """)
        
        # Create test prediction visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=y_true_plot.flatten(),
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=y_pred_plot.flatten(),
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add prediction confidence bands (simplified)
        if len(y_pred_plot.flatten()) > 0 and not np.isnan(y_pred_plot.flatten()).all():
            upper_bound = y_pred_plot.flatten() * 1.02
            lower_bound = y_pred_plot.flatten() * 0.98
        else:
            # Fallback if predictions are invalid
            upper_bound = np.full_like(y_pred_plot.flatten(), current_price * 1.02)
            lower_bound = np.full_like(y_pred_plot.flatten(), current_price * 0.98)
        
        fig.add_trace(go.Scatter(
            y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            y=lower_bound,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            name='Confidence Interval',
            showlegend=True
        ))
        
        # Calculate robust y-axis range
        def calculate_robust_y_range(y_true, y_pred, current_price):
            """Calculate a robust y-axis range that handles edge cases"""
            all_values = np.concatenate([y_true.flatten(), y_pred.flatten()])
            
            # Remove outliers using IQR method
            q1, q3 = np.percentile(all_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter out extreme outliers
            filtered_values = all_values[(all_values >= lower_bound) & (all_values <= upper_bound)]
            
            if len(filtered_values) > 0:
                y_min = np.min(filtered_values)
                y_max = np.max(filtered_values)
            else:
                # Fallback to original values if filtering removes everything
                y_min = np.min(all_values)
                y_max = np.max(all_values)
            
            # Ensure reasonable range
            if y_max - y_min < current_price * 0.01:  # If range is too small
                center = (y_max + y_min) / 2
                y_min = center - current_price * 0.05
                y_max = center + current_price * 0.05
            
            # Add padding
            padding = (y_max - y_min) * 0.1
            y_min = max(0, y_min - padding)  # Don't go below 0 for stock prices
            y_max = y_max + padding
            
            return [y_min, y_max]
        
        # Make sure both lines are on the same axis without auto-scaling
        y_range = calculate_robust_y_range(y_true_plot, y_pred_plot, current_price)
        
        fig.update_layout(
            title=f"{ticker_symbol} Price Prediction - Test Data",
            xaxis_title="Time (Days)",
            yaxis_title="Price ($)",
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=y_range)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        progress.progress(80)
        
        # Step 6: Future predictions (100%)
        status_text.markdown("#### 🔮 Generating future price predictions...")
        
        # Make future predictions
        st.markdown('<h2 class="sub-header">Future Price Predictions</h2>', unsafe_allow_html=True)
        
        # Add a note about data quality if using synthetic data
        if using_synthetic_data:
            st.markdown(f"""
            <div style="background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; margin-bottom: 20px;">
                <p><strong>Note:</strong> These predictions use synthetic sentiment data (real data fetch failed).</p>
                <p style="font-size: 0.9rem;">While the model is still analyzing real market price data, the sentiment component 
                is using synthetically generated data which may affect prediction accuracy.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Store the current price for scaling
        model.last_actual_price = combined_df['close'].iloc[-1]
        model.original_price_data = np.array([[model.last_actual_price]])
        
        # Predict future prices with a visual progress bar
        prediction_progress = st.progress(0)
        prediction_status = st.empty()
        
        future_prices = []
        for i in range(prediction_days):
            prediction_progress.progress((i + 1) / prediction_days)
            prediction_status.markdown(f"Predicting day {i+1} of {prediction_days}...")
            
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
                
                # Remove random variance to ensure consistent predictions
                # variance = day_pred * np.random.normal(0, 0.005)  # 0.5% standard deviation
                # day_pred += variance
                
                future_prices.append(day_pred)
            
            # Slight delay for visual effect
            time.sleep(0.1)
        
        prediction_status.empty()
        
        # Validate future predictions
        def validate_future_predictions(future_prices, current_price, ticker_symbol):
            """Validate future predictions for reasonableness"""
            if not future_prices:
                return future_prices
                
            future_array = np.array(future_prices)
            
            # Check for negative prices
            if np.any(future_array <= 0):
                st.warning(f"⚠️ **Invalid Predictions Detected** - Some future prices are negative or zero. Model may need retraining.")
                # Clip to minimum of 1% of current price
                future_array = np.clip(future_array, current_price * 0.01, None)
            
            # Check for extreme volatility (more than 50% change per day)
            if len(future_array) > 1:
                daily_changes = np.abs(np.diff(future_array) / future_array[:-1])
                max_daily_change = np.max(daily_changes)
                
                if max_daily_change > 0.5:  # 50% daily change
                    st.warning(f"⚠️ **High Volatility Predicted** - Model predicts up to {max_daily_change*100:.1f}% daily price changes.")
            
            # Check if predictions are too far from current price
            avg_prediction = np.mean(future_array)
            price_deviation = abs(avg_prediction - current_price) / current_price
            
            if price_deviation > 2.0:  # More than 200% deviation
                st.info(f"📊 **Large Price Movement Predicted** - Average predicted price (${avg_prediction:.2f}) differs significantly from current price (${current_price:.2f}).")
            
            return future_array.tolist()
        
        # Validate the predictions
        current_price = combined_df['close'].iloc[-1]
        future_prices = validate_future_predictions(future_prices, current_price, ticker_symbol)
        
        # Generate future dates
        last_date = combined_df['date'].iloc[-1]
        future_dates = [(last_date + timedelta(days=i+1)) for i in range(prediction_days)]
        
        # Create fancy prediction display
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Overview of predictions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create main prediction chart
            fig = go.Figure()
            
            # Get historical data for context (last 30 days)
            hist_dates = combined_df['date'].iloc[-30:]
            hist_prices = combined_df['close'].iloc[-30:]
            
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
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Calculate predicted returns
            starting_price = hist_prices.iloc[-1]
            final_price = future_prices[-1]
            overall_return = ((final_price - starting_price) / starting_price) * 100
            
            # Display overall predicted return with a big number
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
                <h3>Predicted {prediction_days}-Day Return</h3>
                <div style="font-size: 2.5rem; font-weight: bold; color: {'green' if overall_return >= 0 else 'red'};">
                    {overall_return:.2f}%
                </div>
                <div style="font-size: 1.2rem; margin-top: 10px;">
                    Starting: ${starting_price:.2f} → Final: ${final_price:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a risk assessment
            volatility = np.std([((future_prices[i] - future_prices[i-1]) / future_prices[i-1]) * 100 
                               for i in range(1, len(future_prices))])
            
            if volatility < 1.0:
                risk_level = "Low"
                risk_color = "green"
            elif volatility < 2.0:
                risk_level = "Moderate"
                risk_color = "orange"
            else:
                risk_level = "High"
                risk_color = "red"
                
            st.markdown(f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
                <h4>Volatility Assessment</h4>
                <div style="font-size: 1.2rem; font-weight: bold; color: {risk_color};">
                    {risk_level} Risk
                </div>
                <div>
                    Expected Volatility: {volatility:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction confidence
            r2_score = metrics['r2']
            
            # Adjust confidence level if using synthetic data
            if using_synthetic_data and r2_score > 0.5:
                # Lower confidence score by one level when using synthetic data
                r2_score = r2_score * 0.8  # Reduce the R² score to reflect lower confidence
                
            if r2_score > 0.7:
                confidence = "High"
                conf_color = "green"
            elif r2_score > 0.5:
                confidence = "Moderate"
                conf_color = "orange"
            else:
                confidence = "Low"
                conf_color = "red"
                
            st.markdown(f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 10px;">
                <h4>Prediction Confidence</h4>
                <div style="font-size: 1.2rem; font-weight: bold; color: {conf_color};">
                    {confidence} Confidence
                </div>
                <div>
                    Model R² Score: {r2_score:.4f}
                </div>
                {"<div style='margin-top:5px; font-size:0.9rem; color:#ff9800;'><i>*Using synthetic sentiment data</i></div>" if using_synthetic_data else ""}
            </div>
            """, unsafe_allow_html=True)
            
        # Display predicted prices in a fancy table
        st.subheader("Daily Price Forecast")
        
        # Prepare data for table
        prediction_data = []
        for i, (date, price) in enumerate(zip(future_dates, future_prices)):
            if i == 0:
                prev_price = hist_prices.iloc[-1]
            else:
                prev_price = future_prices[i-1]
                
            # Calculate daily change
            change = ((price - prev_price) / prev_price) * 100
            change_color = "green" if change >= 0 else "red"
            
            prediction_data.append({
                "Day": i+1,
                "Date": date.strftime('%Y-%m-%d'),
                "Predicted Price": f"${price:.2f}",
                "Change": f"<span style='color:{change_color};'>{change:+.2f}%</span>"
            })
        
        # Create a DataFrame for the table
        prediction_df = pd.DataFrame(prediction_data)
        
        # Display the table with alternating row colors
        st.markdown(
            """
            <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th {
                background-color: #4CAF50;
                color: white;
                text-align: left;
                padding: 12px;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            td {
                padding: 12px;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        # Display the table
        st.write(prediction_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a conclusion
        st.markdown('<h2 class="sub-header">Analysis Conclusion</h2>', unsafe_allow_html=True)
        
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
        
        # Create a styled container for the conclusion
        st.markdown(f"""<div style="background-color: #f9f9f9; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="color: {sentiment_color};">{sentiment_icon} {sentiment} Outlook</h3>
        </div>""", unsafe_allow_html=True)
        
        # Add factors using Streamlit native components
        st.markdown("**Key factors influencing this prediction:**")
        
        # Create bullet points using Streamlit's native components
        trend_text = "a bearish" if overall_return < 0 else "a bullish" if overall_return > 2 else "a neutral"
        sentiment_text = "positive" if avg_pos > 50 else "mixed" if avg_pos > 30 else "negative"
        volatility_text = "low" if volatility < 1.0 else "moderate" if volatility < 2.0 else "high"
        
        st.markdown(f"• Technical indicators suggest {trend_text} trend")
        st.markdown(f"• Sentiment analysis is {sentiment_text}")
        st.markdown(f"• Historical volatility is {volatility_text}")
        
        # Add disclaimer using Streamlit components
        st.warning(
            "**Disclaimer:** These predictions are based on historical data and machine learning models, " 
            "which have inherent limitations. Past performance is not indicative of future results. "
            "This information is for educational purposes only and should not be considered financial advice. "
            "Always consult with a qualified financial advisor before making investment decisions."
        )
        
        # Add synthetic data warning if applicable
        if using_synthetic_data:
            st.warning(
                "**Note:** This analysis uses synthetic sentiment data because real data fetch failed, " 
                "which may affect prediction accuracy."
            )
        
        # Complete the progress bar
        progress.progress(100)
        status_text.markdown("#### ✅ Analysis completed successfully!")
        
        # Add footer
        st.markdown("""
        <div class="footer">
            <p>Stock AI Assistant © 5 | Powered by Machine Learning and Sentiment Analysis</p>
            <p>Data sourced from Yahoo Finance | Not financial advice</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        # Show a nice error card if something goes wrong
        progress.empty()
        status_text.empty()
        
        st.error(f"An error occurred during the analysis: {str(e)}")
        st.markdown("""
        <div style="background-color: #ffebee; border-radius: 10px; padding: 20px; margin-top: 20px;">
            <h3>Troubleshooting Tips</h3>
            <ul>
                <li>Try a different stock ticker symbol</li>
                <li>Select a different time period or interval</li>
                <li>Refresh the page and try again</li>
                <li>For some stocks, there may be limited data available</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed error for debugging
        with st.expander("Show Technical Error Details"):
            import traceback
            st.code(traceback.format_exc())

else:
    st.markdown("""
    <div class="card">
        <h3>🚀 Getting Started</h3>
        <p>Welcome to Stock AI Assistant! Here's how to use this powerful tool:</p>
        <ol>
            <li>Enter a stock ticker symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)</li>
            <li>Choose the time period and interval for analysis</li>
            <li>Set the number of days you want to predict</li>
            <li>Click "Run Analysis" to generate AI-powered predictions</li>
        </ol>
        <p>Our system uses advanced machine learning techniques to analyze historical data, market trends, technical indicators, and financial news sentiment to forecast future stock prices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample stocks in a nice grid
    st.markdown('<h2 class="sub-header">Popular Stocks</h2>', unsafe_allow_html=True)
    
    # Define popular stocks with more details
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
    
    # Create a 2x4 grid of stock cards
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    # Display the stock cards
    for i, stock in enumerate(popular_stocks):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 150px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">{stock['icon']}</div>
                <div style="font-size: 1.2rem; font-weight: bold;">{stock['ticker']}</div>
                <div style="font-size: 0.9rem;">{stock['name']}</div>
                <div style="font-size: 0.8rem; color: #666;">{stock['sector']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add explanation of key features
    st.markdown('<h2 class="sub-header">Key Features</h2>', unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div style="background-color: #e8f5e9; border-radius: 10px; padding: 20px; height: 220px;">
            <h3 style="color: #2e7d32;">📊 Technical Analysis</h3>
            <p>Comprehensive analysis of price patterns, support and resistance levels, moving averages, and momentum indicators to identify market trends.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with feature_col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; border-radius: 10px; padding: 20px; height: 220px;">
            <h3 style="color: #1565c0;">🧠 AI Prediction Model</h3>
            <p>Advanced neural networks with multi-head attention mechanisms that learn complex patterns in market data to forecast future price movements.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with feature_col3:
        st.markdown("""
        <div style="background-color: #fff8e1; border-radius: 10px; padding: 20px; height: 220px;">
            <h3 style="color: #ff8f00;">📰 Sentiment Analysis</h3>
            <p>Analyzes financial news, social media, and market sentiment to understand public perception and its impact on stock prices.</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Add a "How it Works" section
    st.markdown('<h2 class="sub-header">How It Works</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f9f9f9; border-radius: 10px; padding: 20px; margin-bottom: 30px;">
        <ol>
            <li><b>Data Collection:</b> Historical price data and trading volumes are fetched from reliable financial sources.</li>
            <li><b>Feature Engineering:</b> Technical indicators like RSI, MACD, and moving averages are calculated.</li>
            <li><b>Sentiment Analysis:</b> Financial news articles and social media posts are analyzed for market sentiment.</li>
            <li><b>Model Training:</b> Deep learning models are trained on historical data to recognize patterns.</li>
            <li><b>Prediction Generation:</b> The trained model forecasts future price movements with confidence intervals.</li>
            <li><b>Visualization:</b> Results are presented in interactive charts and actionable insights.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Stock AI Assistant © 2025 | Powered by Machine Learning and Sentiment Analysis</p>
        <p>Data sourced from Yahoo Finance | Not financial advice</p>
    </div>
    """, unsafe_allow_html=True)

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

if __name__ == "__main__":
    # This will only execute when the script is run directly, not when imported
    pass
