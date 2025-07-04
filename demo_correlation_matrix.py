#!/usr/bin/env python3
"""
Demonstration of the correlation matrix feature added to the stock analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def demo_correlation_matrix():
    """Create a demonstration correlation matrix"""
    
    # Create sample data similar to what the app would generate
    np.random.seed(42)
    n_days = 100
    
    # Generate sample stock data
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    volume = np.random.lognormal(15, 0.5, n_days)
    
    # Generate sample technical indicators
    rsi = 30 + 40 * np.random.random(n_days)  # RSI between 30-70
    macd = np.random.randn(n_days) * 2
    volatility = np.abs(np.random.randn(n_days) * 0.02)
    
    # Generate sample sentiment data
    sentiment_positive = np.random.beta(2, 1, n_days)  # Skewed towards positive
    sentiment_negative = np.random.beta(1, 2, n_days)  # Skewed towards low negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'Close Price': close_prices,
        'Volume': volume,
        'RSI (14)': rsi,
        'MACD': macd,
        'Volatility': volatility,
        'Positive Sentiment': sentiment_positive,
        'Negative Sentiment': sentiment_negative,
        'Price Change': np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]]),
        'Volume Change': np.concatenate([[0], np.diff(volume) / volume[:-1]]),
        'Momentum': np.concatenate([[0] * 5, close_prices[5:] - close_prices[:-5]])
    })
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0.0, '#d73027'],    # Strong negative correlation - red
            [0.25, '#f46d43'],   # Moderate negative - orange-red
            [0.5, '#ffffff'],    # No correlation - white
            [0.75, '#74add1'],   # Moderate positive - light blue
            [1.0, '#313695']     # Strong positive correlation - dark blue
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Demo: Stock Analysis Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=700,
        width=900,
        template="plotly_white",
        font=dict(size=12)
    )
    
    # Update colorbar
    fig.update_traces(
        colorbar=dict(
            title="Correlation<br>Coefficient",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
        )
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickangle=0)
    
    # Make the layout square for better visual appeal
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=120, r=120, t=100, b=120)
    )
    
    return fig

def print_correlation_insights(df):
    """Print insights about correlations"""
    corr_matrix = df.corr()
    
    print("=== CORRELATION MATRIX INSIGHTS ===")
    print()
    
    # Find strongest positive correlations (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.where(mask)
    
    # Flatten and get top correlations
    corr_flat = corr_values.stack()
    top_positive = corr_flat.nlargest(5)
    top_negative = corr_flat.nsmallest(5)
    
    print("🔗 Strongest POSITIVE Correlations:")
    for (var1, var2), corr in top_positive.items():
        print(f"   • {var1} ↔ {var2}: {corr:.3f}")
    
    print()
    print("🔗 Strongest NEGATIVE Correlations:")
    for (var1, var2), corr in top_negative.items():
        print(f"   • {var1} ↔ {var2}: {corr:.3f}")
    
    print()
    print("📊 What these correlations tell us:")
    print("   • Strong positive correlations (>0.7) suggest features move together")
    print("   • Strong negative correlations (<-0.7) suggest features move oppositely")
    print("   • Weak correlations (~0) suggest features are independent")
    print("   • This helps identify which factors influence stock price movement")

if __name__ == "__main__":
    print("🚀 Generating demonstration correlation matrix...")
    
    # Create sample data
    np.random.seed(42)
    n_days = 100
    
    df = pd.DataFrame({
        'Close Price': 100 + np.cumsum(np.random.randn(n_days) * 0.5),
        'Volume': np.random.lognormal(15, 0.5, n_days),
        'RSI (14)': 30 + 40 * np.random.random(n_days),
        'MACD': np.random.randn(n_days) * 2,
        'Volatility': np.abs(np.random.randn(n_days) * 0.02),
        'Positive Sentiment': np.random.beta(2, 1, n_days),
        'Negative Sentiment': np.random.beta(1, 2, n_days),
    })
    
    # Add derived features
    df['Price Change'] = df['Close Price'].pct_change().fillna(0)
    df['Volume Change'] = df['Volume'].pct_change().fillna(0)
    df['Momentum'] = df['Close Price'] - df['Close Price'].shift(5)
    df = df.fillna(0)
    
    # Generate and show the correlation matrix
    fig = demo_correlation_matrix()
    
    # Save as HTML file
    output_file = "demo_correlation_matrix.html"
    fig.write_html(output_file)
    print(f"📈 Correlation matrix saved to: {output_file}")
    
    # Print insights
    print_correlation_insights(df)
    
    print()
    print("🎯 This correlation matrix feature has been added to your Flask app!")
    print("   Visit http://localhost:8183 and run an analysis to see it in action.")
    print("   It will appear as a new tab in the chart gallery called 'Feature Correlation Matrix'")
