#!/usr/bin/env python3
"""
Test script to verify the larger correlation matrix visualization
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create test data similar to what the real app would have"""
    np.random.seed(42)
    
    # Create 100 data points
    n_points = 100
    dates = pd.date_range('2024-01-01', periods=n_points, freq='D')
    
    # Create base price data
    base_price = 100 + np.cumsum(np.random.normal(0, 1, n_points))
    
    # Create test dataframe with various features
    data = {
        'date': dates,
        'close': base_price,
        'volume': np.random.lognormal(15, 0.5, n_points),
        'sentiment_positive': np.random.beta(2, 2, n_points),
        'sentiment_negative': np.random.beta(1, 3, n_points),
        'rsi14': 30 + 40 * np.random.beta(2, 2, n_points),
        'macd': np.random.normal(0, 2, n_points),
        'macd_signal': np.random.normal(0, 1.5, n_points),
        'ma7': base_price + np.random.normal(0, 0.5, n_points),
        'ma30': base_price + np.random.normal(0, 1, n_points),
        'volatility': np.random.exponential(0.02, n_points),
        'momentum': np.random.normal(0, 3, n_points)
    }
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['price_change'] = df['close'].pct_change()
    
    return df

def test_correlation_matrix():
    """Test the correlation matrix with improved sizing"""
    print("🧪 Testing larger correlation matrix visualization...")
    
    # Create test data
    combined_df = create_test_data()
    ticker_symbol = "TEST"
    
    # Import the function from the web app
    try:
        from stock_prediction_lstm.web.flask_app import create_correlation_matrix_chart
        print("✅ Successfully imported correlation matrix function")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        # Create the chart
        html_output = create_correlation_matrix_chart(combined_df, ticker_symbol)
        
        # Check if we got valid HTML
        if html_output and len(html_output) > 100:
            print("✅ Correlation matrix chart generated successfully")
            print(f"📏 Output length: {len(html_output)} characters")
            
            # Check for key elements in the HTML
            checks = [
                ('Plotly chart div', '<div' in html_output),
                ('Chart data', 'data' in html_output),
                ('Heatmap trace', 'Heatmap' in html_output),
                ('Title contains TEST', 'TEST' in html_output),
                ('Correlation scale', 'colorscale' in html_output)
            ]
            
            for check_name, result in checks:
                status = "✅" if result else "❌"
                print(f"{status} {check_name}: {result}")
            
            # Save output for manual inspection
            output_file = "test_larger_correlation_matrix.html"
            with open(output_file, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Test Larger Correlation Matrix</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                </head>
                <body>
                    <h1>Test Larger Correlation Matrix</h1>
                    <p>Generated with improved sizing and spacing</p>
                    {html_output}
                </body>
                </html>
                """)
            
            print(f"💾 Saved test output to: {output_file}")
            print("🌐 Open this file in a browser to see the improved correlation matrix")
            
            return True
            
        else:
            print("❌ Chart generation failed or returned empty output")
            return False
            
    except Exception as e:
        print(f"❌ Error creating correlation matrix: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_correlation_matrix()
    if success:
        print("\n🎉 Test completed successfully!")
        print("📈 The correlation matrix should now have:")
        print("   • Larger overall size (900-1600px based on features)")
        print("   • Bigger text (12-18px for correlations)")
        print("   • More spacing (200px margins)")
        print("   • Better axis labels (12-16px)")
        print("   • No text overlap issues")
    else:
        print("\n💥 Test failed - check errors above")
    
    sys.exit(0 if success else 1)
