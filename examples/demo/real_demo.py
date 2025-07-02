#!/usr/bin/env python3
"""
Real Demo Data Generator for Stock Prediction LSTM Web Interface

This script runs the actual Stock Prediction LSTM model and generates
real data that can be used in the web demo instead of fake predictions.

Usage:
    python generate_demo_data.py --ticker AAPL --period 2y --days 5
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from stock_prediction_lstm.analysis import StockAnalyzer
    from stock_prediction_lstm.models import StockSentimentModel
    print("✅ Successfully imported stock_prediction_lstm package")
except ImportError as e:
    print(f"❌ Error importing package: {e}")
    print("Please ensure the stock_prediction_lstm package is installed:")
    print("Run: pip install -e .")
    sys.exit(1)

class DemoDataGenerator:
    """Generates real demo data from the Stock Prediction LSTM model"""

    def __init__(self, ticker: str = 'NVDA', period: str = '5y', prediction_days: int = 5):
        self.ticker = ticker
        self.period = period
        self.prediction_days = prediction_days
        
        # Validate API key configuration
        try:
            from stock_prediction_lstm.config.settings import Config
            if not Config.ALPHA_VANTAGE_API_KEY or Config.ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE":
                print("\n⚠️  API KEY CONFIGURATION REQUIRED")
                print("=" * 60)
                print("Alpha Vantage API key is not properly configured.")
                print("The model will use synthetic sentiment data instead.")
                print("")
                print("To use real sentiment data:")
                print("1. Get a free API key at: https://www.alphavantage.co/support/#api-key")
                print("2. Navigate to: stock_prediction_lstm/config/keys/")
                print("3. Edit api_keys.py and replace 'YOUR_API_KEY_HERE' with your key")
                print("=" * 60)
        except Exception as e:
            print(f"⚠️  Configuration warning: {e}")
        
        self.analyzer = StockAnalyzer()
        self.demo_data = {}
        
    def run_analysis(self):
        """Run the complete analysis and collect demo data"""
        print(f"\n🚀 Running analysis for {self.ticker} ({self.period} period)")
        print("=" * 60)
        
        try:
            # Run the actual model analysis
            model, combined_df, future_prices, future_dates = self.analyzer.run_analysis_for_stock(
                self.ticker, self.period, '1d'
            )
            
            if model is None or combined_df is None:
                raise Exception("Model analysis failed - no data returned")
                
            print(f"✅ Model training completed successfully")
            print(f"✅ Dataset size: {len(combined_df)} days")
            print(f"✅ Future predictions generated: {len(future_prices)} days")
            
            # Store the results
            self.model = model
            self.combined_df = combined_df
            self.future_prices = future_prices
            self.future_dates = future_dates
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return False
    
    def evaluate_model_performance(self):
        """Evaluate the model on test data to get real metrics"""
        print("\n📊 Evaluating model performance...")
        
        try:
            # Prepare data for evaluation
            X_market, X_sentiment, y = self.model.prepare_data(self.combined_df, target_col='close')
            
            # Use last 20% for testing
            split_idx = int(0.8 * len(X_market))
            X_market_test = X_market[split_idx:]
            X_sentiment_test = X_sentiment[split_idx:]
            y_test = y[split_idx:]
            
            # Make predictions on test set
            y_pred = self.model.predict(X_market_test, X_sentiment_test)
            
            # Calculate real metrics
            metrics = self.model.evaluate(y_test, y_pred)
            
            print(f"✅ Real Model Metrics:")
            print(f"   RMSE: ${metrics['rmse']:.2f}")
            print(f"   MAE:  ${metrics['mae']:.2f}")
            print(f"   R²:   {metrics['r2']:.3f} ({metrics['r2']*100:.1f}%)")
            print(f"   MAPE: {metrics['mape']:.2f}%")
            
            self.metrics = metrics
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return False
    
    def generate_historical_context(self, days_back: int = 10):
        """Generate historical price context for the chart"""
        print(f"\n📈 Generating historical context ({days_back} days)...")
        
        try:
            # Get the last N days of actual prices
            last_prices = self.combined_df['close'].tail(days_back).tolist()
            last_dates = self.combined_df['date'].tail(days_back).tolist()
            
            # Convert dates to strings for JSON serialization
            last_dates_str = [date.strftime('%Y-%m-%d') for date in last_dates]
            
            print(f"✅ Historical data: {len(last_prices)} price points")
            print(f"   Date range: {last_dates_str[0]} to {last_dates_str[-1]}")
            print(f"   Price range: ${min(last_prices):.2f} - ${max(last_prices):.2f}")
            
            self.historical_prices = last_prices
            self.historical_dates = last_dates_str
            
            return True
            
        except Exception as e:
            print(f"❌ Historical context generation failed: {str(e)}")
            return False
    
    def generate_feature_statistics(self):
        """Generate statistics about the features used in the model"""
        print("\n🔍 Analyzing model features...")
        
        try:
            # Count different types of features
            market_features = []
            sentiment_features = []
            
            for col in self.combined_df.columns:
                if 'sentiment' in col.lower():
                    sentiment_features.append(col)
                elif col not in ['date', 'close', 'open', 'high', 'low', 'price', 'volume']:
                    market_features.append(col)
            
            # Calculate feature ranges and statistics
            feature_stats = {
                'total_features': len(market_features) + len(sentiment_features),
                'market_features': len(market_features),
                'sentiment_features': len(sentiment_features),
                'market_feature_names': market_features[:10],  # First 10 for display
                'sentiment_feature_names': sentiment_features,
                'data_points': len(self.combined_df),
                'training_sequences': len(self.combined_df) - 20  # Minus lookback window
            }
            
            print(f"✅ Feature Analysis:")
            print(f"   Total features: {feature_stats['total_features']}")
            print(f"   Market features: {feature_stats['market_features']}")
            print(f"   Sentiment features: {feature_stats['sentiment_features']}")
            print(f"   Training sequences: {feature_stats['training_sequences']}")
            
            self.feature_stats = feature_stats
            return True
            
        except Exception as e:
            print(f"❌ Feature analysis failed: {str(e)}")
            return False
    
    def prepare_demo_data(self):
        """Prepare all data for the web demo"""
        print("\n📦 Preparing demo data package...")
        
        try:
            # Current/last actual price info
            last_actual_price = self.combined_df['close'].iloc[-1]
            last_actual_date = self.combined_df['date'].iloc[-1].strftime('%Y-%m-%d')
            
            # Prepare future prediction dates
            future_dates_str = []
            base_date = self.combined_df['date'].iloc[-1]
            for i in range(self.prediction_days):
                next_date = base_date + timedelta(days=i+1)
                # Skip weekends (basic approximation)
                while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    next_date += timedelta(days=1)
                future_dates_str.append(next_date.strftime('%Y-%m-%d'))
            
            # Compile the complete demo data package
            self.demo_data = {
                'metadata': {
                    'ticker': self.ticker,
                    'period': self.period,
                    'generation_time': datetime.now().isoformat(),
                    'model_type': 'StockSentimentModel with Multi-Head Attention',
                    'data_source': 'Real predictions from trained LSTM model'
                },
                'current_state': {
                    'last_actual_price': float(last_actual_price),
                    'last_actual_date': last_actual_date,
                    'ticker_symbol': self.ticker
                },
                'historical_data': {
                    'prices': [float(p) for p in self.historical_prices],
                    'dates': self.historical_dates
                },
                'predictions': {
                    'prices': [float(p) for p in self.future_prices],
                    'dates': future_dates_str,
                    'confidence_level': float(self.metrics['r2']) if hasattr(self, 'metrics') else 0.85
                },
                'model_metrics': {
                    'rmse': float(self.metrics['rmse']),
                    'mae': float(self.metrics['mae']),
                    'r2_score': float(self.metrics['r2']),
                    'r2_percentage': float(self.metrics['r2'] * 100),
                    'mape': float(self.metrics['mape']),
                    'mse': float(self.metrics['mse'])
                },
                'feature_info': self.feature_stats,
                'chart_config': {
                    'historical_color': '#7a6b60',
                    'prediction_color': '#7c3aed',
                    'confidence_band': True,
                    'show_legend': True
                }
            }
            
            print(f"✅ Demo data package prepared:")
            print(f"   Last price: ${self.demo_data['current_state']['last_actual_price']:.2f}")
            print(f"   Predictions: {len(self.demo_data['predictions']['prices'])} days")
            print(f"   Historical context: {len(self.demo_data['historical_data']['prices'])} days")
            print(f"   Model R²: {self.demo_data['model_metrics']['r2_percentage']:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Demo data preparation failed: {str(e)}")
            return False
    
    def save_demo_data(self, output_path: str = None):
        """Save the demo data to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"demo_data_{self.ticker}_{timestamp}.json"
        
        print(f"\n💾 Saving demo data to: {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.demo_data, f, indent=2, default=str)
            
            print(f"✅ Demo data saved successfully")
            print(f"   File size: {os.path.getsize(output_path)} bytes")
            
            # Also save a JavaScript-ready version
            js_output_path = output_path.replace('.json', '_js.js')
            with open(js_output_path, 'w') as f:
                f.write(f"// Real demo data generated from Stock Prediction LSTM model\n")
                f.write(f"// Generated: {datetime.now().isoformat()}\n")
                f.write(f"// Ticker: {self.ticker}, Period: {self.period}\n\n")
                f.write(f"const realDemoData = ")
                json.dump(self.demo_data, f, indent=2, default=str)
                f.write(";\n\n")
                f.write("// Usage: Replace the chartData object in your HTML with realDemoData\n")
            
            print(f"✅ JavaScript version saved: {js_output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to save demo data: {str(e)}")
            return None
    
    def generate_usage_instructions(self, output_path: str):
        """Generate instructions for using the demo data"""
        instructions_path = output_path.replace('.json', '_instructions.md')
        
        instructions = f"""# Real Demo Data Usage Instructions

## Generated Data
- **File**: `{output_path}`
- **Ticker**: {self.ticker}
- **Period**: {self.period}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance
- **RMSE**: ${self.demo_data['model_metrics']['rmse']:.2f}
- **R² Score**: {self.demo_data['model_metrics']['r2_percentage']:.1f}%
- **MAPE**: {self.demo_data['model_metrics']['mape']:.2f}%

## How to Use in Web Demo

### Option 1: Replace JavaScript Data
1. Open your HTML file
2. Replace the `chartData` object with the data from `{output_path.replace('.json', '_js.js')}`

### Option 2: Load JSON Dynamically
```javascript
// Load the real demo data
fetch('{output_path}')
    .then(response => response.json())
    .then(data => {{
        // Update chart with real data
        updateChartWithRealData(data);
        
        // Update metrics
        document.getElementById('rmse-value').textContent = '$' + data.model_metrics.rmse.toFixed(2);
        document.getElementById('r2-value').textContent = data.model_metrics.r2_percentage.toFixed(1) + '%';
        document.getElementById('mape-value').textContent = data.model_metrics.mape.toFixed(2) + '%';
        document.getElementById('mae-value').textContent = '$' + data.model_metrics.mae.toFixed(2);
    }});
```

### Option 3: Update Chart Data Structure
```javascript
const chartData = {{
    lastActualPrice: {self.demo_data['current_state']['last_actual_price']},
    lastActualDate: '{self.demo_data['current_state']['last_actual_date']}',
    predictions: {self.demo_data['predictions']['prices']},
    historicalPrices: {self.demo_data['historical_data']['prices']}
}};
```

## Data Structure
```json
{{
    "metadata": {{ ... }},
    "current_state": {{ ... }},
    "historical_data": {{ ... }},
    "predictions": {{ ... }},
    "model_metrics": {{ ... }},
    "feature_info": {{ ... }}
}}
```

## Notes
- All prices are in USD
- Predictions include realistic constraints applied by the model
- Historical data shows actual {self.ticker} prices
- Metrics are calculated on real test data, not training data
"""
        
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        print(f"✅ Usage instructions saved: {instructions_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate real demo data for Stock Prediction LSTM')
    parser.add_argument('--ticker', default='NVDA', help='Stock ticker symbol (default: NVDA)')
    parser.add_argument('--period', default='4y', help='Data period (default: 4y)')
    parser.add_argument('--days', type=int, default=5, help='Prediction days (default: 5)')
    parser.add_argument('--output', help='Output file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("🎯 Stock Prediction LSTM - Real Demo Data Generator")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.period}")
    print(f"Prediction Days: {args.days}")
    
    # Initialize the generator
    generator = DemoDataGenerator(args.ticker, args.period, args.days)
    
    # Run the complete pipeline
    success_steps = []
    
    # Step 1: Run model analysis
    if generator.run_analysis():
        success_steps.append("✅ Model Analysis")
    else:
        print("❌ Pipeline failed at model analysis")
        return
    
    # Step 2: Evaluate model performance
    if generator.evaluate_model_performance():
        success_steps.append("✅ Model Evaluation")
    else:
        print("❌ Pipeline failed at model evaluation")
        return
    
    # Step 3: Generate historical context
    if generator.generate_historical_context():
        success_steps.append("✅ Historical Context")
    else:
        print("❌ Pipeline failed at historical context generation")
        return
    
    # Step 4: Analyze features
    if generator.generate_feature_statistics():
        success_steps.append("✅ Feature Analysis")
    else:
        print("❌ Pipeline failed at feature analysis")
        return
    
    # Step 5: Prepare demo data
    if generator.prepare_demo_data():
        success_steps.append("✅ Demo Data Preparation")
    else:
        print("❌ Pipeline failed at demo data preparation")
        return
    
    # Step 6: Save demo data
    output_path = generator.save_demo_data(args.output)
    if output_path:
        success_steps.append("✅ Data Export")
        generator.generate_usage_instructions(output_path)
    else:
        print("❌ Pipeline failed at data export")
        return
    
    # Success summary
    print("\n" + "=" * 60)
    print("🎉 Demo Data Generation Complete!")
    print("=" * 60)
    
    for step in success_steps:
        print(f"  {step}")
    
    print(f"\n📁 Output Files:")
    print(f"  • JSON Data: {output_path}")
    print(f"  • JavaScript: {output_path.replace('.json', '_js.js')}")
    print(f"  • Instructions: {output_path.replace('.json', '_instructions.md')}")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Copy the generated files to your web project")
    print(f"  2. Update your HTML to use the real data")
    print(f"  3. Test the demo with authentic predictions!")
    
    print(f"\n💡 Pro Tip: Run this script daily to keep your demo data fresh!")

if __name__ == "__main__":
    main()