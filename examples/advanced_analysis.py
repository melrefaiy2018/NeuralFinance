#!/usr/bin/env python3
"""
Example script showing how to use the stock_prediction_lstm package for advanced analysis
with organized output directories.
"""

import os
from datetime import datetime
from pathlib import Path

from stock_prediction_lstm.analysis import StockAnalyzer
from stock_prediction_lstm.visualization import visualize_stock_data, visualize_future_predictions
from stock_prediction_lstm.config import Config

def setup_advanced_calculations_directory():
    """Create and configure an advanced calculations directory for organized output."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Create timestamped calculations directory in the same location as the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calc_dir = script_dir / f"advanced_analysis_{timestamp}"
    calc_dir.mkdir(exist_ok=True)
    
    # Create comprehensive subdirectories for advanced analysis
    subdirs = [
        "data",              # Raw and processed data
        "models",            # Saved models
        "visualizations",    # Charts and plots
        "predictions",       # Prediction results
        "analysis_reports",  # Detailed analysis reports
        "comparisons",       # Multi-stock comparisons
        "logs"              # Log files
    ]
    
    for subdir in subdirs:
        (calc_dir / subdir).mkdir(exist_ok=True)
    
    # Configure package to use our directory
    Config.DATA_CACHE_DIR = calc_dir / "data"
    Config.MODELS_DIR = calc_dir / "models"
    Config.LOGS_DIR = calc_dir / "logs"
    
    print(f"📁 Created advanced analysis directory: {calc_dir.absolute()}")
    print(f"📊 Advanced analysis outputs will be organized in subdirectories")
    
    return calc_dir

def save_analysis_metadata(calc_dir, ticker, period, interval):
    """Save metadata about the analysis session."""
    metadata_file = calc_dir / "analysis_metadata.txt"
    
    metadata_content = f"""Advanced Stock Analysis Session
{'=' * 40}

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Working Directory: {os.getcwd()}
Output Directory: {calc_dir.absolute()}

Stock Analysis Parameters:
- Ticker Symbol: {ticker}
- Data Period: {period}
- Data Interval: {interval}

Configuration Settings:
- API Key Configured: {Config.ALPHA_VANTAGE_API_KEY is not None}
- Cache Enabled: {Config.CACHE_ENABLED}
- Sentiment Analysis: {Config.SENTIMENT_ANALYSIS_ENABLED}
- LSTM Units: {Config.DEFAULT_LSTM_UNITS}
- Epochs: {Config.DEFAULT_EPOCHS}
- Batch Size: {Config.DEFAULT_BATCH_SIZE}

Output Structure:
- data/              # Raw and processed data files
- models/            # Saved ML models
- visualizations/    # Charts and plots (PNG, HTML)
- predictions/       # Prediction results and analysis
- analysis_reports/  # Detailed analysis reports
- comparisons/       # Multi-stock comparison data
- logs/              # System and error logs
"""
    
    metadata_file.write_text(metadata_content)
    print(f"📄 Analysis metadata saved to: analysis_metadata.txt")

def run_advanced_analysis(calc_dir):
    """Run comprehensive advanced analysis with organized outputs."""
    print("\n" + "=" * 60)
    print("🚀 Starting Advanced Stock Analysis")
    print("=" * 60)
    
    # Analysis parameters
    ticker = 'GOOGL'
    period = '2y'
    interval = '1d'
    
    # Save analysis metadata
    save_analysis_metadata(calc_dir, ticker, period, interval)
    
    # Change to visualizations directory for chart outputs
    original_cwd = os.getcwd()
    viz_dir = calc_dir / "visualizations"
    
    # Convert calc_dir to absolute path before changing directories
    calc_dir = calc_dir.absolute()
    
    os.chdir(viz_dir)
    
    try:
        # Create analyzer
        analyzer = StockAnalyzer()
        
        print(f"\n📈 Running comprehensive analysis for {ticker}")
        print(f"📊 Parameters: {period} period, {interval} interval")
        print("-" * 50)
        
        # Run the analysis
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(ticker, period, interval)
        
        if df is not None and future_prices is not None:
            print("✅ Analysis completed successfully")
            
            # Save processed data
            print("\n💾 Saving processed data...")
            data_file = calc_dir / "data" / f"{ticker}_processed_data.csv"
            df.to_csv(str(data_file.absolute()), index=True)
            print(f"📊 Data saved to: data/{ticker}_processed_data.csv")
            
            # Create visualizations
            print(f"\n📊 Creating visualizations for {ticker}...")
            print("   Generating stock data visualization...")
            visualize_stock_data(df, ticker)
            
            print("   Generating future predictions visualization...")
            visualize_future_predictions(future_prices, future_dates, df, ticker)
            
            # Save predictions
            print(f"\n💰 Saving prediction results...")
            predictions_file = calc_dir / "predictions" / f"{ticker}_predictions.txt"
            with open(str(predictions_file.absolute()), 'w') as f:
                f.write(f"{ticker} Future Price Predictions\n")
                f.write("=" * 40 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Period: {period}\n")
                f.write(f"Data Interval: {interval}\n")
                f.write(f"Model Type: LSTM Neural Network\n\n")
                f.write("Predictions:\n")
                f.write("-" * 20 + "\n")
                
                for i, (price, date) in enumerate(zip(future_prices, future_dates)):
                    prediction_line = f"Day {i+1} ({date}): ${price:.2f}"
                    f.write(f"{prediction_line}\n")
                    if i < 10:  # Print first 10 to console
                        print(f"   {prediction_line}")
                
                if len(future_prices) > 10:
                    print(f"   ... and {len(future_prices) - 10} more predictions")
            
            print(f"📄 Predictions saved to: predictions/{ticker}_predictions.txt")
            
            # Save model if available
            if model is not None:
                print(f"\n🤖 Saving trained model...")
                try:
                    model_file = calc_dir / "models" / f"{ticker}_lstm_model"
                    model.save(str(model_file.absolute()))
                    print(f"🤖 Model saved to: models/{ticker}_lstm_model")
                except Exception as e:
                    print(f"⚠️  Could not save model: {str(e)}")
            
            # Create detailed analysis report
            print(f"\n📋 Creating detailed analysis report...")
            create_analysis_report(calc_dir, ticker, df, future_prices, future_dates, model)
            
            print(f"\n✅ Advanced analysis for {ticker} completed successfully!")
            
        else:
            print(f"❌ Analysis failed - no data or predictions generated")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        # Save error details
        error_file = calc_dir / "logs" / "analysis_error.txt"
        with open(str(error_file.absolute()), 'w') as f:
            f.write(f"Analysis Error: {str(e)}\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Ticker: {ticker}\n")
            f.write(f"Parameters: {period}, {interval}\n")
            import traceback
            f.write(f"\nTraceback:\n{traceback.format_exc()}")
        print(f"📄 Error details saved to: logs/analysis_error.txt")
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)

def create_analysis_report(calc_dir, ticker, df, future_prices, future_dates, model):
    """Create a comprehensive analysis report."""
    report_file = calc_dir / "analysis_reports" / f"{ticker}_analysis_report.txt"
    
    with open(str(report_file.absolute()), 'w') as f:
        f.write(f"Comprehensive Stock Analysis Report: {ticker}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data Summary
        f.write("DATA SUMMARY\n")
        f.write("-" * 20 + "\n")
        if df is not None:
            f.write(f"Data Points: {len(df)} records\n")
            f.write(f"Date Range: {df.index[0]} to {df.index[-1]}\n")
            f.write(f"Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}\n")
            f.write(f"Average Volume: {df['Volume'].mean():,.0f}\n")
            f.write(f"Average Price: ${df['Close'].mean():.2f}\n\n")
        
        # Prediction Summary
        f.write("PREDICTION SUMMARY\n")
        f.write("-" * 20 + "\n")
        if future_prices is not None:
            f.write(f"Predictions Generated: {len(future_prices)}\n")
            f.write(f"Current Price: ${df['Close'].iloc[-1]:.2f}\n")
            f.write(f"First Prediction: ${future_prices[0]:.2f}\n")
            f.write(f"Last Prediction: ${future_prices[-1]:.2f}\n")
            f.write(f"Predicted Change: {((future_prices[-1] / df['Close'].iloc[-1]) - 1) * 100:.2f}%\n\n")
        
        # Model Information
        f.write("MODEL INFORMATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model Type: LSTM Neural Network\n")
        f.write(f"LSTM Units: {Config.DEFAULT_LSTM_UNITS}\n")
        f.write(f"Training Epochs: {Config.DEFAULT_EPOCHS}\n")
        f.write(f"Batch Size: {Config.DEFAULT_BATCH_SIZE}\n")
        f.write(f"Sequence Length: {Config.DEFAULT_SEQUENCE_LENGTH}\n")
        f.write(f"Dropout Rate: {Config.DEFAULT_DROPOUT_RATE}\n\n")
        
        # Files Generated
        f.write("FILES GENERATED\n")
        f.write("-" * 20 + "\n")
        f.write(f"📊 Data: data/{ticker}_processed_data.csv\n")
        f.write(f"💰 Predictions: predictions/{ticker}_predictions.txt\n")
        f.write(f"📈 Visualizations: visualizations/{ticker}_*.png\n")
        if model is not None:
            f.write(f"🤖 Model: models/{ticker}_lstm_model/\n")
        f.write(f"📋 Report: analysis_reports/{ticker}_analysis_report.txt\n")
    
    print(f"📋 Analysis report saved to: analysis_reports/{ticker}_analysis_report.txt")

def create_session_summary(calc_dir):
    """Create a final session summary."""
    summary_file = calc_dir / "session_summary.txt"
    
    # Count generated files
    file_counts = {}
    total_files = 0
    
    for subdir in calc_dir.iterdir():
        if subdir.is_dir():
            count = len(list(subdir.glob("*")))
            if count > 0:
                file_counts[subdir.name] = count
                total_files += count
    
    with open(str(summary_file.absolute()), 'w') as f:
        f.write("Advanced Stock Analysis Session Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {calc_dir.absolute()}\n")
        f.write(f"Total files generated: {total_files}\n\n")
        
        f.write("Files by category:\n")
        for category, count in file_counts.items():
            f.write(f"  {category}: {count} files\n")
        
        f.write(f"\nDirectory structure:\n")
        f.write(f"  {calc_dir.name}/\n")
        for subdir in sorted(calc_dir.iterdir()):
            if subdir.is_dir():
                f.write(f"  ├── {subdir.name}/\n")
                for file in sorted(subdir.glob("*"))[:3]:  # Show first 3 files
                    f.write(f"  │   ├── {file.name}\n")
                if len(list(subdir.glob("*"))) > 3:
                    f.write(f"  │   └── ... and {len(list(subdir.glob('*'))) - 3} more\n")
    
    print(f"📋 Session summary saved to: session_summary.txt")

def main():
    """Main function for advanced analysis with organized output."""
    print("🔮 Stock Prediction LSTM - Advanced Analysis Example")
    print("=" * 60)
    
    try:
        # Setup organized output directory
        calc_dir = setup_advanced_calculations_directory()
        
        # Run advanced analysis
        run_advanced_analysis(calc_dir)
        
        # Create session summary
        create_session_summary(calc_dir)
        
        print("\n" + "=" * 60)
        print("🎉 Advanced Analysis Complete!")
        print("=" * 60)
        print(f"📁 All files saved in: {calc_dir.absolute()}")
        print("\n📂 Generated content:")
        print("   📊 Data files and processed datasets")
        print("   📈 Interactive visualizations and charts")
        print("   💰 Detailed prediction results")
        print("   🤖 Trained machine learning models")
        print("   📋 Comprehensive analysis reports")
        print(f"\n💡 To explore results: cd {calc_dir}")
        
    except Exception as e:
        print(f"\n❌ Advanced analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
