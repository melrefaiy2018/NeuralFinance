#!/usr/bin/env python3
"""
Example script showing how to use the stock_prediction_lstm package programmatically.

This version creates a calculations directory to organize outputs instead of 
scattering files throughout the file system.
"""

import os
from datetime import datetime
from pathlib import Path

from stock_prediction_lstm.analysis import StockAnalyzer
from stock_prediction_lstm.config import Config

def setup_calculations_directory():
    """Create and configure a calculations directory for organized output."""
    # Create timestamped calculations directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calc_dir = Path(f"calculations_{timestamp}")
    calc_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (calc_dir / "data").mkdir(exist_ok=True)
    (calc_dir / "models").mkdir(exist_ok=True)
    (calc_dir / "outputs").mkdir(exist_ok=True)
    
    # Configure package to use our directory
    Config.DATA_CACHE_DIR = calc_dir / "data"
    Config.MODELS_DIR = calc_dir / "models"
    
    print(f"📁 Created calculations directory: {calc_dir.absolute()}")
    print(f"📊 Data will be saved in organized subdirectories")
    
    return calc_dir

def main():
    """Main analysis function with organized output."""
    print("🔮 Stock Prediction LSTM - Basic Usage Example")
    print("=" * 50)
    
    # Setup organized output directory
    calc_dir = setup_calculations_directory()
    
    # Change to the calculations directory for this session
    original_cwd = os.getcwd()
    os.chdir(calc_dir)
    
    try:
        # Create analyzer
        analyzer = StockAnalyzer()

        # Example 1: Run diagnostic for NVDA
        print("\n📊 Example 1: Running diagnostic for NVDA")
        print("-" * 40)
        analyzer.self_diagnostic('NVDA', '1y')
        print("✅ NVDA diagnostic completed")
        
        # Example 2: Run full analysis for AAPL
        print("\n📈 Example 2: Running full analysis for AAPL")
        print("-" * 40)
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock('AAPL', '6mo', '1d')
        
        if future_prices is not None:
            print("\n💰 Future price predictions for AAPL:")
            
            # Save predictions to file
            predictions_file = calc_dir / "outputs" / "AAPL_predictions.txt"
            with open(predictions_file, 'w') as f:
                f.write("AAPL Future Price Predictions\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write("-" * 30 + "\n")
                
                for i, price in enumerate(future_prices):
                    prediction_line = f"Day {i+1}: ${price:.2f}"
                    print(f"   {prediction_line}")
                    f.write(f"{prediction_line}\n")
            
            print(f"\n📄 Predictions saved to: outputs/AAPL_predictions.txt")
            
            # Save processed data if available
            if df is not None:
                data_file = calc_dir / "outputs" / "AAPL_data.csv"
                df.to_csv(data_file)
                print(f"📊 Data saved to: outputs/AAPL_data.csv")
        
        print("✅ AAPL analysis completed")
        
        # Create simple summary
        summary_file = calc_dir / "session_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Analysis Session Summary\n")
            f.write(f"Completed: {datetime.now()}\n")
            f.write(f"Location: {calc_dir.absolute()}\n")
            f.write(f"Stocks analyzed: NVDA (diagnostic), AAPL (full analysis)\n")
        
        print(f"\n📋 Session summary: session_summary.txt")
        print(f"📁 All files saved in: {calc_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        # Save error log
        error_file = calc_dir / "error_log.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Time: {datetime.now()}\n")
        print(f"📄 Error logged to: error_log.txt")
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)
        
        print(f"\n💡 To view results: cd {calc_dir}")

if __name__ == "__main__":
    main()
