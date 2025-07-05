#!/usr/bin/env python3
"""
Enhanced example script showing how to use the stock_prediction_lstm package programmatically.

This script creates a dedicated calculations directory and saves all outputs there
for better organization and to avoid cluttering the file system.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import shutil

from stock_prediction_lstm.analysis import StockAnalyzer
from stock_prediction_lstm.config import Config


def create_calculations_directory():
    """
    Create a timestamped calculations directory for this analysis session.

    Returns:
        Path: Path to the created calculations directory
    """
    # Create a timestamp for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create calculations directory in the current working directory
    calc_dir = Path(f"calculations_{timestamp}")
    calc_dir.mkdir(exist_ok=True)

    # Create subdirectories for organization
    subdirs = [
        "models",  # Saved models
        "predictions",  # Prediction results and charts
        "data",  # Processed data files
        "diagnostics",  # Diagnostic outputs
        "logs",  # Log files
        "visualizations",  # Charts and plots
    ]

    for subdir in subdirs:
        (calc_dir / subdir).mkdir(exist_ok=True)

    print(f"📁 Created calculations directory: {calc_dir.absolute()}")
    return calc_dir


def setup_output_directories(calc_dir):
    """
    Configure the package to use our calculations directory for outputs.

    Args:
        calc_dir (Path): The calculations directory path
    """
    # Update configuration to use our calculations directory
    Config.DATA_CACHE_DIR = calc_dir / "data"
    Config.MODELS_DIR = calc_dir / "models"
    Config.LOGS_DIR = calc_dir / "logs"

    # Ensure directories exist
    Config.create_directories()

    print(f"📋 Configured output directories:")
    print(f"   Data cache: {Config.DATA_CACHE_DIR}")
    print(f"   Models: {Config.MODELS_DIR}")
    print(f"   Logs: {Config.LOGS_DIR}")


def save_session_info(calc_dir):
    """
    Save information about this analysis session.

    Args:
        calc_dir (Path): The calculations directory path
    """
    session_info = f"""Stock Prediction LSTM Analysis Session
{'=' * 50}

Session Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Working Directory: {os.getcwd()}
Calculations Directory: {calc_dir.absolute()}

Configuration:
- Data Cache: {Config.DATA_CACHE_DIR}
- Models Directory: {Config.MODELS_DIR}
- Logs Directory: {Config.LOGS_DIR}
- API Key Configured: {Config.ALPHA_VANTAGE_API_KEY is not None}
- Cache Enabled: {Config.CACHE_ENABLED}
- Sentiment Analysis: {Config.SENTIMENT_ANALYSIS_ENABLED}

Analysis Examples:
1. NVDA Diagnostic (1 year data)
2. AAPL Full Analysis (6 months data)

Files generated in this session will be saved in the subdirectories above.
"""

    session_file = calc_dir / "session_info.txt"
    session_file.write_text(session_info)
    print(f"📄 Session info saved to: {session_file}")


def run_enhanced_analysis(calc_dir):
    """
    Run the stock analysis with enhanced output handling.

    Args:
        calc_dir (Path): The calculations directory path
    """
    print("\n" + "=" * 60)
    print("🚀 Starting Enhanced Stock Analysis")
    print("=" * 60)

    # Create analyzer with our configured directories
    analyzer = StockAnalyzer()

    # Example 1: Run diagnostic for NVDA
    print("\n📊 Example 1: Running diagnostic for NVDA")
    print("-" * 40)

    try:
        # Save current working directory
        original_cwd = os.getcwd()

        # Change to diagnostics directory for this analysis
        diagnostics_dir = calc_dir / "diagnostics"
        os.chdir(diagnostics_dir)

        print(f"Running diagnostic in: {diagnostics_dir}")
        analyzer.self_diagnostic("NVDA", "1y")

        print("✅ NVDA diagnostic completed")
        print(f"📁 Diagnostic files saved in: {diagnostics_dir}")

    except Exception as e:
        print(f"❌ Error in NVDA diagnostic: {str(e)}")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Example 2: Run full analysis for AAPL
    print("\n📈 Example 2: Running full analysis for AAPL")
    print("-" * 40)

    try:
        # Change to predictions directory for this analysis
        predictions_dir = calc_dir / "predictions"
        os.chdir(predictions_dir)

        print(f"Running analysis in: {predictions_dir}")
        model, df, future_prices, future_dates = analyzer.run_analysis_for_stock(
            "AAPL", "6mo", "1d"
        )

        if future_prices is not None:
            print("\n💰 Future price predictions for AAPL:")
            predictions_file = predictions_dir / "AAPL_predictions.txt"

            with open(predictions_file, "w") as f:
                f.write("AAPL Future Price Predictions\n")
                f.write("=" * 30 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Period: 6 months\n")
                f.write(f"Data Interval: 1 day\n\n")
                f.write("Predictions:\n")

                for i, (price, date) in enumerate(zip(future_prices, future_dates)):
                    prediction_line = f"Day {i+1} ({date}): ${price:.2f}"
                    print(f"   {prediction_line}")
                    f.write(f"{prediction_line}\n")

            print(f"\n📄 Predictions saved to: {predictions_file}")

            # Save model if available
            if model is not None:
                model_file = calc_dir / "models" / "AAPL_model"
                try:
                    # Save the model (assuming it's a TensorFlow/Keras model)
                    model.save(str(model_file))
                    print(f"🤖 Model saved to: {model_file}")
                except Exception as e:
                    print(f"⚠️  Could not save model: {str(e)}")

            # Save processed data
            if df is not None:
                data_file = calc_dir / "data" / "AAPL_processed_data.csv"
                df.to_csv(data_file, index=True)
                print(f"📊 Processed data saved to: {data_file}")

        else:
            print("⚠️  No predictions generated")

        print("✅ AAPL analysis completed")

    except Exception as e:
        print(f"❌ Error in AAPL analysis: {str(e)}")
        import traceback

        error_file = calc_dir / "logs" / "error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"Error in AAPL analysis: {str(e)}\n")
            f.write(traceback.format_exc())
        print(f"📄 Error details saved to: {error_file}")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def create_summary_report(calc_dir):
    """
    Create a summary report of all generated files.

    Args:
        calc_dir (Path): The calculations directory path
    """
    summary_file = calc_dir / "analysis_summary.txt"

    with open(summary_file, "w") as f:
        f.write("Stock Prediction LSTM Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # List all generated files
        f.write("Generated Files:\n")
        f.write("-" * 20 + "\n")

        for root, dirs, files in os.walk(calc_dir):
            level = root.replace(str(calc_dir), "").count(os.sep)
            indent = " " * 2 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            sub_indent = " " * 2 * (level + 1)
            for file in files:
                if file != "analysis_summary.txt":  # Don't include the summary file itself
                    file_path = Path(root) / file
                    file_size = file_path.stat().st_size
                    f.write(f"{sub_indent}{file} ({file_size} bytes)\n")

    print(f"\n📋 Analysis summary saved to: {summary_file}")


def main():
    """
    Main function that orchestrates the enhanced analysis with organized output.
    """
    print("🔮 Stock Prediction LSTM - Enhanced Analysis Example")
    print("=" * 60)

    try:
        # Step 1: Create calculations directory
        calc_dir = create_calculations_directory()

        # Step 2: Setup output directories
        setup_output_directories(calc_dir)

        # Step 3: Save session information
        save_session_info(calc_dir)

        # Step 4: Run the enhanced analysis
        run_enhanced_analysis(calc_dir)

        # Step 5: Create summary report
        create_summary_report(calc_dir)

        print("\n" + "=" * 60)
        print("🎉 Analysis Complete!")
        print("=" * 60)
        print(f"📁 All files saved in: {calc_dir.absolute()}")
        print("\n📋 Directory structure:")
        print(f"   {calc_dir}/")
        print("   ├── session_info.txt       # Session details")
        print("   ├── analysis_summary.txt   # Summary of generated files")
        print("   ├── data/                  # Processed data files")
        print("   ├── models/                # Saved models")
        print("   ├── predictions/           # Prediction results")
        print("   ├── diagnostics/           # Diagnostic outputs")
        print("   ├── logs/                  # Log files")
        print("   └── visualizations/        # Charts and plots")

        print(f"\n💡 To view results, navigate to: {calc_dir.absolute()}")

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
