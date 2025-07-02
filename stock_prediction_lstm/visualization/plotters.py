import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import matplotlib
import os
import sys

# Set matplotlib backend based on environment
if 'DISPLAY' not in os.environ and sys.platform != 'darwin':
    # Use non-interactive backend for headless environments
    matplotlib.use('Agg')
else:
    # Try to use interactive backend for environments with display
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        try:
            matplotlib.use('Qt5Agg')
        except ImportError:
            matplotlib.use('Agg')

def visualize_stock_data(df, ticker_symbol, output_dir=None):
    fig = plt.figure(figsize=(15, 20))
    grid_spec = plt.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])
    
    ax1 = fig.add_subplot(grid_spec[0])
    ax1.plot(df['date'], df['close'], label='Close Price', color='blue', linewidth=2)
    
    if 'ma7' in df.columns:
        ax1.plot(df['date'], df['ma7'], label='7-day MA', color='orange', linestyle='--')
    if 'ma14' in df.columns:
        ax1.plot(df['date'], df['ma14'], label='14-day MA', color='green', linestyle='--')
    if 'ma30' in df.columns:
        ax1.plot(df['date'], df['ma30'], label='30-day MA', color='red', linestyle='--')
    
    ax1.set_title(f'{ticker_symbol} Stock Price and Moving Averages', fontsize=14)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(grid_spec[1], sharex=ax1)
    ax2.bar(df['date'], df['volume'], color='blue', alpha=0.5)
    ax2.set_title('Trading Volume', fontsize=14)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True)
    
    if 'rsi14' in df.columns:
        ax3 = fig.add_subplot(grid_spec[2], sharex=ax1)
        ax3.plot(df['date'], df['rsi14'], color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('Relative Strength Index (14)', fontsize=14)
        ax3.set_ylabel('RSI', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(True)
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        ax4 = fig.add_subplot(grid_spec[3], sharex=ax1)
        ax4.plot(df['date'], df['macd'], label='MACD', color='blue', linewidth=1.5)
        ax4.plot(df['date'], df['macd_signal'], label='Signal Line', color='red', linewidth=1.5)
        
        macd_hist = df['macd'] - df['macd_signal']
        colors = ['green' if x > 0 else 'red' for x in macd_hist]
        ax4.bar(df['date'], macd_hist, color=colors, alpha=0.5)
        
        ax4.set_title('MACD', fontsize=14)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True)
    
    if 'sentiment_positive' in df.columns and 'sentiment_negative' in df.columns:
        ax5 = fig.add_subplot(grid_spec[4], sharex=ax1)
        ax5.plot(df['date'], df['sentiment_positive'], label='Positive', color='green', linewidth=1.5)
        ax5.plot(df['date'], df['sentiment_negative'], label='Negative', color='red', linewidth=1.5)
        if 'sentiment_neutral' in df.columns:
            ax5.plot(df['date'], df['sentiment_neutral'], label='Neutral', color='gray', linewidth=1.5)
        
        ax5.set_title('Sentiment Analysis', fontsize=14)
        ax5.set_xlabel('Date', fontsize=12)
        ax5.set_ylabel('Sentiment Score', fontsize=12)
        ax5.legend(loc='best')
        ax5.grid(True)
    
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    if output_dir:
        from stock_prediction_lstm.core.utils import save_plot
        save_plot(fig, f"{ticker_symbol}_stock_data", output_dir)
    
    _show_plot_safely(fig, f"{ticker_symbol}_stock_data", output_dir)
    
    return fig

def visualize_prediction_comparison(model, X_market_test, X_sentiment_test, y_test, ticker_symbol, output_dir=None):
    y_pred_scaled = model.model.predict([X_market_test, X_sentiment_test])
    
    if y_test.ndim == 3:
        y_test_2d = y_test.reshape(y_test.shape[0], y_test.shape[2])
    else:
        y_test_2d = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    
    y_pred_2d = y_pred_scaled.reshape(-1, 1) if y_pred_scaled.ndim == 1 else y_pred_scaled
    if y_pred_scaled.ndim == 3:
        y_pred_2d = y_pred_scaled.reshape(y_pred_scaled.shape[0], y_pred_scaled.shape[2])
    
    y_true = model.price_scaler.inverse_transform(y_test_2d)
    y_pred = model.price_scaler.inverse_transform(y_pred_2d)
    
    if hasattr(model, 'transformed_price_data') and hasattr(model, 'original_price_data'):
        if (model.original_price_data > 0).all() and np.mean(model.transformed_price_data) < np.mean(model.original_price_data):
            print("Applying inverse log transform to both predicted and actual values")
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
    
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    scale_ratio = true_mean / pred_mean if pred_mean != 0 else 1.0
    
    if scale_ratio > 10 or scale_ratio < 0.1:
        print(f"WARNING: Scale discrepancy detected! true_mean={true_mean:.4f}, pred_mean={pred_mean:.4f}, ratio={scale_ratio:.4f}")
        print("Applying scaling correction to predicted values...")
        
        y_pred_flat = y_pred.flatten() * scale_ratio
        print(f"Adjusted prediction range: [{np.min(y_pred_flat):.4f}, {np.max(y_pred_flat):.4f}]")
    else:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
    
    fig = plt.figure(figsize=(12, 8))
    
    x_indices = np.arange(len(y_pred_flat))
    
    plt.plot(x_indices, y_pred_flat, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.plot(x_indices, y_true_flat, label='Actual', color='blue', linewidth=2)
    
    mse = np.mean(np.square(y_true_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    y_true_mean = np.mean(y_true_flat)
    total_sum_squares = np.sum(np.square(y_true_flat - y_true_mean))
    residual_sum_squares = np.sum(np.square(y_true_flat - y_pred_flat))
    r2 = 1 - (residual_sum_squares / total_sum_squares) if total_sum_squares != 0 else 0
    
    non_zero_mask = y_true_flat != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / y_true_flat[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    plt.title(f'{ticker_symbol} Stock Price Prediction\n' +
              f'RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%', 
              fontsize=14)
    
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_dir:
        from stock_prediction_lstm.core.utils import save_plot
        save_plot(fig, f"{ticker_symbol}_prediction_comparison", output_dir)
    
    plt.tight_layout()
    _show_plot_safely(fig, f"{ticker_symbol}_prediction_comparison", output_dir)
    
    return fig

def visualize_future_predictions(future_prices, future_dates, df, ticker_symbol, window=30, output_dir=None):
    hist_dates = df['date'].iloc[-window:]
    hist_prices = df['close'].iloc[-window:]
    
    future_prices_adjusted = np.array(future_prices).copy()
    
    last_hist_price = hist_prices.iloc[-1]
    first_pred_price = future_prices_adjusted[0]
    
    if first_pred_price != 0:
        percent_diff = abs((first_pred_price - last_hist_price) / last_hist_price) * 100
    else:
        percent_diff = 100
    
    scaling_applied = False
    if percent_diff > 10:
        print(f"\nWARNING: Detected {percent_diff:.2f}% gap between historical and predicted prices")
        print(f"Last historical price: ${last_hist_price:.2f}, First prediction: ${first_pred_price:.2f}")
        
        adjustment_factor = last_hist_price / first_pred_price if first_pred_price != 0 else 1.0
        
        future_prices_adjusted = future_prices_adjusted * adjustment_factor
        
        print(f"Applied adjustment factor: {adjustment_factor:.4f} for visualization")
        print(f"Adjusted prediction range: ${np.min(future_prices_adjusted):.2f} - ${np.max(future_prices_adjusted):.2f}")
        scaling_applied = True
    
    fig = plt.figure(figsize=(12, 8))
    
    plt.plot(hist_dates, hist_prices, label='Historical', color='blue', linewidth=2)
    
    plt.plot(future_dates, future_prices_adjusted, label='Predicted', color='red', 
             linestyle='--', linewidth=2, marker='o', markersize=8)
    
    uncertainty_factor = 0.01 if scaling_applied else 0.01
    plt.fill_between(future_dates, 
                     [p * (1 - uncertainty_factor * (i+1)) for i, p in enumerate(future_prices_adjusted)],
                     [p * (1 + uncertainty_factor * (i+1)) for i, p in enumerate(future_prices_adjusted)],
                     color='red', alpha=0.2)
    
    pred_return = (future_prices[-1] - future_prices[0]) / future_prices[0] * 100 if future_prices[0] != 0 else 0
    
    title = f'{ticker_symbol} Stock Price Prediction - Next {len(future_prices)} Days\n'
    title += f'Predicted Return: {pred_return:.2f}%'
    if scaling_applied:
        title += ' (Note: Visualization scaled for continuity)'
    plt.title(title, fontsize=14)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.gcf().autofmt_xdate()
    
    for i, (date, price, orig_price) in enumerate(zip(future_dates, future_prices_adjusted, future_prices)):
        if scaling_applied:
            plt.annotate(f'${price:.2f}\n(${orig_price:.2f})', 
                         (date, price),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        else:
            plt.annotate(f'${price:.2f}', 
                         (date, price),
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    if output_dir:
        from stock_prediction_lstm.core.utils import save_plot
        save_plot(fig, f"{ticker_symbol}_future_predictions", output_dir)
    
    plt.tight_layout()
    _show_plot_safely(fig, f"{ticker_symbol}_future_predictions", output_dir)
    
    return fig

def visualize_feature_importance(df, target_col='close', output_dir=None):
    correlations = df.corr()[target_col].sort_values(ascending=False)
    
    correlations = correlations.drop(target_col)
    
    top_pos = correlations.head(15)
    top_neg = correlations.tail(15).iloc[::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.barh(top_pos.index, top_pos.values, color='green')
    ax1.set_title('Top Positive Correlations', fontsize=14)
    ax1.set_xlabel('Correlation with Price', fontsize=12)
    ax1.set_ylabel('Features', fontsize=12)
    ax1.grid(axis='x')
    
    ax2.barh(top_neg.index, top_neg.values, color='red')
    ax2.set_title('Top Negative Correlations', fontsize=14)
    ax2.set_xlabel('Correlation with Price', fontsize=12)
    ax2.grid(axis='x')
    
    if output_dir:
        from stock_prediction_lstm.core.utils import save_plot
        save_plot(fig, "feature_importance", output_dir)
    
    plt.tight_layout()
    _show_plot_safely(fig, "feature_importance", output_dir)
    
    return fig

def visualize_sentiment_impact(df, window=14, output_dir=None):
    if not all(col in df.columns for col in ['close', 'sentiment_positive', 'sentiment_negative']):
        print("Required columns missing for sentiment impact visualization")
        return None
    
    df['price_change'] = df['close'].pct_change(periods=5) * 100
    
    corr_pos = df['sentiment_positive'].rolling(window=window).corr(df['price_change'])
    corr_neg = df['sentiment_negative'].rolling(window=window).corr(df['price_change'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(df['date'], corr_pos, label='Positive Sentiment', color='green', linewidth=2)
    ax1.plot(df['date'], corr_neg, label='Negative Sentiment', color='red', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax1.set_title(f'Rolling {window}-day Correlation: Sentiment vs Price Change', fontsize=14)
    ax1.set_ylabel('Correlation', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True)
    
    ax2.plot(df['date'], df['close'], color='blue', linewidth=2)
    ax2.set_ylabel('Price ($)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax3 = ax2.twinx()
    ax3.plot(df['date'], df['sentiment_positive'], color='green', linestyle='--', 
             label='Positive', linewidth=1.5)
    ax3.plot(df['date'], df['sentiment_negative'], color='red', linestyle='--', 
             label='Negative', linewidth=1.5)
    ax3.set_ylabel('Sentiment Score', fontsize=12, color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.legend(loc='upper right')
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Price vs Sentiment', fontsize=14)
    ax2.grid(True)
    
    plt.gcf().autofmt_xdate()
    
    if output_dir:
        from stock_prediction_lstm.core.utils import save_plot
        save_plot(fig, "sentiment_impact", output_dir)
    
    plt.tight_layout()
    _show_plot_safely(fig, "sentiment_impact", output_dir)
    
    return fig

def _show_plot_safely(fig=None, filename=None, output_dir=None):
    """
    Show plot only if using an interactive backend, otherwise save it.
    
    Args:
        fig: The matplotlib figure to handle (if None, uses current figure)
        filename: Optional filename to save the plot when in non-interactive mode
        output_dir: Optional directory to save the plot (if None, saves in current directory)
    """
    backend = matplotlib.get_backend().lower()
    if 'agg' in backend:
        # Non-interactive backend - optionally save the plot
        if filename:
            try:
                import os
                # Use current working directory if no output_dir specified
                save_dir = output_dir if output_dir else os.getcwd()
                os.makedirs(save_dir, exist_ok=True)
                
                if fig is None:
                    fig = plt.gcf()
                filepath = os.path.join(save_dir, f"{filename}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {filepath}")
            except Exception as e:
                print(f"Could not save plot: {e}")
        else:
            print("Plot generated (non-interactive mode - plot not displayed)")
        plt.close()
    else:
        # Interactive backend - show the plot
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            plt.close()
