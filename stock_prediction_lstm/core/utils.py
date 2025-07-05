import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def create_output_directory(config):
    """
    Create a timestamped output directory for the current analysis run.

    Args:
        config (dict): A dictionary containing the configuration for the analysis run.

    Returns:
        str: The path to the output directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.get("mode") == 1:
        base_name = f"diagnostic_{config.get('ticker', 'unknown')}"
    elif config.get("mode") == 2:
        base_name = f"analysis_{config.get('ticker', 'unknown')}"
    elif config.get("mode") == 3:
        tickers = config.get("tickers", [])
        if len(tickers) > 3:
            ticker_str = f"{tickers[0]}_{tickers[1]}_{tickers[2]}_plus{len(tickers)-3}"
        else:
            ticker_str = "_".join(tickers)
        base_name = f"comparison_{ticker_str}"
    else:
        base_name = "unknown_analysis"

    output_dir = os.path.join(os.getcwd(), "analysis_outputs", f"{base_name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write(f"Analysis run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

    return output_dir


def save_plot(fig, filename, output_dir, dpi=300):
    """
    Save a matplotlib figure to the output directory.

    Args:
        fig: The matplotlib figure to save.
        filename (str): The name of the file.
        output_dir (str): The path to the output directory.
        dpi (int, optional): The resolution of the plot. Defaults to 300.

    Returns:
        str: The path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")

    return fig_path


def save_dataframe(df, filename, output_dir):
    """
    Save a pandas DataFrame to the output directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file.
        output_dir (str): The path to the output directory.

    Returns:
        str: The path to the saved DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=False)

    return csv_path
