import click
from ..analysis.stock_analyzer import StockAnalyzer


@click.group()
def cli():
    """Neural Finance CLI"""
    pass


@cli.command()
@click.option("--ticker", default="NVDA", help="Stock ticker symbol.")
@click.option("--period", default="1y", help="Data period.")
@click.option("--interval", default="1d", help="Data interval.")
def analyze(ticker, period, interval):
    """
    Run a full analysis for a single stock.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The data period.
        interval (str): The data interval.
    """
    analyzer = StockAnalyzer()
    analyzer.run_analysis_for_stock(ticker, period, interval)


@cli.command()
@click.option("--ticker", default="NVDA", help="Stock ticker symbol.")
@click.option("--period", default="1y", help="Data period.")
def diagnostic(ticker, period):
    """
    Run a self-diagnostic for a stock.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The data period.
    """
    analyzer = StockAnalyzer()
    analyzer.self_diagnostic(ticker, period)


if __name__ == "__main__":
    cli()
