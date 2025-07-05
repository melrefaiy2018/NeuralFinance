"""
Portfolio Analysis Module

This module provides comprehensive portfolio analysis capabilities including
risk metrics, performance analysis, optimization, and rebalancing strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings

try:
    from ..data.fetchers import StockDataFetcher
    from ..config.model_config import ModelConfig
    from .stock_analyzer import StockAnalyzer
except ImportError:
    # Fallback for direct script execution
    from data.fetchers import StockDataFetcher
    from config.model_config import ModelConfig
    from stock_analyzer import StockAnalyzer

warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """
    Comprehensive portfolio analysis tool for multi-asset portfolios.
    
    Provides risk analysis, performance metrics, optimization capabilities,
    and rebalancing strategies for stock portfolios.
    """
    
    def __init__(self, portfolio: Dict[str, float], config: Optional[ModelConfig] = None):
        """
        Initialize Portfolio Analyzer.
        
        Args:
            portfolio (Dict[str, float]): Dictionary mapping stock symbols to allocation weights.
                                        Weights should sum to 1.0.
            config (Optional[ModelConfig]): Configuration for individual stock analysis.
        """
        self.portfolio = portfolio
        self.config = config if config is not None else ModelConfig.default()
        
        # Validate portfolio weights
        total_weight = sum(portfolio.values())
        if abs(total_weight - 1.0) > 0.01:
            print(f"Warning: Portfolio weights sum to {total_weight:.3f}, not 1.0")
            
        self.tickers = list(portfolio.keys())
        self.weights = np.array(list(portfolio.values()))
        
        # Initialize data storage
        self.price_data = None
        self.returns_data = None
        self.correlation_matrix = None
        self.cov_matrix = None
        
        print(f"Initialized PortfolioAnalyzer with {len(self.tickers)} assets: {self.tickers}")
    
    def fetch_portfolio_data(self, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical price data for all portfolio assets.
        
        Args:
            period (str): Time period for data ('1y', '2y', '5y', etc.)
            interval (str): Data interval ('1d', '1wk', '1mo')
            
        Returns:
            pd.DataFrame: Combined price data for all assets
        """
        print(f"Fetching data for {len(self.tickers)} assets...")
        
        all_data = {}
        
        for ticker in self.tickers:
            try:
                # Use yfinance for simpler data fetching
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=interval)
                
                if len(hist) > 0:
                    all_data[ticker] = hist['Close']
                    print(f"✓ {ticker}: {len(hist)} data points")
                else:
                    print(f"✗ {ticker}: No data available")
                    
            except Exception as e:
                print(f"✗ {ticker}: Error fetching data - {str(e)}")
        
        if not all_data:
            raise ValueError("No data could be fetched for any portfolio assets")
        
        # Combine all price data
        self.price_data = pd.DataFrame(all_data)
        self.price_data = self.price_data.dropna()
        
        # Calculate returns
        self.returns_data = self.price_data.pct_change().dropna()
        
        # Calculate correlation and covariance matrices
        self.correlation_matrix = self.returns_data.corr()
        self.cov_matrix = self.returns_data.cov()
        
        print(f"Portfolio data prepared: {len(self.price_data)} trading days")
        return self.price_data
    
    def calculate_portfolio_metrics(self, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dict[str, float]: Dictionary of portfolio metrics
        """
        if self.returns_data is None:
            raise ValueError("No data available. Call fetch_portfolio_data() first.")
        
        # Filter weights for available assets
        available_tickers = list(self.returns_data.columns)
        available_weights = np.array([self.portfolio[ticker] for ticker in available_tickers])
        available_weights = available_weights / available_weights.sum()  # Renormalize
        
        # Portfolio returns
        portfolio_returns = (self.returns_data[available_tickers] * available_weights).sum(axis=1)
        
        # Calculate metrics
        metrics = {}
        
        # Return metrics
        metrics['expected_return'] = portfolio_returns.mean() * 252  # Annualized
        metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = (metrics['expected_return'] - risk_free_rate) / metrics['volatility']
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdowns.min()
        
        # Additional metrics
        metrics['value_at_risk_95'] = portfolio_returns.quantile(0.05)  # 5% VaR
        metrics['conditional_var_95'] = portfolio_returns[portfolio_returns <= metrics['value_at_risk_95']].mean()
        
        # Asset-specific metrics
        individual_returns = self.returns_data[available_tickers].mean() * 252
        individual_volatility = self.returns_data[available_tickers].std() * np.sqrt(252)
        
        metrics['individual_returns'] = individual_returns.to_dict()
        metrics['individual_volatility'] = individual_volatility.to_dict()
        
        # Portfolio composition
        metrics['portfolio_weights'] = dict(zip(available_tickers, available_weights))
        metrics['correlation_matrix'] = self.correlation_matrix.to_dict()
        
        return metrics
    
    def optimize_portfolio(self, target_return: Optional[float] = None, 
                          risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Optimize portfolio weights using Modern Portfolio Theory.
        
        Args:
            target_return (Optional[float]): Target annual return. If None, optimizes for maximum Sharpe ratio.
            risk_free_rate (float): Risk-free rate for optimization
            
        Returns:
            Dict[str, Any]: Optimization results including optimal weights and metrics
        """
        if self.returns_data is None:
            raise ValueError("No data available. Call fetch_portfolio_data() first.")
        
        available_tickers = list(self.returns_data.columns)
        returns = self.returns_data[available_tickers].mean() * 252
        cov_matrix = self.returns_data[available_tickers].cov() * 252
        
        n_assets = len(available_tickers)
        
        def portfolio_stats(weights):
            port_return = np.sum(returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility
            return port_return, port_volatility, sharpe_ratio
        
        def negative_sharpe(weights):
            return -portfolio_stats(weights)[2]
        
        def portfolio_variance(weights):
            return portfolio_stats(weights)[1] ** 2
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only portfolio
        
        if target_return is not None:
            # Minimize variance for target return
            constraints.append({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - target_return})
            objective = portfolio_variance
        else:
            # Maximize Sharpe ratio
            objective = negative_sharpe
        
        # Initial guess (equal weights)
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            opt_return, opt_volatility, opt_sharpe = portfolio_stats(optimal_weights)
            
            optimization_result = {
                'success': True,
                'optimal_weights': dict(zip(available_tickers, optimal_weights)),
                'expected_return': opt_return,
                'volatility': opt_volatility,
                'sharpe_ratio': opt_sharpe,
                'original_weights': dict(zip(available_tickers, 
                                           [self.portfolio[t] for t in available_tickers]))
            }
        else:
            optimization_result = {
                'success': False,
                'message': 'Optimization failed',
                'error': result.message
            }
        
        return optimization_result
    
    def analyze_portfolio(self, period: str = '2y', interval: str = '1d',
                         rebalance_frequency: str = 'monthly',
                         risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Comprehensive portfolio analysis including data fetching, metrics calculation,
        and optimization suggestions.
        
        Args:
            period (str): Historical data period
            interval (str): Data interval
            rebalance_frequency (str): Portfolio rebalancing frequency
            risk_free_rate (float): Risk-free rate for calculations
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        print("=== Starting Portfolio Analysis ===")
        
        # Fetch data
        self.fetch_portfolio_data(period, interval)
        
        # Calculate current portfolio metrics
        current_metrics = self.calculate_portfolio_metrics(risk_free_rate)
        
        # Optimize portfolio
        optimization_result = self.optimize_portfolio(risk_free_rate=risk_free_rate)
        
        # Compile results
        analysis_result = {
            'portfolio_composition': self.portfolio,
            'analysis_period': period,
            'data_points': len(self.price_data),
            'risk_free_rate': risk_free_rate,
            **current_metrics,
            'optimization': optimization_result,
            'rebalance_frequency': rebalance_frequency
        }
        
        print("=== Portfolio Analysis Complete ===")
        return analysis_result
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a formatted text report of portfolio analysis results.
        
        Args:
            analysis_results (Dict[str, Any]): Results from analyze_portfolio()
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("📊 PORTFOLIO ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Portfolio composition
        report.append("\n🎯 PORTFOLIO COMPOSITION")
        for ticker, weight in self.portfolio.items():
            report.append(f"  {ticker}: {weight:.1%}")
        
        # Performance metrics
        report.append(f"\n📈 PERFORMANCE METRICS")
        report.append(f"  Expected Annual Return: {analysis_results['expected_return']:.2%}")
        report.append(f"  Annual Volatility: {analysis_results['volatility']:.2%}")
        report.append(f"  Sharpe Ratio: {analysis_results['sharpe_ratio']:.3f}")
        report.append(f"  Maximum Drawdown: {analysis_results['max_drawdown']:.2%}")
        report.append(f"  Value at Risk (95%): {analysis_results['value_at_risk_95']:.2%}")
        
        # Optimization results
        if analysis_results['optimization']['success']:
            opt = analysis_results['optimization']
            report.append(f"\n🎯 OPTIMIZATION RESULTS")
            report.append(f"  Optimized Sharpe Ratio: {opt['sharpe_ratio']:.3f}")
            report.append(f"  Optimized Return: {opt['expected_return']:.2%}")
            report.append(f"  Optimized Volatility: {opt['volatility']:.2%}")
            report.append("  Suggested Weights:")
            for ticker, weight in opt['optimal_weights'].items():
                report.append(f"    {ticker}: {weight:.1%}")
        
        return "\n".join(report)
