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
import os
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot

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

# Set style for matplotlib
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


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
        
        # Setup calculation directories
        self.base_dir = "calculations/portfolio_analysis"
        self.data_dir = f"{self.base_dir}/data"
        self.viz_dir = f"{self.base_dir}/visualizations"
        self.reports_dir = f"{self.base_dir}/reports"
        
        # Create timestamp for this analysis session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"{self.base_dir}/session_{self.timestamp}"
        
        # Ensure directories exist
        for directory in [self.data_dir, self.viz_dir, self.reports_dir, self.session_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print(f"Initialized PortfolioAnalyzer with {len(self.tickers)} assets: {self.tickers}")
        print(f"Results will be saved to: {self.session_dir}")

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
            'timestamp': self.timestamp,
            'session_directory': self.session_dir,
            **current_metrics,
            'optimization': optimization_result,
            'rebalance_frequency': rebalance_frequency
        }
        
        # Automatically save all data
        print("💾 Saving analysis data...")
        saved_files = self.save_analysis_data(analysis_result)
        analysis_result['saved_files'] = saved_files
        
        print("=== Portfolio Analysis Complete ===")
        print(f"📁 All data saved to: {self.session_dir}")
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
    
    def save_analysis_data(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save all analysis data to files in the calculations directory.
        
        Args:
            analysis_results (Dict[str, Any]): Results from analyze_portfolio()
            
        Returns:
            Dict[str, str]: Dictionary of saved file paths
        """
        saved_files = {}
        
        # Save main results as JSON
        results_file = f"{self.session_dir}/portfolio_analysis_results.json"
        # Convert numpy arrays and other non-serializable objects to lists/dicts
        serializable_results = self._make_serializable(analysis_results.copy())
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        saved_files['results_json'] = results_file
        
        # Save results as pickle for full object preservation
        pickle_file = f"{self.session_dir}/portfolio_analysis_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(analysis_results, f)
        saved_files['results_pickle'] = pickle_file
        
        # Save price data
        if self.price_data is not None:
            price_file = f"{self.session_dir}/price_data.csv"
            self.price_data.to_csv(price_file)
            saved_files['price_data'] = price_file
        
        # Save returns data
        if self.returns_data is not None:
            returns_file = f"{self.session_dir}/returns_data.csv"
            self.returns_data.to_csv(returns_file)
            saved_files['returns_data'] = returns_file
        
        # Save correlation matrix
        if self.correlation_matrix is not None:
            corr_file = f"{self.session_dir}/correlation_matrix.csv"
            self.correlation_matrix.to_csv(corr_file)
            saved_files['correlation_matrix'] = corr_file
        
        # Save portfolio composition
        portfolio_file = f"{self.session_dir}/portfolio_composition.json"
        with open(portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
        saved_files['portfolio_composition'] = portfolio_file
        
        # Save formatted report
        report_file = f"{self.session_dir}/portfolio_report.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_report(analysis_results))
        saved_files['text_report'] = report_file
        
        return saved_files
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj
    
    def plot_portfolio_performance(self, results: Dict[str, Any], show_individual: bool = True, save: bool = True):
        """
        Create comprehensive portfolio performance dashboard with optimized styling.
        
        Args:
            results (Dict[str, Any]): Analysis results
            show_individual (bool): Whether to show individual asset performance
            save (bool): Whether to save the plot
        """
        # Set style for professional appearance
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        
        # Create custom color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0', '#FF5722']
        
        # 1. Portfolio Value Over Time (Top Left)
        ax1 = plt.subplot(2, 3, 1)
        if self.price_data is not None:
            portfolio_value = (self.price_data * self.weights).sum(axis=1)
            normalized_portfolio = portfolio_value / portfolio_value.iloc[0] * 100
            
            ax1.plot(self.price_data.index, normalized_portfolio, 
                    linewidth=3, color=colors[0], label='Portfolio Value')
            ax1.set_title('📈 Portfolio Performance (Normalized to 100)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Normalized Value', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Individual Asset Performance (Top Middle)
        ax2 = plt.subplot(2, 3, 2)
        if show_individual and self.price_data is not None:
            for i, ticker in enumerate(self.tickers):
                if ticker in self.price_data.columns:
                    normalized_asset = self.price_data[ticker] / self.price_data[ticker].iloc[0] * 100
                    ax2.plot(self.price_data.index, normalized_asset, 
                            linewidth=2, color=colors[i % len(colors)], 
                            label=ticker, alpha=0.8)
            
            ax2.set_title('📊 Individual Asset Performance', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Normalized Value', fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Portfolio Composition (Top Right)
        ax3 = plt.subplot(2, 3, 3)
        wedges, texts, autotexts = ax3.pie(self.weights, labels=self.tickers, autopct='%1.1f%%',
                                          colors=colors[:len(self.tickers)], startangle=90,
                                          textprops={'fontsize': 10})
        ax3.set_title('🎯 Portfolio Allocation', fontsize=14, fontweight='bold')
        
        # 4. Risk-Return Scatter (Bottom Left)
        ax4 = plt.subplot(2, 3, 4)
        if 'individual_returns' in results and 'individual_volatility' in results:
            returns = [results['individual_returns'][ticker] for ticker in self.tickers 
                      if ticker in results['individual_returns']]
            volatilities = [results['individual_volatility'][ticker] for ticker in self.tickers 
                           if ticker in results['individual_volatility']]
            
            scatter = ax4.scatter(volatilities, returns, s=200, c=colors[:len(returns)], 
                                 alpha=0.7, edgecolors='black', linewidth=2)
            
            # Add portfolio point
            ax4.scatter(results['volatility'], results['expected_return'], 
                       s=300, c='red', marker='*', edgecolors='black', 
                       linewidth=2, label='Portfolio', zorder=5)
            
            # Add labels for each point
            for i, ticker in enumerate([t for t in self.tickers if t in results['individual_returns']]):
                ax4.annotate(ticker, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            ax4.set_xlabel('Annual Volatility', fontsize=12)
            ax4.set_ylabel('Expected Annual Return', fontsize=12)
            ax4.set_title('⚖️ Risk-Return Profile', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # 5. Key Metrics (Bottom Middle)
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        metrics_text = [
            f"Expected Return: {results['expected_return']:.2%}",
            f"Volatility: {results['volatility']:.2%}",
            f"Sharpe Ratio: {results['sharpe_ratio']:.3f}",
            f"Max Drawdown: {results['max_drawdown']:.2%}",
            f"VaR (95%): {results['value_at_risk_95']:.2%}",
            f"Data Points: {results['data_points']}"
        ]
        
        for i, text in enumerate(metrics_text):
            ax5.text(0.1, 0.9 - i*0.12, text, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)], alpha=0.3))
        
        ax5.set_title('📊 Key Metrics', fontsize=14, fontweight='bold')
        
        # 6. Optimization Comparison (Bottom Right)
        ax6 = plt.subplot(2, 3, 6)
        if results['optimization']['success']:
            categories = ['Current\nPortfolio', 'Optimized\nPortfolio']
            sharpe_ratios = [results['sharpe_ratio'], results['optimization']['sharpe_ratio']]
            
            bars = ax6.bar(categories, sharpe_ratios, color=[colors[0], colors[1]], alpha=0.8, edgecolor='black')
            ax6.set_ylabel('Sharpe Ratio', fontsize=12)
            ax6.set_title('🎯 Optimization Impact', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, sharpe_ratios):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = f"{self.session_dir}/portfolio_performance_dashboard.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, save: bool = True):
        """
        Create an enhanced correlation matrix heatmap.
        
        Args:
            save (bool): Whether to save the plot
        """
        if self.correlation_matrix is None:
            print("No correlation data available")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(self.correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        plt.title('🔗 Asset Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Assets', fontsize=12, fontweight='bold')
        plt.ylabel('Assets', fontsize=12, fontweight='bold')
        
        if save:
            save_path = f"{self.session_dir}/correlation_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔗 Correlation matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_optimization_comparison(self, results: Dict[str, Any], save: bool = True):
        """
        Create detailed optimization comparison visualization.
        
        Args:
            results (Dict[str, Any]): Analysis results
            save (bool): Whether to save the plot
        """
        if not results['optimization']['success']:
            print("No optimization results available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50']
        
        # 1. Weight Comparison
        tickers = list(self.portfolio.keys())
        current_weights = [self.portfolio[t] for t in tickers]
        optimal_weights = [results['optimization']['optimal_weights'].get(t, 0) for t in tickers]
        
        x = np.arange(len(tickers))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, current_weights, width, label='Current', color=colors[0], alpha=0.8)
        bars2 = ax1.bar(x + width/2, optimal_weights, width, label='Optimal', color=colors[1], alpha=0.8)
        
        ax1.set_xlabel('Assets', fontweight='bold')
        ax1.set_ylabel('Weight', fontweight='bold')
        ax1.set_title('⚖️ Portfolio Weight Comparison', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(tickers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. Performance Metrics Comparison
        metrics = ['Return', 'Volatility', 'Sharpe Ratio']
        current_values = [results['expected_return'], results['volatility'], results['sharpe_ratio']]
        optimal_values = [results['optimization']['expected_return'], 
                         results['optimization']['volatility'], 
                         results['optimization']['sharpe_ratio']]
        
        x_metrics = np.arange(len(metrics))
        bars1 = ax2.bar(x_metrics - width/2, current_values, width, label='Current', color=colors[2], alpha=0.8)
        bars2 = ax2.bar(x_metrics + width/2, optimal_values, width, label='Optimal', color=colors[3], alpha=0.8)
        
        ax2.set_xlabel('Metrics', fontweight='bold')
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('📊 Performance Metrics Comparison', fontweight='bold', fontsize=14)
        ax2.set_xticks(x_metrics)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficient Frontier (if possible to compute)
        ax3.set_title('📈 Risk-Return Profile', fontweight='bold', fontsize=14)
        ax3.scatter(results['volatility'], results['expected_return'], 
                   s=200, c=colors[0], label='Current Portfolio', marker='o', edgecolors='black')
        ax3.scatter(results['optimization']['volatility'], results['optimization']['expected_return'], 
                   s=200, c=colors[1], label='Optimal Portfolio', marker='*', edgecolors='black')
        ax3.set_xlabel('Volatility', fontweight='bold')
        ax3.set_ylabel('Expected Return', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Weight Changes
        weight_changes = [optimal_weights[i] - current_weights[i] for i in range(len(tickers))]
        colors_changes = [colors[2] if change >= 0 else colors[3] for change in weight_changes]
        
        bars = ax4.bar(tickers, weight_changes, color=colors_changes, alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Assets', fontweight='bold')
        ax4.set_ylabel('Weight Change', fontweight='bold')
        ax4.set_title('📈 Suggested Weight Changes', fontweight='bold', fontsize=14)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, change in zip(bars, weight_changes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{change:+.1%}', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            save_path = f"{self.session_dir}/optimization_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"⚖️ Optimization comparison saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict[str, Any], save: bool = True):
        """
        Create interactive Plotly dashboard.
        
        Args:
            results (Dict[str, Any]): Analysis results
            save (bool): Whether to save the HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Performance Over Time', 'Asset Correlation Heatmap',
                           'Portfolio Composition', 'Risk-Return Profile'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # 1. Portfolio Performance Over Time
        if self.price_data is not None:
            portfolio_value = (self.price_data * self.weights).sum(axis=1)
            normalized_portfolio = portfolio_value / portfolio_value.iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(x=self.price_data.index, y=normalized_portfolio,
                          mode='lines', name='Portfolio Value',
                          line=dict(width=3, color='#2E86AB')),
                row=1, col=1
            )
        
        # 2. Correlation Heatmap
        if self.correlation_matrix is not None:
            fig.add_trace(
                go.Heatmap(z=self.correlation_matrix.values,
                          x=self.correlation_matrix.columns,
                          y=self.correlation_matrix.index,
                          colorscale='RdBu', zmid=0,
                          showscale=True),
                row=1, col=2
            )
        
        # 3. Portfolio Composition
        fig.add_trace(
            go.Pie(labels=self.tickers, values=self.weights,
                   name="Portfolio Allocation"),
            row=2, col=1
        )
        
        # 4. Risk-Return Scatter
        if 'individual_returns' in results and 'individual_volatility' in results:
            returns = [results['individual_returns'][ticker] for ticker in self.tickers 
                      if ticker in results['individual_returns']]
            volatilities = [results['individual_volatility'][ticker] for ticker in self.tickers 
                           if ticker in results['individual_volatility']]
            tickers_available = [t for t in self.tickers if t in results['individual_returns']]
            
            fig.add_trace(
                go.Scatter(x=volatilities, y=returns, mode='markers+text',
                          text=tickers_available, textposition="top center",
                          marker=dict(size=12, color='#F18F01'),
                          name='Individual Assets'),
                row=2, col=2
            )
            
            # Add portfolio point
            fig.add_trace(
                go.Scatter(x=[results['volatility']], y=[results['expected_return']],
                          mode='markers+text', text=['Portfolio'], textposition="top center",
                          marker=dict(size=15, color='red', symbol='star'),
                          name='Portfolio'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="📊 Interactive Portfolio Analysis Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            height=800
        )
        
        if save:
            save_path = f"{self.session_dir}/interactive_dashboard.html"
            plot(fig, filename=save_path, auto_open=False)
            print(f"🌐 Interactive dashboard saved to: {save_path}")
        
        fig.show()