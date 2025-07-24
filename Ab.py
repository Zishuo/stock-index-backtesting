#!/usr/bin/env python3
"""
Algorithmic Backtesting Framework (Ab.py)

This is the main compatibility layer for the refactored backtesting framework.
It maintains backward compatibility with existing notebooks while providing
access to the new modular structure.

For new development, consider importing modules directly:
    from core.data_handler import StockData
    from strategies.basic_strategies import BuyAndHold
    etc.

Installation requirements:
    pip install yfinance --upgrade --no-cache-dir
"""

# Standard library imports
import yfinance as yf
import datetime as dt
import pytz
import warnings
import numpy as np
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
from abc import abstractmethod, ABCMeta

# Suppress common pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*get_loc.*deprecated.*')
warnings.filterwarnings('ignore', message='.*Setting an item of incompatible dtype.*')

# GPU acceleration check
try:
    import cupy as cp
    import cudf as cd
    print('GPU acceleration is available')
except ImportError:
    print('GPU acceleration is NOT available')
    pass

# Import all components from the refactored modules
# This maintains backward compatibility with existing code

# Core components
from core.data_handler import StockData
from core.indicators import get_sma, get_ema, get_stochastic
from core.strategy_base import Strategy
from core.portfolio import Portfolio, BackTest

# Phase 2 enhancements
from core.strategy_factory import StrategyFactory, create_strategy
from core.config import ConfigManager, get_config
from core.validators import DataValidator, ParameterValidator
from core.exceptions import get_error_handler
from core.performance import get_performance_monitor, timer, cached

# Phase 3 advanced features
from core.plugins import get_plugin_manager, PluginManager, calculate_custom_indicator
from core.multi_asset import AssetAllocation, MultiAssetPortfolio, BuyAndHoldMultiAsset
from core.advanced_metrics import AdvancedMetrics, calculate_portfolio_metrics
from core.risk_management import RiskManager, FixedFractionSizer, StopLossControl
from core.parallel_processing import ParallelOptimizer, ParameterGrid

# Basic strategies
from strategies.basic_strategies import BuyAndHold, MACross, MAThreshold
from strategies.threshold_strategies import Threshold
from strategies.stochastic_strategy import StochasticCross
from strategies.custom_strategies import fftyspy_stg, fftynaa200r_stg, CustomizedStrategy

# Utilities
from utils.constants import *
from utils.helpers import *

# Backward compatibility: Export all classes and functions at module level
__all__ = [
    # Core classes
    'StockData',
    'Strategy',
    'Portfolio',
    'BackTest',
    
    # Strategy classes
    'BuyAndHold',
    'MACross',
    'MAThreshold',
    'Threshold',
    'StochasticCross',
    'fftyspy_stg',
    'fftynaa200r_stg',
    'CustomizedStrategy',
    
    # Technical indicators
    'get_sma',
    'get_ema', 
    'get_stochastic',
    
    # Phase 2 enhancements
    'StrategyFactory',
    'create_strategy',
    'ConfigManager',
    'get_config',
    'DataValidator',
    'ParameterValidator',
    'get_error_handler',
    'get_performance_monitor',
    'timer',
    'cached',
    
    # Phase 3 advanced features
    'get_plugin_manager',
    'PluginManager',
    'calculate_custom_indicator',
    'AssetAllocation',
    'MultiAssetPortfolio',
    'BuyAndHoldMultiAsset',
    'AdvancedMetrics',
    'calculate_portfolio_metrics',
    'RiskManager',
    'FixedFractionSizer',
    'StopLossControl',
    'ParallelOptimizer',
    'ParameterGrid',
    
    # Utility functions (from utils.helpers)
    'validate_date_range',
    'format_percentage',
    'format_currency',
    'calculate_returns',
    'calculate_log_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'resample_data',
    'merge_signals',
    'validate_strategy_parameters',
    'clean_dataframe',
    'print_trade_summary'
]

# Module metadata
__version__ = "3.0.0"
__author__ = "Backtesting Framework Team"
__description__ = "Advanced algorithmic backtesting framework with multi-asset, risk management, and parallel processing"

def get_version():
    """Return the current version of the backtesting framework."""
    return __version__

def list_strategies():
    """List all available strategy classes."""
    strategies = [
        'BuyAndHold - Simple buy and hold strategy',
        'MACross - Moving average crossover strategy', 
        'MAThreshold - Moving average threshold strategy',
        'Threshold - Price threshold with MA confirmation',
        'StochasticCross - Stochastic oscillator strategy',
        'fftyspy_stg - FFTY-SPY combined strategy',
        'fftynaa200r_stg - FFTY-NAA200R combined strategy',
        'CustomizedStrategy - User-defined signal strategy'
    ]
    
    print("Available Strategies:")
    print("=" * 50)
    for strategy in strategies:
        print(f"• {strategy}")
    print("=" * 50)
    print(f"Total: {len(strategies)} strategies available")

def quick_backtest(ticker, strategy_class, start_date, end_date, **strategy_params):
    """
    Quick backtest function for rapid strategy testing.
    
    Args:
        ticker (str): Stock ticker symbol
        strategy_class: Strategy class to use
        start_date: Backtest start date
        end_date: Backtest end date
        **strategy_params: Additional strategy parameters
        
    Returns:
        BackTest: Completed backtest object
    """
    # Load data
    stock_data = StockData(ticker)
    stock_data.get_data_from_yfinance(ticker, start_date, end_date)
    
    # Initialize strategy
    if strategy_params:
        strategy = strategy_class(**strategy_params)
    else:
        strategy = strategy_class()
    
    # Run strategy
    strategy.run_strategy(stock_data, start_date, end_date)
    
    # Run backtest
    backtest = BackTest()
    backtest.run_backtest(strategy, stock_data, start_date, end_date)
    
    return backtest

def create_strategy_by_name(strategy_name, **params):
    """
    Create strategy using the factory pattern (Phase 2 enhancement).
    
    Args:
        strategy_name (str): Name of the strategy
        **params: Strategy parameters
        
    Returns:
        Strategy: Created strategy instance
    """
    return StrategyFactory.create_strategy(strategy_name, **params)

def optimized_backtest(ticker, strategy_name, start_date, end_date, **params):
    """
    Run an optimized backtest with Phase 2 enhancements.
    
    Args:
        ticker (str): Stock ticker symbol
        strategy_name (str): Name of the strategy
        start_date: Backtest start date
        end_date: Backtest end date
        **params: Strategy and backtest parameters
        
    Returns:
        BackTest: Completed backtest object with performance monitoring
    """
    monitor = get_performance_monitor()
    
    monitor.start_timer("total_backtest")
    
    try:
        # Load configuration
        config = get_config()
        
        # Load and validate data
        monitor.start_timer("data_loading")
        stock_data = StockData(ticker)
        stock_data.get_data_from_yfinance(ticker, start_date, end_date)
        
        # Validate data
        validator = DataValidator()
        report = validator.validate_price_data(stock_data.data, ticker)
        if not report['valid']:
            print(f"Data validation warnings for {ticker}: {report['warnings']}")
        
        monitor.stop_timer("data_loading")
        
        # Create strategy using factory
        monitor.start_timer("strategy_creation")
        strategy = StrategyFactory.create_strategy(strategy_name, **params)
        monitor.stop_timer("strategy_creation")
        
        # Run strategy
        monitor.start_timer("strategy_execution")
        strategy.run_strategy(stock_data, start_date, end_date)
        monitor.stop_timer("strategy_execution")
        
        # Run backtest
        monitor.start_timer("backtest_execution")
        backtest = BackTest(
            principal=config.get_backtest_config().initial_capital,
            trade_size=config.get_backtest_config().trade_size
        )
        
        backtest.run_backtest(
            strategy, stock_data, start_date, end_date, 
            long_term_tax_rate=config.get_backtest_config().long_term_tax_rate,
            short_term_tax_rate=config.get_backtest_config().short_term_tax_rate,
            verbose=config.get_backtest_config().verbose
        )
        monitor.stop_timer("backtest_execution")
        
        return backtest
        
    finally:
        monitor.stop_timer("total_backtest")
        
        # Print performance summary if verbose
        if get_config().get_backtest_config().verbose:
            monitor.print_summary()

def show_framework_info():
    """Show comprehensive framework information including all phases."""
    print(f"Advanced Algorithmic Backtesting Framework v{__version__}")
    print("=" * 70)
    print("PHASE 1 - Core Framework:")
    print("• core.data_handler - StockData class")
    print("• core.indicators - Technical indicators") 
    print("• core.strategy_base - Strategy base class")
    print("• core.portfolio - Portfolio and BackTest classes")
    print("• strategies.* - Various trading strategies")
    print("• utils.* - Helper functions and constants")
    print()
    print("PHASE 2 - Enhanced Features:")
    print("• core.strategy_factory - Factory pattern for strategy creation")
    print("• core.config - Centralized configuration management")
    print("• core.validators - Data validation and parameter checking")
    print("• core.exceptions - Improved error handling")
    print("• core.performance - Performance monitoring and optimization")
    print()
    print("PHASE 3 - Advanced Features:")
    print("• core.plugins - Plugin architecture for custom indicators")
    print("• core.multi_asset - Multi-asset portfolio management")
    print("• core.advanced_metrics - Advanced performance analytics")
    print("• core.risk_management - Comprehensive risk controls")
    print("• core.parallel_processing - Parameter optimization & Monte Carlo")
    print()
    print("Key Functions:")
    print("• list_strategies() - Show available strategies")
    print("• optimize_strategy() - Parameter optimization with parallel processing")
    print("• create_multi_asset_portfolio() - Multi-asset portfolio management")
    print("• calculate_advanced_metrics() - Comprehensive performance analysis")
    print("• setup_risk_management() - Risk management configuration")
    print("=" * 70)

def optimize_strategy(strategy_class, param_grid_dict, ticker, start_date, end_date, 
                     n_jobs=-1, optimization_metric='sharpe_ratio'):
    """
    Optimize strategy parameters using parallel processing (Phase 3 feature).
    
    Args:
        strategy_class: Strategy class to optimize
        param_grid_dict (Dict): Parameter grid dictionary
        ticker (str): Stock ticker
        start_date: Start date
        end_date: End date
        n_jobs (int): Number of parallel jobs
        optimization_metric (str): Metric to optimize
        
    Returns:
        List: Best optimization results
    """
    param_grid = ParameterGrid(param_grid_dict)
    optimizer = ParallelOptimizer(n_jobs=n_jobs, verbose=True)
    
    return optimizer.optimize_parameters(
        strategy_class, param_grid, ticker, start_date, end_date,
        optimization_metric=optimization_metric
    )

def create_multi_asset_portfolio(allocations_dict, rebalance_frequency='monthly'):
    """
    Create a multi-asset portfolio (Phase 3 feature).
    
    Args:
        allocations_dict (Dict[str, float]): Asset allocations
        rebalance_frequency (str): Rebalancing frequency
        
    Returns:
        MultiAssetPortfolio: Multi-asset portfolio instance
    """
    allocation = AssetAllocation(allocations_dict, rebalance_frequency)
    return MultiAssetPortfolio()

def calculate_advanced_metrics(balance_df, benchmark_data=None, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics (Phase 3 feature).
    
    Args:
        balance_df (pd.DataFrame): Portfolio balance data
        benchmark_data (pd.DataFrame, optional): Benchmark data
        risk_free_rate (float): Risk-free rate
        
    Returns:
        Dict: Advanced performance metrics
    """
    return calculate_portfolio_metrics(balance_df, benchmark_data, risk_free_rate)

def setup_risk_management(stop_loss_pct=0.1, max_drawdown=0.2, position_size_method='fixed'):
    """
    Setup risk management system (Phase 3 feature).
    
    Args:
        stop_loss_pct (float): Stop loss percentage
        max_drawdown (float): Maximum drawdown threshold
        position_size_method (str): Position sizing method
        
    Returns:
        RiskManager: Configured risk manager
    """
    risk_manager = RiskManager()
    
    # Add position sizer
    if position_size_method == 'fixed':
        risk_manager.position_sizer = FixedFractionSizer(0.1)
    
    # Add risk controls
    risk_manager.add_risk_control(StopLossControl(stop_loss_pct))
    
    return risk_manager

def list_custom_indicators():
    """List available custom indicators from the plugin system."""
    plugin_manager = get_plugin_manager()
    indicators = plugin_manager.list_indicators()
    
    print("Available Custom Indicators:")
    print("=" * 50)
    for name, description in indicators.items():
        print(f"• {name}: {description}")
    print("=" * 50)
    print(f"Total: {len(indicators)} custom indicators available")

def monte_carlo_backtest(strategy_class, strategy_params, ticker, start_date, end_date, 
                        n_simulations=1000):
    """
    Run Monte Carlo analysis on a strategy (Phase 3 feature).
    
    Args:
        strategy_class: Strategy class
        strategy_params (Dict): Strategy parameters
        ticker (str): Stock ticker
        start_date: Start date
        end_date: End date
        n_simulations (int): Number of simulations
        
    Returns:
        Dict: Monte Carlo analysis results
    """
    optimizer = ParallelOptimizer(verbose=True)
    return optimizer.monte_carlo_analysis(
        strategy_class, strategy_params, ticker, start_date, end_date, n_simulations
    )

# Backward compatibility aliases and wrapper functions
# These ensure that existing notebooks continue to work without modification

def create_stock_data(ticker):
    """Create StockData instance (backward compatibility)."""
    return StockData(ticker)

def create_backtest(principal=100000):
    """Create BackTest instance (backward compatibility)."""
    return BackTest(principal=principal)

# Legacy function wrappers for standalone indicator functions
def sma_indicator(df, column, window, output_col):
    """Legacy wrapper for SMA calculation."""
    get_sma(df, column, output_col, window)

def ema_indicator(df, column, window, output_col):
    """Legacy wrapper for EMA calculation.""" 
    get_ema(df, column, output_col, window)

def stochastic_indicator(df, column, out_fastk, out_k, out_d, k_window, fk_window, fd_window):
    """Legacy wrapper for stochastic calculation."""
    get_stochastic(df, column, out_fastk, out_k, out_d, k_window, fk_window, fd_window)

# Print framework information on import
print(f"Advanced Algorithmic Backtesting Framework v{__version__} loaded successfully")
print("=" * 70)
print("PHASE 1 - Core modular structure:")
print("• core.data_handler - StockData class")
print("• core.indicators - Technical indicators") 
print("• core.strategy_base - Strategy base class")
print("• core.portfolio - Portfolio and BackTest classes")
print("• strategies.* - Various trading strategies")
print()
print("PHASE 2 - Enhanced capabilities:")
print("• Strategy factory pattern & configuration management")
print("• Data validation & enhanced error handling")
print("• Performance monitoring & optimization")
print()
print("PHASE 3 - Advanced features:")
print("• Plugin architecture for custom indicators")
print("• Multi-asset portfolio management")
print("• Advanced performance metrics & risk management")
print("• Parallel parameter optimization & Monte Carlo analysis")
print("=" * 70)
print("Quick Start:")
print("• show_framework_info() - Comprehensive feature overview")
print("• optimize_strategy(BuyAndHold, {'param': [1,2,3]}, 'TQQQ', '2023-01-01', '2023-12-31')")
print("• create_multi_asset_portfolio({'TQQQ': 0.6, 'QLD': 0.4})")
print("=" * 70)