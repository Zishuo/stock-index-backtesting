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
__version__ = "2.0.0"
__author__ = "Backtesting Framework Team"
__description__ = "Algorithmic backtesting framework for financial markets"

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
print(f"Algorithmic Backtesting Framework v{__version__} loaded successfully")
print("=" * 60)
print("New modular structure available:")
print("• core.data_handler - StockData class")
print("• core.indicators - Technical indicators") 
print("• core.strategy_base - Strategy base class")
print("• core.portfolio - Portfolio and BackTest classes")
print("• strategies.* - Various trading strategies")
print("• utils.* - Helper functions and constants")
print("=" * 60)
print("Use list_strategies() to see available strategies")
print("Use quick_backtest() for rapid strategy testing")
print("=" * 60)