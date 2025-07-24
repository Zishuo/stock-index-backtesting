"""
Core backtesting framework components.

This module provides the fundamental classes and utilities for financial backtesting,
including data handling, technical indicators, strategies, and portfolio management.
"""

# Import all core components to maintain compatibility
from .data_handler import StockData
from .indicators import get_sma, get_ema, get_stochastic
from .strategy_base import Strategy
from .portfolio import Portfolio, BackTest

# Phase 2 enhancements
from .strategy_factory import StrategyFactory, create_strategy
from .config import ConfigManager, BacktestConfig, TechnicalIndicatorConfig, DataConfig, get_config
from .validators import DataValidator, ParameterValidator, ValidationError
from .exceptions import (
    BacktestingError, DataError, StrategyError, PortfolioError,
    ConfigurationError, IndicatorError, NetworkError,
    ErrorHandler, get_error_handler
)
from .performance import (
    PerformanceMonitor, get_performance_monitor, timer, cached,
    DataFrameOptimizer, ParallelProcessor
)

__all__ = [
    # Core Phase 1 components
    'StockData',
    'get_sma', 'get_ema', 'get_stochastic',
    'Strategy',
    'Portfolio', 'BackTest',
    
    # Phase 2 enhancements
    'StrategyFactory', 'create_strategy',
    'ConfigManager', 'BacktestConfig', 'TechnicalIndicatorConfig', 'DataConfig', 'get_config',
    'DataValidator', 'ParameterValidator', 'ValidationError',
    'BacktestingError', 'DataError', 'StrategyError', 'PortfolioError',
    'ConfigurationError', 'IndicatorError', 'NetworkError',
    'ErrorHandler', 'get_error_handler',
    'PerformanceMonitor', 'get_performance_monitor', 'timer', 'cached',
    'DataFrameOptimizer', 'ParallelProcessor'
]