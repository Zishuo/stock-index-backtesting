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

__all__ = [
    'StockData',
    'get_sma', 'get_ema', 'get_stochastic',
    'Strategy',
    'Portfolio', 'BackTest'
]