"""
Trading strategy implementations.

This module contains various trading strategies that can be used with the backtesting framework.
"""

from .basic_strategies import BuyAndHold, MACross, MAThreshold
from .threshold_strategies import Threshold
from .stochastic_strategy import StochasticCross
from .custom_strategies import fftyspy_stg, fftynaa200r_stg, CustomizedStrategy

__all__ = [
    # Basic strategies
    'BuyAndHold', 'MACross', 'MAThreshold',
    # Threshold strategies
    'Threshold',
    # Advanced strategies
    'StochasticCross',
    # Custom strategies
    'fftyspy_stg', 'fftynaa200r_stg', 'CustomizedStrategy'
]