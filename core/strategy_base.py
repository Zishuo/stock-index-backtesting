"""
Base strategy class for trading strategies.

This module provides the abstract base class that all trading strategies
should inherit from, defining the common interface and structure.
"""

import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod


class Strategy(metaclass=ABCMeta):
    """
    Abstract base class for all trading strategies.

    This class defines the common interface that all trading strategies must implement.
    It provides basic structure for managing strategy parameters, trade signals, and results.

    Attributes:
        name (str): Strategy name
        stop_loss (float): Stop loss percentage (0.0 = no stop loss)
        take_profit (float): Take profit percentage (0.0 = no take profit)
        trades (pd.DataFrame): DataFrame containing trade signals
        joined_data (pd.DataFrame): Combined data from strategy execution
    """

    def __init__(self, name: str, stop_loss: float, take_profit: float):
        """
        Initialize the strategy with basic parameters.

        Args:
            name (str): Name of the strategy
            stop_loss (float): Stop loss percentage (e.g., 0.1 for 10%)
            take_profit (float): Take profit percentage (e.g., 0.2 for 20%)
        """
        self.name = name
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = pd.DataFrame(columns=['Date', 'Signal'])
        self.trades.set_index(['Date'], inplace=True)
        self.joined_data = None
        # Action signals: Buy(1), Sell(-1), StopLoss(-2), TakeProfit, BuyAll, SellAll

    @abstractmethod
    def run_strategy(self, indicators, start_date: dt.datetime, end_date: dt.datetime,
                    verbose: bool = False):
        """
        Execute the trading strategy.

        This method must be implemented by all concrete strategy classes.
        It should analyze the input data and generate trading signals.

        Args:
            indicators: Input data or indicators (can be single or list of data sources)
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
            verbose (bool): Whether to print debug information

        Returns:
            None: Results should be stored in self.trades and self.joined_data
        """
        pass