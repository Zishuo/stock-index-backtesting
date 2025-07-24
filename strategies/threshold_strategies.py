"""
Threshold-based trading strategies.

This module contains strategies that use various threshold values
to generate trading signals based on price levels and moving averages.
"""

import datetime as dt
import numpy as np
import pandas as pd

from core.strategy_base import Strategy


class Threshold(Strategy):
    """
    Threshold Strategy using price levels and moving average confirmation.

    This strategy generates buy signals when price is above a buy threshold
    and above its moving average (upward trend), and sell signals when price
    is below a sell threshold and below its moving average (downward trend).
    """

    def __init__(self, name: str = 'TH', buy_threshold: float = 15, sell_threshold: float = 30,
                 signal_ma_window: int = 20, stop_loss: float = 0, take_profit: float = 0):
        """
        Initialize Threshold strategy.

        Args:
            name (str): Strategy name
            buy_threshold (float): Price level above which to consider buying
            sell_threshold (float): Price level below which to consider selling
            signal_ma_window (int): Moving average window for trend confirmation
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
        """
        stg_name = '{} {}/{} MA {}'.format(name, buy_threshold, sell_threshold, signal_ma_window)
        super().__init__(stg_name, stop_loss, take_profit)

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.signal_ma_window = signal_ma_window

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        """
        Execute threshold strategy.

        Args:
            indicator: StockData object containing price data
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
        """
        # Clear the trades
        self.trades = pd.DataFrame(columns=['Date', 'Signal'])
        self.trades.set_index('Date', inplace=True)

        self.joined_data = indicator.data.copy()
        # Filter for date range - include dates where both signal data and stock data are available
        self.joined_data = self.joined_data.loc[start_date:end_date]

        ma_str = 'MA{}'.format(self.signal_ma_window)
        self.joined_data[ma_str] = self.joined_data['Close'].rolling(
            window=self.signal_ma_window).mean()
        # Fix the NA value in the MA column
        self.joined_data[ma_str] = self.joined_data[ma_str].bfill()

        # Calculate the signal
        # Buy signal: price > buy_threshold AND price > MA (upward trend)
        # Sell signal: price < sell_threshold AND price < MA (downward trend)
        # This ensures we don't have conflicting signals when buy_threshold < sell_threshold

        self.joined_data['Signal'] = 0
        self.joined_data['Signal'] = np.where(
            (self.joined_data['Close'] > self.buy_threshold) &
            (self.joined_data[ma_str] < self.joined_data['Close']), 1.0, 0.0)
        self.joined_data['Signal'] = np.where(
            (self.joined_data['Close'] < self.sell_threshold) &
            (self.joined_data[ma_str] > self.joined_data['Close']), -1.0,
            self.joined_data['Signal'])

        # Rename the 'Close' column to the ticker symbol
        self.joined_data.rename(columns={'Close': indicator.ticker}, inplace=True)

        # Keep only the Date and Signal columns for trades
        self.trades = self.joined_data[['Signal']]