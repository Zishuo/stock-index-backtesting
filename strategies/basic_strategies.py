"""
Basic trading strategies implementation.

This module contains fundamental trading strategies including buy-and-hold,
moving average crossover, and moving average threshold strategies.
"""

import datetime as dt
import numpy as np
import pandas as pd

from core.strategy_base import Strategy


class BuyAndHold(Strategy):
    """
    Simple buy-and-hold strategy.

    This strategy buys at the start date and sells at the end date,
    representing a passive investment approach.
    """

    def __init__(self, name: str = 'Buy and Hold', stop_loss: float = 0, take_profit: float = 0):
        """
        Initialize Buy and Hold strategy.

        Args:
            name (str): Strategy name
            stop_loss (float): Stop loss percentage (typically 0 for buy-and-hold)
            take_profit (float): Take profit percentage (typically 0 for buy-and-hold)
        """
        super().__init__(name, stop_loss, take_profit)

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        """
        Execute buy-and-hold strategy.

        Args:
            indicator: StockData object containing price data
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
        """
        # Get the start and end date, if the start date is before the stock data start date,
        # use the stock data start date
        sd = max(start_date, indicator.data.index[0])
        ed = min(end_date, indicator.data.index[-1])

        # Check if sd is in indicator index
        if sd not in indicator.data.index:
            sd = indicator.data.index[indicator.data.index.get_indexer([sd], method='backfill')[0]]

        self.trades = pd.DataFrame({'Date': [sd, ed], 'Signal': [1, -1]})
        self.trades.set_index('Date', inplace=True)


class MACross(Strategy):
    """
    Moving Average Crossover Strategy.

    This strategy generates buy signals when the short-term moving average
    crosses above the long-term moving average, and sell signals when it
    crosses below.
    """

    def __init__(self, name: str = 'MA Cross', short_window: int = 50, long_window: int = 200,
                 stop_loss: float = 0, take_profit: float = 0):
        """
        Initialize MA Crossover strategy.

        Args:
            name (str): Strategy name
            short_window (int): Short-term MA window size
            long_window (int): Long-term MA window size
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
        """
        stg_name = '{} {}/{}'.format(name, short_window, long_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.short_window = short_window
        self.long_window = long_window

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        """
        Execute MA crossover strategy.

        Args:
            indicator: StockData object containing price data
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
        """
        # Clear the trades
        self.trades = pd.DataFrame(columns=['Date', 'Signal'])
        self.trades.set_index('Date', inplace=True)

        # Check if the columns are present in the indicator
        if 'MA{}'.format(self.short_window) not in indicator.data.columns:
            indicator.get_indicators('Close', ma_windows=[self.short_window])
        if 'MA{}'.format(self.long_window) not in indicator.data.columns:
            indicator.get_indicators('Close', ma_windows=[self.long_window])

        # Filter data for the date range
        if start_date:
            self.joined_data = indicator.data.loc[
                (indicator.data.index >= start_date) & (indicator.data.index <= end_date)
            ].copy()
        else:
            self.joined_data = indicator.data.copy()

        # Calculate the short and long moving average
        self.joined_data['ShortMA'] = indicator.data['MA{}'.format(self.short_window)]
        self.joined_data['LongMA'] = indicator.data['MA{}'.format(self.long_window)]

        # Calculate the signal
        self.joined_data['Signal'] = 0.0
        # Buy signal: short MA > long MA
        self.joined_data['Signal'] = np.where(
            self.joined_data['ShortMA'] > self.joined_data['LongMA'], 1.0, 0.0)
        # Sell signal: short MA < long MA
        self.joined_data['Signal'] = np.where(
            self.joined_data['ShortMA'] < self.joined_data['LongMA'], -1.0,
            self.joined_data['Signal'])

        self.trades = self.joined_data[['Signal']]


class MAThreshold(Strategy):
    """
    Moving Average Threshold Strategy.

    This strategy generates signals based on the price's relationship to a
    moving average threshold, buying when price exceeds threshold and selling
    when it falls below.
    """

    def __init__(self, name='MA TH', ma_window: int = 20, buy_threshold: float = 1,
                 sell_threshold: float = 1, stop_loss: float = 0, take_profit: float = 0):
        """
        Initialize MA Threshold strategy.

        Args:
            name (str): Strategy name
            ma_window (int): Moving average window size
            buy_threshold (float): Buy threshold ratio (e.g., 1.02 for 2% above MA)
            sell_threshold (float): Sell threshold ratio (e.g., 0.98 for 2% below MA)
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
        """
        stg_name = '{} {}/{} MA {}'.format(name, buy_threshold, sell_threshold, ma_window)
        super().__init__(stg_name, stop_loss, take_profit)
        self.ma_window = ma_window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run_strategy(self, indicator, start_date: dt.datetime, end_date: dt.datetime):
        """
        Execute MA threshold strategy.

        Args:
            indicator: StockData object containing price data
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
        """
        self.trades = pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])
        self.trades.set_index(['Date'], inplace=True)
        self.stock_ticker = indicator.ticker

        ma_str = 'MA{}'.format(self.ma_window)
        if ma_str not in indicator.data.columns:
            indicator.get_indicators('Close', ma_windows=[self.ma_window])

        if start_date:
            self.joined_data = indicator.data.loc[
                (indicator.data.index >= start_date) & (indicator.data.index <= end_date)
            ].copy()
        else:
            self.joined_data = indicator.data.copy()

        self.joined_data['price_to_MA'] = self.joined_data['Close'] / self.joined_data[ma_str]

        # Calculate the signal
        self.joined_data['Signal'] = np.where(
            (self.joined_data['price_to_MA'] < self.sell_threshold), -1.0,
            np.where((self.joined_data['price_to_MA'] > self.buy_threshold), 1.0, 0.0)
        )
        self.trades = self.joined_data[['Signal']]