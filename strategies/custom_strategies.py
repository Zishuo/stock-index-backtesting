"""
Custom trading strategies.

This module contains specialized trading strategies developed for specific
market conditions and combinations of indicators.
"""

import datetime as dt
import numpy as np
import pandas as pd

from core.strategy_base import Strategy
from core.indicators import get_sma
from core.data_handler import StockData


class fftyspy_stg(Strategy):
    """
    FFTY-SPY Combined Strategy.

    This strategy combines FFTY (Russell 2000) signals with SPY conditions
    to generate trading signals based on market breadth and momentum.
    """

    def __init__(self, name: str = 'fftyspy_stg', stop_loss: float = 0, take_profit: float = 0,
                 ffty_sell_threshold: float = 0.95, ffty_buy_threshold: float = 1.02,
                 spy_consecutive_buy_threshold: int = 1, spy_consecutive_days: int = 10,
                 spy_max_off_new_high_pct: float = -0.2):
        """
        Initialize FFTY-SPY strategy.

        Args:
            name (str): Strategy name
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            ffty_sell_threshold (float): FFTY sell threshold vs 200MA
            ffty_buy_threshold (float): FFTY buy threshold vs 200MA
            spy_consecutive_buy_threshold (int): SPY consecutive threshold
            spy_consecutive_days (int): Number of consecutive days to check
            spy_max_off_new_high_pct (float): Maximum drawdown from new high
        """
        super().__init__(name, stop_loss, take_profit)
        self.ffty_sell_threshold = ffty_sell_threshold
        self.ffty_buy_threshold = ffty_buy_threshold
        self.spy_consecutive_buy_threshold = spy_consecutive_buy_threshold
        self.spy_consecutive_days = spy_consecutive_days
        self.spy_max_off_new_high_pct = spy_max_off_new_high_pct

    def run_strategy(self, indicators, sd: dt.datetime, ed: dt.datetime):
        """
        Execute FFTY-SPY strategy.

        Args:
            indicators: List containing [ffty_data, spy_data] StockData objects
            sd (dt.datetime): Strategy start date
            ed (dt.datetime): Strategy end date
        """
        ffty = indicators[0]
        spy = indicators[1]

        ## FFTY signals
        ffty.get_sma('Close', 200, 'Close-SMA200')
        ffty_signals_df = ffty.data[['Close', 'Close-SMA200']].copy()

        # Rename columns
        ffty_signals_df.rename(columns={'Close': 'FFTY', 'Close-SMA200': 'FFTY-SMA200'}, inplace=True)
        ffty_signals_df['FFTY_to_SMA200'] = ffty_signals_df['FFTY'] / ffty_signals_df['FFTY-SMA200']
        ffty_signals_df['FFTY_Signal'] = np.where(
            (ffty_signals_df['FFTY_to_SMA200'] > self.ffty_buy_threshold) &
            (ffty_signals_df['FFTY_to_SMA200'].shift(1) < self.ffty_buy_threshold), 1.0, 0.0)
        ffty_signals_df['FFTY_Signal'] = np.where(
            (ffty_signals_df['FFTY_to_SMA200'] < self.ffty_sell_threshold) &
            (ffty_signals_df['FFTY_to_SMA200'].shift(1) > self.ffty_sell_threshold),
            -1.0, ffty_signals_df['FFTY_Signal'])

        ## SPY signals
        spy.get_sma('Close', 200, 'Close-SMA200')
        spy.data['SPY-to-SMA200'] = (spy.data['Close'] - spy.data['Close-SMA200']) / spy.data['Close-SMA200']
        spy.data['new_high'] = spy.data['Close'].cummax()
        spy.data['off_new_high'] = spy.data['Close'] / spy.data['new_high'] - 1
        # The down is negative, so we need to take the min
        spy.data['max_off_new_high'] = spy.data['off_new_high'].rolling(252, min_periods=1).min()
        spy.data['SPY-to-SMA200_prev'] = spy.data['SPY-to-SMA200'].shift(self.spy_consecutive_days)

        spy_signals_df = spy.data[[
            'Close', 'Close-SMA200', 'new_high', 'off_new_high',
            'max_off_new_high', 'SPY-to-SMA200', 'SPY-to-SMA200_prev']].copy()
        spy_signals_df.rename(columns={'Close': 'SPY', 'Close-SMA200': 'SPY-SMA200'}, inplace=True)

        # Buy rule: two consecutive weeks of above 200 AND previously SPY DOWN 20%
        spy_signals_df['spy-ready-to-buy'] = spy_signals_df['max_off_new_high'] < self.spy_max_off_new_high_pct
        spy_buy_rule = (spy_signals_df['spy-ready-to-buy'] &
                       (spy_signals_df['SPY-to-SMA200'] + 1 > self.spy_consecutive_buy_threshold) &
                       (spy_signals_df['SPY-to-SMA200_prev'].shift(1) + 1 < self.spy_consecutive_buy_threshold))
        # Fill in the first 10 days with 0
        spy_buy_rule.iloc[0:self.spy_consecutive_days] = False
        spy_signals_df['SPY_Signal'] = np.where(spy_buy_rule, 1, 0)

        signals_df = ffty_signals_df[['FFTY_Signal', 'FFTY_to_SMA200']].merge(
            spy_signals_df[['SPY_Signal', 'spy-ready-to-buy']],
            how='left', left_index=True, right_index=True).sort_index()
        signals_df['Signal'] = np.where(
            (signals_df['SPY_Signal'] > 0) | (signals_df['FFTY_Signal'] > 0), 1,
            np.where(signals_df['FFTY_Signal'] < 0, -1, 0))

        self.joined_data = ffty_signals_df.merge(
            spy_signals_df, how='left', left_index=True, right_index=True).sort_index()
        self.joined_data.merge(signals_df, how='left', left_index=True, right_index=True).sort_index()

        self.trades = signals_df[['Signal']]


class fftynaa200r_stg(Strategy):
    """
    FFTY-NAA200R Combined Strategy.

    This strategy uses FFTY < 200MA as sell rules and NAA200R as buy rules,
    combining trend following with market breadth indicators.
    """

    def __init__(self, name: str = 'fftynaa200r_stg', stop_loss: float = 0, take_profit: float = 0,
                 ffty_ma_window: int = 200, ffty_sell_threshold: float = 1, ffty_buy_threshold: float = 1,
                 naa200r_buy_threshold: float = 15, naa200r_sell_threshold: float = 30):
        """
        Initialize FFTY-NAA200R strategy.

        Args:
            name (str): Strategy name
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            ffty_ma_window (int): FFTY moving average window
            ffty_sell_threshold (float): FFTY sell threshold
            ffty_buy_threshold (float): FFTY buy threshold
            naa200r_buy_threshold (float): NAA200R buy threshold
            naa200r_sell_threshold (float): NAA200R sell threshold
        """
        super().__init__(name, stop_loss, take_profit)
        self.ffty_ma_window = ffty_ma_window
        self.ffty_sell_threshold = ffty_sell_threshold
        self.ffty_buy_threshold = ffty_buy_threshold
        self.naa200r_buy_threshold = naa200r_buy_threshold
        self.naa200r_sell_threshold = naa200r_sell_threshold

    def run_strategy(self, indicators, start_date: dt.datetime, end_date: dt.datetime, verbose: bool = False):
        """
        Execute FFTY-NAA200R strategy.

        Args:
            indicators: List containing [ffty_data, naa200r_data] StockData objects
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
            verbose (bool): Whether to print debug information
        """
        ffty = indicators[0]
        naa200r = indicators[1]

        ## FFTY signals
        # Handle multi-level columns from yfinance data
        if isinstance(ffty.data.columns, pd.MultiIndex):
            close_col = ('Close', ffty.ticker)
            ffty.data[('Close-SMA200', '')] = ffty.data[close_col].rolling(
                window=self.ffty_ma_window).mean()
            ffty_signals_df = ffty.data[[close_col, ('Close-SMA200', '')]].copy()
            ffty_signals_df.columns = ['FFTY', 'FFTY-SMA200']
        else:
            get_sma(ffty.data, 'Close', 'Close-SMA200', self.ffty_ma_window)
            ffty_signals_df = ffty.data[['Close', 'Close-SMA200']].copy()
            ffty_signals_df.rename(columns={'Close': 'FFTY', 'Close-SMA200': 'FFTY-SMA200'}, inplace=True)

        ffty_signals_df['FFTY_Signal'] = np.where(
            (ffty_signals_df['FFTY'] >= ffty_signals_df['FFTY-SMA200'] * self.ffty_buy_threshold) &
            (ffty_signals_df['FFTY'].shift(1) < ffty_signals_df['FFTY-SMA200'].shift(1)), 1.0, 0.0)
        ffty_signals_df['FFTY_Signal'] = np.where(
            (ffty_signals_df['FFTY'] < ffty_signals_df['FFTY-SMA200'] * self.ffty_sell_threshold) &
            (ffty_signals_df['FFTY'].shift(1) > ffty_signals_df['FFTY-SMA200'].shift(1)),
            -1, ffty_signals_df['FFTY_Signal'])
        ffty_signals_df['FFTY_TO_SMA200'] = (
            (ffty_signals_df['FFTY'] - ffty_signals_df['FFTY-SMA200']) / ffty_signals_df['FFTY-SMA200']
        )

        ## NAA200R as buy and sell signals
        get_sma(naa200r.data, 'Close', 'Close-SMA20', 20)
        naa200r_signals_df = naa200r.data[['Close', 'Close-SMA20']].copy()
        naa200r_signals_df.rename(columns={'Close': 'NAA200R', 'Close-SMA20': 'NAA200R-SMA20'}, inplace=True)
        naa200r_signals_df['NAA200R_Signal'] = np.where(
            (naa200r_signals_df['NAA200R'] > self.naa200r_buy_threshold) &
            (naa200r_signals_df['NAA200R'] > naa200r_signals_df['NAA200R-SMA20']), 1.0, 0.0)
        naa200r_signals_df['NAA200R_Signal'] = np.where(
            (naa200r_signals_df['NAA200R'] < self.naa200r_sell_threshold) &
            (naa200r_signals_df['NAA200R'] < naa200r_signals_df['NAA200R-SMA20']),
            -1, naa200r_signals_df['NAA200R_Signal'])

        self.joined_data = ffty_signals_df.merge(
            naa200r_signals_df, how='left', left_index=True, right_index=True).sort_index()

        self.joined_data['Signal'] = ffty_signals_df['FFTY_Signal']
        self.joined_data['Signal'] = np.where(
            self.joined_data['FFTY_TO_SMA200'] <= 0,
            self.joined_data['NAA200R_Signal'],
            self.joined_data['Signal'])

        self.trades = self.joined_data[['Signal']]


class CustomizedStrategy(Strategy):
    """
    Customized Strategy using external signals.

    This strategy allows users to provide their own signal DataFrame
    and combines it with stock price data for backtesting.
    """

    def __init__(self, signals_df: pd.DataFrame, name: str = 'Customized',
                 stop_loss: float = 0, take_profit: float = 0):
        """
        Initialize Customized strategy.

        Args:
            signals_df (pd.DataFrame): DataFrame containing trading signals
            name (str): Strategy name
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
        """
        super().__init__(name, stop_loss, take_profit)
        self.signals_df = signals_df

    def run_strategy(self, stock_data: StockData, start_date: dt.datetime, end_date: dt.datetime):
        """
        Execute customized strategy.

        Args:
            stock_data (StockData): Stock price data
            start_date (dt.datetime): Strategy start date
            end_date (dt.datetime): Strategy end date
        """
        # Make sure dates and rows are aligned in signal data and stock data
        # Include only dates where both signal data and stock data are available
        self.joined_data = stock_data.data[['Close']].rename(
            columns={'Close': 'Price'}).merge(
            self.signals_df[['Signal']],
            how='inner', left_index=True, right_index=True).sort_index()

        # Filter for date range if specified
        if start_date:
            self.joined_data = self.joined_data.loc[
                (self.joined_data.index >= start_date) &
                (self.joined_data.index <= end_date)].copy()

        self.trades = self.joined_data[['Signal']]