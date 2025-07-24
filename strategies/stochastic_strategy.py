"""
Stochastic Oscillator based trading strategy.

This module implements a complex stochastic oscillator strategy that combines
daily and weekly stochastic indicators with various filtering rules.

Reference: https://www.investopedia.com/terms/s/stochasticoscillator.asp
"""

import datetime as dt
import numpy as np
import pandas as pd

from core.strategy_base import Strategy
from core.indicators import get_stochastic, get_sma


class StochasticCross(Strategy):
    """
    Stochastic Oscillator Crossover Strategy.

    This strategy uses both daily and weekly stochastic oscillators to generate
    trading signals with multiple filtering conditions.

    Buy Rules:
    - Daily %K above %D
    - Weekly %K Going Up or Flat
    - Daily %K above oversold threshold (e.g., 20)

    Sell Rules:
    - Weekly %K under %D
    - Weekly %K under overbought threshold (e.g., 80)

    Stop Loss Rule:
    - Asset drops 10% or more in a day (signal = -2)
    """

    def __init__(self, name: str = 'Stochastic', k_window: int = 14, full_k_window: int = 5,
                 full_d_window: int = 5, overbought: int = 80, oversold: int = 10,
                 stop_loss: float = 0, take_profit: float = 0, var: int = 0, ma_notrade: int = 0):
        """
        Initialize Stochastic Crossover strategy.

        Args:
            name (str): Strategy name
            k_window (int): Window for stochastic %K calculation
            full_k_window (int): Window for %K smoothing
            full_d_window (int): Window for %D calculation
            overbought (int): Overbought threshold (typically 80)
            oversold (int): Oversold threshold (typically 20)
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            var (int): Strategy variation (0=original, 1=daily only, 3=experimental)
            ma_notrade (int): Moving average filter window (0=disabled)
        """
        name_cb = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            name, k_window, full_k_window, full_d_window, overbought,
            oversold, var, ma_notrade, stop_loss)
        super().__init__(name_cb, stop_loss, take_profit)

        self.k_window = k_window
        self.full_k_window = full_k_window
        self.full_d_window = full_d_window
        self.oversold = oversold
        self.overbought = overbought
        self.var = var
        self.ma_notrade = ma_notrade

    def run_strategy(self, indicators, sd: dt.datetime, ed: dt.datetime):
        """
        Execute stochastic crossover strategy.

        Args:
            indicators: List containing [daily_data, weekly_data] StockData objects
            sd (dt.datetime): Strategy start date
            ed (dt.datetime): Strategy end date
        """
        d_df = indicators[0].data.copy()
        w_df = indicators[1].data.copy()

        # Daily Stochastic Oscillator
        get_stochastic(d_df, 'Close', 'FastD%K', 'D%K', 'D%D',
                      self.k_window, self.full_k_window, self.full_d_window)
        d_df.rename(columns={'Close': 'DClose'}, inplace=True)

        # Weekly Stochastic Oscillator
        get_stochastic(w_df, 'Close', 'FastW%K', 'W%K', 'W%D',
                      self.k_window, self.full_k_window, self.full_d_window)
        w_df.rename(columns={'Close': 'WClose'}, inplace=True)

        # Determine if Weekly %K is going up or flat
        w_df['W%K-UP'] = np.where(w_df['W%K'] > w_df['W%K'].shift(1), 1, 0)
        w_df['13MIN'] = w_df['WClose'].rolling(window=13).min()
        w_df['13MAX'] = w_df['WClose'].rolling(window=13).max()

        # Join daily and weekly data
        self.joined_data = d_df[['DClose', 'D%K', 'D%D']].merge(
            w_df[['WClose', 'W%K', 'W%D', 'W%K-UP', 'FastW%K', '13MIN', '13MAX', 'Weekday']],
            how='left', left_index=True, right_index=True)

        # Fill in the NA values with forward fill
        self.joined_data['W%K'] = self.joined_data['W%K'].ffill()
        self.joined_data['W%K-UP'] = self.joined_data['W%K-UP'].ffill()
        self.joined_data['W%D'] = self.joined_data['W%D'].ffill()
        self.joined_data['WClose'] = self.joined_data['WClose'].ffill()
        self.joined_data['13MIN'] = self.joined_data['13MIN'].ffill()
        self.joined_data['13MAX'] = self.joined_data['13MAX'].ffill()

        # Intra-week weekly %K calculation
        # Use 13 weeks WClose and today's DClose to calculate 14 weeks intra-week weekly FAST-WD%K
        self.joined_data['14MIN'] = np.where(
            self.joined_data['DClose'] < self.joined_data['13MIN'],
            self.joined_data['DClose'], self.joined_data['13MIN'])
        self.joined_data['14MAX'] = np.where(
            self.joined_data['DClose'] > self.joined_data['13MAX'],
            self.joined_data['DClose'], self.joined_data['13MAX'])
        self.joined_data['FAST-WD%K'] = (
            (self.joined_data['DClose'] - self.joined_data['14MIN']) /
            (self.joined_data['14MAX'] - self.joined_data['14MIN']) * 100
        )

        self.joined_data['BSignal'] = 0
        self.joined_data['SSignal'] = 0

        # Apply different strategy variations
        if self.var == 0:
            # Original strategy: cross event + status
            # Sell rule: Weekly %K under %D (crossover) & Weekly %K under overbought
            self.joined_data['SSignal'] = np.where(
                (self.joined_data['W%K'] < self.joined_data['W%D']) &
                (self.joined_data['W%K'].shift(1) > self.joined_data['W%D'].shift(1)) &
                (self.joined_data['W%K'] < self.overbought),
                -1.0, self.joined_data['SSignal'])

            # Buy rule: Daily %K above %D (crossover) & Weekly %K Going Up & Daily %K above oversold
            self.joined_data['BSignal'] = np.where(
                (self.joined_data['D%K'] > self.joined_data['D%D']) &
                (self.joined_data['D%K'].shift(1) < self.joined_data['D%D'].shift(1)) &
                (self.joined_data['W%K-UP'] == 1.0) &
                (self.joined_data['D%K'] > self.oversold),
                1.0, self.joined_data['BSignal'])

        elif self.var == 1:
            print(self.var)
            # Daily only strategy
            # Buy rule: Daily %K above %D & Daily %K above oversold
            self.joined_data['BSignal'] = np.where(
                (self.joined_data['D%K'] > self.joined_data['D%D']) &
                (self.joined_data['D%K'] > self.oversold), 1.0, 0.0)
            # Sell rule: Daily %K under %D & Daily %K under overbought
            self.joined_data['SSignal'] = np.where(
                (self.joined_data['D%K'] < self.joined_data['D%D']) &
                (self.joined_data['D%K'] < self.overbought), -1.0, 0.0)

        elif self.var == 3:
            # Experimental variation (placeholder)
            pass

        # Apply moving average filter if enabled
        if self.ma_notrade != 0:
            # Don't trade when price is below moving average
            get_sma(self.joined_data, 'DClose', 'DClose-SMA{}'.format(self.ma_notrade),
                   self.ma_notrade, 1)
            # Wipe buy signal when under MA
            self.joined_data['BSignal'] = np.where(
                (self.joined_data['DClose'] < self.joined_data['DClose-SMA{}'.format(self.ma_notrade)]),
                0, self.joined_data['BSignal'])
            # Set sell signal to -1.5 when under MA
            self.joined_data['SSignal'] = np.where(
                (self.joined_data['DClose'] < self.joined_data['DClose-SMA{}'.format(self.ma_notrade)]),
                -1.5, self.joined_data['SSignal'])

        # Stop loss rule: Asset drops 10% or more in a day
        self.joined_data['SSignal'] = np.where(
            (self.joined_data['DClose'] < self.joined_data['DClose'].shift(1) * 0.9),
            -2.0, self.joined_data['SSignal'])

        # Filter for date range
        self.joined_data = self.joined_data.loc[sd:ed]
        self.trades = self.joined_data[['BSignal', 'SSignal']]