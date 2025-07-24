"""
Portfolio management and backtesting implementation.

This module provides classes for managing portfolios and running backtests,
including performance analysis, trade recording, and tax calculations.
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

from .strategy_base import Strategy
from .data_handler import StockData


class Portfolio(metaclass=ABCMeta):
    """
    Abstract base class for portfolio management.

    This class defines the interface for portfolio management systems,
    including trade execution, performance tracking, and analysis.

    Attributes:
        principal (float): Initial investment amount
        trade_size (float): Size of each trade
        pyramiding (int): Maximum number of concurrent positions
        margin (float): Margin requirement
        balance (pd.DataFrame): Portfolio balance over time
        trade_records (pd.DataFrame): Record of all trades
        joined_data (pd.DataFrame): Combined strategy and price data
    """

    def __init__(self, principal: float, trade_size: float, pyramiding: int, margin: float):
        """
        Initialize portfolio with basic parameters.

        Args:
            principal (float): Initial investment amount
            trade_size (float): Size of each trade (0-1 for percentage, >1 for fixed amount)
            pyramiding (int): Maximum number of concurrent positions
            margin (float): Margin requirement
        """
        self.principal = principal
        self.pyramiding = pyramiding
        self.trade_size = trade_size
        self.margin = margin

        # Real account balance should be Cash + Stock - Margin = Total
        self.balance = pd.DataFrame(columns=['Date', 'Cash', 'Stock', 'Total', 'Margin'])
        self.balance.set_index('Date', inplace=True)
        self.joined_data = None
        self.name = None

        # Initialize trade records with proper data types
        self.trade_records = pd.DataFrame(columns=[
            'Buy Date', 'Sell Date', 'Ticker', 'Quant', 'Buy Price', 'Sell Price',
            'Profit', 'Profit %', 'HoldingDays', 'LongTermProfit', 'ShortTermProfit',
            'TaxCollectYear', 'TaxCollected'])

        # Set proper dtypes to avoid warnings
        self.trade_records = self.trade_records.astype({
            'Buy Date': 'datetime64[ns]',
            'Sell Date': 'datetime64[ns]',
            'Ticker': 'object',
            'Quant': 'float64',
            'Buy Price': 'float64',
            'Sell Price': 'float64',
            'Profit': 'float64',
            'Profit %': 'float64',
            'HoldingDays': 'float64',
            'LongTermProfit': 'float64',
            'ShortTermProfit': 'float64',
            'TaxCollectYear': 'float64',
            'TaxCollected': 'float64'
        })

    @abstractmethod
    def run_backtest(self, strategy: Strategy, stock_data: StockData):
        """
        Execute backtest with given strategy and data.

        Args:
            strategy (Strategy): Trading strategy to test
            stock_data (StockData): Historical price data
        """
        pass

    @abstractmethod
    def performance_summary(self, verbose: bool = True):
        """
        Calculate and display performance metrics.

        Args:
            verbose (bool): Whether to print detailed summary

        Returns:
            pd.DataFrame: Summary statistics
        """
        portValue = self.balance[['Total']]

        # Cumulative return
        self.cumulative_return = portValue.iloc[-1] / portValue.iloc[0] - 1

        # Max drawdown
        drawdown_window = 252
        rolling_max = portValue.rolling(drawdown_window, min_periods=1).max()
        daily_drawdown = portValue / rolling_max - 1.0
        max_daily_drawdown = daily_drawdown.rolling(drawdown_window, min_periods=1).min()
        self.max_drawdown = max_daily_drawdown.min()

        # Daily return and sharpe ratio
        daily_return = (portValue / portValue.shift(1) - 1)[1:]
        self.avg_return = daily_return.mean()
        self.std_return = daily_return.std()
        self.sharp_ratio = self.avg_return / self.std_return

        # Trading date calculations
        start_date = self.balance.index[0]
        end_date = self.balance.index[-1]
        trading_dates = end_date - start_date

        # Annual return
        if round((trading_dates.days / 365)) == 0:
            self.annual_return = self.cumulative_return.values[0] + 1
        else:
            self.annual_return = np.power(
                self.cumulative_return.values[0] + 1, 1 / round((trading_dates.days / 365)))

        # Number of trades
        self.num_trades = self.balance.Stock.nunique()

        # Trade record analysis
        # Batting average
        gain = len(self.trade_records.loc[self.trade_records['Profit %'] > 0])
        loss = len(self.trade_records.loc[self.trade_records['Profit %'] <= 0])
        bat_avg = gain / (gain + loss) if (gain + loss) > 0 else 0

        gain_avg = self.trade_records.loc[self.trade_records['Profit %'] > 0, 'Profit %'].mean()
        loss_avg = self.trade_records.loc[self.trade_records['Profit %'] <= 0, 'Profit %'].mean()
        gain_std = self.trade_records.loc[self.trade_records['Profit %'] > 0, 'Profit %'].std()
        loss_std = self.trade_records.loc[self.trade_records['Profit %'] <= 0, 'Profit %'].std()

        if verbose:
            print(f"""
{self.name}:
cumulative return      : {self.cumulative_return.values[0]:.2%}
compound anual return  : {self.annual_return - 1:.4%}
max_drawdown           : {self.max_drawdown.values[0]:.2%}
sharp_ratio            : {self.sharp_ratio.values[0]:.2%}
average of daily return: {self.avg_return.values[0]:.4%}
std of daily return    : {self.std_return.values[0]:.4%}
number of trades       : {self.num_trades},
trading days           : {trading_dates.days},
batting Average        : {bat_avg:.2%}
Gain Average           : {gain_avg:.2%}
Loss Average           : {loss_avg:.2%}
Risk Reward Ratio      : {gain_avg / (-loss_avg) if loss_avg < 0 else 0:.2f}
Gain STD               : {gain_std:.2%}
Loss STD               : {loss_std:.2%}
            """)

        stats_names = [
            'name', 'num_trades', 'cumulative_return', 'annual_return', 'max_drawdown',
            'sharp_ratio', 'avg_daily_return', 'std_daily_return', 'num_trading_days',
            'batting Average', 'Gain Average', 'Loss Average', 'Risk Reward Ratio',
            'Gain STD', 'Loss STD'
        ]

        stats = [
            self.name, self.num_trades, self.cumulative_return.values[0],
            self.annual_return - 1, self.max_drawdown.values[0],
            self.sharp_ratio.values[0], self.avg_return.values[0], self.std_return.values[0],
            trading_dates.days, bat_avg, gain_avg, loss_avg,
            gain_avg / (-loss_avg) if loss_avg < 0 else 0, gain_std, loss_std
        ]

        self.summary_result = pd.DataFrame([stats], columns=stats_names)


class BackTest(Portfolio):
    """
    Concrete implementation of portfolio for backtesting.

    This class provides a complete backtesting framework with trade execution,
    tax calculation, and performance analysis capabilities.
    """

    def __init__(self, principal: float = 1, trade_size: float = 1, pyramiding: int = 1):
        """
        Initialize BackTest portfolio.

        Args:
            principal (float): Initial investment amount
            trade_size (float): Size of each trade
            pyramiding (int): Maximum number of concurrent positions
        """
        super().__init__(principal, trade_size, pyramiding, 0)
        self.pyramiding_count = 0

    def _record_buy(self, ticker: str, date, price: float, quantity: float):
        """
        Record a buy transaction.

        Args:
            ticker (str): Stock ticker symbol
            date: Transaction date
            price (float): Buy price
            quantity (float): Number of shares
        """
        self.trade_records.loc[len(self.trade_records)] = [
            date, np.nan, ticker, quantity, price, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan]

    def _record_sell(self, ticker: str, date, price: float, quantity: float):
        """
        Record a sell transaction.

        Args:
            ticker (str): Stock ticker symbol
            date: Transaction date
            price (float): Sell price
            quantity (float): Number of shares
        """
        # Find the last buy order with empty sell date
        i = self.trade_records[
            (self.trade_records['Ticker'] == ticker) &
            (self.trade_records['Sell Date'].isnull())].index

        if i.empty:
            print(f'No buy order for ticker {ticker} on date {date}')
            return

        self.trade_records.loc[i, 'Sell Date'] = pd.to_datetime(date)
        self.trade_records.loc[i, 'Sell Price'] = price
        self.trade_records.loc[i, 'Profit'] = (
            (price - self.trade_records.loc[i, 'Buy Price'].values[0]) * quantity)
        self.trade_records.loc[i, 'Profit %'] = (
            (price - self.trade_records.loc[i, 'Buy Price'].values[0]) /
            self.trade_records.loc[i, 'Buy Price'].values[0])
        self.trade_records.loc[i, 'HoldingDays'] = (
            pd.to_datetime(self.trade_records.loc[i, 'Sell Date']) -
            self.trade_records.loc[i, 'Buy Date']).dt.days

        # Tax calculations
        self.trade_records.loc[i, 'LongTermProfit'] = np.where(
            (self.trade_records.loc[i, 'Profit'] > 0) &
            (self.trade_records.loc[i, 'HoldingDays'] > 365),
            self.trade_records.loc[i, 'Profit'], 0)
        self.trade_records.loc[i, 'ShortTermProfit'] = np.where(
            (self.trade_records.loc[i, 'Profit'] > 0) &
            (self.trade_records.loc[i, 'HoldingDays'] < 365),
            self.trade_records.loc[i, 'Profit'], 0)
        self.trade_records.loc[i, 'TaxCollectYear'] = (
            pd.to_datetime(self.trade_records.loc[i, 'Sell Date']).dt.year + 1)
        self.trade_records.loc[i, 'TaxCollected'] = 0

    def _copy_balance(self, i, cash: float, stock: float, total: float):
        """
        Copy balance values to the current row.

        Args:
            i: Row index
            cash (float): Cash balance
            stock (float): Stock value
            total (float): Total portfolio value
        """
        self.balance.loc[i[0], 'Cash'] = cash
        self.balance.loc[i[0], 'Stock'] = stock
        self.balance.loc[i[0], 'Total'] = total

    def _collect_tax(self, i, tax_collect_year: int, c_price: float,
                    long_term_tax_rate: float, short_term_tax_rate: float, verbose: bool = False):
        """
        Collect taxes on realized gains.

        Args:
            i: Row index
            tax_collect_year (int): Year to collect taxes for
            c_price (float): Current stock price
            long_term_tax_rate (float): Long-term capital gains tax rate
            short_term_tax_rate (float): Short-term capital gains tax rate
            verbose (bool): Whether to print tax collection details
        """
        # Check if taxes for this year have already been collected
        if self.trade_records.loc[
            self.trade_records['TaxCollectYear'] == tax_collect_year, 'TaxCollected'].max() == 0:

            # Calculate tax to be collected
            long_term_gains = self.trade_records.loc[
                self.trade_records['TaxCollectYear'] == tax_collect_year, 'LongTermProfit'].sum()
            short_term_gains = self.trade_records.loc[
                self.trade_records['TaxCollectYear'] == tax_collect_year, 'ShortTermProfit'].sum()

            tax_to_collect = (long_term_gains * long_term_tax_rate +
                            short_term_gains * short_term_tax_rate)

            # Collect from cash or by selling stocks
            if self.balance.loc[i[0], 'Cash'] >= tax_to_collect:
                self.balance.loc[i[0], 'Cash'] -= tax_to_collect
            else:
                self.balance.loc[i[0], 'Stock'] -= tax_to_collect / c_price

            self.balance.loc[i[0], 'Total'] -= tax_to_collect

            # Mark as collected
            self.trade_records.loc[
                self.trade_records['TaxCollectYear'] == tax_collect_year, 'TaxCollected'] = tax_to_collect

            if verbose:
                print(f'{tax_to_collect} Tax collected on {i[0]}')

    def run_backtest(self, strategy: Strategy, stock_data: StockData, start_date, end_date,
                    weekly_buy: bool = False, weekly_sell: bool = False,
                    short_term_tax_rate: float = 0, long_term_tax_rate: float = 0, verbose: bool = False):
        """
        Execute backtest with given strategy and parameters.

        Args:
            strategy (Strategy): Trading strategy to test
            stock_data (StockData): Historical price data
            start_date: Backtest start date
            end_date: Backtest end date
            weekly_buy (bool): Whether to restrict buying to weekly intervals
            weekly_sell (bool): Whether to restrict selling to weekly intervals
            short_term_tax_rate (float): Short-term capital gains tax rate
            long_term_tax_rate (float): Long-term capital gains tax rate
            verbose (bool): Whether to print detailed execution log
        """
        self.verbose = verbose
        self.name = strategy.name
        self.ticker = stock_data.ticker
        sd = max(start_date, stock_data.data.index.min())
        ed = min(end_date, stock_data.data.index.max())

        # Handle multi-level columns from yfinance data
        if isinstance(stock_data.data.columns, pd.MultiIndex):
            close_col = ('Close', stock_data.ticker)
            weekday_col = ('Weekday', '')
            self.balance = stock_data.data[[close_col, weekday_col]].loc[sd:ed].copy()
            # Flatten the multi-level columns
            self.balance.columns = [stock_data.ticker, 'Weekday']
        else:
            self.balance = stock_data.data[['Close', 'Weekday']].loc[sd:ed].copy()
            self.balance.rename(columns={'Close': stock_data.ticker}, inplace=True)

        # Merge the strategy signal to the balance
        if 'BSignal' in strategy.trades.columns and 'SSignal' in strategy.trades.columns:
            self.balance = self.balance.merge(
                strategy.trades[['BSignal', 'SSignal']], how='left', left_index=True, right_index=True)
            self.balance['Signal'] = 0
        elif 'Signal' in strategy.trades.columns:
            self.balance = self.balance.merge(
                strategy.trades[['Signal']], how='left', left_index=True, right_index=True)
            self.balance['BSignal'] = 0
            self.balance['SSignal'] = 0

        # Fill Signal NaN with 0
        self.balance['Signal'].fillna(0, inplace=True)

        # Initialize balance columns
        self.balance['Cash'] = 0.0
        self.balance['Stock'] = 0.0
        self.balance['Total'] = 0.0
        self.balance['Margin'] = 0.0
        self.balance['Trade'] = 0.0
        self.balance['Buy Price'] = 0.0
        self.balance['Profit'] = 0.0

        self.balance.loc[self.balance.index[0], 'Cash'] = self.principal
        self.balance.loc[self.balance.index[0], 'Total'] = self.principal

        p_cash = self.principal
        p_stock = 0
        p_price = self.balance.iloc[0][stock_data.ticker]

        # Iterate through the balance
        for i in self.balance.iterrows():
            c_price = i[1][stock_data.ticker]

            # Buy logic
            if ((i[1]['Signal'] > 0 and self.pyramiding_count < self.pyramiding) or
                (i[1]['BSignal'] > 0 and self.pyramiding_count < self.pyramiding and p_stock == 0)):

                self.pyramiding_count += 1
                self.balance.loc[i[0], 'Stock'] = p_stock + p_cash / c_price
                self.balance.loc[i[0], 'Cash'] = 0
                self.balance.loc[i[0], 'Total'] = (self.balance.loc[i[0], 'Cash'] +
                                                 self.balance.loc[i[0], 'Stock'] * c_price)
                self.balance.loc[i[0], 'Buy Price'] = self.balance.loc[i[0], 'Total']
                self.balance.loc[i[0], 'Trade'] = i[1]['Signal'] + i[1]['BSignal']
                self._record_buy(stock_data.ticker, i[0], c_price, self.balance.loc[i[0], 'Stock'])

                if verbose:
                    print(f'{i[0]} Buy {self.balance.loc[i[0], "Stock"]}')

            # Sell logic
            elif ((i[1]['Signal'] < 0 and self.pyramiding_count > 0) or
                  (i[1]['SSignal'] < 0 and self.pyramiding_count > 0 and p_stock != 0)):

                if i[1]['Signal'] == -2 or i[1]['SSignal'] == -2:
                    # Stop loss sell, cap the loss at yesterday's price -10%
                    sell_price = p_price * 0.9
                else:
                    sell_price = c_price

                self.balance.loc[i[0], 'Stock'] = 0
                self.balance.loc[i[0], 'Cash'] = p_cash + p_stock * sell_price
                self.balance.loc[i[0], 'Total'] = (self.balance.loc[i[0], 'Cash'] +
                                                 self.balance.loc[i[0], 'Stock'] * sell_price)
                self.balance.loc[i[0], 'Trade'] = i[1]['Signal'] + i[1]['SSignal']
                self._record_sell(stock_data.ticker, i[0], sell_price, p_stock)
                self.pyramiding_count -= 1

                if verbose:
                    print(f'{i[0]} Sell {p_stock}')
            else:
                self._copy_balance(i, p_cash, p_stock, p_cash + p_stock * c_price)
                if verbose:
                    print(f'{i[0]} No trading action')

                # Collect tax only when there's no trade on that day
                # Check if it's April and tax rate > 0
                if (i[0].month == 4) and (short_term_tax_rate + long_term_tax_rate > 0):
                    self._collect_tax(i, i[0].year, c_price, long_term_tax_rate,
                                    short_term_tax_rate, verbose)

            p_stock = self.balance.loc[i[0], 'Stock']
            p_cash = self.balance.loc[i[0], 'Cash']
            p_price = c_price

        # Force sell remaining stock at the end
        if self.balance.iloc[-1]['Stock'] > 0:
            final_price = self.balance.iloc[-1][stock_data.ticker]
            self.balance.loc[self.balance.index[-1], 'Total'] = (
                self.balance.iloc[-1]['Stock'] * final_price)
            self.balance.loc[self.balance.index[-1], 'Stock'] = 0
            self.balance.loc[self.balance.index[-1], 'Cash'] = 0
            self._record_sell(stock_data.ticker, self.balance.index[-1],
                            final_price, self.balance.iloc[-1]['Stock'])

        # Settle final year taxes
        for i in self.balance.iloc[[-1]].iterrows():
            self._collect_tax(i, i[0].year + 1, c_price, long_term_tax_rate,
                            short_term_tax_rate, verbose)

        # Calculate profit tracking
        self.balance['Buy Price'] = self.balance.loc[
            (self.balance['Stock'] != 0) | (self.balance['Trade'] != 0), 'Buy Price'
        ].replace(0, np.nan).ffill()
        self.balance['Profit'] = (
            (self.balance['Total'] - self.balance['Buy Price']) / self.balance['Buy Price']
        )

        # Join with strategy data if available
        if strategy.joined_data is not None:
            self.joined_data = self.balance.merge(
                strategy.joined_data, how='left', left_index=True, right_index=True)

    def plot_records(self):
        """Plot trade records as a bar chart."""
        plt.figure(figsize=(16, 4))
        plt.bar(self.trade_records.index, self.trade_records['Profit %'], label=self.name)
        plt.title(f'{self.name} Trade Records on {self.ticker}')
        plt.legend()

    def plot_balance(self):
        """Plot portfolio balance over time."""
        plt.figure(figsize=(16, 4))
        plt.plot(self.balance.index, self.balance['Total'], label=self.name)
        plt.title(f'{self.name} Portfolio Balance on {self.ticker}')
        plt.legend()

    def plot_joined_data(self, indicator_column: list, start_date, end_date,
                        ydash_low=None, ydash_high=None):
        """
        Plot joined data with indicators and trade signals.

        Args:
            indicator_column (list): List of indicator column names to plot
            start_date: Start date for plotting
            end_date: End date for plotting
            ydash_low: Lower horizontal line value
            ydash_high: Upper horizontal line value
        """
        plt.figure(figsize=(16, 3))

        # Plot indicators
        for i in indicator_column:
            plt.plot(self.joined_data.loc[start_date:end_date][i], label=i)

        # Plot trade signals
        for idx, row in self.joined_data.iterrows():
            if idx < start_date or idx > end_date:
                continue
            if row['Trade'] < 0:
                plt.axvline(x=idx, color='red', linestyle='dashed',
                          linewidth=abs(row['Trade']))
            if row['Trade'] > 0:
                plt.axvline(x=idx, color='green', linestyle='dashed',
                          linewidth=abs(row['Trade']))

        if ydash_low is not None:
            plt.axhline(y=ydash_low, color='black', linestyle='solid')
        if ydash_high is not None:
            plt.axhline(y=ydash_high, color='black', linestyle='solid')

        plt.title(f'{self.name} Analysis on {self.ticker}')
        plt.legend()

    def performance_summary(self, v: bool = True, verbose: bool = None):
        """
        Calculate and display performance summary.

        Args:
            v (bool): Whether to print verbose output (backward compatibility)
            verbose (bool): Whether to print verbose output (new parameter name)

        Returns:
            Performance summary result
        """
        # Handle both parameter names for backward compatibility
        if verbose is not None:
            v = verbose
        return super().performance_summary(verbose=v)