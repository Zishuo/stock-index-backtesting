"""
Data handling and processing for financial backtesting.

This module provides the StockData class for fetching, loading, and processing
financial market data from various sources including Yahoo Finance, CSV files,
and SQLite databases.
"""

import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import sqlite3 as sql


class StockData(object):
    """
    A class for handling stock market data from various sources.

    This class provides methods to fetch data from Yahoo Finance, load from CSV files,
    or retrieve from SQLite databases. It also includes methods for calculating
    technical indicators and processing the data for backtesting.

    Attributes:
        ticker (str): Stock ticker symbol
        data (pd.DataFrame): Stock price and volume data
        start_date (dt.datetime): Start date of the data
        end_date (dt.datetime): End date of the data
        index (pd.DatetimeIndex): DateTime index of the data
    """

    def __init__(self, ticker: str):
        """
        Initialize StockData with a ticker symbol.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        """
        self.ticker = ticker
        self.data = None
        self.start_date = None
        self.end_date = None

    def get_data_from_yfinance(self, ticker: str, start_date: dt.datetime,
                             end_date: dt.datetime, interval: str = '1d'):
        """
        Fetch stock data from Yahoo Finance.

        Args:
            ticker (str): Stock ticker symbol
            start_date (dt.datetime): Start date for data retrieval
            end_date (dt.datetime): End date for data retrieval
            interval (str): Data interval ('1d', '1wk', '1mo', etc.)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(self.ticker, start=self.start_date,
                               end=self.end_date, interval=interval)

        # Handle MultiIndex columns that yfinance sometimes returns
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.droplevel(1)

        self.start_date = self.data.index[0]
        self.end_date = self.data.index[-1]

        # BUG!!! when getting 1wk data, the index is on Monday, but the data is on Friday
        if interval == '1wk':
            self.data.index = self.data.index + dt.timedelta(days=4)

        self.data['Weekday'] = self.data.index.weekday
        self.index = pd.to_datetime(self.data.index)

    def get_data_history_from_yfinance(self, ticker: str, period: str, interval: str,
                                     start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance using period parameter.

        Args:
            ticker (str): Stock ticker symbol
            period (str): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str): Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            start_date: Download start date (YYYY-MM-DD) or datetime
            end_date: Download end date (YYYY-MM-DD) or datetime

        Note:
            Either use period parameter or use start and end dates.
            Intraday data cannot extend last 60 days.
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

        self.data = yf.Ticker(self.ticker).history(
            period=self.period, interval=self.interval,
            start=self.start_date, end=self.end_date,
            prepost=False, actions=True, auto_adjust=True,
            back_adjust=False, proxy=None, rounding=False, timeout=None
        )

        if interval == '1wk':
            self.data.index = self.data.index - dt.timedelta(days=2)

        self.data['Weekday'] = self.data.index.weekday
        self.data.index = pd.to_datetime(self.data.index)

    def get_data_from_csv(self, path: str):
        """
        Load stock data from a CSV file.

        Args:
            path (str): Path to the CSV file

        Note:
            CSV file should have a 'Date' column and standard OHLCV columns.
        """
        self.data = pd.read_csv(path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index(['Date'], inplace=True)
        self.data.sort_index(inplace=True)

    def get_data_from_db(self, db_path: str = 'data/stock_data.db', limit: int = 100000):
        """
        Load stock data from SQLite database.

        Args:
            db_path (str): Path to the SQLite database file
            limit (int): Maximum number of records to retrieve
        """
        conn = sql.connect(db_path)
        print("SELECT * FROM stock_history Where Ticker='{}' limit {}".format(self.ticker, limit))
        self.data = pd.read_sql_query(
            "SELECT * FROM stock_history Where Ticker='{}' limit {}".format(self.ticker, limit),
            conn
        )
        self.data.set_index(['Date'], inplace=True)
        conn.close()

    def get_indicators(self, column='Close', ma_windows=[5, 10, 20, 50, 200],
                      below_thresholds=[30], above_thresholds=[15]):
        """
        Calculate moving averages and price-to-MA ratios.

        Args:
            column (str): Column name to calculate indicators for
            ma_windows (list): List of moving average window sizes
            below_thresholds (list): Threshold values for below indicators
            above_thresholds (list): Threshold values for above indicators
        """
        for ma_window in ma_windows:
            ma_col = 'MA{}'.format(ma_window)
            self.data[ma_col] = self.data[column].rolling(
                window=ma_window, min_periods=1).mean()
            self.data['price_to_MA{}'.format(ma_window)] = (
                self.data[column] / self.data[ma_col]
            ).squeeze()

    def get_thresholds(self, column='Close', ma_windows=[5, 10, 20, 50, 200],
                      below_thresholds=[30], above_thresholds=[15]):
        """
        Calculate threshold indicators.

        Args:
            column (str): Column name to calculate thresholds for
            ma_windows (list): List of moving average window sizes (unused in current implementation)
            below_thresholds (list): Threshold values for below indicators
            above_thresholds (list): Threshold values for above indicators
        """
        for below_threshold in below_thresholds:
            self.data['below{}'.format(below_threshold)] = np.where(
                self.data[column] < below_threshold, 1, 0)
        for above_threshold in above_thresholds:
            self.data['above{}'.format(above_threshold)] = np.where(
                self.data[column] > above_threshold, 1, 0)

    def get_sma(self, input_column='Close', sma_window=21, output_column='SMA'):
        """
        Calculate Simple Moving Average.

        Args:
            input_column (str): Input column name
            sma_window (int): Moving average window size
            output_column (str): Output column name
        """
        self.data[output_column] = self.data[input_column].rolling(
            window=sma_window, min_periods=1).mean()

    def get_ema(self, input_column='Close', ema_window=21, output_column='EMA'):
        """
        Calculate Exponential Moving Average.

        Args:
            input_column (str): Input column name
            ema_window (int): EMA window size
            output_column (str): Output column name
        """
        self.data[output_column] = self.data[input_column].ewm(
            span=ema_window, adjust=False).mean()

    def get_k(self, input_column='Close', k_window=14, output_column='K'):
        """
        Calculate Stochastic %K.

        Args:
            input_column (str): Input column name (typically 'Close')
            k_window (int): Window size for %K calculation
            output_column (str): Output column name

        Note:
            %K = (Close - 14Days low) / (14Days high - 14Days low) * 100
        """
        low_14 = self.data[input_column].rolling(window=k_window, min_periods=1).min()
        high_14 = self.data[input_column].rolling(window=k_window, min_periods=1).max()
        self.data[output_column] = (
            (self.data[input_column] - low_14) / (high_14 - low_14) * 100
        )

    def get_stochastic(self, input_column='Close', k_window=14, fk_window=5, fd_window=5):
        """
        Calculate Stochastic Oscillator (%K and %D).

        Args:
            input_column (str): Input column name
            k_window (int): Window size for fast %K calculation
            fk_window (int): Window size for %K smoothing
            fd_window (int): Window size for %D calculation

        Note:
            Fast %K = basic %K calculation
            %K = EMA of Fast %K
            %D = EMA of %K
        """
        # Fast Stochastic Oscillator:
        # Fast %K = %K basic calculation
        # Fast %D = 5-period SMA of Fast %K
        self.get_k(input_column, k_window, '%K-FAST')
        self.get_ema('%K-FAST', fk_window, '%K')

        # Full Stochastic Oscillator:
        # Full %K = Fast %D
        # Full %D = 5-period EMA of Full %K
        self.get_ema('%K', fd_window, '%D')

    def get_close_range(self, high='High', low='Low', close='Close', range='Range'):
        """
        Calculate close range indicator.

        Args:
            high (str): High price column name
            low (str): Low price column name
            close (str): Close price column name
            range (str): Output column name
        """
        self.data[range] = (
            (self.data[close] - self.data[low]) / (self.data[high] - self.data[low])
        )

    def get_to_percentage(self, input='Close', target='SMA', output='Output'):
        """
        Calculate percentage difference between input and target.

        Args:
            input (str): Input column name
            target (str): Target column name
            output (str): Output column name
        """
        self.data[output] = (
            (self.data[input] - self.data[target]) / self.data[target]
        )