"""
Technical indicators for financial analysis.

This module provides various technical analysis indicators including
moving averages, stochastic oscillators, and other commonly used
financial market indicators.
"""

import pandas as pd


def get_sma(df: pd.DataFrame, in_c: str, out_c: str, window: int, min_periods: int = 1):
    """
    Calculate Simple Moving Average.

    Args:
        df (pd.DataFrame): Input DataFrame
        in_c (str): Input column name
        out_c (str): Output column name
        window (int): Moving average window size
        min_periods (int): Minimum number of observations required
    """
    df[out_c] = df[in_c].rolling(window=window, min_periods=min_periods).mean()


def get_ema(df: pd.DataFrame, in_c: str, out_c: str, window: int, min_periods: int = 1):
    """
    Calculate Exponential Moving Average.

    Args:
        df (pd.DataFrame): Input DataFrame
        in_c (str): Input column name
        out_c (str): Output column name
        window (int): EMA span
        min_periods (int): Minimum number of observations required
    """
    df[out_c] = df[in_c].ewm(span=window, min_periods=min_periods).mean()


def get_stochastic(df: pd.DataFrame, in_c: str, out_fastk: str, out_fk: str,
                  out_fd: str, k: int, fk: int, fd: int):
    """
    Calculate Stochastic Oscillator.

    Args:
        df (pd.DataFrame): Input DataFrame
        in_c (str): Input column name (typically 'Close')
        out_fastk (str): Output column name for Fast %K
        out_fk (str): Output column name for %K
        out_fd (str): Output column name for %D
        k (int): Window size for Fast %K calculation
        fk (int): Window size for %K smoothing
        fd (int): Window size for %D calculation

    Note:
        Fast %K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100
        %K = EMA of Fast %K
        %D = EMA of %K
    """
    def stoch_calc(x):
        """Helper function to calculate stochastic %K"""
        return (x[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50.0

    df[out_fastk] = df[in_c].rolling(window=k).apply(stoch_calc, raw=True)
    get_ema(df, out_fastk, out_fk, fk)
    get_ema(df, out_fk, out_fd, fd)