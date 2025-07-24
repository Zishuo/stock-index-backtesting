"""
Helper functions and utilities for the backtesting framework.

This module provides various utility functions for data validation,
formatting, and common operations used throughout the system.
"""

import datetime as dt
import pandas as pd
import numpy as np


def validate_date_range(start_date, end_date, data_index):
    """
    Validate and adjust date range to fit available data.
    
    Args:
        start_date: Desired start date
        end_date: Desired end date
        data_index: Available data index (pd.DatetimeIndex)
        
    Returns:
        tuple: (adjusted_start_date, adjusted_end_date)
    """
    sd = max(start_date, data_index.min()) if start_date else data_index.min()
    ed = min(end_date, data_index.max()) if end_date else data_index.max()
    
    # Ensure start date exists in data
    if sd not in data_index:
        sd = data_index[data_index.get_indexer([sd], method='backfill')[0]]
    
    return sd, ed


def format_percentage(value, decimals=2):
    """
    Format a decimal value as a percentage string.
    
    Args:
        value (float): Decimal value (e.g., 0.15 for 15%)
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}%}"


def format_currency(value, symbol="$"):
    """
    Format a number as currency.
    
    Args:
        value (float): Monetary value
        symbol (str): Currency symbol
        
    Returns:
        str: Formatted currency string
    """
    if pd.isna(value):
        return "N/A"
    return f"{symbol}{value:,.2f}"


def calculate_returns(prices):
    """
    Calculate simple returns from price series.
    
    Args:
        prices (pd.Series): Price series
        
    Returns:
        pd.Series: Return series
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices):
    """
    Calculate logarithmic returns from price series.
    
    Args:
        prices (pd.Series): Price series
        
    Returns:
        pd.Series: Log return series
    """
    return np.log(prices / prices.shift(1)).dropna()


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio.
    
    Args:
        returns (pd.Series): Return series
        risk_free_rate (float): Risk-free rate (annualized)
        periods_per_year (int): Number of periods per year
        
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown from price series.
    
    Args:
        prices (pd.Series): Price series
        
    Returns:
        float: Maximum drawdown (negative value)
    """
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def resample_data(data, freq='D', method='last'):
    """
    Resample data to different frequency.
    
    Args:
        data (pd.DataFrame): Input data with datetime index
        freq (str): Target frequency ('D', 'W', 'M', etc.)
        method (str): Resampling method ('last', 'first', 'mean', etc.)
        
    Returns:
        pd.DataFrame: Resampled data
    """
    if method == 'last':
        return data.resample(freq).last()
    elif method == 'first':
        return data.resample(freq).first()
    elif method == 'mean':
        return data.resample(freq).mean()
    elif method == 'ohlc':
        # For OHLC data
        resampled = data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        return resampled
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def merge_signals(*signal_dfs, how='outer'):
    """
    Merge multiple signal DataFrames.
    
    Args:
        *signal_dfs: Variable number of signal DataFrames
        how (str): How to merge ('outer', 'inner', 'left', 'right')
        
    Returns:
        pd.DataFrame: Merged signals
    """
    result = signal_dfs[0]
    for df in signal_dfs[1:]:
        result = pd.merge(result, df, left_index=True, right_index=True, how=how)
    
    return result.fillna(0)


def validate_strategy_parameters(**kwargs):
    """
    Validate common strategy parameters.
    
    Args:
        **kwargs: Strategy parameters to validate
        
    Raises:
        ValueError: If parameters are invalid
    """
    if 'stop_loss' in kwargs and kwargs['stop_loss'] < 0:
        raise ValueError("Stop loss must be non-negative")
    
    if 'take_profit' in kwargs and kwargs['take_profit'] < 0:
        raise ValueError("Take profit must be non-negative")
    
    if 'ma_window' in kwargs and kwargs['ma_window'] <= 0:
        raise ValueError("Moving average window must be positive")
    
    if 'buy_threshold' in kwargs and 'sell_threshold' in kwargs:
        if kwargs['buy_threshold'] <= kwargs['sell_threshold']:
            print("Warning: Buy threshold should typically be higher than sell threshold")


def clean_dataframe(df, fill_method='ffill'):
    """
    Clean DataFrame by handling missing values and infinite values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        fill_method (str): Method to fill missing values
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    if fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'bfill':
        df = df.bfill()
    elif fill_method == 'drop':
        df = df.dropna()
    elif isinstance(fill_method, (int, float)):
        df = df.fillna(fill_method)
    
    return df


def print_trade_summary(trade_records):
    """
    Print a summary of trade records.
    
    Args:
        trade_records (pd.DataFrame): Trade records DataFrame
    """
    if trade_records.empty:
        print("No trades recorded")
        return
    
    total_trades = len(trade_records)
    profitable_trades = len(trade_records[trade_records['Profit %'] > 0])
    losing_trades = len(trade_records[trade_records['Profit %'] <= 0])
    
    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
    avg_profit = trade_records['Profit %'].mean() * 100
    avg_winning_trade = trade_records[trade_records['Profit %'] > 0]['Profit %'].mean() * 100
    avg_losing_trade = trade_records[trade_records['Profit %'] <= 0]['Profit %'].mean() * 100
    
    print(f"""
Trade Summary:
Total trades: {total_trades}
Profitable trades: {profitable_trades}
Losing trades: {losing_trades}
Win rate: {win_rate:.1f}%
Average profit per trade: {avg_profit:.2f}%
Average winning trade: {avg_winning_trade:.2f}%
Average losing trade: {avg_losing_trade:.2f}%
    """)


__all__ = [
    'validate_date_range',
    'format_percentage',
    'format_currency',
    'calculate_returns',
    'calculate_log_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'resample_data',
    'merge_signals',
    'validate_strategy_parameters',
    'clean_dataframe',
    'print_trade_summary'
]