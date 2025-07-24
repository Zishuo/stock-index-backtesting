"""
Data validation utilities for the backtesting framework.

This module provides comprehensive validation functions for data integrity,
parameter validation, and input sanitization.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime, date
import warnings


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """
    Comprehensive data validation for financial data.
    
    This class provides methods to validate price data, indicators,
    and other financial datasets used in backtesting.
    """
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame, ticker: str = None) -> Dict[str, Any]:
        """
        Validate price data for completeness and quality.
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            ticker (str, optional): Ticker symbol for error messages
            
        Returns:
            Dict[str, Any]: Validation report
            
        Raises:
            ValidationError: If data is invalid
        """
        ticker_str = f" for {ticker}" if ticker else ""
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Check if DataFrame is empty
        if data.empty:
            report['valid'] = False
            report['errors'].append(f"Price data{ticker_str} is empty")
            return report
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            report['valid'] = False
            report['errors'].append(f"Missing required columns{ticker_str}: {missing_columns}")
        
        # Check for valid price relationships
        if report['valid']:
            # High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
            invalid_high_low = (data['High'] < data['Low']).sum()
            invalid_high_open = (data['High'] < data['Open']).sum()
            invalid_high_close = (data['High'] < data['Close']).sum()
            invalid_low_open = (data['Low'] > data['Open']).sum()
            invalid_low_close = (data['Low'] > data['Close']).sum()
            
            if invalid_high_low > 0:
                report['warnings'].append(f"Found {invalid_high_low} rows where High < Low{ticker_str}")
            if invalid_high_open > 0:
                report['warnings'].append(f"Found {invalid_high_open} rows where High < Open{ticker_str}")
            if invalid_high_close > 0:
                report['warnings'].append(f"Found {invalid_high_close} rows where High < Close{ticker_str}")
            if invalid_low_open > 0:
                report['warnings'].append(f"Found {invalid_low_open} rows where Low > Open{ticker_str}")
            if invalid_low_close > 0:
                report['warnings'].append(f"Found {invalid_low_close} rows where Low > Close{ticker_str}")
        
        # Check for negative prices
        if report['valid']:
            for col in required_columns:
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    report['warnings'].append(f"Found {negative_prices} non-positive {col} prices{ticker_str}")
        
        # Check for missing values
        if report['valid']:
            for col in data.columns:
                missing_values = data[col].isnull().sum()
                if missing_values > 0:
                    report['warnings'].append(f"Found {missing_values} missing values in {col}{ticker_str}")
        
        # Check for infinite values
        if report['valid']:
            for col in data.select_dtypes(include=[np.number]).columns:
                infinite_values = np.isinf(data[col]).sum()
                if infinite_values > 0:
                    report['warnings'].append(f"Found {infinite_values} infinite values in {col}{ticker_str}")
        
        # Check for duplicate dates
        if isinstance(data.index, pd.DatetimeIndex):
            duplicate_dates = data.index.duplicated().sum()
            if duplicate_dates > 0:
                report['warnings'].append(f"Found {duplicate_dates} duplicate dates{ticker_str}")
        
        # Calculate statistics
        if report['valid']:
            report['statistics'] = {
                'total_rows': len(data),
                'date_range': (data.index.min(), data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else None,
                'avg_volume': data['Volume'].mean() if 'Volume' in data.columns else None,
                'price_range': (data['Close'].min(), data['Close'].max()) if 'Close' in data.columns else None,
                'missing_data_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            }
        
        return report
    
    @staticmethod
    def validate_signal_data(signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate trading signal data.
        
        Args:
            signals (pd.DataFrame): Signal data
            
        Returns:
            Dict[str, Any]: Validation report
        """
        report = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if signals.empty:
            report['valid'] = False
            report['errors'].append("Signal data is empty")
            return report
        
        # Check for valid signal values
        signal_columns = [col for col in signals.columns if 'signal' in col.lower()]
        
        for col in signal_columns:
            unique_values = signals[col].unique()
            valid_signals = {-2, -1, 0, 1}  # Stop loss, sell, hold, buy
            
            invalid_signals = set(unique_values) - valid_signals
            if invalid_signals:
                report['warnings'].append(f"Found invalid signal values in {col}: {invalid_signals}")
        
        # Check signal frequency
        if signal_columns:
            total_signals = 0
            buy_signals = 0
            sell_signals = 0
            
            for col in signal_columns:
                total_signals += (signals[col] != 0).sum()
                buy_signals += (signals[col] > 0).sum()
                sell_signals += (signals[col] < 0).sum()
            
            report['statistics'] = {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'signal_frequency': total_signals / len(signals) if len(signals) > 0 else 0
            }
        
        return report
    
    @staticmethod
    def clean_price_data(data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Clean price data by handling missing and invalid values.
        
        Args:
            data (pd.DataFrame): Raw price data
            method (str): Method for handling missing data ('ffill', 'bfill', 'interpolate', 'drop')
            
        Returns:
            pd.DataFrame: Cleaned price data
        """
        cleaned_data = data.copy()
        
        # Replace infinite values with NaN
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        
        # Handle negative prices (replace with NaN)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in cleaned_data.columns:
                cleaned_data.loc[cleaned_data[col] <= 0, col] = np.nan
        
        # Handle missing values
        if method == 'ffill':
            cleaned_data = cleaned_data.ffill()
        elif method == 'bfill':
            cleaned_data = cleaned_data.bfill()
        elif method == 'interpolate':
            cleaned_data = cleaned_data.interpolate()
        elif method == 'drop':
            cleaned_data = cleaned_data.dropna()
        
        # Fix OHLC relationships
        for idx in cleaned_data.index:
            row = cleaned_data.loc[idx]
            if all(col in row.index for col in price_columns):
                # Ensure High is the maximum
                max_price = max(row['Open'], row['Close'])
                if row['High'] < max_price:
                    cleaned_data.loc[idx, 'High'] = max_price
                
                # Ensure Low is the minimum
                min_price = min(row['Open'], row['Close'])
                if row['Low'] > min_price:
                    cleaned_data.loc[idx, 'Low'] = min_price
        
        return cleaned_data


class ParameterValidator:
    """
    Parameter validation for strategy and backtest configuration.
    """
    
    @staticmethod
    def validate_date_range(start_date: Any, end_date: Any) -> Tuple[datetime, datetime]:
        """
        Validate and convert date range.
        
        Args:
            start_date: Start date (various formats)
            end_date: End date (various formats)
            
        Returns:
            Tuple[datetime, datetime]: Validated date range
            
        Raises:
            ValidationError: If dates are invalid
        """
        # Convert to datetime if needed
        if isinstance(start_date, str):
            try:
                start_date = pd.to_datetime(start_date)
            except:
                raise ValidationError(f"Invalid start_date format: {start_date}")
        
        if isinstance(end_date, str):
            try:
                end_date = pd.to_datetime(end_date)
            except:
                raise ValidationError(f"Invalid end_date format: {end_date}")
        
        # Validate date order
        if start_date >= end_date:
            raise ValidationError("start_date must be before end_date")
        
        # Check if dates are reasonable
        if start_date.year < 1900:
            raise ValidationError("start_date is too early (before 1900)")
        
        if end_date > datetime.now():
            warnings.warn("end_date is in the future")
        
        return start_date, end_date
    
    @staticmethod
    def validate_numeric_parameter(value: Any, name: str, min_val: float = None, 
                                 max_val: float = None, allow_zero: bool = True) -> float:
        """
        Validate numeric parameters.
        
        Args:
            value: Value to validate
            name (str): Parameter name for error messages
            min_val (float, optional): Minimum allowed value
            max_val (float, optional): Maximum allowed value
            allow_zero (bool): Whether zero is allowed
            
        Returns:
            float: Validated numeric value
            
        Raises:
            ValidationError: If value is invalid
        """
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be a numeric value, got {type(value).__name__}")
        
        if not allow_zero and numeric_value == 0:
            raise ValidationError(f"{name} cannot be zero")
        
        if min_val is not None and numeric_value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {numeric_value}")
        
        if max_val is not None and numeric_value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {numeric_value}")
        
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            raise ValidationError(f"{name} cannot be NaN or infinite")
        
        return numeric_value
    
    @staticmethod
    def validate_integer_parameter(value: Any, name: str, min_val: int = None, 
                                 max_val: int = None, allow_zero: bool = True) -> int:
        """
        Validate integer parameters.
        
        Args:
            value: Value to validate
            name (str): Parameter name for error messages
            min_val (int, optional): Minimum allowed value
            max_val (int, optional): Maximum allowed value
            allow_zero (bool): Whether zero is allowed
            
        Returns:
            int: Validated integer value
            
        Raises:
            ValidationError: If value is invalid
        """
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{name} must be an integer value, got {type(value).__name__}")
        
        if not allow_zero and int_value == 0:
            raise ValidationError(f"{name} cannot be zero")
        
        if min_val is not None and int_value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {int_value}")
        
        if max_val is not None and int_value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {int_value}")
        
        return int_value
    
    @staticmethod
    def validate_choice_parameter(value: Any, name: str, choices: List[Any]) -> Any:
        """
        Validate choice parameters.
        
        Args:
            value: Value to validate
            name (str): Parameter name for error messages
            choices (List[Any]): List of valid choices
            
        Returns:
            Any: Validated choice value
            
        Raises:
            ValidationError: If value is not in choices
        """
        if value not in choices:
            raise ValidationError(f"{name} must be one of {choices}, got {value}")
        
        return value


# Validation decorators
def validate_data(func):
    """Decorator to validate data inputs."""
    def wrapper(*args, **kwargs):
        # Basic validation - can be extended
        return func(*args, **kwargs)
    return wrapper


def validate_parameters(**param_validators):
    """Decorator to validate function parameters."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Apply parameter validators
            for param_name, validator in param_validators.items():
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator(kwargs[param_name])
                    except Exception as e:
                        raise ValidationError(f"Parameter validation failed for {param_name}: {e}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    'ValidationError', 'DataValidator', 'ParameterValidator',
    'validate_data', 'validate_parameters'
]