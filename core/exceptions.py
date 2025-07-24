"""
Custom exceptions and error handling for the backtesting framework.

This module defines custom exception classes and error handling utilities
to provide better error messages and debugging information.
"""

import traceback
import logging
from typing import Optional, Any, Dict
from functools import wraps


class BacktestingError(Exception):
    """Base exception for all backtesting framework errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message
            details (dict, optional): Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class DataError(BacktestingError):
    """Exception raised for data-related errors."""
    pass


class StrategyError(BacktestingError):
    """Exception raised for strategy-related errors."""
    pass


class PortfolioError(BacktestingError):
    """Exception raised for portfolio management errors."""
    pass


class ConfigurationError(BacktestingError):
    """Exception raised for configuration errors."""
    pass


class ValidationError(BacktestingError):
    """Exception raised for validation errors.""" 
    pass


class IndicatorError(BacktestingError):
    """Exception raised for technical indicator calculation errors."""
    pass


class NetworkError(BacktestingError):
    """Exception raised for network/data source errors."""
    pass


class InsufficientDataError(DataError):
    """Exception raised when there is insufficient data for analysis."""
    pass


class MissingDataError(DataError):
    """Exception raised when required data is missing."""
    pass


class InvalidSignalError(StrategyError):
    """Exception raised for invalid trading signals."""
    pass


class InsufficientCapitalError(PortfolioError):
    """Exception raised when there is insufficient capital for trading."""
    pass


class ErrorHandler:
    """
    Centralized error handling and logging system.
    """
    
    def __init__(self, logger_name: str = "backtesting", log_level: int = logging.WARNING):
        """
        Initialize the error handler.
        
        Args:
            logger_name (str): Name of the logger
            log_level (int): Logging level
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def handle_exception(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> BacktestingError:
        """
        Handle and convert exceptions to framework-specific errors.
        
        Args:
            exc (Exception): Original exception
            context (dict, optional): Additional context information
            
        Returns:
            BacktestingError: Framework-specific exception
        """
        context = context or {}
        
        # Log the original exception
        self.logger.error(f"Exception occurred: {exc}", exc_info=True)
        
        # Convert to appropriate framework exception
        if isinstance(exc, BacktestingError):
            return exc
        elif isinstance(exc, (FileNotFoundError, IOError)):
            return DataError(f"File/IO error: {exc}", {"original_type": type(exc).__name__, **context})
        elif isinstance(exc, KeyError):
            return DataError(f"Missing required data field: {exc}", {"original_type": type(exc).__name__, **context})
        elif isinstance(exc, ValueError):
            return ValidationError(f"Invalid value: {exc}", {"original_type": type(exc).__name__, **context})
        elif isinstance(exc, TypeError):
            return ValidationError(f"Type error: {exc}", {"original_type": type(exc).__name__, **context})
        elif isinstance(exc, ImportError):
            return ConfigurationError(f"Missing dependency: {exc}", {"original_type": type(exc).__name__, **context})
        elif "network" in str(exc).lower() or "connection" in str(exc).lower():
            return NetworkError(f"Network error: {exc}", {"original_type": type(exc).__name__, **context})
        else:
            return BacktestingError(f"Unexpected error: {exc}", {"original_type": type(exc).__name__, **context})
    
    def log_warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        if details:
            message += f" Details: {details}"
        self.logger.warning(message)
    
    def log_error(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an error message."""
        if details:
            message += f" Details: {details}"
        self.logger.error(message)
    
    def log_info(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        if details:
            message += f" Details: {details}"
        self.logger.info(message)


# Global error handler instance
_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


def handle_errors(reraise: bool = True, default_return: Any = None, context: Optional[Dict[str, Any]] = None):
    """
    Decorator for automatic error handling.
    
    Args:
        reraise (bool): Whether to reraise the exception after handling
        default_return: Default return value if exception is caught and not reraised
        context (dict, optional): Additional context for error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or {}
                error_context.update({
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
                
                handled_error = _error_handler.handle_exception(e, error_context)
                
                if reraise:
                    raise handled_error
                else:
                    _error_handler.log_error(f"Error in {func.__name__}", {"error": str(handled_error)})
                    return default_return
        
        return wrapper
    return decorator


def safe_execute(func, *args, default=None, context=None, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default return value on error
        context (dict, optional): Additional context
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = context or {}
        error_context.update({
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'args': str(args),
            'kwargs': str(kwargs)
        })
        
        handled_error = _error_handler.handle_exception(e, error_context)
        _error_handler.log_error(f"Safe execution failed", {"error": str(handled_error)})
        return default


def validate_not_none(value: Any, name: str) -> Any:
    """
    Validate that a value is not None.
    
    Args:
        value: Value to validate
        name (str): Name of the value for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_positive(value: float, name: str) -> float:
    """
    Validate that a numeric value is positive.
    
    Args:
        value (float): Value to validate
        name (str): Name of the value for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_in_range(value: float, name: str, min_val: float = None, max_val: float = None) -> float:
    """
    Validate that a value is within a specified range.
    
    Args:
        value (float): Value to validate
        name (str): Name of the value for error messages
        min_val (float, optional): Minimum allowed value
        max_val (float, optional): Maximum allowed value
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is outside the range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return value


class ContextualError:
    """
    Context manager for adding context to errors.
    """
    
    def __init__(self, context: Dict[str, Any]):
        """
        Initialize with context information.
        
        Args:
            context (dict): Context information to add to errors
        """
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            handled_error = _error_handler.handle_exception(exc_val, self.context)
            raise handled_error
        return False


__all__ = [
    # Exception classes
    'BacktestingError', 'DataError', 'StrategyError', 'PortfolioError',
    'ConfigurationError', 'ValidationError', 'IndicatorError', 'NetworkError',
    'InsufficientDataError', 'MissingDataError', 'InvalidSignalError',
    'InsufficientCapitalError',
    
    # Error handling utilities
    'ErrorHandler', 'get_error_handler', 'handle_errors', 'safe_execute',
    'ContextualError',
    
    # Validation utilities
    'validate_not_none', 'validate_positive', 'validate_in_range'
]