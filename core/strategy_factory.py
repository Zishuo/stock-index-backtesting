"""
Strategy Factory for creating trading strategies.

This module implements the Factory pattern for creating strategy instances,
providing a centralized way to instantiate strategies with validation
and configuration management.
"""

from typing import Dict, Any, Type, Optional
import inspect

from .strategy_base import Strategy
# Import strategies will be done dynamically to avoid circular imports


class StrategyFactory:
    """
    Factory class for creating trading strategy instances.
    
    This factory provides a centralized way to create strategies with
    parameter validation and configuration management.
    """
    
    # Registry of available strategies (will be populated dynamically)
    _strategies: Dict[str, Type[Strategy]] = {}
    _initialized = False
    
    @classmethod
    def _initialize_strategies(cls):
        """Initialize the strategy registry dynamically."""
        if cls._initialized:
            return
        
        try:
            # Import strategies dynamically to avoid circular imports
            from strategies.basic_strategies import BuyAndHold, MACross, MAThreshold
            from strategies.threshold_strategies import Threshold
            from strategies.stochastic_strategy import StochasticCross
            from strategies.custom_strategies import fftyspy_stg, fftynaa200r_stg, CustomizedStrategy
            
            cls._strategies.update({
                'buyandhold': BuyAndHold,
                'buy_and_hold': BuyAndHold,
                'macross': MACross,
                'ma_cross': MACross,
                'moving_average_cross': MACross,
                'mathreshold': MAThreshold,
                'ma_threshold': MAThreshold,
                'moving_average_threshold': MAThreshold,
                'threshold': Threshold,
                'stochastic': StochasticCross,
                'stochastic_cross': StochasticCross,
                'fftyspy': fftyspy_stg,
                'ffty_spy': fftyspy_stg,
                'fftynaa200r': fftynaa200r_stg,
                'ffty_naa200r': fftynaa200r_stg,
                'customized': CustomizedStrategy,
                'custom': CustomizedStrategy
            })
            cls._initialized = True
        except ImportError as e:
            print(f"Warning: Could not initialize all strategies: {e}")
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> Strategy:
        """
        Create a strategy instance by name.
        
        Args:
            strategy_name (str): Name of the strategy to create
            **kwargs: Strategy-specific parameters
            
        Returns:
            Strategy: Instantiated strategy object
            
        Raises:
            ValueError: If strategy name is not recognized
            TypeError: If required parameters are missing
        """
        # Initialize strategies if not already done
        cls._initialize_strategies()
        
        # Normalize strategy name
        normalized_name = strategy_name.lower().replace('-', '_').replace(' ', '_')
        
        if normalized_name not in cls._strategies:
            available = ', '.join(sorted(set(cls._strategies.keys())))
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
        
        strategy_class = cls._strategies[normalized_name]
        
        # Validate parameters
        cls._validate_parameters(strategy_class, kwargs)
        
        try:
            return strategy_class(**kwargs)
        except Exception as e:
            raise TypeError(f"Failed to create strategy '{strategy_name}': {str(e)}")
    
    @classmethod
    def _validate_parameters(cls, strategy_class: Type[Strategy], params: Dict[str, Any]):
        """
        Validate parameters for strategy creation.
        
        Args:
            strategy_class: The strategy class to validate against
            params: Parameters to validate
            
        Raises:
            TypeError: If required parameters are missing or types are incorrect
        """
        # Get the __init__ signature
        init_signature = inspect.signature(strategy_class.__init__)
        
        # Check for required parameters (no default value)
        required_params = []
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
        
        # Check if all required parameters are provided
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise TypeError(f"Missing required parameters: {missing_params}")
        
        # Type hints validation (if available)
        for param_name, param in init_signature.parameters.items():
            if param_name in params and param.annotation != inspect.Parameter.empty:
                value = params[param_name]
                expected_type = param.annotation
                
                # Basic type checking
                if not isinstance(value, expected_type):
                    # Handle Union types and Optional
                    if hasattr(expected_type, '__origin__'):
                        continue  # Skip complex type checking for now
                    raise TypeError(f"Parameter '{param_name}' expected {expected_type.__name__}, got {type(value).__name__}")
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[Strategy]):
        """
        Register a new strategy with the factory.
        
        Args:
            name (str): Name to register the strategy under
            strategy_class: Strategy class to register
            
        Raises:
            TypeError: If strategy_class is not a subclass of Strategy
        """
        if not issubclass(strategy_class, Strategy):
            raise TypeError("Strategy class must inherit from Strategy base class")
        
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, str]:
        """
        Get a list of all available strategies.
        
        Returns:
            Dict[str, str]: Dictionary mapping strategy names to class names
        """
        # Initialize strategies if not already done
        cls._initialize_strategies()
        
        unique_strategies = {}
        seen_classes = set()
        
        for name, strategy_class in cls._strategies.items():
            if strategy_class not in seen_classes:
                unique_strategies[name] = strategy_class.__name__
                seen_classes.add(strategy_class)
        
        return unique_strategies
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            Dict[str, Any]: Strategy information including parameters and description
            
        Raises:
            ValueError: If strategy name is not recognized
        """
        # Initialize strategies if not already done
        cls._initialize_strategies()
        
        normalized_name = strategy_name.lower().replace('-', '_').replace(' ', '_')
        
        if normalized_name not in cls._strategies:
            raise ValueError(f"Unknown strategy '{strategy_name}'")
        
        strategy_class = cls._strategies[normalized_name]
        init_signature = inspect.signature(strategy_class.__init__)
        
        # Extract parameter information
        parameters = {}
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            
            param_info = {
                'type': param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any',
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
            parameters[param_name] = param_info
        
        return {
            'class_name': strategy_class.__name__,
            'description': strategy_class.__doc__.strip() if strategy_class.__doc__ else 'No description available',
            'parameters': parameters
        }


# Convenience function for backward compatibility
def create_strategy(strategy_name: str, **kwargs) -> Strategy:
    """
    Create a strategy instance using the factory.
    
    Args:
        strategy_name (str): Name of the strategy to create
        **kwargs: Strategy-specific parameters
        
    Returns:
        Strategy: Instantiated strategy object
    """
    return StrategyFactory.create_strategy(strategy_name, **kwargs)


__all__ = ['StrategyFactory', 'create_strategy']