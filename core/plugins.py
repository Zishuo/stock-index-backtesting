"""
Plugin architecture for custom indicators and extensions.

This module provides a flexible plugin system that allows users to create
custom technical indicators, strategy components, and other extensions
to the backtesting framework.
"""

import inspect
import importlib.util
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, Type
import pandas as pd
import numpy as np
from pathlib import Path

from .exceptions import BacktestingError, ValidationError


class IndicatorPlugin(ABC):
    """
    Base class for custom technical indicator plugins.
    
    All custom indicators must inherit from this class and implement
    the required methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the indicator."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the indicator does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return the required parameters and their default values."""
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate the indicator values.
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
            **kwargs: Indicator-specific parameters
            
        Returns:
            pd.Series: Calculated indicator values
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and return processed parameters.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Dict[str, Any]: Validated parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        validated = {}
        required_params = self.parameters
        
        for param_name, default_value in required_params.items():
            if param_name in kwargs:
                validated[param_name] = kwargs[param_name]
            elif default_value is not None:
                validated[param_name] = default_value
            else:
                raise ValidationError(f"Required parameter '{param_name}' not provided for {self.name}")
        
        return validated
    
    def validate_data(self, data: pd.DataFrame):
        """
        Validate input data.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        if data.empty:
            raise ValidationError(f"Empty data provided to {self.name}")
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValidationError(f"{self.name} requires columns {missing_columns}")


class StrategyPlugin(ABC):
    """
    Base class for custom strategy plugins.
    
    Allows users to create modular strategy components that can be
    combined or used independently.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the strategy component."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the strategy component."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            **kwargs: Strategy-specific parameters
            
        Returns:
            pd.Series: Trading signals (-1, 0, 1)
        """
        pass


class PluginManager:
    """
    Manager for loading, registering, and using plugins.
    
    This class handles the discovery and management of custom indicators
    and strategy components.
    """
    
    def __init__(self):
        """Initialize the plugin manager."""
        self._indicator_plugins: Dict[str, IndicatorPlugin] = {}
        self._strategy_plugins: Dict[str, StrategyPlugin] = {}
        self._plugin_directories: List[str] = []
        
        # Register built-in plugins
        self._register_builtin_indicators()
    
    def _register_builtin_indicators(self):
        """Register built-in indicator plugins."""
        
        # Built-in RSI indicator
        class RSIPlugin(IndicatorPlugin):
            @property
            def name(self) -> str:
                return "rsi"
            
            @property
            def description(self) -> str:
                return "Relative Strength Index - momentum oscillator (0-100)"
            
            @property
            def parameters(self) -> Dict[str, Any]:
                return {"period": 14}
            
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                params = self.validate_parameters(**kwargs)
                period = params["period"]
                
                close = data['Close']
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
        
        # Built-in MACD indicator
        class MACDPlugin(IndicatorPlugin):
            @property
            def name(self) -> str:
                return "macd"
            
            @property
            def description(self) -> str:
                return "Moving Average Convergence Divergence"
            
            @property
            def parameters(self) -> Dict[str, Any]:
                return {"fast_period": 12, "slow_period": 26, "signal_period": 9}
            
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                params = self.validate_parameters(**kwargs)
                close = data['Close']
                
                ema_fast = close.ewm(span=params["fast_period"]).mean()
                ema_slow = close.ewm(span=params["slow_period"]).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=params["signal_period"]).mean()
                histogram = macd_line - signal_line
                
                # Return MACD line, but could be extended to return all components
                return macd_line
        
        # Built-in Bollinger Bands indicator
        class BollingerBandsPlugin(IndicatorPlugin):
            @property
            def name(self) -> str:
                return "bollinger_bands"
            
            @property
            def description(self) -> str:
                return "Bollinger Bands - volatility indicator"
            
            @property
            def parameters(self) -> Dict[str, Any]:
                return {"period": 20, "std_dev": 2}
            
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                params = self.validate_parameters(**kwargs)
                close = data['Close']
                
                sma = close.rolling(window=params["period"]).mean()
                std = close.rolling(window=params["period"]).std()
                
                upper_band = sma + (std * params["std_dev"])
                lower_band = sma - (std * params["std_dev"])
                
                # Return middle band (SMA), but could return all bands
                return sma
        
        # Register built-in indicators
        self.register_indicator_plugin(RSIPlugin())
        self.register_indicator_plugin(MACDPlugin())
        self.register_indicator_plugin(BollingerBandsPlugin())
    
    def register_indicator_plugin(self, plugin: IndicatorPlugin):
        """
        Register a custom indicator plugin.
        
        Args:
            plugin (IndicatorPlugin): The indicator plugin to register
            
        Raises:
            ValidationError: If plugin is invalid
        """
        if not isinstance(plugin, IndicatorPlugin):
            raise ValidationError("Plugin must inherit from IndicatorPlugin")
        
        if plugin.name in self._indicator_plugins:
            print(f"Warning: Overriding existing indicator plugin '{plugin.name}'")
        
        self._indicator_plugins[plugin.name] = plugin
    
    def register_strategy_plugin(self, plugin: StrategyPlugin):
        """
        Register a custom strategy plugin.
        
        Args:
            plugin (StrategyPlugin): The strategy plugin to register
        """
        if not isinstance(plugin, StrategyPlugin):
            raise ValidationError("Plugin must inherit from StrategyPlugin")
        
        if plugin.name in self._strategy_plugins:
            print(f"Warning: Overriding existing strategy plugin '{plugin.name}'")
        
        self._strategy_plugins[plugin.name] = plugin
    
    def add_plugin_directory(self, directory: str):
        """
        Add a directory to search for plugins.
        
        Args:
            directory (str): Path to the plugin directory
        """
        if os.path.exists(directory):
            self._plugin_directories.append(directory)
            self.discover_plugins(directory)
        else:
            print(f"Warning: Plugin directory '{directory}' does not exist")
    
    def discover_plugins(self, directory: str):
        """
        Discover and load plugins from a directory.
        
        Args:
            directory (str): Directory to search for plugins
        """
        plugin_dir = Path(directory)
        
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip private files
            
            try:
                self._load_plugin_from_file(str(py_file))
            except Exception as e:
                print(f"Warning: Failed to load plugin from {py_file}: {e}")
    
    def _load_plugin_from_file(self, file_path: str):
        """Load plugins from a Python file."""
        spec = importlib.util.spec_from_file_location("plugin_module", file_path)
        if spec is None or spec.loader is None:
            raise BacktestingError(f"Could not load plugin from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for plugin classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, IndicatorPlugin) and obj != IndicatorPlugin and 
                not inspect.isabstract(obj)):
                self.register_indicator_plugin(obj())
            elif (issubclass(obj, StrategyPlugin) and obj != StrategyPlugin and 
                  not inspect.isabstract(obj)):
                self.register_strategy_plugin(obj())
    
    def get_indicator(self, name: str) -> IndicatorPlugin:
        """
        Get an indicator plugin by name.
        
        Args:
            name (str): Name of the indicator
            
        Returns:
            IndicatorPlugin: The indicator plugin
            
        Raises:
            ValidationError: If indicator not found
        """
        if name not in self._indicator_plugins:
            available = list(self._indicator_plugins.keys())
            raise ValidationError(f"Indicator '{name}' not found. Available: {available}")
        
        return self._indicator_plugins[name]
    
    def get_strategy(self, name: str) -> StrategyPlugin:
        """
        Get a strategy plugin by name.
        
        Args:
            name (str): Name of the strategy plugin
            
        Returns:
            StrategyPlugin: The strategy plugin
        """
        if name not in self._strategy_plugins:
            available = list(self._strategy_plugins.keys())
            raise ValidationError(f"Strategy plugin '{name}' not found. Available: {available}")
        
        return self._strategy_plugins[name]
    
    def calculate_indicator(self, name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate an indicator using a plugin.
        
        Args:
            name (str): Name of the indicator
            data (pd.DataFrame): Price data
            **kwargs: Indicator parameters
            
        Returns:
            pd.Series: Calculated indicator values
        """
        plugin = self.get_indicator(name)
        plugin.validate_data(data)
        return plugin.calculate(data, **kwargs)
    
    def list_indicators(self) -> Dict[str, str]:
        """
        List all available indicator plugins.
        
        Returns:
            Dict[str, str]: Dictionary mapping names to descriptions
        """
        return {name: plugin.description for name, plugin in self._indicator_plugins.items()}
    
    def list_strategies(self) -> Dict[str, str]:
        """
        List all available strategy plugins.
        
        Returns:
            Dict[str, str]: Dictionary mapping names to descriptions
        """
        return {name: plugin.description for name, plugin in self._strategy_plugins.items()}
    
    def get_indicator_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an indicator.
        
        Args:
            name (str): Name of the indicator
            
        Returns:
            Dict[str, Any]: Indicator information
        """
        plugin = self.get_indicator(name)
        return {
            "name": plugin.name,
            "description": plugin.description,
            "parameters": plugin.parameters
        }


class IndicatorComposer:
    """
    Utility class for composing multiple indicators into complex signals.
    
    This allows users to combine multiple indicators to create sophisticated
    trading strategies.
    """
    
    def __init__(self, plugin_manager: PluginManager):
        """
        Initialize the indicator composer.
        
        Args:
            plugin_manager (PluginManager): Plugin manager instance
        """
        self.plugin_manager = plugin_manager
    
    def create_composite_indicator(self, indicators: List[Dict[str, Any]], 
                                 composition_func: Callable) -> Callable:
        """
        Create a composite indicator from multiple indicators.
        
        Args:
            indicators (List[Dict]): List of indicator configurations
            composition_func (Callable): Function to combine indicator values
            
        Returns:
            Callable: Function that calculates the composite indicator
        """
        def composite_indicator(data: pd.DataFrame) -> pd.Series:
            indicator_values = []
            
            for indicator_config in indicators:
                name = indicator_config["name"]
                params = indicator_config.get("params", {})
                values = self.plugin_manager.calculate_indicator(name, data, **params)
                indicator_values.append(values)
            
            return composition_func(*indicator_values)
        
        return composite_indicator
    
    def create_signal_from_indicators(self, indicators: List[Dict[str, Any]], 
                                    signal_func: Callable) -> Callable:
        """
        Create trading signals from multiple indicators.
        
        Args:
            indicators (List[Dict]): List of indicator configurations
            signal_func (Callable): Function to generate signals from indicators
            
        Returns:
            Callable: Function that generates trading signals
        """
        def signal_generator(data: pd.DataFrame) -> pd.Series:
            indicator_values = {}
            
            for indicator_config in indicators:
                name = indicator_config["name"]
                params = indicator_config.get("params", {})
                values = self.plugin_manager.calculate_indicator(name, data, **params)
                indicator_values[name] = values
            
            return signal_func(indicator_values)
        
        return signal_generator


# Global plugin manager instance
_plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def register_indicator(plugin: IndicatorPlugin):
    """Register a custom indicator plugin."""
    get_plugin_manager().register_indicator_plugin(plugin)

def register_strategy(plugin: StrategyPlugin):
    """Register a custom strategy plugin."""
    get_plugin_manager().register_strategy_plugin(plugin)

def calculate_custom_indicator(name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
    """Calculate a custom indicator."""
    return get_plugin_manager().calculate_indicator(name, data, **kwargs)


__all__ = [
    'IndicatorPlugin', 'StrategyPlugin', 'PluginManager', 'IndicatorComposer',
    'get_plugin_manager', 'register_indicator', 'register_strategy', 
    'calculate_custom_indicator'
]