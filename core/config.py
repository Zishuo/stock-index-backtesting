"""
Configuration management system for the backtesting framework.

This module provides a centralized configuration system that can load
settings from files, environment variables, and programmatic overrides.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Try to import yaml, make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import constants directly to avoid circular imports
from utils.constants import *


@dataclass
class BacktestConfig:
    """Configuration settings for backtesting operations."""
    
    # Portfolio settings
    initial_capital: float = DEFAULT_PRINCIPAL
    trade_size: float = 1.0
    pyramiding: int = DEFAULT_PYRAMIDING
    commission: float = DEFAULT_COMMISSION
    market_impact: float = DEFAULT_IMPACT
    
    # Risk management
    stop_loss: float = DEFAULT_STOP_LOSS
    take_profit: float = DEFAULT_TAKE_PROFIT
    
    # Tax settings
    long_term_tax_rate: float = LONG_TERM_TAX_RATE
    short_term_tax_rate: float = SHORT_TERM_TAX_RATE
    
    # Data settings
    data_interval: str = DEFAULT_DATA_INTERVAL
    database_path: str = DEFAULT_DB_PATH
    database_limit: int = DEFAULT_DB_LIMIT
    
    # Performance calculation
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    drawdown_window: int = DRAWDOWN_WINDOW
    
    # Output settings
    verbose: bool = False
    plot_results: bool = True
    save_results: bool = False
    results_directory: str = "results"


@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators."""
    
    # Moving averages
    ma_windows: list = None
    default_ma_window: int = 20
    
    # Stochastic oscillator
    stochastic_k: int = DEFAULT_STOCHASTIC_K
    stochastic_fk: int = DEFAULT_STOCHASTIC_FK
    stochastic_fd: int = DEFAULT_STOCHASTIC_FD
    overbought_level: int = DEFAULT_OVERBOUGHT
    oversold_level: int = DEFAULT_OVERSOLD
    
    # Threshold values
    threshold_values: dict = None
    
    def __post_init__(self):
        if self.ma_windows is None:
            self.ma_windows = DEFAULT_MA_WINDOWS.copy()
        if self.threshold_values is None:
            self.threshold_values = THRESHOLD_VALUES.copy()


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    # Data sources
    default_source: str = "yfinance"
    cache_data: bool = True
    cache_directory: str = "cache"
    
    # Data validation
    validate_data: bool = True
    fill_missing_data: bool = True
    missing_data_method: str = "ffill"
    
    # Date handling
    timezone: str = "UTC"
    business_days_only: bool = True


class ConfigManager:
    """
    Centralized configuration management system.
    
    This class handles loading, validating, and providing access to
    configuration settings from various sources.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str, optional): Path to configuration file
        """
        self.backtest_config = BacktestConfig()
        self.indicator_config = TechnicalIndicatorConfig()
        self.data_config = DataConfig()
        
        self._config_file = config_file
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from various sources in priority order."""
        
        # 1. Load from default config file if it exists
        default_config_files = [
            "config.yaml", "config.yml", "config.json",
            ".backtesting.yaml", ".backtesting.yml", ".backtesting.json"
        ]
        
        for config_file in default_config_files:
            if os.path.exists(config_file):
                self._load_from_file(config_file)
                break
        
        # 2. Load from specified config file
        if self._config_file and os.path.exists(self._config_file):
            self._load_from_file(self._config_file)
        
        # 3. Override with environment variables
        self._load_from_environment()
    
    def _load_from_file(self, config_file: str):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to configuration file
        """
        try:
            file_path = Path(config_file)
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        raise ImportError("PyYAML is required to load YAML configuration files")
                    config_data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
            
            self._apply_config_data(config_data)
            
        except Exception as e:
            print(f"Warning: Failed to load configuration from {config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        
        env_mappings = {
            # Backtest config
            'BT_INITIAL_CAPITAL': ('backtest_config', 'initial_capital', float),
            'BT_STOP_LOSS': ('backtest_config', 'stop_loss', float),
            'BT_TAKE_PROFIT': ('backtest_config', 'take_profit', float),
            'BT_COMMISSION': ('backtest_config', 'commission', float),
            'BT_VERBOSE': ('backtest_config', 'verbose', lambda x: x.lower() == 'true'),
            
            # Data config
            'BT_DATA_SOURCE': ('data_config', 'default_source', str),
            'BT_CACHE_DATA': ('data_config', 'cache_data', lambda x: x.lower() == 'true'),
            'BT_DATABASE_PATH': ('backtest_config', 'database_path', str),
            
            # Tax rates
            'BT_LONG_TERM_TAX': ('backtest_config', 'long_term_tax_rate', float),
            'BT_SHORT_TERM_TAX': ('backtest_config', 'short_term_tax_rate', float),
        }
        
        for env_var, (config_section, config_key, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    config_obj = getattr(self, config_section)
                    setattr(config_obj, config_key, value)
                except Exception as e:
                    print(f"Warning: Failed to parse environment variable {env_var}: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """
        Apply configuration data to config objects.
        
        Args:
            config_data (dict): Configuration data dictionary
        """
        for section, values in config_data.items():
            if section == 'backtest' and isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(self.backtest_config, key):
                        setattr(self.backtest_config, key, value)
            
            elif section == 'indicators' and isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(self.indicator_config, key):
                        setattr(self.indicator_config, key, value)
            
            elif section == 'data' and isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(self.data_config, key):
                        setattr(self.data_config, key, value)
    
    def get_backtest_config(self) -> BacktestConfig:
        """Get backtest configuration."""
        return self.backtest_config
    
    def get_indicator_config(self) -> TechnicalIndicatorConfig:
        """Get technical indicator configuration."""
        return self.indicator_config
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.data_config
    
    def update_config(self, section: str, **kwargs):
        """
        Update configuration values programmatically.
        
        Args:
            section (str): Configuration section ('backtest', 'indicators', 'data')
            **kwargs: Configuration values to update
        """
        if section == 'backtest':
            config_obj = self.backtest_config
        elif section == 'indicators':
            config_obj = self.indicator_config
        elif section == 'data':
            config_obj = self.data_config
        else:
            raise ValueError(f"Unknown configuration section: {section}")
        
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}' in section '{section}'")
    
    def save_config(self, config_file: str, format: str = 'yaml'):
        """
        Save current configuration to a file.
        
        Args:
            config_file (str): Path to save configuration
            format (str): File format ('yaml' or 'json')
        """
        config_data = {
            'backtest': asdict(self.backtest_config),
            'indicators': asdict(self.indicator_config),
            'data': asdict(self.data_config)
        }
        
        with open(config_file, 'w') as f:
            if format.lower() == 'yaml':
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to save YAML configuration files")
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def print_config(self):
        """Print current configuration settings."""
        print("=" * 60)
        print("BACKTESTING FRAMEWORK CONFIGURATION")
        print("=" * 60)
        
        print("\nBacktest Settings:")
        for key, value in asdict(self.backtest_config).items():
            print(f"  {key}: {value}")
        
        print("\nTechnical Indicator Settings:")
        for key, value in asdict(self.indicator_config).items():
            print(f"  {key}: {value}")
        
        print("\nData Settings:")
        for key, value in asdict(self.data_config).items():
            print(f"  {key}: {value}")
        
        print("=" * 60)


# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reset_config():
    """Reset the global configuration manager."""
    global _config_manager
    _config_manager = None

__all__ = [
    'BacktestConfig', 'TechnicalIndicatorConfig', 'DataConfig',
    'ConfigManager', 'get_config', 'reset_config'
]