"""
Core backtesting framework components.

This module provides the fundamental classes and utilities for financial backtesting,
including data handling, technical indicators, strategies, and portfolio management.
"""

# Import all core components to maintain compatibility
from .data_handler import StockData
from .indicators import get_sma, get_ema, get_stochastic
from .strategy_base import Strategy
from .portfolio import Portfolio, BackTest

# Phase 2 enhancements
from .strategy_factory import StrategyFactory, create_strategy
from .config import ConfigManager, BacktestConfig, TechnicalIndicatorConfig, DataConfig, get_config
from .validators import DataValidator, ParameterValidator, ValidationError
from .exceptions import (
    BacktestingError, DataError, StrategyError, PortfolioError,
    ConfigurationError, IndicatorError, NetworkError,
    ErrorHandler, get_error_handler
)
from .performance import (
    PerformanceMonitor, get_performance_monitor, timer, cached,
    DataFrameOptimizer, ParallelProcessor
)

# Phase 3 advanced features
from .plugins import (
    PluginManager, IndicatorPlugin, StrategyPlugin, IndicatorComposer,
    get_plugin_manager, register_indicator, register_strategy, calculate_custom_indicator
)
from .multi_asset import AssetAllocation, MultiAssetStrategy, MultiAssetPortfolio, BuyAndHoldMultiAsset
from .advanced_metrics import AdvancedMetrics, RiskMetrics, PerformanceAttribution, calculate_portfolio_metrics
from .risk_management import (
    RiskManager, RiskControl, PositionSizer,
    FixedFractionSizer, VolatilityTargetSizer, KellyCriterionSizer,
    StopLossControl, TakeProfitControl, DrawdownControl, VolatilityControl
)
from .parallel_processing import ParameterGrid, ParallelOptimizer, BacktestJob, BacktestResult

__all__ = [
    # Core Phase 1 components
    'StockData',
    'get_sma', 'get_ema', 'get_stochastic',
    'Strategy',
    'Portfolio', 'BackTest',
    
    # Phase 2 enhancements
    'StrategyFactory', 'create_strategy',
    'ConfigManager', 'BacktestConfig', 'TechnicalIndicatorConfig', 'DataConfig', 'get_config',
    'DataValidator', 'ParameterValidator', 'ValidationError',
    'BacktestingError', 'DataError', 'StrategyError', 'PortfolioError',
    'ConfigurationError', 'IndicatorError', 'NetworkError',
    'ErrorHandler', 'get_error_handler',
    'PerformanceMonitor', 'get_performance_monitor', 'timer', 'cached',
    'DataFrameOptimizer', 'ParallelProcessor',
    
    # Phase 3 advanced features
    'PluginManager', 'IndicatorPlugin', 'StrategyPlugin', 'IndicatorComposer',
    'get_plugin_manager', 'register_indicator', 'register_strategy', 'calculate_custom_indicator',
    'AssetAllocation', 'MultiAssetStrategy', 'MultiAssetPortfolio', 'BuyAndHoldMultiAsset',
    'AdvancedMetrics', 'RiskMetrics', 'PerformanceAttribution', 'calculate_portfolio_metrics',
    'RiskManager', 'RiskControl', 'PositionSizer',
    'FixedFractionSizer', 'VolatilityTargetSizer', 'KellyCriterionSizer',
    'StopLossControl', 'TakeProfitControl', 'DrawdownControl', 'VolatilityControl',
    'ParameterGrid', 'ParallelOptimizer', 'BacktestJob', 'BacktestResult'
]