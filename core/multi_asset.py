"""
Multi-asset portfolio management and backtesting.

This module extends the backtesting framework to support portfolios
containing multiple assets with sophisticated allocation and rebalancing
strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
import warnings
from abc import ABC, abstractmethod

from .data_handler import StockData
from .strategy_base import Strategy
from .portfolio import Portfolio
from .exceptions import PortfolioError, ValidationError, validate_positive, validate_not_none
from .performance import timer


class AssetAllocation:
    """
    Represents asset allocation information for a multi-asset portfolio.
    """
    
    def __init__(self, allocations: Dict[str, float], rebalance_frequency: str = 'monthly'):
        """
        Initialize asset allocation.
        
        Args:
            allocations (Dict[str, float]): Asset symbols and their target weights
            rebalance_frequency (str): How often to rebalance ('daily', 'weekly', 'monthly', 'quarterly')
        """
        self.allocations = self._normalize_allocations(allocations)
        self.rebalance_frequency = rebalance_frequency
        self._validate_allocations()
    
    def _normalize_allocations(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Normalize allocations to sum to 1.0."""
        total = sum(allocations.values())
        if total == 0:
            raise ValidationError("Total allocation cannot be zero")
        return {asset: weight / total for asset, weight in allocations.items()}
    
    def _validate_allocations(self):
        """Validate allocation weights."""
        for asset, weight in self.allocations.items():
            if weight < 0:
                raise ValidationError(f"Allocation weight for {asset} cannot be negative: {weight}")
        
        total = sum(self.allocations.values())
        if not (0.99 <= total <= 1.01):  # Allow small rounding errors
            warnings.warn(f"Allocations sum to {total:.4f}, expected 1.0")
    
    def get_asset_symbols(self) -> List[str]:
        """Get list of asset symbols."""
        return list(self.allocations.keys())
    
    def get_weight(self, asset: str) -> float:
        """Get allocation weight for an asset."""
        return self.allocations.get(asset, 0.0)
    
    def update_allocation(self, asset: str, weight: float):
        """Update allocation for a specific asset."""
        self.allocations[asset] = weight
        self.allocations = self._normalize_allocations(self.allocations)
        self._validate_allocations()


class MultiAssetStrategy(Strategy):
    """
    Base class for multi-asset trading strategies.
    
    This class extends the single-asset Strategy to support multiple assets
    with individual signals and allocation decisions.
    """
    
    def __init__(self, name: str, allocation: AssetAllocation, 
                 stop_loss: float = 0.1, take_profit: float = 0.0):
        """
        Initialize multi-asset strategy.
        
        Args:
            name (str): Strategy name
            allocation (AssetAllocation): Asset allocation configuration
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
        """
        super().__init__(name, stop_loss, take_profit)
        self.allocation = allocation
        self.asset_data: Dict[str, pd.DataFrame] = {}
        self.asset_signals: Dict[str, pd.DataFrame] = {}
    
    def add_asset_data(self, asset: str, data: pd.DataFrame):
        """
        Add price data for an asset.
        
        Args:
            asset (str): Asset symbol
            data (pd.DataFrame): Price data
        """
        self.asset_data[asset] = data.copy()
    
    def run_strategy(self, start_date, end_date):
        """
        Run the multi-asset strategy.
        
        Args:
            start_date: Strategy start date
            end_date: Strategy end date
        """
        # Validate that we have data for all allocated assets
        missing_assets = set(self.allocation.get_asset_symbols()) - set(self.asset_data.keys())
        if missing_assets:
            raise ValidationError(f"Missing data for assets: {missing_assets}")
        
        # Generate signals for each asset
        for asset in self.allocation.get_asset_symbols():
            self.asset_signals[asset] = self.generate_asset_signals(
                asset, self.asset_data[asset], start_date, end_date
            )
        
        # Combine signals into portfolio-level trades
        self.trades = self.combine_asset_signals(start_date, end_date)
    
    @abstractmethod
    def generate_asset_signals(self, asset: str, data: pd.DataFrame, 
                             start_date, end_date) -> pd.DataFrame:
        """
        Generate trading signals for a specific asset.
        
        Args:
            asset (str): Asset symbol
            data (pd.DataFrame): Asset price data
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: Trading signals for the asset
        """
        pass
    
    def combine_asset_signals(self, start_date, end_date) -> pd.DataFrame:
        """
        Combine individual asset signals into portfolio-level trades.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: Combined trading signals
        """
        # Create a unified date index
        all_dates = set()
        for signals in self.asset_signals.values():
            all_dates.update(signals.index)
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        date_index = date_index[(date_index >= start_date) & (date_index <= end_date)]
        
        # Create combined trades DataFrame
        trades = pd.DataFrame(index=date_index)
        
        # Add allocation columns for each asset
        for asset in self.allocation.get_asset_symbols():
            trades[f'{asset}_Signal'] = 0
            trades[f'{asset}_Weight'] = self.allocation.get_weight(asset)
        
        # Fill in signals from individual assets
        for asset, signals in self.asset_signals.items():
            signal_col = f'{asset}_Signal'
            if 'Signal' in signals.columns:
                trades[signal_col] = signals['Signal'].reindex(date_index, fill_value=0)
            elif 'BSignal' in signals.columns:
                trades[signal_col] = (signals['BSignal'].reindex(date_index, fill_value=0) + 
                                    signals['SSignal'].reindex(date_index, fill_value=0))
        
        return trades


class MultiAssetPortfolio(Portfolio):
    """
    Portfolio implementation for multi-asset backtesting.
    
    This class manages multiple assets with proper allocation,
    rebalancing, and performance tracking.
    """
    
    def __init__(self, principal: float = 100000, trade_size: float = 1, 
                 pyramiding: int = 1, rebalance_threshold: float = 0.05):
        """
        Initialize multi-asset portfolio.
        
        Args:
            principal (float): Initial capital
            trade_size (float): Trade size (not used in multi-asset)
            pyramiding (int): Not used in multi-asset
            rebalance_threshold (float): Threshold for triggering rebalancing
        """
        super().__init__(principal, trade_size, pyramiding, 0)
        self.rebalance_threshold = rebalance_threshold
        self.asset_positions: Dict[str, float] = {}
        self.asset_values: Dict[str, float] = {}
        self.asset_data: Dict[str, pd.DataFrame] = {}
        self.rebalance_dates: List[datetime] = []
    
    def add_asset_data(self, asset: str, data: pd.DataFrame):
        """
        Add price data for an asset.
        
        Args:
            asset (str): Asset symbol
            data (pd.DataFrame): Price data
        """
        self.asset_data[asset] = data.copy()
        self.asset_positions[asset] = 0.0
        self.asset_values[asset] = 0.0
    
    @timer("multi_asset_backtest")
    def run_backtest(self, strategy: MultiAssetStrategy, start_date, end_date,
                    rebalance_frequency: str = 'monthly', **kwargs):
        """
        Run multi-asset backtest.
        
        Args:
            strategy (MultiAssetStrategy): Multi-asset strategy
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency (str): Rebalancing frequency
            **kwargs: Additional parameters
        """
        self.name = strategy.name
        self.rebalance_frequency = rebalance_frequency
        
        # Validate that we have data for all assets
        strategy_assets = set(strategy.allocation.get_asset_symbols())
        available_assets = set(self.asset_data.keys())
        
        if not strategy_assets.issubset(available_assets):
            missing = strategy_assets - available_assets
            raise PortfolioError(f"Missing asset data for: {missing}")
        
        # Create unified balance DataFrame
        self._create_unified_balance(strategy, start_date, end_date)
        
        # Run the backtest
        self._execute_multi_asset_backtest(strategy, start_date, end_date)
    
    def _create_unified_balance(self, strategy: MultiAssetStrategy, start_date, end_date):
        """Create unified balance DataFrame with all asset prices."""
        
        # Collect all dates from all assets
        all_dates = set()
        for asset_data in self.asset_data.values():
            asset_dates = asset_data.loc[start_date:end_date].index
            all_dates.update(asset_dates)
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        
        # Create balance DataFrame
        columns = ['Cash', 'Total']
        for asset in strategy.allocation.get_asset_symbols():
            columns.extend([f'{asset}_Price', f'{asset}_Shares', f'{asset}_Value'])
        
        self.balance = pd.DataFrame(index=date_index, columns=columns)
        self.balance = self.balance.fillna(0.0)
        
        # Fill in asset prices
        for asset in strategy.allocation.get_asset_symbols():
            asset_data = self.asset_data[asset].loc[start_date:end_date]
            price_col = f'{asset}_Price'
            
            if 'Close' in asset_data.columns:
                self.balance[price_col] = asset_data['Close'].reindex(date_index, method='ffill')
            else:
                raise PortfolioError(f"Missing 'Close' column for asset {asset}")
        
        # Initialize first row
        self.balance.loc[self.balance.index[0], 'Cash'] = self.principal
        self.balance.loc[self.balance.index[0], 'Total'] = self.principal
    
    def _execute_multi_asset_backtest(self, strategy: MultiAssetStrategy, start_date, end_date):
        """Execute the multi-asset backtesting logic."""
        
        # Initial allocation
        self._initial_allocation(strategy)
        
        # Process each day
        for i, (date, row) in enumerate(self.balance.iterrows()):
            if i == 0:
                continue  # Skip first day (already initialized)
            
            # Check for rebalancing
            if self._should_rebalance(date, strategy):
                self._rebalance_portfolio(date, strategy)
                self.rebalance_dates.append(date)
            
            # Process strategy signals
            self._process_strategy_signals(date, strategy, i)
            
            # Update portfolio values
            self._update_portfolio_values(date, strategy)
    
    def _initial_allocation(self, strategy: MultiAssetStrategy):
        """Perform initial portfolio allocation."""
        first_date = self.balance.index[0]
        available_cash = self.principal
        
        for asset in strategy.allocation.get_asset_symbols():
            target_weight = strategy.allocation.get_weight(asset)
            target_value = available_cash * target_weight
            
            price_col = f'{asset}_Price'
            shares_col = f'{asset}_Shares'
            value_col = f'{asset}_Value'
            
            asset_price = self.balance.loc[first_date, price_col]
            if asset_price > 0:
                shares = target_value / asset_price
                self.balance.loc[first_date, shares_col] = shares
                self.balance.loc[first_date, value_col] = shares * asset_price
                available_cash -= shares * asset_price
        
        self.balance.loc[first_date, 'Cash'] = available_cash
        self.balance.loc[first_date, 'Total'] = self.principal
    
    def _should_rebalance(self, date: datetime, strategy: MultiAssetStrategy) -> bool:
        """Determine if portfolio should be rebalanced."""
        if self.rebalance_frequency == 'daily':
            return True
        elif self.rebalance_frequency == 'weekly':
            return date.weekday() == 4  # Friday
        elif self.rebalance_frequency == 'monthly':
            return date.day == 1 or (date.month != (date - pd.Timedelta(days=1)).month)
        elif self.rebalance_frequency == 'quarterly':
            return date.month in [1, 4, 7, 10] and date.day == 1
        
        return False
    
    def _rebalance_portfolio(self, date: datetime, strategy: MultiAssetStrategy):
        """Rebalance portfolio to target allocations."""
        total_portfolio_value = self._calculate_total_value(date, strategy)
        
        # Calculate target values for each asset
        for asset in strategy.allocation.get_asset_symbols():
            target_weight = strategy.allocation.get_weight(asset)
            target_value = total_portfolio_value * target_weight
            
            price_col = f'{asset}_Price'
            shares_col = f'{asset}_Shares'
            value_col = f'{asset}_Value'
            
            current_price = self.balance.loc[date, price_col]
            if current_price > 0:
                target_shares = target_value / current_price
                self.balance.loc[date, shares_col] = target_shares
                self.balance.loc[date, value_col] = target_shares * current_price
        
        # Update cash position
        total_asset_value = sum(self.balance.loc[date, f'{asset}_Value'] 
                              for asset in strategy.allocation.get_asset_symbols())
        self.balance.loc[date, 'Cash'] = total_portfolio_value - total_asset_value
    
    def _process_strategy_signals(self, date: datetime, strategy: MultiAssetStrategy, row_index: int):
        """Process trading signals from the strategy."""
        if strategy.trades is not None and date in strategy.trades.index:
            signals = strategy.trades.loc[date]
            
            # Process signals for each asset
            for asset in strategy.allocation.get_asset_symbols():
                signal_col = f'{asset}_Signal'
                if signal_col in signals and signals[signal_col] != 0:
                    # For now, signals don't override rebalancing in multi-asset
                    # This could be extended to handle more complex signal processing
                    pass
    
    def _update_portfolio_values(self, date: datetime, strategy: MultiAssetStrategy):
        """Update portfolio values based on current prices."""
        # Carry forward positions from previous day if not explicitly set
        prev_date = self.balance.index[self.balance.index.get_loc(date) - 1]
        
        for asset in strategy.allocation.get_asset_symbols():
            shares_col = f'{asset}_Shares'
            value_col = f'{asset}_Value'
            price_col = f'{asset}_Price'
            
            # Carry forward shares if not updated
            if self.balance.loc[date, shares_col] == 0:
                self.balance.loc[date, shares_col] = self.balance.loc[prev_date, shares_col]
            
            # Update value based on current price
            current_shares = self.balance.loc[date, shares_col]
            current_price = self.balance.loc[date, price_col]
            self.balance.loc[date, value_col] = current_shares * current_price
        
        # Update total portfolio value
        total_asset_value = sum(self.balance.loc[date, f'{asset}_Value'] 
                              for asset in strategy.allocation.get_asset_symbols())
        
        # Carry forward cash if not updated
        if self.balance.loc[date, 'Cash'] == 0:
            self.balance.loc[date, 'Cash'] = self.balance.loc[prev_date, 'Cash']
        
        self.balance.loc[date, 'Total'] = self.balance.loc[date, 'Cash'] + total_asset_value
    
    def _calculate_total_value(self, date: datetime, strategy: MultiAssetStrategy) -> float:
        """Calculate total portfolio value on a specific date."""
        total_value = self.balance.loc[date, 'Cash']
        
        for asset in strategy.allocation.get_asset_symbols():
            shares_col = f'{asset}_Shares'
            price_col = f'{asset}_Price'
            
            shares = self.balance.loc[date, shares_col]
            price = self.balance.loc[date, price_col]
            total_value += shares * price
        
        return total_value
    
    def get_asset_performance(self, asset: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific asset.
        
        Args:
            asset (str): Asset symbol
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        value_col = f'{asset}_Value'
        if value_col not in self.balance.columns:
            raise ValidationError(f"Asset {asset} not found in portfolio")
        
        asset_values = self.balance[value_col]
        initial_value = asset_values.iloc[0] if asset_values.iloc[0] > 0 else asset_values[asset_values > 0].iloc[0]
        final_value = asset_values.iloc[-1]
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': (final_value - initial_value) / initial_value if initial_value > 0 else 0,
            'contribution_to_portfolio': final_value / self.balance['Total'].iloc[-1]
        }
    
    def get_allocation_drift(self) -> pd.DataFrame:
        """
        Calculate how actual allocations drifted from targets over time.
        
        Returns:
            pd.DataFrame: Allocation drift over time
        """
        drift_data = {}
        
        for asset in self.balance.columns:
            if asset.endswith('_Value') and not asset.startswith('Total'):
                asset_name = asset.replace('_Value', '')
                asset_weights = self.balance[asset] / self.balance['Total']
                drift_data[f'{asset_name}_Weight'] = asset_weights
        
        return pd.DataFrame(drift_data, index=self.balance.index)
    
    def performance_summary(self, verbose: bool = True):
        """
        Calculate and display performance metrics for multi-asset portfolio.
        
        Args:
            verbose (bool): Whether to print detailed summary
            
        Returns:
            Performance summary result
        """
        if len(self.balance) == 0:
            return {}
        
        return super().performance_summary(verbose=verbose)


class BuyAndHoldMultiAsset(MultiAssetStrategy):
    """
    Simple buy-and-hold strategy for multiple assets.
    
    This strategy maintains the target allocation and rebalances
    according to the specified frequency.
    """
    
    def __init__(self, allocation: AssetAllocation):
        super().__init__("BuyAndHoldMultiAsset", allocation)
    
    def generate_asset_signals(self, asset: str, data: pd.DataFrame, 
                             start_date, end_date) -> pd.DataFrame:
        """Generate buy-and-hold signals (always hold)."""
        asset_data = data.loc[start_date:end_date].copy()
        asset_data['Signal'] = 1  # Always hold
        return asset_data[['Signal']]


__all__ = [
    'AssetAllocation', 'MultiAssetStrategy', 'MultiAssetPortfolio', 
    'BuyAndHoldMultiAsset'
]