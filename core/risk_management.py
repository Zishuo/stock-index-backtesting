"""
Risk management systems and controls for trading strategies.

This module provides comprehensive risk management tools including
position sizing, stop losses, portfolio risk controls, and real-time
risk monitoring capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings

from .exceptions import PortfolioError, ValidationError, validate_positive, validate_not_none
from .advanced_metrics import RiskMetrics


class RiskControl(ABC):
    """
    Abstract base class for risk control mechanisms.
    
    Risk controls can be applied at the strategy, position, or portfolio level
    to manage various types of risk.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the risk control."""
        pass
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the risk control given the current context.
        
        Args:
            context (Dict[str, Any]): Current market and portfolio context
            
        Returns:
            Dict[str, Any]: Risk control result with action and details
        """
        pass


class PositionSizer(ABC):
    """
    Abstract base class for position sizing algorithms.
    
    Position sizers determine how much capital to allocate to each trade
    based on risk parameters and portfolio constraints.
    """
    
    @abstractmethod
    def calculate_position_size(self, context: Dict[str, Any]) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            context (Dict[str, Any]): Trading context including price, volatility, etc.
            
        Returns:
            float: Position size as a fraction of portfolio or absolute amount
        """
        pass


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction position sizer.
    
    Allocates a fixed percentage of portfolio to each position.
    """
    
    def __init__(self, fraction: float = 0.1):
        """
        Initialize fixed fraction sizer.
        
        Args:
            fraction (float): Fraction of portfolio to allocate (0-1)
        """
        self.fraction = validate_positive(fraction, "fraction")
        if fraction > 1.0:
            warnings.warn(f"Fraction {fraction} > 1.0, consider using values < 1")
    
    def calculate_position_size(self, context: Dict[str, Any]) -> float:
        """Calculate position size as fixed fraction of portfolio."""
        return self.fraction


class VolatilityTargetSizer(PositionSizer):
    """
    Volatility targeting position sizer.
    
    Adjusts position size to target a specific level of volatility contribution.
    """
    
    def __init__(self, target_volatility: float = 0.1, lookback_window: int = 20):
        """
        Initialize volatility target sizer.
        
        Args:
            target_volatility (float): Target annualized volatility
            lookback_window (int): Lookback window for volatility estimation
        """
        self.target_volatility = validate_positive(target_volatility, "target_volatility")
        self.lookback_window = validate_positive(lookback_window, "lookback_window")
    
    def calculate_position_size(self, context: Dict[str, Any]) -> float:
        """Calculate position size based on volatility target."""
        if 'returns' not in context:
            raise ValidationError("Volatility sizer requires 'returns' in context")
        
        returns = context['returns']
        if len(returns) < self.lookback_window:
            return 0.1  # Default small position if insufficient data
        
        # Calculate realized volatility
        recent_returns = returns.tail(self.lookback_window)
        realized_vol = recent_returns.std() * np.sqrt(252)  # Annualized
        
        if realized_vol == 0:
            return 0.1
        
        # Scale position to target volatility
        position_size = self.target_volatility / realized_vol
        return min(max(position_size, 0.01), 1.0)  # Clamp between 1% and 100%


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion position sizer.
    
    Calculates optimal position size based on win rate and average win/loss.
    """
    
    def __init__(self, lookback_window: int = 50, max_kelly: float = 0.25):
        """
        Initialize Kelly criterion sizer.
        
        Args:
            lookback_window (int): Number of trades to analyze
            max_kelly (float): Maximum Kelly fraction to prevent over-leveraging
        """
        self.lookback_window = validate_positive(lookback_window, "lookback_window")
        self.max_kelly = validate_positive(max_kelly, "max_kelly")
    
    def calculate_position_size(self, context: Dict[str, Any]) -> float:
        """Calculate Kelly optimal position size."""
        if 'trade_results' not in context:
            return 0.1  # Default if no trade history
        
        trade_results = context['trade_results']
        if len(trade_results) < 10:  # Need minimum trades
            return 0.1
        
        recent_trades = trade_results.tail(self.lookback_window)
        
        # Calculate win rate and average win/loss
        wins = recent_trades[recent_trades > 0]
        losses = recent_trades[recent_trades < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.1
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return self.max_kelly
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply maximum Kelly limit and ensure non-negative
        return min(max(kelly_fraction, 0), self.max_kelly)


class StopLossControl(RiskControl):
    """
    Stop loss risk control.
    
    Monitors positions for stop loss triggers based on price movements
    or portfolio drawdown.
    """
    
    def __init__(self, stop_loss_pct: float = 0.1, trailing_stop: bool = False):
        """
        Initialize stop loss control.
        
        Args:
            stop_loss_pct (float): Stop loss percentage (0-1)
            trailing_stop (bool): Whether to use trailing stop loss
        """
        self.stop_loss_pct = validate_positive(stop_loss_pct, "stop_loss_pct")
        self.trailing_stop = trailing_stop
        self._highest_value = None
    
    @property
    def name(self) -> str:
        return f"StopLoss_{self.stop_loss_pct:.1%}"
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate stop loss conditions."""
        current_price = context.get('current_price')
        entry_price = context.get('entry_price')
        current_value = context.get('current_value')
        
        if current_price is None or entry_price is None:
            return {'action': 'hold', 'reason': 'insufficient_data'}
        
        # Calculate current return
        current_return = (current_price - entry_price) / entry_price
        
        if self.trailing_stop:
            # Trailing stop logic
            if self._highest_value is None or current_value > self._highest_value:
                self._highest_value = current_value
            
            drawdown_from_peak = (current_value - self._highest_value) / self._highest_value
            
            if drawdown_from_peak <= -self.stop_loss_pct:
                return {
                    'action': 'sell',
                    'reason': 'trailing_stop_triggered',
                    'drawdown': drawdown_from_peak,
                    'trigger_level': -self.stop_loss_pct
                }
        else:
            # Fixed stop loss
            if current_return <= -self.stop_loss_pct:
                return {
                    'action': 'sell',
                    'reason': 'stop_loss_triggered',
                    'current_return': current_return,
                    'trigger_level': -self.stop_loss_pct
                }
        
        return {'action': 'hold', 'current_return': current_return}


class TakeProfitControl(RiskControl):
    """
    Take profit risk control.
    
    Monitors positions for take profit opportunities.
    """
    
    def __init__(self, take_profit_pct: float = 0.2):
        """
        Initialize take profit control.
        
        Args:
            take_profit_pct (float): Take profit percentage (0-1)
        """
        self.take_profit_pct = validate_positive(take_profit_pct, "take_profit_pct")
    
    @property
    def name(self) -> str:
        return f"TakeProfit_{self.take_profit_pct:.1%}"
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate take profit conditions."""
        current_price = context.get('current_price')
        entry_price = context.get('entry_price')
        
        if current_price is None or entry_price is None:
            return {'action': 'hold', 'reason': 'insufficient_data'}
        
        current_return = (current_price - entry_price) / entry_price
        
        if current_return >= self.take_profit_pct:
            return {
                'action': 'sell',
                'reason': 'take_profit_triggered',
                'current_return': current_return,
                'trigger_level': self.take_profit_pct
            }
        
        return {'action': 'hold', 'current_return': current_return}


class DrawdownControl(RiskControl):
    """
    Portfolio-level drawdown control.
    
    Monitors overall portfolio drawdown and can halt trading
    if drawdown exceeds limits.
    """
    
    def __init__(self, max_drawdown: float = 0.2, lookback_window: int = 252):
        """
        Initialize drawdown control.
        
        Args:
            max_drawdown (float): Maximum allowed drawdown
            lookback_window (int): Lookback window for drawdown calculation
        """
        self.max_drawdown = validate_positive(max_drawdown, "max_drawdown")
        self.lookback_window = validate_positive(lookback_window, "lookback_window")
    
    @property
    def name(self) -> str:
        return f"DrawdownControl_{self.max_drawdown:.1%}"
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate portfolio drawdown."""
        portfolio_values = context.get('portfolio_values')
        
        if portfolio_values is None or len(portfolio_values) < 2:
            return {'action': 'continue', 'reason': 'insufficient_data'}
        
        # Calculate rolling maximum and current drawdown
        recent_values = portfolio_values.tail(self.lookback_window)
        running_max = recent_values.expanding().max()
        current_drawdown = (recent_values.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        
        if current_drawdown <= -self.max_drawdown:
            return {
                'action': 'halt_trading',
                'reason': 'max_drawdown_exceeded',
                'current_drawdown': current_drawdown,
                'max_allowed': -self.max_drawdown
            }
        
        return {
            'action': 'continue',
            'current_drawdown': current_drawdown,
            'max_allowed': -self.max_drawdown
        }


class VolatilityControl(RiskControl):
    """
    Volatility-based risk control.
    
    Monitors portfolio volatility and adjusts position sizes or
    halts trading if volatility exceeds limits.
    """
    
    def __init__(self, max_volatility: float = 0.3, lookback_window: int = 20):
        """
        Initialize volatility control.
        
        Args:
            max_volatility (float): Maximum allowed annualized volatility
            lookback_window (int): Lookback window for volatility calculation
        """
        self.max_volatility = validate_positive(max_volatility, "max_volatility")
        self.lookback_window = validate_positive(lookback_window, "lookback_window")
    
    @property
    def name(self) -> str:
        return f"VolatilityControl_{self.max_volatility:.1%}"
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate portfolio volatility."""
        returns = context.get('returns')
        
        if returns is None or len(returns) < self.lookback_window:
            return {'action': 'continue', 'reason': 'insufficient_data'}
        
        # Calculate realized volatility
        recent_returns = returns.tail(self.lookback_window)
        realized_vol = recent_returns.std() * np.sqrt(252)  # Annualized
        
        scaling_factor = 1.0
        if realized_vol > self.max_volatility:
            scaling_factor = self.max_volatility / realized_vol
            
            return {
                'action': 'scale_positions',
                'reason': 'volatility_exceeded',
                'scaling_factor': scaling_factor,
                'current_volatility': realized_vol,
                'max_allowed': self.max_volatility
            }
        
        return {
            'action': 'continue',
            'current_volatility': realized_vol,
            'max_allowed': self.max_volatility
        }


class RiskManager:
    """
    Central risk management system.
    
    Coordinates multiple risk controls and position sizing algorithms
    to provide comprehensive risk management for trading strategies.
    """
    
    def __init__(self, position_sizer: Optional[PositionSizer] = None):
        """
        Initialize risk manager.
        
        Args:
            position_sizer (PositionSizer, optional): Position sizing algorithm
        """
        self.position_sizer = position_sizer or FixedFractionSizer(0.1)
        self.risk_controls: List[RiskControl] = []
        self.risk_metrics_history: List[Dict[str, Any]] = []
        self.active_positions: Dict[str, Dict[str, Any]] = {}
    
    def add_risk_control(self, control: RiskControl):
        """
        Add a risk control to the manager.
        
        Args:
            control (RiskControl): Risk control to add
        """
        if not isinstance(control, RiskControl):
            raise ValidationError("Control must inherit from RiskControl")
        
        self.risk_controls.append(control)
    
    def calculate_position_size(self, context: Dict[str, Any]) -> float:
        """
        Calculate position size using the configured position sizer.
        
        Args:
            context (Dict[str, Any]): Trading context
            
        Returns:
            float: Recommended position size
        """
        return self.position_sizer.calculate_position_size(context)
    
    def evaluate_risk_controls(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all risk controls and return aggregated results.
        
        Args:
            context (Dict[str, Any]): Current market and portfolio context
            
        Returns:
            Dict[str, Any]: Aggregated risk control results
        """
        results = {
            'timestamp': datetime.now(),
            'controls': {},
            'overall_action': 'continue',
            'risk_level': 'low'
        }
        
        # Evaluate each risk control
        for control in self.risk_controls:
            try:
                control_result = control.evaluate(context)
                results['controls'][control.name] = control_result
                
                # Update overall action based on control results
                if control_result.get('action') in ['sell', 'halt_trading']:
                    results['overall_action'] = control_result['action']
                    results['risk_level'] = 'high'
                elif control_result.get('action') == 'scale_positions':
                    if results['overall_action'] == 'continue':
                        results['overall_action'] = 'scale_positions'
                    results['risk_level'] = 'medium'
                
            except Exception as e:
                results['controls'][control.name] = {
                    'action': 'error',
                    'error': str(e)
                }
        
        # Store results for historical analysis
        self.risk_metrics_history.append(results)
        
        return results
    
    def update_position(self, symbol: str, entry_price: float, current_price: float,
                       quantity: float, entry_date: datetime):
        """
        Update position information for risk monitoring.
        
        Args:
            symbol (str): Asset symbol
            entry_price (float): Entry price
            current_price (float): Current price
            quantity (float): Position quantity
            entry_date (datetime): Entry date
        """
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'current_price': current_price,
            'quantity': quantity,
            'entry_date': entry_date,
            'current_value': current_price * quantity,
            'unrealized_pnl': (current_price - entry_price) * quantity
        }
    
    def close_position(self, symbol: str):
        """
        Remove a position from active monitoring.
        
        Args:
            symbol (str): Asset symbol to close
        """
        if symbol in self.active_positions:
            del self.active_positions[symbol]
    
    def get_portfolio_risk_metrics(self, portfolio_values: pd.Series,
                                 returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_values (pd.Series): Portfolio value time series
            returns (pd.Series): Portfolio return time series
            
        Returns:
            Dict[str, Any]: Portfolio risk metrics
        """
        if len(returns) < 10:
            return {'error': 'Insufficient data for risk calculations'}
        
        # Basic risk metrics
        current_value = portfolio_values.iloc[-1]
        peak_value = portfolio_values.max()
        current_drawdown = (current_value - peak_value) / peak_value
        
        # Volatility metrics
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # VaR calculations
        var_95 = RiskMetrics.calculate_var(returns, 0.05)
        var_99 = RiskMetrics.calculate_var(returns, 0.01)
        cvar_95 = RiskMetrics.calculate_cvar(returns, 0.05)
        
        # Risk-adjusted performance
        sharpe_ratio = returns.mean() / daily_vol if daily_vol > 0 else 0
        
        return {
            'current_drawdown': current_drawdown,
            'annualized_volatility': annualized_vol,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'sharpe_ratio': sharpe_ratio,
            'active_positions': len(self.active_positions),
            'total_unrealized_pnl': sum(pos['unrealized_pnl'] 
                                      for pos in self.active_positions.values())
        }
    
    def generate_risk_report(self, portfolio_values: pd.Series,
                           returns: pd.Series) -> str:
        """
        Generate a comprehensive risk report.
        
        Args:
            portfolio_values (pd.Series): Portfolio values
            returns (pd.Series): Portfolio returns
            
        Returns:
            str: Formatted risk report
        """
        metrics = self.get_portfolio_risk_metrics(portfolio_values, returns)
        
        report = f"""
RISK MANAGEMENT REPORT
{'='*50}

Portfolio Risk Metrics:
  Current Drawdown: {metrics.get('current_drawdown', 0):.2%}
  Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}
  VaR (95%): {metrics.get('var_95', 0):.2%}
  VaR (99%): {metrics.get('var_99', 0):.2%}
  CVaR (95%): {metrics.get('cvar_95', 0):.2%}
  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}

Active Positions: {metrics.get('active_positions', 0)}
Total Unrealized P&L: ${metrics.get('total_unrealized_pnl', 0):,.2f}

Risk Controls Active: {len(self.risk_controls)}
"""
        
        # Add individual risk control status
        if self.risk_metrics_history:
            latest_evaluation = self.risk_metrics_history[-1]
            report += f"\nRisk Control Status:\n"
            for control_name, result in latest_evaluation['controls'].items():
                action = result.get('action', 'unknown')
                report += f"  {control_name}: {action}\n"
            
            report += f"\nOverall Risk Level: {latest_evaluation['risk_level'].upper()}\n"
            report += f"Recommended Action: {latest_evaluation['overall_action'].upper()}\n"
        
        report += "="*50
        
        return report


__all__ = [
    'RiskControl', 'PositionSizer', 'RiskManager',
    'FixedFractionSizer', 'VolatilityTargetSizer', 'KellyCriterionSizer',
    'StopLossControl', 'TakeProfitControl', 'DrawdownControl', 'VolatilityControl'
]