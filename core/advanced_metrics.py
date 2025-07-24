"""
Advanced performance metrics and risk analysis.

This module provides sophisticated performance measurement tools including
risk-adjusted returns, drawdown analysis, and advanced statistical metrics
for comprehensive strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import warnings
from scipy import stats
from sklearn.metrics import r2_score

from .exceptions import ValidationError, validate_positive


class AdvancedMetrics:
    """
    Advanced performance metrics calculator.
    
    This class provides comprehensive performance analysis including
    risk metrics, drawdown analysis, and statistical measures.
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize advanced metrics calculator.
        
        Args:
            returns (pd.Series): Strategy returns
            benchmark_returns (pd.Series, optional): Benchmark returns for comparison
            risk_free_rate (float): Annual risk-free rate
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
        
        if len(self.returns) == 0:
            raise ValidationError("Returns series cannot be empty")
        
        # Align benchmark returns if provided
        if self.benchmark_returns is not None:
            common_dates = self.returns.index.intersection(self.benchmark_returns.index)
            if len(common_dates) == 0:
                warnings.warn("No common dates between returns and benchmark")
                self.benchmark_returns = None
            else:
                self.returns = self.returns.loc[common_dates]
                self.benchmark_returns = self.benchmark_returns.loc[common_dates]
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all available performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.basic_metrics())
        
        # Risk metrics
        metrics.update(self.risk_metrics())
        
        # Drawdown metrics
        metrics.update(self.drawdown_metrics())
        
        # Higher moment metrics
        metrics.update(self.higher_moment_metrics())
        
        # Benchmark comparison metrics (if benchmark available)
        if self.benchmark_returns is not None:
            metrics.update(self.benchmark_metrics())
        
        # Rolling metrics
        metrics.update(self.rolling_metrics())
        
        return metrics
    
    def basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        total_return = (1 + self.returns).prod() - 1
        n_periods = len(self.returns)
        n_years = n_periods / self.trading_days_per_year
        
        # Annualized return
        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = 0
        
        # Volatility
        annualized_volatility = self.returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Win rate
        positive_returns = (self.returns > 0).sum()
        win_rate = positive_returns / len(self.returns)
        
        # Average win/loss
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = -losses.sum() if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trading_periods': n_periods,
            'years': n_years
        }
    
    def risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-based performance metrics."""
        # Sortino ratio (downside deviation)
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year)
        
        annualized_return = (1 + ((1 + self.returns).prod() - 1)) ** (
            self.trading_days_per_year / len(self.returns)) - 1
        excess_return = annualized_return - self.risk_free_rate
        
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        max_dd = self.calculate_max_drawdown()
        calmar_ratio = annualized_return / abs(max_dd) if max_dd < 0 else np.inf
        
        # Value at Risk (VaR)
        var_95 = np.percentile(self.returns, 5)
        var_99 = np.percentile(self.returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = self.returns[self.returns <= var_95].mean()
        cvar_99 = self.returns[self.returns <= var_99].mean()
        
        # Tail ratio
        tail_ratio = abs(np.percentile(self.returns, 95)) / abs(np.percentile(self.returns, 5))
        
        return {
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'tail_ratio': tail_ratio
        }
    
    def drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        # Calculate drawdown series
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Average drawdown
        drawdown_periods = drawdown[drawdown < 0]
        avg_drawdown = drawdown_periods.mean() if len(drawdown_periods) > 0 else 0
        
        # Drawdown duration analysis
        dd_durations = self._calculate_drawdown_durations(drawdown)
        max_dd_duration = max(dd_durations) if dd_durations else 0
        avg_dd_duration = np.mean(dd_durations) if dd_durations else 0
        
        # Recovery factor
        total_return = (1 + self.returns).prod() - 1
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_dd_date,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'avg_drawdown_duration': avg_dd_duration,
            'recovery_factor': recovery_factor,
            'drawdown_periods': len(dd_durations)
        }
    
    def higher_moment_metrics(self) -> Dict[str, float]:
        """Calculate higher moment statistics."""
        # Skewness
        skewness = stats.skew(self.returns)
        
        # Kurtosis
        kurtosis = stats.kurtosis(self.returns)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal_distribution': jb_pvalue > 0.05
        }
    
    def benchmark_metrics(self) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        if self.benchmark_returns is None:
            return {}
        
        # Information ratio
        excess_returns = self.returns - self.benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)
        information_ratio = excess_returns.mean() * self.trading_days_per_year / tracking_error if tracking_error > 0 else 0
        
        # Beta
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (Jensen's alpha)
        strategy_return = self.returns.mean() * self.trading_days_per_year
        benchmark_return = self.benchmark_returns.mean() * self.trading_days_per_year
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Treynor ratio
        excess_return = strategy_return - self.risk_free_rate
        treynor_ratio = excess_return / beta if beta > 0 else 0
        
        # R-squared
        try:
            r_squared = r2_score(self.returns, self.benchmark_returns)
        except:
            r_squared = 0
        
        # Correlation
        correlation = self.returns.corr(self.benchmark_returns)
        
        # Up/Down capture ratios
        up_markets = self.benchmark_returns > 0
        down_markets = self.benchmark_returns < 0
        
        up_capture = (self.returns[up_markets].mean() / self.benchmark_returns[up_markets].mean() 
                     if up_markets.sum() > 0 and self.benchmark_returns[up_markets].mean() != 0 else 0)
        down_capture = (self.returns[down_markets].mean() / self.benchmark_returns[down_markets].mean() 
                       if down_markets.sum() > 0 and self.benchmark_returns[down_markets].mean() != 0 else 0)
        
        return {
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha,
            'treynor_ratio': treynor_ratio,
            'r_squared': r_squared,
            'correlation': correlation,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture
        }
    
    def rolling_metrics(self, window: int = 252) -> Dict[str, Any]:
        """Calculate rolling performance metrics."""
        if len(self.returns) < window:
            return {}
        
        # Rolling Sharpe ratio
        rolling_returns = self.returns.rolling(window=window).mean() * self.trading_days_per_year
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(self.trading_days_per_year)
        rolling_sharpe = (rolling_returns - self.risk_free_rate) / rolling_vol
        
        # Rolling maximum drawdown
        rolling_cumulative = (1 + self.returns).rolling(window=window).apply(lambda x: x.prod(), raw=False)
        rolling_max = rolling_cumulative.rolling(window=window).max()
        rolling_dd = (rolling_cumulative - rolling_max) / rolling_max
        rolling_max_dd = rolling_dd.rolling(window=window).min()
        
        return {
            'rolling_sharpe_mean': rolling_sharpe.mean(),
            'rolling_sharpe_std': rolling_sharpe.std(),
            'rolling_max_dd_mean': rolling_max_dd.mean(),
            'rolling_max_dd_std': rolling_max_dd.std(),
            'rolling_sharpe_series': rolling_sharpe,
            'rolling_max_dd_series': rolling_max_dd
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_drawdown_durations(self, drawdown: pd.Series) -> List[int]:
        """Calculate the duration of each drawdown period."""
        durations = []
        in_drawdown = False
        duration = 0
        
        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    duration = 1
                else:
                    duration += 1
            else:
                if in_drawdown:
                    durations.append(duration)
                    in_drawdown = False
                    duration = 0
        
        # Handle case where series ends in drawdown
        if in_drawdown:
            durations.append(duration)
        
        return durations
    
    def monte_carlo_analysis(self, n_simulations: int = 1000, 
                           confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis of returns.
        
        Args:
            n_simulations (int): Number of Monte Carlo simulations
            confidence_levels (List[float]): Confidence levels for analysis
            
        Returns:
            Dict[str, Any]: Monte Carlo analysis results
        """
        # Bootstrap returns
        n_periods = len(self.returns)
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Randomly sample returns with replacement
            bootstrap_sample = np.random.choice(self.returns, size=n_periods, replace=True)
            total_return = (1 + pd.Series(bootstrap_sample)).prod() - 1
            simulated_returns.append(total_return)
        
        simulated_returns = np.array(simulated_returns)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in confidence_levels:
            ci_lower = np.percentile(simulated_returns, level * 100)
            ci_upper = np.percentile(simulated_returns, (1 - level) * 100)
            confidence_intervals[f'ci_{int(level*100)}'] = (ci_lower, ci_upper)
        
        return {
            'simulated_returns': simulated_returns,
            'mean_simulated_return': simulated_returns.mean(),
            'std_simulated_return': simulated_returns.std(),
            'confidence_intervals': confidence_intervals,
            'probability_of_loss': (simulated_returns < 0).mean()
        }


class RiskMetrics:
    """
    Specialized risk measurement and analysis tools.
    """
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            float: VaR value
        """
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level
            
        Returns:
            float: CVaR value
        """
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
        """
        Calculate risk parity portfolio weights.
        
        Args:
            returns (pd.DataFrame): Multi-asset returns
            
        Returns:
            pd.Series: Risk parity weights
        """
        # Calculate inverse volatility weights
        volatilities = returns.std()
        inv_vol_weights = 1 / volatilities
        risk_parity_weights = inv_vol_weights / inv_vol_weights.sum()
        
        return risk_parity_weights
    
    @staticmethod
    def calculate_maximum_diversification_weights(returns: pd.DataFrame) -> pd.Series:
        """
        Calculate maximum diversification portfolio weights.
        
        Args:
            returns (pd.DataFrame): Multi-asset returns
            
        Returns:
            pd.Series: Maximum diversification weights
        """
        # This is a simplified version - full implementation would use optimization
        correlation_matrix = returns.corr()
        volatilities = returns.std()
        
        # Inverse correlation weighted by volatility
        inv_corr_sum = (1 / correlation_matrix).sum(axis=1)
        weights = (inv_corr_sum / volatilities) / (inv_corr_sum / volatilities).sum()
        
        return weights


class PerformanceAttribution:
    """
    Performance attribution analysis for multi-asset portfolios.
    """
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                 asset_weights: pd.DataFrame, benchmark_weights: pd.Series):
        """
        Initialize performance attribution analysis.
        
        Args:
            portfolio_returns (pd.Series): Portfolio returns
            benchmark_returns (pd.Series): Benchmark returns
            asset_weights (pd.DataFrame): Asset weights over time
            benchmark_weights (pd.Series): Benchmark asset weights
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.asset_weights = asset_weights
        self.benchmark_weights = benchmark_weights
    
    def brinson_attribution(self) -> Dict[str, pd.Series]:
        """
        Perform Brinson performance attribution.
        
        Returns:
            Dict[str, pd.Series]: Attribution components
        """
        # This is a simplified implementation
        # Full Brinson attribution requires individual asset returns
        
        active_return = self.portfolio_returns - self.benchmark_returns
        
        return {
            'total_active_return': active_return,
            'asset_allocation_effect': pd.Series(index=active_return.index, dtype=float),
            'stock_selection_effect': pd.Series(index=active_return.index, dtype=float),
            'interaction_effect': pd.Series(index=active_return.index, dtype=float)
        }


def calculate_portfolio_metrics(balance_df: pd.DataFrame, 
                              benchmark_data: Optional[pd.DataFrame] = None,
                              risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a portfolio balance DataFrame.
    
    Args:
        balance_df (pd.DataFrame): Portfolio balance with 'Total' column
        benchmark_data (pd.DataFrame, optional): Benchmark price data
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        Dict[str, Any]: Comprehensive performance metrics
    """
    if 'Total' not in balance_df.columns:
        raise ValidationError("Balance DataFrame must contain 'Total' column")
    
    # Calculate returns
    portfolio_values = balance_df['Total']
    returns = portfolio_values.pct_change().dropna()
    
    # Calculate benchmark returns if provided
    benchmark_returns = None
    if benchmark_data is not None and 'Close' in benchmark_data.columns:
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
    
    # Create metrics calculator
    metrics_calc = AdvancedMetrics(returns, benchmark_returns, risk_free_rate)
    
    # Calculate all metrics
    all_metrics = metrics_calc.calculate_all_metrics()
    
    # Add portfolio-specific metrics
    all_metrics.update({
        'initial_value': portfolio_values.iloc[0],
        'final_value': portfolio_values.iloc[-1],
        'peak_value': portfolio_values.max(),
        'trough_value': portfolio_values.min()
    })
    
    return all_metrics


__all__ = [
    'AdvancedMetrics', 'RiskMetrics', 'PerformanceAttribution',
    'calculate_portfolio_metrics'
]