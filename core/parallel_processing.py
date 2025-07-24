"""
Parallel processing capabilities for backtesting optimization.

This module provides parallel execution of backtests, parameter optimization,
and Monte Carlo simulations to improve performance and enable large-scale
analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Iterator
from datetime import datetime
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import time
from functools import partial
import pickle
import os

from .data_handler import StockData
from .strategy_base import Strategy
from .portfolio import BackTest
from .exceptions import BacktestingError, ValidationError
from .performance import get_performance_monitor


class ParameterGrid:
    """
    Parameter grid generator for optimization tasks.
    
    Creates all combinations of parameters for grid search optimization.
    """
    
    def __init__(self, param_dict: Dict[str, List[Any]]):
        """
        Initialize parameter grid.
        
        Args:
            param_dict (Dict[str, List[Any]]): Dictionary of parameter names and values
        """
        self.param_dict = param_dict
        self.param_names = list(param_dict.keys())
        self.param_values = list(param_dict.values())
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all parameter combinations."""
        for combination in itertools.product(*self.param_values):
            yield dict(zip(self.param_names, combination))
    
    def __len__(self) -> int:
        """Return total number of parameter combinations."""
        total = 1
        for values in self.param_values:
            total *= len(values)
        return total
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Randomly sample parameter combinations.
        
        Args:
            n_samples (int): Number of samples to generate
            random_state (int, optional): Random seed
            
        Returns:
            List[Dict[str, Any]]: Sampled parameter combinations
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        all_combinations = list(self)
        n_samples = min(n_samples, len(all_combinations))
        
        sampled_indices = np.random.choice(len(all_combinations), size=n_samples, replace=False)
        return [all_combinations[i] for i in sampled_indices]


class BacktestJob:
    """
    Represents a single backtest job for parallel execution.
    """
    
    def __init__(self, job_id: str, strategy_class: type, strategy_params: Dict[str, Any],
                 ticker: str, start_date, end_date, backtest_params: Optional[Dict[str, Any]] = None):
        """
        Initialize backtest job.
        
        Args:
            job_id (str): Unique job identifier
            strategy_class (type): Strategy class to instantiate
            strategy_params (Dict[str, Any]): Strategy parameters
            ticker (str): Stock ticker
            start_date: Backtest start date
            end_date: Backtest end date
            backtest_params (Dict[str, Any], optional): Additional backtest parameters
        """
        self.job_id = job_id
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.backtest_params = backtest_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'strategy_class_name': self.strategy_class.__name__,
            'strategy_params': self.strategy_params,
            'ticker': self.ticker,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'backtest_params': self.backtest_params
        }


class BacktestResult:
    """
    Container for backtest results.
    """
    
    def __init__(self, job_id: str, success: bool, metrics: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None, execution_time: Optional[float] = None):
        """
        Initialize backtest result.
        
        Args:
            job_id (str): Job identifier
            success (bool): Whether backtest succeeded
            metrics (Dict[str, Any], optional): Performance metrics
            error (str, optional): Error message if failed
            execution_time (float, optional): Execution time in seconds
        """
        self.job_id = job_id
        self.success = success
        self.metrics = metrics or {}
        self.error = error
        self.execution_time = execution_time
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'job_id': self.job_id,
            'success': self.success,
            'metrics': self.metrics,
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


def run_single_backtest(job: BacktestJob, data_cache: Optional[Dict[str, pd.DataFrame]] = None) -> BacktestResult:
    """
    Run a single backtest job.
    
    Args:
        job (BacktestJob): Backtest job to execute
        data_cache (Dict[str, pd.DataFrame], optional): Pre-loaded data cache
        
    Returns:
        BacktestResult: Backtest result
    """
    start_time = time.time()
    
    try:
        # Load or get data from cache
        if data_cache and job.ticker in data_cache:
            stock_data = StockData(job.ticker)
            stock_data.data = data_cache[job.ticker].copy()
        else:
            stock_data = StockData(job.ticker)
            stock_data.get_data_from_yfinance(job.ticker, job.start_date, job.end_date)
        
        # Create strategy
        strategy = job.strategy_class(**job.strategy_params)
        
        # Run strategy
        strategy.run_strategy(stock_data, job.start_date, job.end_date)
        
        # Run backtest
        backtest = BackTest()
        backtest.run_backtest(strategy, stock_data, job.start_date, job.end_date, **job.backtest_params)
        
        # Calculate performance metrics
        backtest.performance_summary(verbose=False)
        
        execution_time = time.time() - start_time
        
        # Extract key metrics
        metrics = {
            'cumulative_return': backtest.cumulative_return.values[0] if hasattr(backtest, 'cumulative_return') else 0,
            'annual_return': backtest.annual_return - 1 if hasattr(backtest, 'annual_return') else 0,
            'max_drawdown': backtest.max_drawdown.values[0] if hasattr(backtest, 'max_drawdown') else 0,
            'sharpe_ratio': backtest.sharp_ratio.values[0] if hasattr(backtest, 'sharp_ratio') else 0,
            'num_trades': len(backtest.trade_records),
            'strategy_params': job.strategy_params
        }
        
        return BacktestResult(
            job_id=job.job_id,
            success=True,
            metrics=metrics,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        return BacktestResult(
            job_id=job.job_id,
            success=False,
            error=str(e),
            execution_time=execution_time
        )


class ParallelOptimizer:
    """
    Parallel parameter optimization for trading strategies.
    
    This class enables efficient optimization of strategy parameters
    using parallel processing.
    """
    
    def __init__(self, n_jobs: int = -1, backend: str = 'multiprocessing',
                 cache_data: bool = True, verbose: bool = True):
        """
        Initialize parallel optimizer.
        
        Args:
            n_jobs (int): Number of parallel jobs (-1 for all CPUs)
            backend (str): Parallel backend ('multiprocessing' or 'threading')
            cache_data (bool): Whether to cache price data
            verbose (bool): Whether to print progress updates
        """
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.backend = backend
        self.cache_data = cache_data
        self.verbose = verbose
        self.results_history: List[BacktestResult] = []
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def optimize_parameters(self, strategy_class: type, param_grid: ParameterGrid,
                          ticker: str, start_date, end_date,
                          optimization_metric: str = 'sharpe_ratio',
                          n_best: int = 10) -> List[BacktestResult]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_class (type): Strategy class to optimize
            param_grid (ParameterGrid): Parameter grid for optimization
            ticker (str): Stock ticker
            start_date: Backtest start date
            end_date: Backtest end date
            optimization_metric (str): Metric to optimize
            n_best (int): Number of best results to return
            
        Returns:
            List[BacktestResult]: Best optimization results
        """
        if self.verbose:
            print(f"Starting parameter optimization with {len(param_grid)} combinations")
            print(f"Using {self.n_jobs} parallel workers")
        
        # Prepare data cache if enabled
        if self.cache_data:
            self._cache_data(ticker, start_date, end_date)
        
        # Create jobs
        jobs = []
        for i, params in enumerate(_safe_parameter_grid_iter(param_grid)):
            job_id = f"{strategy_class.__name__}_{ticker}_{i}"
            job = BacktestJob(
                job_id=job_id,
                strategy_class=strategy_class,
                strategy_params=params,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            jobs.append(job)
        
        # Execute jobs in parallel
        results = self._execute_parallel_jobs(jobs)
        
        # Filter successful results and sort by optimization metric
        successful_results = [r for r in results if r.success]
        if not successful_results:
            raise BacktestingError("No successful backtests in optimization")
        
        # Sort by optimization metric (descending for most metrics, ascending for max_drawdown)
        reverse_sort = optimization_metric != 'max_drawdown'
        sorted_results = sorted(
            successful_results,
            key=lambda r: r.metrics.get(optimization_metric, -np.inf if reverse_sort else np.inf),
            reverse=reverse_sort
        )
        
        # Store results
        self.results_history.extend(results)
        
        if self.verbose:
            print(f"Optimization complete. {len(successful_results)} successful backtests")
            self._print_optimization_summary(sorted_results[:n_best], optimization_metric)
        
        return sorted_results[:n_best]
    
    def random_search(self, strategy_class: type, param_grid: ParameterGrid,
                     ticker: str, start_date, end_date, n_iterations: int = 100,
                     optimization_metric: str = 'sharpe_ratio', n_best: int = 10,
                     random_state: Optional[int] = None) -> List[BacktestResult]:
        """
        Random parameter search optimization.
        
        Args:
            strategy_class (type): Strategy class to optimize
            param_grid (ParameterGrid): Parameter grid for sampling
            ticker (str): Stock ticker
            start_date: Backtest start date
            end_date: Backtest end date
            n_iterations (int): Number of random samples
            optimization_metric (str): Metric to optimize
            n_best (int): Number of best results to return
            random_state (int, optional): Random seed
            
        Returns:
            List[BacktestResult]: Best optimization results
        """
        if self.verbose:
            print(f"Starting random search with {n_iterations} iterations")
        
        # Sample parameter combinations
        sampled_params = param_grid.sample(n_iterations, random_state)
        
        # Create parameter grid from samples
        sampled_grid = ParameterGrid({})
        sampled_grid.param_dict = {}
        sampled_grid._combinations = sampled_params
        
        # Use the regular optimize_parameters method
        return self.optimize_parameters(
            strategy_class, sampled_grid, ticker, start_date, end_date,
            optimization_metric, n_best
        )
    
    def monte_carlo_analysis(self, strategy_class: type, strategy_params: Dict[str, Any],
                           ticker: str, start_date, end_date, n_simulations: int = 1000,
                           simulation_method: str = 'bootstrap') -> Dict[str, Any]:
        """
        Monte Carlo analysis of strategy performance.
        
        Args:
            strategy_class (type): Strategy class
            strategy_params (Dict[str, Any]): Strategy parameters
            ticker (str): Stock ticker
            start_date: Analysis start date
            end_date: Analysis end date
            n_simulations (int): Number of Monte Carlo simulations
            simulation_method (str): Simulation method ('bootstrap' or 'parametric')
            
        Returns:
            Dict[str, Any]: Monte Carlo analysis results
        """
        if self.verbose:
            print(f"Running Monte Carlo analysis with {n_simulations} simulations")
        
        # Load original data
        stock_data = StockData(ticker)
        stock_data.get_data_from_yfinance(ticker, start_date, end_date)
        original_data = stock_data.data.copy()
        
        # Generate simulation jobs
        jobs = []
        for i in range(n_simulations):
            # Create simulated data
            if simulation_method == 'bootstrap':
                simulated_data = self._bootstrap_data(original_data)
            else:
                simulated_data = self._parametric_simulation(original_data)
            
            # Cache the simulation data
            sim_ticker = f"{ticker}_sim_{i}"
            self.data_cache[sim_ticker] = simulated_data
            
            job = BacktestJob(
                job_id=f"mc_{ticker}_{i}",
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                ticker=sim_ticker,
                start_date=start_date,
                end_date=end_date
            )
            jobs.append(job)
        
        # Execute simulations
        results = self._execute_parallel_jobs(jobs)
        successful_results = [r for r in results if r.success]
        
        # Analyze results
        returns = [r.metrics['cumulative_return'] for r in successful_results]
        sharpe_ratios = [r.metrics['sharpe_ratio'] for r in successful_results]
        max_drawdowns = [r.metrics['max_drawdown'] for r in successful_results]
        
        analysis = {
            'n_simulations': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentiles': {
                    '5': np.percentile(returns, 5),
                    '25': np.percentile(returns, 25),
                    '50': np.percentile(returns, 50),
                    '75': np.percentile(returns, 75),
                    '95': np.percentile(returns, 95)
                }
            },
            'sharpe_ratios': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios)
            },
            'max_drawdowns': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'worst': np.min(max_drawdowns)
            },
            'probability_of_profit': (np.array(returns) > 0).mean()
        }
        
        if self.verbose:
            self._print_monte_carlo_summary(analysis)
        
        return analysis
    
    def _cache_data(self, ticker: str, start_date, end_date):
        """Cache price data for reuse."""
        if ticker not in self.data_cache:
            if self.verbose:
                print(f"Caching data for {ticker}")
            
            stock_data = StockData(ticker)
            stock_data.get_data_from_yfinance(ticker, start_date, end_date)
            self.data_cache[ticker] = stock_data.data.copy()
    
    def _execute_parallel_jobs(self, jobs: List[BacktestJob]) -> List[BacktestResult]:
        """Execute jobs in parallel."""
        if self.backend == 'multiprocessing':
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        
        with executor_class(max_workers=self.n_jobs) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(run_single_backtest, job, self.data_cache): job 
                for job in jobs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                completed += 1
                
                if self.verbose and completed % max(1, len(jobs) // 20) == 0:
                    print(f"Completed {completed}/{len(jobs)} backtests")
        
        return results
    
    def _bootstrap_data(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create bootstrap sample of price data."""
        returns = original_data['Close'].pct_change().dropna()
        
        # Bootstrap returns
        bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
        
        # Reconstruct price series
        initial_price = original_data['Close'].iloc[0]
        bootstrap_prices = [initial_price]
        
        for ret in bootstrap_returns:
            next_price = bootstrap_prices[-1] * (1 + ret)
            bootstrap_prices.append(next_price)
        
        # Create new DataFrame
        bootstrap_data = original_data.copy()
        bootstrap_data['Close'] = bootstrap_prices[:-1]  # Remove extra price
        
        # Adjust other OHLC data proportionally
        price_ratio = bootstrap_data['Close'] / original_data['Close']
        for col in ['Open', 'High', 'Low']:
            if col in bootstrap_data.columns:
                bootstrap_data[col] = original_data[col] * price_ratio
        
        return bootstrap_data
    
    def _parametric_simulation(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Create parametric simulation of price data."""
        returns = original_data['Close'].pct_change().dropna()
        
        # Estimate parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mu, sigma, len(returns))
        
        # Reconstruct price series
        initial_price = original_data['Close'].iloc[0]
        simulated_prices = [initial_price]
        
        for ret in simulated_returns:
            next_price = simulated_prices[-1] * (1 + ret)
            simulated_prices.append(next_price)
        
        # Create new DataFrame
        simulated_data = original_data.copy()
        simulated_data['Close'] = simulated_prices[:-1]
        
        # Adjust other OHLC data proportionally
        price_ratio = simulated_data['Close'] / original_data['Close']
        for col in ['Open', 'High', 'Low']:
            if col in simulated_data.columns:
                simulated_data[col] = original_data[col] * price_ratio
        
        return simulated_data
    
    def _print_optimization_summary(self, results: List[BacktestResult], metric: str):
        """Print optimization summary."""
        print(f"\nTop Results (optimized for {metric}):")
        print("-" * 80)
        print(f"{'Rank':<5} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<7} Parameters")
        print("-" * 80)
        
        for i, result in enumerate(results[:10], 1):
            metrics = result.metrics
            params_str = str(metrics.get('strategy_params', {}))[:30] + "..."
            
            print(f"{i:<5} "
                  f"{metrics.get('cumulative_return', 0):<10.2%} "
                  f"{metrics.get('sharpe_ratio', 0):<8.2f} "
                  f"{metrics.get('max_drawdown', 0):<8.2%} "
                  f"{metrics.get('num_trades', 0):<7} "
                  f"{params_str}")
    
    def _print_monte_carlo_summary(self, analysis: Dict[str, Any]):
        """Print Monte Carlo analysis summary."""
        print(f"\nMonte Carlo Analysis Results:")
        print("-" * 50)
        print(f"Simulations: {analysis['n_simulations']}")
        print(f"Success Rate: {analysis['success_rate']:.1%}")
        print(f"Probability of Profit: {analysis['probability_of_profit']:.1%}")
        print(f"\nReturn Distribution:")
        print(f"  Mean: {analysis['returns']['mean']:.2%}")
        print(f"  Std:  {analysis['returns']['std']:.2%}")
        print(f"  5th percentile:  {analysis['returns']['percentiles']['5']:.2%}")
        print(f"  95th percentile: {analysis['returns']['percentiles']['95']:.2%}")
        print(f"\nSharpe Ratio: {analysis['sharpe_ratios']['mean']:.2f} ± {analysis['sharpe_ratios']['std']:.2f}")
        print(f"Worst Drawdown: {analysis['max_drawdowns']['worst']:.2%}")


# Helper function to avoid recursion in parameter grid
def _safe_parameter_grid_iter(param_grid):
    """Safely iterate over parameter combinations."""
    if hasattr(param_grid, '_combinations'):
        return iter(param_grid._combinations)
    else:
        # Use the original implementation
        for combination in itertools.product(*param_grid.param_values):
            yield dict(zip(param_grid.param_names, combination))


__all__ = [
    'ParameterGrid', 'BacktestJob', 'BacktestResult', 'ParallelOptimizer',
    'run_single_backtest'
]