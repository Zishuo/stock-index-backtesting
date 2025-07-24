"""
Performance optimization utilities for the backtesting framework.

This module provides caching, memoization, and other performance
optimizations to improve the speed and memory efficiency of backtests.
"""

import functools
import hashlib
import pickle
import time
import psutil
import os
from typing import Any, Dict, Optional, Callable, Union
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Try to import optional performance libraries
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class PerformanceMonitor:
    """
    Monitor and track performance metrics.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start a timer for a specific operation."""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record the elapsed time.
        
        Args:
            name (str): Name of the operation
            
        Returns:
            float: Elapsed time in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = {'times': [], 'total_time': 0, 'count': 0}
        
        self.metrics[name]['times'].append(elapsed)
        self.metrics[name]['total_time'] += elapsed
        self.metrics[name]['count'] += 1
        
        del self.start_times[name]
        return elapsed
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.
        
        Args:
            name (str): Name of the operation
            
        Returns:
            Dict[str, float]: Performance statistics
        """
        if name not in self.metrics:
            return {}
        
        times = self.metrics[name]['times']
        return {
            'count': self.metrics[name]['count'],
            'total_time': self.metrics[name]['total_time'],
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    def print_summary(self):
        """Print a summary of all performance metrics."""
        print("=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for name, stats in [(n, self.get_stats(n)) for n in self.metrics.keys()]:
            if stats:
                print(f"\n{name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Total time: {stats['total_time']:.4f}s")
                print(f"  Average time: {stats['avg_time']:.4f}s")
                print(f"  Min time: {stats['min_time']:.4f}s")
                print(f"  Max time: {stats['max_time']:.4f}s")
        
        memory = self.get_memory_usage()
        print(f"\nMemory Usage:")
        print(f"  RSS: {memory['rss_mb']:.2f} MB")
        print(f"  VMS: {memory['vms_mb']:.2f} MB")
        print(f"  Percent: {memory['percent']:.2f}%")
        print("=" * 60)


# Global performance monitor
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _performance_monitor


def timer(name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        name (str, optional): Name for the timer (defaults to function name)
    """
    def decorator(func):
        timer_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _performance_monitor.start_timer(timer_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _performance_monitor.stop_timer(timer_name)
        
        return wrapper
    return decorator


class Cache:
    """
    Simple in-memory cache with optional persistence.
    """
    
    def __init__(self, max_size: int = 1000, persist_path: Optional[str] = None):
        """
        Initialize the cache.
        
        Args:
            max_size (int): Maximum number of items to cache
            persist_path (str, optional): Path to persist cache to disk
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self._cache = {}
        self._access_times = {}
        
        # Load persisted cache if available
        if persist_path and os.path.exists(persist_path):
            self._load_cache()
    
    def _make_key(self, func, args, kwargs) -> str:
        """Create a cache key from function and arguments."""
        # Create a hash from function name and arguments
        key_data = {
            'func': func.__name__,
            'args': args,
            'kwargs': kwargs
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Any:
        """Get an item from the cache."""
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set an item in the cache."""
        # Remove oldest item if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_times.clear()
    
    def _save_cache(self):
        """Save cache to disk."""
        if self.persist_path:
            try:
                with open(self.persist_path, 'wb') as f:
                    pickle.dump({'cache': self._cache, 'access_times': self._access_times}, f)
            except Exception as e:
                warnings.warn(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self._cache = data.get('cache', {})
                self._access_times = data.get('access_times', {})
        except Exception as e:
            warnings.warn(f"Failed to load cache: {e}")
    
    def __del__(self):
        """Save cache when object is destroyed."""
        if self.persist_path:
            self._save_cache()


# Global cache instance
_cache = Cache(max_size=1000, persist_path=".backtesting_cache.pkl")

def cached(cache_instance: Optional[Cache] = None, ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_instance (Cache, optional): Cache instance to use
        ttl (int, optional): Time to live in seconds
    """
    def decorator(func):
        cache = cache_instance or _cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._make_key(func, args, kwargs)
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                # Check TTL if specified
                if ttl is None or (time.time() - cache._access_times[key]) < ttl:
                    return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator


def memoize(func):
    """
    Simple memoization decorator.
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


class DataFrameOptimizer:
    """
    Utilities for optimizing pandas DataFrame operations.
    """
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to appropriate dtypes.
        
        Args:
            df (pd.DataFrame): DataFrame to optimize
            
        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        optimized_df[col] = optimized_df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                    else:
                        optimized_df[col] = optimized_df[col].astype(np.float64)
        
        return optimized_df
    
    @staticmethod
    def chunk_processing(df: pd.DataFrame, func: Callable, chunk_size: int = 10000, **kwargs) -> pd.DataFrame:
        """
        Process large DataFrame in chunks to reduce memory usage.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            func (Callable): Function to apply to each chunk
            chunk_size (int): Size of each chunk
            **kwargs: Additional arguments for the function
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        results = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            result = func(chunk, **kwargs)
            results.append(result)
        
        return pd.concat(results, ignore_index=True)


class ParallelProcessor:
    """
    Utilities for parallel processing of backtests and computations.
    """
    
    def __init__(self, n_jobs: int = -1, backend: str = 'threading'):
        """
        Initialize the parallel processor.
        
        Args:
            n_jobs (int): Number of parallel jobs (-1 for all CPUs)
            backend (str): Backend to use ('threading' or 'multiprocessing')
        """
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.backend = backend
    
    def parallel_backtest(self, strategy_configs: list, data: pd.DataFrame, **kwargs) -> list:
        """
        Run multiple backtests in parallel.
        
        Args:
            strategy_configs (list): List of strategy configurations
            data (pd.DataFrame): Price data
            **kwargs: Additional backtest parameters
            
        Returns:
            list: List of backtest results
        """
        if self.backend == 'threading':
            executor_class = ThreadPoolExecutor
        else:
            executor_class = ProcessPoolExecutor
        
        with executor_class(max_workers=self.n_jobs) as executor:
            futures = []
            for config in strategy_configs:
                future = executor.submit(self._run_single_backtest, config, data, **kwargs)
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        return results
    
    def _run_single_backtest(self, config: dict, data: pd.DataFrame, **kwargs):
        """Run a single backtest (for parallel execution)."""
        # This would be implemented to run a backtest with the given configuration
        # For now, this is a placeholder
        return {"config": config, "result": "placeholder"}
    
    def parallel_apply(self, func: Callable, items: list, **kwargs) -> list:
        """
        Apply a function to a list of items in parallel.
        
        Args:
            func (Callable): Function to apply
            items (list): List of items to process
            **kwargs: Additional function arguments
            
        Returns:
            list: List of results
        """
        if self.backend == 'threading':
            executor_class = ThreadPoolExecutor
        else:
            executor_class = ProcessPoolExecutor
        
        with executor_class(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(func, item, **kwargs) for item in items]
            results = [future.result() for future in futures]
        
        return results


# Numba optimizations (if available)
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True)
    def fast_sma(prices: np.ndarray, window: int) -> np.ndarray:
        """Fast SMA calculation using Numba."""
        n = len(prices)
        sma = np.empty(n)
        sma[:window-1] = np.nan
        
        for i in range(window-1, n):
            sma[i] = np.mean(prices[i-window+1:i+1])
        
        return sma
    
    @numba.jit(nopython=True)
    def fast_ema(prices: np.ndarray, window: int) -> np.ndarray:
        """Fast EMA calculation using Numba."""
        alpha = 2.0 / (window + 1.0)
        n = len(prices)
        ema = np.empty(n)
        ema[0] = prices[0]
        
        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
else:
    def fast_sma(prices: np.ndarray, window: int) -> np.ndarray:
        """Fallback SMA calculation."""
        return pd.Series(prices).rolling(window=window).mean().values
    
    def fast_ema(prices: np.ndarray, window: int) -> np.ndarray:
        """Fallback EMA calculation."""
        return pd.Series(prices).ewm(span=window).mean().values


__all__ = [
    'PerformanceMonitor', 'get_performance_monitor', 'timer',
    'Cache', 'cached', 'memoize',
    'DataFrameOptimizer', 'ParallelProcessor',
    'fast_sma', 'fast_ema'
]