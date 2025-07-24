"""
Constants and configuration values for the backtesting framework.

This module contains commonly used constants, default parameters,
and configuration values used throughout the backtesting system.
"""

# Technical indicator default parameters
DEFAULT_MA_WINDOWS = [5, 10, 20, 50, 200]
DEFAULT_STOCHASTIC_K = 14
DEFAULT_STOCHASTIC_FK = 5
DEFAULT_STOCHASTIC_FD = 5
DEFAULT_OVERBOUGHT = 80
DEFAULT_OVERSOLD = 20

# Trading parameters
DEFAULT_STOP_LOSS = 0.10  # 10% stop loss
DEFAULT_TAKE_PROFIT = 0.0  # No take profit by default
DEFAULT_COMMISSION = 0.0
DEFAULT_IMPACT = 0.0

# Tax rates (US federal)
LONG_TERM_TAX_RATE = 0.15  # 15% for long-term capital gains
SHORT_TERM_TAX_RATE = 0.22  # 22% for short-term capital gains (example rate)

# Portfolio parameters
DEFAULT_PRINCIPAL = 100000  # $100,000 starting capital
DEFAULT_PYRAMIDING = 1  # Single position by default

# Data parameters
DEFAULT_DATA_INTERVAL = '1d'  # Daily data by default
DEFAULT_DB_PATH = 'data/stock_data.db'
DEFAULT_DB_LIMIT = 100000

# Performance calculation parameters
TRADING_DAYS_PER_YEAR = 252
DRAWDOWN_WINDOW = 252

# Signal values
BUY_SIGNAL = 1
SELL_SIGNAL = -1
STOP_LOSS_SIGNAL = -2
NO_SIGNAL = 0

# Common threshold values
THRESHOLD_VALUES = {
    'buy_low': 15,
    'sell_high': 30,
    'buy_high': 1.05,  # 5% above moving average
    'sell_low': 0.95   # 5% below moving average
}