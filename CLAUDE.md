# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based financial backtesting system focused on algorithmic trading strategies for leveraged ETFs (particularly TQQQ, QLD, QQQ) and individual stocks. The repository contains custom backtesting frameworks, technical indicator implementations, and various trading strategies including NAA200R-based signals, moving average crossovers, and stochastic oscillators.

## Core Architecture

### Main Framework (`Ab.py`)
The core backtesting engine consists of several key classes:

- **`StockData`**: Data fetching and preprocessing from Yahoo Finance, CSV files, or SQLite databases
- **`Strategy`** (abstract base): Template for implementing trading strategies with common interface
- **`Portfolio`** (abstract base): Portfolio management and performance tracking
- **`BackTest`**: Concrete implementation for running backtests with tax calculation support

### Key Strategy Implementations
- **`BuyAndHold`**: Simple buy-and-hold benchmark strategy
- **`MACross`**: Moving average crossover strategy (short MA vs long MA)
- **`MAThreshold`**: Moving average threshold-based signals
- **`Threshold`**: Price threshold strategy with trend confirmation
- **`StochasticCross`**: Complex stochastic oscillator with daily/weekly timeframes
- **`fftyspy_stg`**: Combined fifty/SPY strategy using 200-day MA
- **`fftynaa200r_stg`**: Combines FFTY moving average with NAA200R signals

### Technical Indicators
Built-in technical analysis functions in `Ab.py`:
- Simple Moving Average (SMA) and Exponential Moving Average (EMA)
- Stochastic Oscillator (Fast %K, %D with customizable periods)
- Price-to-moving-average ratios
- Volume analysis and threshold detection

## Data Structure

### Data Sources
- **Yahoo Finance**: Primary source via `yfinance` library
- **Local CSV files**: Stored in `data/`, `CleanData/`, `Data-*/` directories
- **SQLite Database**: `data/stock_data.db` contains historical stock data with pre-calculated moving averages

### Key Data Files
- **NAA200R**: NASDAQ stocks above 200-day moving average percentage (critical signal)
- **Market Data**: QQQ, TQQQ, QLD, SPY, NDX historical prices
- **Earnings Data**: Individual stock earnings with gap-up analysis in `earnings/` and processed data in `Processed_Data-*/`

## Development Workflow

### Running Jupyter Notebooks
The primary development is done in Jupyter notebooks:
```bash
jupyter notebook
# or
jupyter lab
```

### Key Notebooks
- **`NAA200R_research.ipynb`**: Research and backtesting using NAA200R signals
- **`QQQbacktesting.ipynb`**: QQQ trading strategy development and testing
- **`stock_data_fetch.ipynb`**: Data collection and database population
- **`gapup_analysis.ipynb`**: Earnings gap-up pattern analysis
- **`*_stg.ipynb`**: Strategy-specific backtesting notebooks

### Python Environment
The codebase uses standard scientific Python stack:
- `pandas`, `numpy`: Data manipulation
- `matplotlib`: Plotting and visualization  
- `yfinance`: Yahoo Finance data fetching
- `sqlite3`: Database operations
- Optional: `cupy`, `cudf` for GPU acceleration (CUDA)

### Database Operations
SQLite database at `data/stock_data.db` contains:
- Historical OHLCV data for thousands of stocks
- Pre-calculated moving averages (MA5, MA10, MA20, MA50, MA200)
- Exchange information (NASDAQ, NYSE, AMEX)

## Strategy Development Pattern

1. **Data Preparation**: Use `StockData` class to fetch/load price data
2. **Signal Generation**: Implement strategy logic in a class inheriting from `Strategy`
3. **Backtesting**: Use `BackTest` class to simulate trading with the strategy
4. **Analysis**: Generate performance metrics and visualizations

### Example Strategy Structure
```python
class MyStrategy(Strategy):
    def __init__(self, param1, param2):
        super().__init__(name, stop_loss, take_profit)
        self.param1 = param1
        
    def run_strategy(self, indicators, start_date, end_date):
        # Generate signals based on indicator data
        # Populate self.trades DataFrame with buy/sell signals
        pass
```

### Performance Analysis
All strategies generate standardized metrics:
- Cumulative return, annual return, max drawdown
- Sharpe ratio, batting average
- Trade-by-trade analysis with tax implications
- Risk-reward ratios and gain/loss statistics

## Important Implementation Notes

- **Tax Calculation**: BackTest class includes sophisticated tax treatment for long/short term capital gains
- **Stop Loss**: Built-in 10% daily stop-loss for TQQQ strategies (signal = -2)
- **Weekly Data Handling**: Special handling for weekly data alignment (Fridays)
- **GPU Support**: Optional CUDA acceleration via cupy/cudf for large datasets
- **Multiple Timeframes**: Strategies can combine daily and weekly signals (especially stochastic)

## Data File Conventions

- Stock symbols in uppercase (e.g., `TQQQ.csv`, `QQQ.csv`)
- Date column as index in YYYY-MM-DD format
- Standard OHLCV columns plus technical indicators
- Earnings data includes surprise percentages and performance windows
- Processed data includes gap analysis and various risk metrics (R1-R4 rules)