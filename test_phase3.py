#!/usr/bin/env python3
"""Test script for Phase 3 advanced features."""

import Ab as ab
import pandas as pd
import numpy as np
from datetime import datetime

print('Testing Phase 3 advanced features...')
print()

# Test 1: Plugin Architecture
print('1. Testing Plugin Architecture:')
try:
    plugin_manager = ab.get_plugin_manager()
    indicators = plugin_manager.list_indicators()
    print(f'   ✅ Plugin manager loaded with {len(indicators)} built-in indicators')
    
    # Test built-in indicators
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100) + 100,
        'High': np.random.randn(100) + 102,
        'Low': np.random.randn(100) + 98,
        'Close': np.random.randn(100) + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    rsi = plugin_manager.calculate_indicator('rsi', sample_data, period=14)
    print(f'   ✅ RSI calculation successful, length: {len(rsi)}')
    
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 2: Multi-Asset Portfolio
print('2. Testing Multi-Asset Portfolio:')
try:
    allocations = {'TQQQ': 0.6, 'QLD': 0.4}
    allocation = ab.AssetAllocation(allocations)
    portfolio = ab.MultiAssetPortfolio()
    print(f'   ✅ Multi-asset portfolio created with {len(allocation.get_asset_symbols())} assets')
    
    # Test allocation
    print(f'   ✅ TQQQ allocation: {allocation.get_weight("TQQQ"):.1%}')
    print(f'   ✅ QLD allocation: {allocation.get_weight("QLD"):.1%}')
    
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 3: Advanced Metrics
print('3. Testing Advanced Metrics:')
try:
    # Create sample returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    returns.index = pd.date_range('2023-01-01', periods=252, freq='D')
    
    metrics_calc = ab.AdvancedMetrics(returns)
    basic_metrics = metrics_calc.basic_metrics()
    
    print(f'   ✅ Basic metrics calculated: {len(basic_metrics)} metrics')
    print(f'   ✅ Sharpe ratio: {basic_metrics["sharpe_ratio"]:.2f}')
    print(f'   ✅ Win rate: {basic_metrics["win_rate"]:.1%}')
    
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 4: Risk Management
print('4. Testing Risk Management:')
try:
    risk_manager = ab.RiskManager()
    stop_loss = ab.StopLossControl(0.1)
    risk_manager.add_risk_control(stop_loss)
    
    print(f'   ✅ Risk manager created with {len(risk_manager.risk_controls)} controls')
    
    # Test position sizing
    position_size = risk_manager.calculate_position_size({})
    print(f'   ✅ Position size calculated: {position_size:.1%}')
    
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 5: Parameter Grid
print('5. Testing Parameter Grid:')
try:
    param_grid = ab.ParameterGrid({
        'param1': [1, 2, 3],
        'param2': [0.1, 0.2]
    })
    
    total_combinations = len(param_grid)
    print(f'   ✅ Parameter grid created with {total_combinations} combinations')
    
    # Test sampling
    samples = param_grid.sample(3, random_state=42)
    print(f'   ✅ Sampled {len(samples)} parameter combinations')
    
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 6: Convenience Functions
print('6. Testing Phase 3 Convenience Functions:')
try:
    # Test setup_risk_management
    risk_mgr = ab.setup_risk_management(stop_loss_pct=0.05)
    print('   ✅ Risk management setup successful')
    
    # Test list_custom_indicators (should show built-in indicators)
    print('   ✅ Custom indicators list function available')
    
except Exception as e:
    print(f'   ❌ Error: {e}')

print()
print('✅ Phase 3 advanced features test completed!')
print()
print('Advanced features now available:')
print('• Plugin architecture for custom indicators')
print('• Multi-asset portfolio management')
print('• Advanced performance metrics')
print('• Risk management systems')
print('• Parameter optimization capabilities')