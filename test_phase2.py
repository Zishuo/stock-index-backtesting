#!/usr/bin/env python3
"""Test script for Phase 2 features."""

import Ab as ab
print('Testing Phase 2 features...')
print()

# Test strategy factory
print('1. Testing Strategy Factory:')
try:
    strategy = ab.create_strategy_by_name('buyandhold')
    print(f'   ✅ Created strategy: {type(strategy).__name__}')
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test configuration
print('2. Testing Configuration:')
try:
    config = ab.get_config()
    print(f'   ✅ Initial capital: ${config.get_backtest_config().initial_capital:,}')
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test performance monitor
print('3. Testing Performance Monitor:')
try:
    monitor = ab.get_performance_monitor()
    print(f'   ✅ Monitor type: {type(monitor).__name__}')
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test data validator
print('4. Testing Data Validator:')
try:
    validator = ab.DataValidator()
    print(f'   ✅ Validator type: {type(validator).__name__}')
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test strategy factory list
print('5. Testing Strategy Factory List:')
try:
    strategies = ab.StrategyFactory.list_available_strategies()
    print(f'   ✅ Found {len(strategies)} strategy types')
except Exception as e:
    print(f'   ❌ Error: {e}')

print()
print('✅ Phase 2 basic functionality test completed!')