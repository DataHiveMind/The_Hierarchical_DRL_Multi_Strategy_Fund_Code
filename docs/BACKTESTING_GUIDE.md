# Backtesting Framework Guide

## Overview

This project uses a comprehensive backtesting framework built on top of the `src/backtesting` module. The framework provides realistic simulations of trading strategies with transaction costs, slippage, and detailed performance analytics.

## Architecture

### 1. BacktestEngine (`src/backtesting/engine.py`)

The main backtesting engine that simulates trading on historical data.

**Key Features:**
- Transaction costs (default: 0.1%)
- Slippage simulation (default: 0.05%)
- Support for specialist and master-level agents
- Real-time equity tracking
- Action and position logging

**Main Methods:**

```python
# Initialize the engine
backtest_engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost=0.001,
    slippage=0.0005,
    risk_free_rate=0.02
)

# Run specialist backtest
results = backtest_engine.run_specialist_backtest(
    agent=trained_agent,
    env=test_environment,
    test_data=test_dataframe,
    strategy_name="my_strategy",
    deterministic=True
)

# Run master allocator backtest
master_results = backtest_engine.run_master_backtest(
    master_agent=master_agent,
    specialist_agents=specialist_dict,
    env=master_env,
    test_data=test_dataframe,
    deterministic=True
)
```

### 2. PerformanceMetrics (`src/backtesting/metrics.py`)

Comprehensive performance metrics calculator.

**Calculated Metrics:**

1. **Return Metrics:**
   - Total Return
   - Annual Return
   - Annual Volatility

2. **Risk-Adjusted Returns:**
   - **Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility
   - **Sortino Ratio**: Uses downside deviation instead of total volatility
   - **Calmar Ratio**: Annual Return / Max Drawdown

3. **Risk Measures:**
   - **Max Drawdown**: Largest peak-to-trough decline
   - **Max Drawdown Duration**: Longest period in drawdown
   - **Current Drawdown**: Current drawdown from peak
   - **VaR (95%)**: Value at Risk at 95% confidence
   - **CVaR (95%)**: Conditional VaR (expected shortfall)

4. **Trading Statistics:**
   - **Win Rate**: Percentage of profitable periods
   - **Profit Factor**: Gross profit / Gross loss
   - **Number of Periods**: Total trading periods

**Usage:**

```python
metrics_calc = PerformanceMetrics(risk_free_rate=0.02)

# Calculate all metrics
metrics = metrics_calc.calculate_all_metrics(
    equity_curve=equity_series,
    periods_per_year=252
)

# Individual metrics
sharpe = metrics_calc.sharpe_ratio(returns, periods_per_year=252)
max_dd = metrics_calc.max_drawdown(equity_curve)
win_rate = metrics_calc.win_rate(returns)
```

### 3. StrategyComparison

Compare multiple strategies side-by-side.

**Usage:**

```python
comparison = StrategyComparison(risk_free_rate=0.02)

strategies = {
    'Strategy_A': equity_curve_a,
    'Strategy_B': equity_curve_b,
    'Strategy_C': equity_curve_c
}

# Compare all strategies
comparison_df = comparison.compare_strategies(strategies, periods_per_year=252)

# Rank by specific metric
ranked = comparison.rank_strategies(strategies, metric='sharpe_ratio')
```

### 4. RollingMetrics

Calculate rolling/windowed performance metrics.

**Usage:**

```python
rolling = RollingMetrics(window=252, risk_free_rate=0.02)

# Rolling Sharpe ratio
rolling_sharpe = rolling.rolling_sharpe(returns, periods_per_year=252)

# Rolling volatility
rolling_vol = rolling.rolling_volatility(returns, periods_per_year=252)

# Rolling drawdown
rolling_dd = rolling.rolling_drawdown(equity_curve)
```

## Backtesting Workflow

### Step 1: Train Agents

```python
# Train specialist agents on historical data
trained_specialists = train_all_specialists(
    specialist_datasets=specialist_datasets,
    initial_capital=100000,
    train_timesteps=50000
)
```

### Step 2: Initialize Backtesting Engine

```python
backtest_engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost=0.001,
    slippage=0.0005,
    risk_free_rate=0.02
)
```

### Step 3: Run Backtests

```python
specialist_results = {}

for strategy_name, specialist_data in trained_specialists.items():
    agent = specialist_data['agent']
    env_class = specialist_data['env_class']
    test_data = specialist_datasets[strategy_name]['test']
    
    # Create test environment
    test_env = env_class(data=test_data, initial_capital=100000)
    
    # Run backtest
    results = backtest_engine.run_specialist_backtest(
        agent=agent,
        env=test_env,
        test_data=test_data,
        strategy_name=strategy_name,
        deterministic=True
    )
    
    specialist_results[strategy_name] = results
```

### Step 4: Analyze Results

```python
# View metrics for a specific strategy
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")

# Get formatted summary
summary = backtest_engine.get_summary(strategy_name)
print(summary)

# Export results
backtest_engine.export_results(
    strategy_name=strategy_name,
    output_dir='reports/backtest_results'
)
```

### Step 5: Compare Strategies

```python
# Create equity curve dictionary
equity_curves = {
    name: results['equity_curve']
    for name, results in specialist_results.items()
}

# Compare all strategies
comparison_df = backtest_engine.compare_strategies()
print(comparison_df)
```

## Results Structure

Each backtest returns a dictionary with:

```python
{
    'strategy_name': str,
    'equity_curve': pd.Series,           # Portfolio value over time
    'results_df': pd.DataFrame,          # Detailed results (actions, positions, rewards)
    'metrics': dict,                     # All performance metrics
    'final_value': float,                # Final portfolio value
    'total_return': float,               # Total return percentage
    'num_trades': int,                   # Number of trades executed
    'test_data': pd.DataFrame            # Test dataset used
}
```

## Visualization Examples

### Equity Curves

```python
import matplotlib.pyplot as plt

for name, results in specialist_results.items():
    plt.plot(results['equity_curve'], label=name)

plt.title('Specialist Strategy Equity Curves')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()
```

### Drawdown Analysis

```python
for name, results in specialist_results.items():
    equity = results['equity_curve']
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    plt.plot(drawdown * 100, label=name)

plt.title('Drawdown Analysis')
plt.xlabel('Date')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.show()
```

### Performance Metrics Comparison

```python
metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']

for metric in metrics:
    values = [results['metrics'][metric] for results in specialist_results.values()]
    names = list(specialist_results.keys())
    
    plt.barh(names, values)
    plt.title(f'{metric.replace("_", " ").title()} Comparison')
    plt.xlabel('Value')
    plt.show()
```

## Advanced Features

### Walk-Forward Analysis

```python
from src.backtesting.engine import WalkForwardAnalysis

wf_analysis = WalkForwardAnalysis(
    train_window=252,    # 1 year training
    test_window=63,      # 3 months testing
    step_size=63         # Roll forward 3 months
)

results = wf_analysis.run_walk_forward(
    agent_factory=lambda env: DDPGAgent(env),
    env_factory=lambda data: StatisticalArbitrageEnv(data=data),
    data=full_dataset,
    train_timesteps=10000
)
```

### Rolling Metrics Analysis

```python
rolling_metrics = RollingMetrics(window=252)

returns = results['equity_curve'].pct_change()

rolling_sharpe = rolling_metrics.rolling_sharpe(returns, periods_per_year=252)
rolling_vol = rolling_metrics.rolling_volatility(returns, periods_per_year=252)

plt.plot(rolling_sharpe, label='Rolling Sharpe')
plt.title('Rolling 1-Year Sharpe Ratio')
plt.legend()
plt.show()
```

## Best Practices

1. **Always use deterministic policies** for backtesting to ensure reproducibility
2. **Include realistic transaction costs** (typically 0.1% - 0.5% for equities)
3. **Account for slippage** (typically 0.01% - 0.1%)
4. **Use out-of-sample test data** that agents haven't seen during training
5. **Calculate multiple metrics** - no single metric tells the full story
6. **Compare against benchmarks** - equal-weight, mean-variance, etc.
7. **Analyze drawdowns** - understand worst-case scenarios
8. **Export results** for auditing and further analysis
9. **Use walk-forward analysis** for more robust validation
10. **Plot equity curves** - visual inspection reveals important patterns

## Files and Directories

```
src/backtesting/
├── __init__.py           # Module initialization and exports
├── engine.py             # BacktestEngine and WalkForwardAnalysis
└── metrics.py            # PerformanceMetrics, StrategyComparison, RollingMetrics

reports/
├── backtest_results/     # Detailed CSV results
├── plots/                # Visualization outputs
└── tables/               # Summary metrics tables
```

## References

- **Notebook**: `notebooks/03_results_and_visualization.ipynb`
- **Documentation**: `docs/BACKTESTING_GUIDE.md`
- **Source Code**: `src/backtesting/`

---

*Last Updated: November 26, 2025*
