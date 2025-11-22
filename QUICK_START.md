# Quick Start Guide - Hierarchical DRL Multi-Strategy Fund

## 📁 Project Structure

```
The_Hierarchical_DRL_Multi_Strategy_Fund_Code/
├── src/
│   ├── agents/              # RL agents (PPO, DDPG, DQN)
│   ├── environments/        # Trading environments
│   │   ├── specialist_envs/ # 7 specialist strategies
│   │   └── master_env/      # CIO allocator
│   ├── data/               # Data loading & features
│   ├── backtesting/        # Backtesting engine
│   └── utils/              # Utilities
│       ├── config.py       # Configuration
│       ├── plotting.py     # Visualization
│       └── model_manager.py # Model saving/loading
├── models/                 # Saved models
│   ├── specialists/        # 7 specialist folders
│   ├── master/            # CIO models
│   └── checkpoints/       # Training checkpoints
├── notebooks/             # Jupyter notebooks
├── tests/                # Test scripts
└── data/                 # Raw & processed data
```

## 🎯 Workflow

### Phase 1: Train Specialist Agents
Each specialist learns its own trading strategy independently:

1. **Statistical Arbitrage (PPO)**
   - Pairs trading with mean reversion
   - Models saved to: `models/specialists/statistical_arbitrage/`

2. **Market Making (DDPG)**
   - Quote posting with inventory management
   - Models saved to: `models/specialists/market_making/`

3. **Volatility Trading (PPO)**
   - Delta-hedged vol exposure
   - Models saved to: `models/specialists/volatility_trading/`

4. **Delta Hedging (DDPG)**
   - Optimal hedge ratios
   - Models saved to: `models/specialists/delta_hedging/`

5. **Futures Spreads (PPO)**
   - Calendar/inter-commodity spreads
   - Models saved to: `models/specialists/futures_spreads/`

6. **Factor Tracking (DQN)**
   - Multi-factor portfolio construction
   - Models saved to: `models/specialists/factor_tracking/`

7. **FX Arbitrage (DDPG)**
   - Triangular arbitrage & carry
   - Models saved to: `models/specialists/fx_arbitrage/`

### Phase 2: Train CIO Allocator
After specialists are trained:
- CIO learns to allocate capital across the 7 specialists
- Observes specialist performance metrics
- Outputs allocation weights
- Models saved to: `models/master/`

## 🚀 Quick Commands

### Test All Environments
```powershell
python tests/test_all_environments.py
```

### Test Individual Environment
```powershell
# Statistical Arbitrage
python src/environments/specialist_envs/Statistical_Arbitrage/env_stat_arb.py

# Market Making
python src/environments/specialist_envs/Market_Making/env_market_maker.py

# Volatility Trading
python src/environments/specialist_envs/Volatility_Trading/env_vol_trading.py

# Delta Hedging
python src/environments/specialist_envs/Delta_Hedging/env_delta_hedging.py

# Futures Spreads
python src/environments/specialist_envs/Futures_Spreads/env_futures_spread.py

# Factor Tracking
python src/environments/specialist_envs/Factor_Tracking/env_factor_tracker.py

# FX Arbitrage
python src/environments/specialist_envs/FX_Arbitrage/env_fx_arb.py
```

### Model Manager
```python
from src.utils.model_manager import ModelManager

manager = ModelManager()

# Save specialist model
manager.save_specialist(agent, 'statistical_arbitrage', metrics, is_best=True)

# Load specialist model
manager.load_specialist(agent, 'statistical_arbitrage', best=True)

# Save CIO model
manager.save_cio(agent, metrics, is_best=True)

# Print summary
manager.print_summary()
```

### Plotting
```python
from src.utils.plotting import PerformancePlotter

plotter = PerformancePlotter()

# Training progress
plotter.plot_training_progress(rewards, losses)

# Equity curve
plotter.plot_equity_curve(portfolio_values, timestamps)

# Strategy comparison
plotter.plot_strategy_comparison(strategy_dict)

# Full dashboard
plotter.create_performance_dashboard(
    rewards, portfolio_values, strategy_returns, 
    allocation_weights, metrics, save_path='dashboard.png'
)
```

## 📊 Key Metrics

Each specialist tracks:
- Portfolio value
- Sharpe ratio
- Max drawdown
- Win rate
- Total trades
- Transaction costs

CIO allocator optimizes:
- Overall fund Sharpe ratio
- Risk-adjusted returns
- Diversification
- Downside protection

## 🔧 Configuration

All parameters in `src/utils/strategy_config.py`:
- Training hyperparameters
- Environment parameters
- Agent architectures
- Model save paths

## 📈 Next Steps

1. ✅ All environments created
2. ✅ Model management ready
3. ✅ Plotting utilities ready
4. ⏭️ Train specialist agents
5. ⏭️ Train CIO allocator
6. ⏭️ Backtest full system
7. ⏭️ Generate performance report

## 💡 Tips

- Start with small training runs to verify setup
- Use checkpointing for long training sessions
- Monitor rewards and losses during training
- Compare specialist performance before training CIO
- Adjust hyperparameters based on initial results
- Use visualization tools to debug agent behavior

## 📝 Notes

- PPO agents: On-policy, good for continuous control
- DDPG agents: Off-policy, deterministic policies
- DQN agent: Discrete actions, experience replay
- All environments inherit from `BaseTradingEnv`
- Transaction costs included in all environments
- Reward engineering balances profit vs risk
