# Agent-Strategy Assignment for Hierarchical DRL Multi-Strategy Fund

## Overview

This document outlines the assignment of Deep Reinforcement Learning agents to specialist trading strategies and the CIO allocator in our hierarchical framework.

---

## 🎯 Agent Distribution

### **PPO (Proximal Policy Optimization)** - 3 Strategies + CIO
**Best for:** Continuous action spaces with policy-based learning

1. **Statistical Arbitrage** (Equities)
   - Continuous portfolio weights between cointegrated pairs
   - Handles non-stationary pair relationships
   - PPO stability crucial for mean-reversion

2. **Volatility Trading** (Options)
   - Delta-hedged option position sizing
   - Complex volatility surface dynamics
   - Clipped updates prevent catastrophic losses

3. **Futures Spreads** (Futures)
   - Calendar and inter-commodity spread positions
   - Captures seasonal patterns
   - On-policy updates for evolving spreads

**CIO Allocator:**
- Continuous allocation weights across 7 specialists
- Sample-efficient for portfolio-level decisions
- Adapts to market regime changes

---

### **DDPG (Deep Deterministic Policy Gradient)** - 3 Strategies
**Best for:** Continuous control with deterministic policies

1. **Market Making** (Equities)
   - Precise bid/ask spread control
   - Inventory risk management
   - High-frequency decision making

2. **Delta Hedging** (Options/Equities)
   - Continuous hedge ratio adjustments
   - Precise Greek (delta, gamma, vega) management
   - Off-policy learning from hedging history

3. **FX Arbitrage** (Forex)
   - Currency triangle position sizing
   - Carry trade optimization
   - Cross-currency dynamics

---

### **DQN (Deep Q-Network)** - 1 Strategy
**Best for:** Discrete action spaces with value-based learning

1. **Factor Tracking** (Equities)
   - Discrete factor exposures (long/neutral/short)
   - Factor timing decisions
   - Experience replay from historical data

---

## 📊 Training Pipeline

### **Phase 1: Specialist Training** (Independent)
```
Statistical Arbitrage (PPO)    → 500K timesteps
Market Making (DDPG)           → 1M timesteps
Volatility Trading (PPO)       → 500K timesteps
Delta Hedging (DDPG)           → 750K timesteps
Futures Spreads (PPO)          → 500K timesteps
Factor Tracking (DQN)          → 500K timesteps
FX Arbitrage (DDPG)            → 750K timesteps
```

**Parallel Training:** ✓ Enabled  
**Save Frequency:** Every 10K steps  
**Evaluation:** Every 5K steps (10 episodes)

### **Phase 2: CIO Allocator Training**
```
CIO Allocator (PPO)            → 200K timesteps
```

**Specialist Loading:** Pre-trained specialists loaded  
**Fine-tuning:** Allowed during CIO training  
**Save Frequency:** Every 5K steps  
**Evaluation:** Every 2K steps (5 episodes)

### **Phase 3: Joint Fine-tuning** (Optional)
```
All Agents                     → 50K timesteps
```

**Specialist Learning Rate:** 0.1× original (fine-tuning only)

---

## 🔧 Key Configuration Parameters

### Transaction Costs by Strategy
- Market Making: 0.01 bps (market maker rebates)
- FX Arbitrage: 0.02 bps (tight FX spreads)
- Futures Spreads: 0.05 bps (low futures costs)
- Statistical Arbitrage: 0.10 bps (standard equity costs)
- Delta Hedging: 0.10 bps
- Factor Tracking: 0.10 bps
- Volatility Trading: 0.20 bps (options have higher costs)

### Leverage Limits
- FX Arbitrage: 10× (FX standard)
- Futures Spreads: 3× (futures margin)
- Market Making: 2×
- Statistical Arbitrage: 1.5×
- All Others: 1× (unleveraged)

### Portfolio Constraints (CIO)
- Max Single Strategy: 40%
- Min Single Strategy: 5%
- Max Turnover: 50% per rebalance
- Target Volatility: 12% annual

---

## 💡 Design Rationale

### Why PPO for Statistical Arbitrage?
- Mean-reversion requires stable policy updates
- Continuous positions in pair spreads
- Non-stationary cointegration relationships

### Why DDPG for Market Making?
- Deterministic bid/ask quotes needed
- Inventory management is continuous control
- Off-policy learning from order book data

### Why DQN for Factor Tracking?
- Factor exposures are discrete choices
- Value-based learning for timing
- Replay buffer leverages historical factor data

### Why PPO for CIO Allocator?
- Capital allocation is continuous
- Sample efficiency for portfolio decisions
- On-policy adapts to regime changes
- No catastrophic forgetting with clipping

---

## 📁 Model Storage Structure

```
models/
├── specialists/
│   ├── statistical_arbitrage_best.zip
│   ├── market_making_best.zip
│   ├── volatility_trading_best.zip
│   ├── delta_hedging_best.zip
│   ├── futures_spreads_best.zip
│   ├── factor_tracking_best.zip
│   └── fx_arbitrage_best.zip
├── master/
│   └── cio_allocator_best.zip
├── checkpoints/
│   └── [timestamped checkpoints]
└── logs/
    ├── specialists/
    └── master/
```

---

## 🚀 Quick Start

```python
from src.utils.strategy_config import (
    SPECIALIST_STRATEGIES,
    CIO_ALLOCATOR_CONFIG,
    get_strategy_summary,
    get_agent_class
)

# View configuration
print(get_strategy_summary())

# Get agent for a strategy
config = SPECIALIST_STRATEGIES['statistical_arbitrage']
AgentClass = get_agent_class(config.agent_type)  # Returns PPOAgent

# Access training parameters
print(config.training_params)
print(config.environment_params)
```

---

## 📈 Expected Performance Characteristics

### Specialist Strategies
- **Statistical Arbitrage:** Sharpe 1.5-2.5, low correlation to market
- **Market Making:** Sharpe 2.0-3.0, market-neutral
- **Volatility Trading:** Sharpe 1.0-1.5, negative market beta
- **Delta Hedging:** Sharpe 0.8-1.2, market-neutral
- **Futures Spreads:** Sharpe 1.5-2.0, commodity exposure
- **Factor Tracking:** Sharpe 1.0-1.5, systematic equity
- **FX Arbitrage:** Sharpe 1.5-2.0, uncorrelated

### Portfolio (CIO Allocation)
- **Target Sharpe:** 2.0-2.5 (diversification benefit)
- **Target Volatility:** 12% annual
- **Max Drawdown:** <15%
- **Correlation to SPX:** <0.3

---

*Configuration validated and ready for training pipeline execution.*
