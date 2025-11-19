# Hierarchical Deep Reinforcement Learning Multi-Strategy Fund

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)]()

## Overview

This project implements a **hierarchical deep reinforcement learning (DRL) framework** for systematic multi-strategy portfolio management. Inspired by organizational structures in quantitative hedge funds, the system features a two-level hierarchy:

- **Specialist Agents**: Seven independent DRL agents, each specialized in a distinct trading strategy (statistical arbitrage, market making, volatility trading, delta hedging, futures spreads, factor tracking, and FX arbitrage)
- **Master Agent (CIO Allocator)**: A meta-agent that dynamically allocates capital across specialist strategies based on market conditions and portfolio-level objectives

This architecture mirrors real-world institutional investment processes while leveraging modern deep reinforcement learning techniques to optimize both strategy execution and capital allocation.

---

## Key Features

### Multi-Strategy Framework
- **7 Specialist Trading Strategies** spanning multiple asset classes (equities, futures, FX)
- **Hierarchical Architecture** with independent specialist agents and a master allocator
- **Custom Gymnasium Environments** for each trading strategy with realistic market dynamics
- **Modular Design** allowing for easy extension and strategy addition

### Advanced DRL Algorithms
- **PPO (Proximal Policy Optimization)** for policy-based learning
- **DDPG (Deep Deterministic Policy Gradient)** for continuous action spaces
- **DQN (Deep Q-Network)** for discrete action selection
- **Custom Policy Networks** tailored to trading objectives

### Quantitative Research Pipeline
- **Comprehensive Backtesting Engine** with transaction cost modeling
- **Risk-Adjusted Performance Metrics** (Sharpe ratio, Sortino ratio, max drawdown, etc.)
- **Feature Engineering Pipeline** for market data preprocessing
- **Detailed Performance Attribution** across strategies and time periods

---

## Project Structure

```
├── data/
│   ├── raw/                    # Raw market data (equities, futures, FX)
│   └── processed/              # Preprocessed features and datasets
├── src/
│   ├── agents/                 # DRL agent implementations (PPO, DDPG, DQN)
│   ├── environments/           # Custom trading environments
│   │   ├── specialist_envs/    # 7 specialist strategy environments
│   │   └── master_env/         # CIO allocator environment
│   ├── data/                   # Data loading and feature engineering
│   ├── backtesting/            # Backtesting engine and metrics
│   └── utils/                  # Configuration and utilities
├── models/
│   ├── specialists/            # Trained specialist agent models
│   └── master/                 # Trained master allocator models
├── notebooks/                  # Research and analysis notebooks
├── reports/                    # Research paper and visualizations
└── environment.yml             # Conda environment specification
```

---

## Specialist Trading Strategies

| Strategy | Asset Class | Description | Agent Type |
|----------|-------------|-------------|------------|
| **Statistical Arbitrage** | Equities | Pair trading and mean reversion on cointegrated stocks | PPO |
| **Market Making** | Equities | Limit order placement with inventory management | DDPG |
| **Volatility Trading** | Options | Delta-hedged option positions exploiting vol surface dynamics | PPO |
| **Delta Hedging** | Options/Equities | Dynamic hedging of option Greeks | DDPG |
| **Futures Spreads** | Futures | Calendar and inter-commodity spread trading | PPO |
| **Factor Tracking** | Equities | Smart beta and factor portfolio construction | DQN |
| **FX Arbitrage** | Foreign Exchange | Triangular arbitrage and carry trade strategies | DDPG |

---

## Performance Metrics

### Portfolio-Level Performance

| Metric | Value |
|--------|-------|
| **Annualized Return** | NaN |
| **Annualized Volatility** | NaN |
| **Sharpe Ratio** | NaN |
| **Sortino Ratio** | NaN |
| **Maximum Drawdown** | NaN |
| **Calmar Ratio** | NaN |
| **Win Rate** | NaN |
| **Profit Factor** | NaN |

### Strategy-Level Performance

| Strategy | Sharpe | Max DD | Annual Return | Correlation to SPX |
|----------|--------|--------|---------------|---------------------|
| Statistical Arbitrage | NaN | NaN | NaN | NaN |
| Market Making | NaN | NaN | NaN | NaN |
| Volatility Trading | NaN | NaN | NaN | NaN |
| Delta Hedging | NaN | NaN | NaN | NaN |
| Futures Spreads | NaN | NaN | NaN | NaN |
| Factor Tracking | NaN | NaN | NaN | NaN |
| FX Arbitrage | NaN | NaN | NaN | NaN |

*Note: Project currently in development. Metrics will be updated upon completion.*

---

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/DataHiveMind/The_Hierarchical_DRL_Multi_Strategy_Fund_Code.git
cd The_Hierarchical_DRL_Multi_Strategy_Fund_Code
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate hrl_fund
```

3. **Prepare data** (place raw data in `data/raw/` directories)

---

## Usage

### Training Specialist Agents

```python
# Example: Train the statistical arbitrage specialist
from src.agents.ppo import PPOAgent
from src.environments.specialist_envs.env_stat_arb import StatArbEnv

env = StatArbEnv(data_path="data/processed/equities_pairs.csv")
agent = PPOAgent(env)
agent.train(total_timesteps=1_000_000)
agent.save("models/specialists/stat_arb")
```

### Training Master Allocator

```python
# Train the CIO allocator to manage specialist strategies
from src.agents.ppo import PPOAgent
from src.environments.master_env.env_cio_allocator import CIOAllocatorEnv

env = CIOAllocatorEnv(specialist_models_path="models/specialists/")
master_agent = PPOAgent(env)
master_agent.train(total_timesteps=500_000)
master_agent.save("models/master/cio_allocator")
```

### Backtesting

```python
# Run comprehensive backtest
from src.backtesting.engine import BacktestEngine

engine = BacktestEngine(
    master_model="models/master/cio_allocator",
    specialist_models="models/specialists/",
    test_data_path="data/processed/test_set/"
)
results = engine.run()
engine.generate_report(output_dir="reports/")
```

---

## Methodology

### Hierarchical Architecture

The system employs a two-tier hierarchy inspired by institutional fund management:

1. **Specialist Level**: Each agent independently learns optimal execution for its specific strategy using domain-specific state representations and reward functions

2. **Master Level**: The CIO allocator observes specialist performance, market regime indicators, and portfolio-level risk metrics to dynamically allocate capital

### State Spaces

- **Specialists**: Market microstructure features, technical indicators, position data
- **Master**: Specialist returns, volatilities, correlations, portfolio metrics, regime indicators

### Reward Functions

- **Specialists**: Strategy-specific PnL with transaction costs and risk penalties
- **Master**: Portfolio Sharpe ratio optimization with drawdown constraints

### Training Approach

- **Curriculum Learning**: Specialists trained first on historical data
- **Hierarchical Training**: Master agent trained using pre-trained specialist policies
- **Online Adaptation**: Continuous learning with expanding window backtesting

---

## Research Notebooks

Explore the development process through detailed Jupyter notebooks:

1. `00_data_loading_and_eda.ipynb` - Data exploration and preprocessing
2. `01_specialist_env_testing.ipynb` - Environment validation and debugging
3. `02_specialist_agent_training.ipynb` - Specialist agent training experiments
4. `03_master_env_testing.ipynb` - Master environment validation
5. `04_master_agent_training.ipynb` - Master agent training experiments
6. `05_results_and_visualization.ipynb` - Performance analysis and visualization

---

## Technical Stack

- **Deep Learning**: PyTorch, Stable-Baselines3
- **Environment**: Gymnasium (OpenAI Gym)
- **Data Processing**: Pandas, NumPy, Polars
- **Backtesting**: Custom engine with vectorized operations
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Research**: Jupyter, Scikit-learn

---

## Roadmap

- [x] Design hierarchical architecture
- [x] Implement specialist environments
- [x] Implement master allocator environment
- [ ] Collect and preprocess market data
- [ ] Train specialist agents
- [ ] Train master allocator
- [ ] Comprehensive backtesting
- [ ] Performance attribution analysis
- [ ] Research paper writing
- [ ] Hyperparameter optimization
- [ ] Walk-forward validation

---

## Contributing

This is a research project. Suggestions and feedback are welcome via issues or pull requests.

---

## License

MIT License - see LICENSE file for details

---

## Contact

**Author**: Kenneth (DataHiveMind)  
**Repository**: [github.com/DataHiveMind/The_Hierarchical_DRL_Multi_Strategy_Fund_Code](https://github.com/DataHiveMind/The_Hierarchical_DRL_Multi_Strategy_Fund_Code)

---

## Acknowledgments

This project draws inspiration from:
- Modern quantitative hedge fund organizational structures
- Recent advances in hierarchical reinforcement learning
- Industry best practices in systematic trading and risk management

---

*This project is for research and educational purposes. Past performance does not guarantee future results.*
