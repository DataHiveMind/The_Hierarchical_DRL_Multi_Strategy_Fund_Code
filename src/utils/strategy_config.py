"""
Strategy Configuration for Hierarchical DRL Multi-Strategy Fund

This module defines the mapping between specialist trading strategies
and their corresponding DRL agents, along with training configurations.

Agent-Strategy Mapping Rationale:
---------------------------------

1. STATISTICAL ARBITRAGE -> PPO
   - Continuous action space for portfolio weights between pairs
   - Policy-based learning handles the non-stationary nature of pairs
   - PPO's stability is crucial for mean-reversion strategies

2. MARKET MAKING -> DDPG
   - Continuous bid/ask spread control and inventory management
   - Deterministic policy for precise quote placement
   - Actor-critic suitable for high-frequency decision making

3. VOLATILITY TRADING -> PPO
   - Continuous actions for delta-hedged option positions
   - Policy gradient methods handle complex volatility surface dynamics
   - PPO's clipping prevents catastrophic updates in volatile markets

4. DELTA HEDGING -> DDPG
   - Continuous hedge ratio adjustments
   - Deterministic policy for precise Greek management
   - Off-policy learning from historical hedging data

5. FUTURES SPREADS -> PPO
   - Continuous positions in calendar and inter-commodity spreads
   - Policy-based learning for seasonal patterns
   - On-policy updates align with evolving spread relationships

6. FACTOR TRACKING -> DQN
   - Discrete portfolio construction (factor exposures: long/neutral/short)
   - Value-based learning for factor timing decisions
   - Experience replay benefits from historical factor performance

7. FX ARBITRAGE -> DDPG
   - Continuous position sizing in currency triangles
   - Deterministic policy for carry trade optimization
   - Actor-critic handles complex cross-currency dynamics

CIO ALLOCATOR -> PPO
   - Continuous allocation weights across 7 specialists
   - Policy-based learning for capital allocation under constraints
   - PPO's sample efficiency crucial for portfolio-level decisions
   - On-policy learning adapts to regime changes
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np


@dataclass
class StrategyConfig:
    """Configuration for a specialist trading strategy."""

    name: str
    agent_type: str  # 'PPO', 'DDPG', or 'DQN'
    asset_class: str
    action_space_type: str  # 'continuous' or 'discrete'
    description: str
    training_params: Dict[str, Any]
    environment_params: Dict[str, Any]


# =============================================================================
# SPECIALIST STRATEGY CONFIGURATIONS
# =============================================================================

SPECIALIST_STRATEGIES: Dict[str, StrategyConfig] = {
    "statistical_arbitrage": StrategyConfig(
        name="Statistical Arbitrage",
        agent_type="PPO",
        asset_class="equities",
        action_space_type="continuous",
        description="Pair trading and mean reversion on cointegrated stocks",
        training_params={
            "total_timesteps": 500_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.001,
            "slippage_pct": 0.0005,
            "max_position_size": 100_000,
            "max_leverage": 1.5,
            "lookback_window": 60,
        },
    ),
    "market_making": StrategyConfig(
        name="Market Making",
        agent_type="DDPG",
        asset_class="equities",
        action_space_type="continuous",
        description="Limit order placement with inventory management",
        training_params={
            "total_timesteps": 1_000_000,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 1_000_000,
            "batch_size": 128,
            "warmup_steps": 10_000,
            "update_frequency": 1,
            "noise_theta": 0.15,
            "noise_sigma": 0.2,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.0001,  # Market makers pay less
            "slippage_pct": 0.0001,
            "max_position_size": 50_000,
            "max_leverage": 2.0,
            "lookback_window": 30,
            "inventory_penalty": 0.01,  # Penalize large inventory
        },
    ),
    "volatility_trading": StrategyConfig(
        name="Volatility Trading",
        agent_type="PPO",
        asset_class="options",
        action_space_type="continuous",
        description="Delta-hedged option positions exploiting vol surface dynamics",
        training_params={
            "total_timesteps": 500_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.995,  # Higher gamma for options
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.005,
            "max_grad_norm": 0.5,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.002,  # Options have higher costs
            "slippage_pct": 0.001,
            "max_position_size": 1000,  # Option contracts
            "max_leverage": 1.0,
            "lookback_window": 60,
            "vega_limit": 100_000,  # Volatility risk limit
        },
    ),
    "delta_hedging": StrategyConfig(
        name="Delta Hedging",
        agent_type="DDPG",
        asset_class="options_equities",
        action_space_type="continuous",
        description="Dynamic hedging of option Greeks",
        training_params={
            "total_timesteps": 750_000,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "gamma": 0.995,
            "tau": 0.005,
            "buffer_size": 500_000,
            "batch_size": 128,
            "warmup_steps": 10_000,
            "update_frequency": 1,
            "noise_theta": 0.15,
            "noise_sigma": 0.15,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.001,
            "slippage_pct": 0.0005,
            "max_position_size": 100_000,
            "max_leverage": 1.0,
            "lookback_window": 30,
            "gamma_limit": 50_000,  # Gamma risk limit
        },
    ),
    "futures_spreads": StrategyConfig(
        name="Futures Spreads",
        agent_type="PPO",
        asset_class="futures",
        action_space_type="continuous",
        description="Calendar and inter-commodity spread trading",
        training_params={
            "total_timesteps": 500_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.0005,  # Futures have low costs
            "slippage_pct": 0.0003,
            "max_position_size": 100,  # Contracts
            "max_leverage": 3.0,  # Futures allow higher leverage
            "margin_requirement": 0.05,
            "lookback_window": 60,
        },
    ),
    "factor_tracking": StrategyConfig(
        name="Factor Tracking",
        agent_type="DQN",
        asset_class="equities",
        action_space_type="discrete",
        description="Smart beta and factor portfolio construction",
        training_params={
            "total_timesteps": 500_000,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 50_000,
            "buffer_size": 100_000,
            "batch_size": 64,
            "target_update_frequency": 1000,
            "warmup_steps": 10_000,
            "update_frequency": 4,
            "use_dueling": True,
            "use_double_dqn": True,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.001,
            "slippage_pct": 0.0005,
            "max_position_size": None,  # Portfolio construction
            "max_leverage": 1.0,
            "lookback_window": 60,
            "n_factors": 5,  # Value, Momentum, Quality, Size, Low Vol
            "rebalance_frequency": 5,  # Days
        },
    ),
    "fx_arbitrage": StrategyConfig(
        name="FX Arbitrage",
        agent_type="DDPG",
        asset_class="forex",
        action_space_type="continuous",
        description="Triangular arbitrage and carry trade strategies",
        training_params={
            "total_timesteps": 750_000,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 500_000,
            "batch_size": 128,
            "warmup_steps": 10_000,
            "update_frequency": 1,
            "noise_theta": 0.15,
            "noise_sigma": 0.2,
        },
        environment_params={
            "initial_balance": 1_000_000,
            "transaction_cost_pct": 0.0002,  # Low FX spreads
            "slippage_pct": 0.0001,
            "max_position_size": 10_000_000,  # FX notional
            "max_leverage": 10.0,  # FX allows high leverage
            "margin_requirement": 0.02,
            "lookback_window": 30,
        },
    ),
}


# =============================================================================
# CIO ALLOCATOR CONFIGURATION
# =============================================================================

CIO_ALLOCATOR_CONFIG = {
    "agent_type": "PPO",
    "training_params": {
        "total_timesteps": 200_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.0,  # No entropy for allocator
        "max_grad_norm": 0.5,
    },
    "environment_params": {
        "initial_capital": 10_000_000,
        "rebalance_frequency": 5,  # Days
        "lookback_window": 60,
        "transaction_cost": 0.001,
        "max_single_allocation": 0.40,  # Max 40% to any strategy
        "min_single_allocation": 0.05,  # Min 5% to each strategy
        "max_turnover": 0.50,  # Max 50% portfolio turnover
        "target_volatility": 0.12,  # 12% annual volatility target
        "risk_free_rate": 0.02,
    },
}


# =============================================================================
# TRAINING PIPELINE CONFIGURATION
# =============================================================================

TRAINING_PIPELINE = {
    "phase_1_specialists": {
        "description": "Train specialist agents independently on their strategies",
        "strategies": list(SPECIALIST_STRATEGIES.keys()),
        "parallel_training": True,
        "save_frequency": 10_000,
        "eval_frequency": 5_000,
        "n_eval_episodes": 10,
    },
    "phase_2_cio": {
        "description": "Train CIO allocator using pre-trained specialists",
        "load_specialists": True,
        "freeze_specialists": False,  # Allow fine-tuning during CIO training
        "save_frequency": 5_000,
        "eval_frequency": 2_000,
        "n_eval_episodes": 5,
    },
    "phase_3_joint": {
        "description": "Optional: Joint fine-tuning of all agents",
        "enabled": False,
        "total_timesteps": 50_000,
        "specialist_lr_multiplier": 0.1,  # Lower LR for specialists
    },
}


# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_PATHS = {
    "specialists": {
        strategy: f"models/specialists/{strategy}_best.zip"
        for strategy in SPECIALIST_STRATEGIES.keys()
    },
    "cio": "models/master/cio_allocator_best.zip",
    "checkpoints": "models/checkpoints/",
    "logs": "logs/",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_agent_class(agent_type: str):
    """Get agent class based on type string."""
    from src.agents.ppo import PPOAgent
    from src.agents.ddpg import DDPGAgent
    from src.agents.dqn import DQNAgent

    agent_map = {
        "PPO": PPOAgent,
        "DDPG": DDPGAgent,
        "DQN": DQNAgent,
    }

    return agent_map.get(agent_type)


def get_strategies_by_agent_type(agent_type: str) -> List[str]:
    """Get list of strategies using a specific agent type."""
    return [
        strategy_name
        for strategy_name, config in SPECIALIST_STRATEGIES.items()
        if config.agent_type == agent_type
    ]


def get_strategy_summary() -> str:
    """Generate a formatted summary of strategy-agent mapping."""
    summary = "\n" + "=" * 80 + "\n"
    summary += "HIERARCHICAL DRL MULTI-STRATEGY FUND - AGENT ASSIGNMENT\n"
    summary += "=" * 80 + "\n\n"

    # Group by agent type
    for agent_type in ["PPO", "DDPG", "DQN"]:
        strategies = get_strategies_by_agent_type(agent_type)
        if strategies:
            summary += f"\n{agent_type} Agent ({len(strategies)} strategies):\n"
            summary += "-" * 40 + "\n"
            for strategy in strategies:
                config = SPECIALIST_STRATEGIES[strategy]
                summary += f"  • {config.name}\n"
                summary += f"    Asset Class: {config.asset_class}\n"
                summary += f"    Action Space: {config.action_space_type}\n"
                summary += f"    Description: {config.description}\n\n"

    summary += "\n" + "=" * 80 + "\n"
    summary += "CIO ALLOCATOR\n"
    summary += "=" * 80 + "\n"
    summary += f"  Agent Type: {CIO_ALLOCATOR_CONFIG['agent_type']}\n"
    summary += f"  Action Space: Continuous (7 allocation weights)\n"
    summary += f"  Description: Dynamic capital allocation across specialists\n"
    summary += "=" * 80 + "\n"

    return summary


def validate_configuration():
    """Validate configuration consistency."""
    errors = []

    # Check all strategies have valid agent types
    valid_agents = ["PPO", "DDPG", "DQN"]
    for name, config in SPECIALIST_STRATEGIES.items():
        if config.agent_type not in valid_agents:
            errors.append(f"Invalid agent type for {name}: {config.agent_type}")

    # Check action space consistency
    for name, config in SPECIALIST_STRATEGIES.items():
        if config.agent_type == "DQN" and config.action_space_type != "discrete":
            errors.append(f"{name} uses DQN but has continuous action space")
        if (
            config.agent_type in ["PPO", "DDPG"]
            and config.action_space_type != "continuous"
        ):
            errors.append(
                f"{name} uses {config.agent_type} but has discrete action space"
            )

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

    return True


# Validate on import
validate_configuration()


if __name__ == "__main__":
    # Print configuration summary
    print(get_strategy_summary())

    # Print training pipeline
    print("\nTRAINING PIPELINE:")
    print("=" * 80)
    for phase, config in TRAINING_PIPELINE.items():
        print(f"\n{phase.upper().replace('_', ' ')}:")
        print(f"  {config['description']}")
        if "strategies" in config:
            print(f"  Strategies: {len(config['strategies'])}")
