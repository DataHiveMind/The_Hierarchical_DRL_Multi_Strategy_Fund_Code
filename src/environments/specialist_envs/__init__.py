"""
Specialist Environments Package

All trading environments for the hierarchical DRL multi-strategy fund.
Each environment is designed for specific agent types (PPO, DDPG, DQN).
"""

# Statistical Arbitrage (PPO)
from .stats_arb.env_stat_arb import StatisticalArbitrageEnv

# Market Making (DDPG)
from .Market_Making.env_market_maker import MarketMakingEnv

# Volatility Trading (PPO)
from .Volatility_Trading.env_vol_trading import VolatilityTradingEnv

# Delta Hedging (DDPG)
from .Delta_Hedging.env_delta_hedging import DeltaHedgingEnv

# Futures Spreads (PPO)
from .Futures_Spreads.env_futures_spread import FuturesSpreadsEnv

# Factor Tracking (DQN)
from .Factor_Tracking.env_factor_tracker import FactorTrackingEnv

# FX Arbitrage (DDPG)
from .FX_Arbitrage.env_fx_arb import FXArbitrageEnv

# Base environment
from .base_trading_env import BaseTradingEnv


__all__ = [
    "StatisticalArbitrageEnv",
    "MarketMakingEnv",
    "VolatilityTradingEnv",
    "DeltaHedgingEnv",
    "FuturesSpreadsEnv",
    "FactorTrackingEnv",
    "FXArbitrageEnv",
    "BaseTradingEnv",
]


# Environment registry for easy instantiation
SPECIALIST_ENVS = {
    "statistical_arbitrage": {
        "class": StatisticalArbitrageEnv,
        "agent_type": "PPO",
        "description": "Pairs trading with mean reversion",
    },
    "market_making": {
        "class": MarketMakingEnv,
        "agent_type": "DDPG",
        "description": "Quote posting with inventory management",
    },
    "volatility_trading": {
        "class": VolatilityTradingEnv,
        "agent_type": "PPO",
        "description": "Delta-hedged volatility trading",
    },
    "delta_hedging": {
        "class": DeltaHedgingEnv,
        "agent_type": "DDPG",
        "description": "Dynamic option delta hedging",
    },
    "futures_spreads": {
        "class": FuturesSpreadsEnv,
        "agent_type": "PPO",
        "description": "Calendar and inter-commodity spreads",
    },
    "factor_tracking": {
        "class": FactorTrackingEnv,
        "agent_type": "DQN",
        "description": "Multi-factor portfolio construction",
    },
    "fx_arbitrage": {
        "class": FXArbitrageEnv,
        "agent_type": "DDPG",
        "description": "FX triangular arbitrage and carry",
    },
}


def get_specialist_env(strategy_name: str, **kwargs):
    """
    Get specialist environment by strategy name.

    Args:
        strategy_name: Name of strategy
        **kwargs: Environment initialization parameters

    Returns:
        Environment instance
    """
    if strategy_name not in SPECIALIST_ENVS:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(SPECIALIST_ENVS.keys())}"
        )

    env_class = SPECIALIST_ENVS[strategy_name]["class"]
    return env_class(**kwargs)


def list_specialist_envs():
    """Print all available specialist environments."""
    print("\nSpecialist Trading Environments")
    print("=" * 80)

    for name, info in SPECIALIST_ENVS.items():
        print(f"\n{name.upper().replace('_', ' ')}")
        print(f"  Agent: {info['agent_type']}")
        print(f"  Description: {info['description']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    list_specialist_envs()
