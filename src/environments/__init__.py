"""
Environments Package

Contains all trading environments for the hierarchical DRL multi-strategy fund:
- Specialist environments (7 trading strategies)
- Master environment (CIO allocator)
"""

from .specialist_envs import (
    StatisticalArbitrageEnv,
    MarketMakingEnv,
    VolatilityTradingEnv,
    DeltaHedgingEnv,
    FuturesSpreadsEnv,
    FactorTrackingEnv,
    FXArbitrageEnv,
    BaseTradingEnv,
    SPECIALIST_ENVS,
    get_specialist_env,
    list_specialist_envs,
)

from .master_env import CIOAllocatorEnv


__all__ = [
    # Specialist environments
    "StatisticalArbitrageEnv",
    "MarketMakingEnv",
    "VolatilityTradingEnv",
    "DeltaHedgingEnv",
    "FuturesSpreadsEnv",
    "FactorTrackingEnv",
    "FXArbitrageEnv",
    "BaseTradingEnv",
    # Master environment
    "CIOAllocatorEnv",
    # Utilities
    "SPECIALIST_ENVS",
    "get_specialist_env",
    "list_specialist_envs",
]
