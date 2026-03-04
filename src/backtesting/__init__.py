"""
Backtesting module for evaluating trading strategies and RL agents.
"""

from .engine import BacktestEngine, WalkForwardAnalysis
from .metrics import PerformanceMetrics, StrategyComparison, RollingMetrics
from .validation import SplitConfig, PurgedWalkForwardSplitter

__all__ = [
    "BacktestEngine",
    "WalkForwardAnalysis",
    "PerformanceMetrics",
    "StrategyComparison",
    "RollingMetrics",
    "SplitConfig",
    "PurgedWalkForwardSplitter",
]
