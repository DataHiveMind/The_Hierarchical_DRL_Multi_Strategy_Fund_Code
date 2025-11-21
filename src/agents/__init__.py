"""
Agent implementations for the Hierarchical DRL Multi-Strategy Fund.

This module provides various deep reinforcement learning agents including:
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- DQN (Deep Q-Network)

Each agent is designed to work with custom trading environments and can be
trained for different specialist trading strategies.
"""

from src.agents.ppo import PPOAgent
from src.agents.ddpg import DDPGAgent
from src.agents.dqn import DQNAgent
from src.agents.policy_networks import (
    ActorNetwork,
    CriticNetwork,
    QNetwork,
    PolicyValueNetwork,
)

__all__ = [
    "PPOAgent",
    "DDPGAgent",
    "DQNAgent",
    "ActorNetwork",
    "CriticNetwork",
    "QNetwork",
    "PolicyValueNetwork",
]

__version__ = "0.1.0"
