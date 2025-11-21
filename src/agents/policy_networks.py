"""
Neural network architectures for policy and value functions.

This module provides custom PyTorch neural networks tailored for trading:
- ActorNetwork: Policy network for continuous action spaces
- CriticNetwork: Value network for actor-critic methods
- QNetwork: Q-value network for DQN
- PolicyValueNetwork: Combined policy and value network for PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor network for continuous action spaces (DDPG, TD3).
    Outputs deterministic actions with tanh activation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize Actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout rate (0.0 means no dropout)
        """
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Activation function
        self.activation = self._get_activation(activation)

        # Build layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, action_dim)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        # Small initialization for output layer
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Action tensor with tanh activation (range [-1, 1])
        """
        x = self.feature_extractor(state)
        action = torch.tanh(self.output_layer(x))
        return action


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation (DDPG, TD3, SAC).
    Estimates Q-value given state and action.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
        """
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.activation = self._get_activation(activation)

        # First layer processes state
        self.state_layer = nn.Linear(state_dim, hidden_dims[0])
        self.state_norm = nn.LayerNorm(hidden_dims[0])

        # Second layer combines state features with action
        self.combined_layer = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.combined_norm = nn.LayerNorm(hidden_dims[1])

        # Additional hidden layers
        layers = []
        prev_dim = hidden_dims[1]
        for hidden_dim in hidden_dims[2:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

        self._initialize_weights()

    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Q-value estimate
        """
        # Process state
        x = self.activation(self.state_norm(self.state_layer(state)))

        # Combine with action
        x = torch.cat([x, action], dim=-1)
        x = self.activation(self.combined_norm(self.combined_layer(x)))

        # Additional hidden layers
        x = self.hidden_layers(x)

        # Output Q-value
        q_value = self.output_layer(x)
        return q_value


class QNetwork(nn.Module):
    """
    Q-Network for DQN (discrete action spaces).
    Outputs Q-values for all actions given a state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        dueling: bool = False,
    ):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            dueling: Whether to use dueling architecture
        """
        super(QNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling

        self.activation = self._get_activation(activation)

        # Feature extraction layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                self.activation,
                nn.Linear(prev_dim // 2, 1),
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                self.activation,
                nn.Linear(prev_dim // 2, action_dim),
            )
        else:
            # Standard DQN
            self.output_layer = nn.Linear(prev_dim, action_dim)

        self._initialize_weights()

    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Q-values for all actions
        """
        features = self.feature_extractor(state)

        if self.dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            q_values = self.output_layer(features)

        return q_values


class PolicyValueNetwork(nn.Module):
    """
    Combined policy and value network for PPO.
    Shares feature extraction between policy and value functions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        dropout: float = 0.0,
        continuous: bool = True,
    ):
        """
        Initialize PolicyValue network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            continuous: Whether action space is continuous
        """
        super(PolicyValueNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        self.activation = self._get_activation(activation)

        # Shared feature extraction
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Policy head
        if continuous:
            # Output mean and log_std for continuous actions
            self.policy_mean = nn.Linear(prev_dim, action_dim)
            self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Output action probabilities for discrete actions
            self.policy_head = nn.Linear(prev_dim, action_dim)

        # Value head
        self.value_head = nn.Linear(prev_dim, 1)

        self._initialize_weights()

    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Small initialization for policy output
        if self.continuous:
            nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

        # Initialize value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Tuple of (policy_output, value)
            - For continuous: (action_mean, value)
            - For discrete: (action_logits, value)
        """
        features = self.shared_layers(state)

        if self.continuous:
            action_mean = self.policy_mean(features)
            value = self.value_head(features)
            return action_mean, value
        else:
            action_logits = self.policy_head(features)
            value = self.value_head(features)
            return action_logits, value

    def get_action_distribution(self, state: torch.Tensor):
        """
        Get action distribution for the policy.

        Args:
            state: State tensor

        Returns:
            torch.distributions object
        """
        features = self.shared_layers(state)

        if self.continuous:
            action_mean = self.policy_mean(features)
            action_std = torch.exp(self.policy_log_std)
            return torch.distributions.Normal(action_mean, action_std)
        else:
            action_logits = self.policy_head(features)
            return torch.distributions.Categorical(logits=action_logits)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """
        Evaluate actions for PPO update.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        dist = self.get_action_distribution(state)
        log_probs = dist.log_prob(action)

        if self.continuous:
            log_probs = log_probs.sum(dim=-1, keepdim=True)

        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1, keepdim=True)

        _, values = self.forward(state)

        return log_probs, values, entropy
