"""
Proximal Policy Optimization (PPO) Agent implementation.

PPO is a policy gradient method that uses a clipped objective function to
prevent overly large policy updates. It's well-suited for continuous control
tasks like trading strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
import gymnasium as gym
from collections import deque
import os

from src.agents.policy_networks import PolicyValueNetwork


class PPOAgent:
    """
    PPO Agent for continuous and discrete action spaces.

    Suitable for:
    - Statistical Arbitrage
    - Volatility Trading
    - Futures Spreads
    - Factor Tracking (if using continuous actions)
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "auto",
    ):
        """
        Initialize PPO Agent.

        Args:
            env: Gymnasium environment
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            n_steps: Number of steps to collect before update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]

        # Determine if continuous or discrete
        if isinstance(env.action_space, gym.spaces.Box):
            self.continuous = True
            self.action_dim = env.action_space.shape[0]
            self.action_scale = torch.FloatTensor(
                (env.action_space.high - env.action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (env.action_space.high + env.action_space.low) / 2.0
            )
        else:
            self.continuous = False
            self.action_dim = env.action_space.n
            self.action_scale = None
            self.action_bias = None

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize network
        self.policy_value_net = PolicyValueNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            continuous=self.continuous,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(), lr=learning_rate
        )

        # Move action scale/bias to device if continuous
        if self.continuous:
            self.action_scale = self.action_scale.to(self.device)
            self.action_bias = self.action_bias.to(self.device)

        # Storage for trajectory
        self.reset_storage()

        # Metrics
        self.training_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "episode_rewards": [],
        }

    def reset_storage(self):
        """Reset trajectory storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given state.

        Args:
            state: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.policy_value_net.get_action_distribution(state_tensor)
            _, value = self.policy_value_net(state_tensor)

            if deterministic:
                if self.continuous:
                    action = dist.mean
                else:
                    action = dist.probs.argmax(dim=-1, keepdim=True)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            if self.continuous:
                log_prob = log_prob.sum(dim=-1)

        # Scale action if continuous
        if self.continuous:
            action_scaled = action * self.action_scale + self.action_bias
            action_np = action_scaled.cpu().numpy()[0]
        else:
            action_np = action.cpu().numpy()[0]

        return action_np, log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self):
        """Update policy using collected trajectory."""
        # Get next value for GAE calculation
        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.policy_value_net(last_state)
            last_value = last_value.cpu().numpy()[0][0]

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.rewards, self.values, self.dones, last_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).to(
            self.device
        )
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO update
        dataset_size = len(self.states)

        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy_value_net.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = nn.MSELoss()(values, batch_returns.unsqueeze(-1))

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_value_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Record metrics
                self.training_metrics["policy_loss"].append(policy_loss.item())
                self.training_metrics["value_loss"].append(value_loss.item())
                self.training_metrics["entropy"].append(-entropy_loss.item())
                self.training_metrics["total_loss"].append(loss.item())

        # Reset storage
        self.reset_storage()

    def train(self, total_timesteps: int, callback=None, log_interval: int = 10):
        """
        Train the agent.

        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback function
            log_interval: Logging interval
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_count = 0
        timestep = 0

        print(f"Training PPO agent for {total_timesteps} timesteps...")
        print(f"Device: {self.device}")

        while timestep < total_timesteps:
            # Collect trajectory
            for _ in range(self.n_steps):
                action, log_prob, value = self.select_action(state)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.values.append(value)
                self.log_probs.append(log_prob)

                episode_reward += reward
                timestep += 1

                if done:
                    self.training_metrics["episode_rewards"].append(episode_reward)
                    episode_count += 1

                    if episode_count % log_interval == 0:
                        avg_reward = np.mean(
                            self.training_metrics["episode_rewards"][-10:]
                        )
                        print(
                            f"Episode {episode_count}, Timestep {timestep}, Avg Reward: {avg_reward:.2f}"
                        )

                    state, _ = self.env.reset()
                    episode_reward = 0
                else:
                    state = next_state

                if timestep >= total_timesteps:
                    break

            # Update policy
            if len(self.states) > 0:
                self.update()

            if callback:
                callback(self, timestep)

        print(f"Training completed! Total episodes: {episode_count}")

    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_value_net": self.policy_value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_metrics": self.training_metrics,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_value_net.load_state_dict(checkpoint["policy_value_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_metrics = checkpoint["training_metrics"]
        print(f"Model loaded from {path}")
