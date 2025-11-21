"""
Deep Deterministic Policy Gradient (DDPG) Agent implementation.

DDPG is an actor-critic algorithm for continuous control that uses
deterministic policies and experience replay. Well-suited for trading
strategies requiring continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import gymnasium as gym
from collections import deque
import random
import os

from src.agents.policy_networks import ActorNetwork, CriticNetwork


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2
    ):
        """
        Initialize OU noise process.

        Args:
            size: Dimension of action space
            mu: Mean reversion level
            theta: Mean reversion rate
            sigma: Volatility parameter
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset internal state."""
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """Generate noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.size
        )
        self.state += dx
        return self.state


class DDPGAgent:
    """
    DDPG Agent for continuous action spaces.

    Suitable for:
    - Market Making
    - Delta Hedging
    - FX Arbitrage
    """

    def __init__(
        self,
        env: gym.Env,
        actor_hidden_dims: Tuple[int, ...] = (400, 300),
        critic_hidden_dims: Tuple[int, ...] = (400, 300),
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1000000,
        batch_size: int = 128,
        noise_theta: float = 0.15,
        noise_sigma: float = 0.2,
        device: str = "auto",
    ):
        """
        Initialize DDPG Agent.

        Args:
            env: Gymnasium environment
            actor_hidden_dims: Actor network hidden dimensions
            critic_hidden_dims: Critic network hidden dimensions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Replay buffer size
            batch_size: Minibatch size
            noise_theta: OU noise theta
            noise_sigma: OU noise sigma
            device: Device to use
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Action scaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Actor networks
        self.actor = ActorNetwork(
            self.state_dim, self.action_dim, actor_hidden_dims
        ).to(self.device)

        self.actor_target = ActorNetwork(
            self.state_dim, self.action_dim, actor_hidden_dims
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = CriticNetwork(
            self.state_dim, self.action_dim, critic_hidden_dims
        ).to(self.device)

        self.critic_target = CriticNetwork(
            self.state_dim, self.action_dim, critic_hidden_dims
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Move action scale/bias to device
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            self.action_dim, theta=noise_theta, sigma=noise_sigma
        )

        # Metrics
        self.training_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "episode_rewards": [],
            "q_values": [],
        }

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise

        Returns:
            Action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor)
            action_scaled = action * self.action_scale + self.action_bias

        action_np = action_scaled.cpu().numpy()[0]

        if add_noise:
            noise = self.noise.sample()
            action_np = np.clip(
                action_np + noise, self.env.action_space.low, self.env.action_space.high
            )

        return action_np

    def update(self):
        """Update networks using a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Normalize actions to [-1, 1] for network input
        actions_normalized = (actions - self.action_bias) / self.action_scale

        # ============ Update Critic ============
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(next_states)
            next_actions_scaled = next_actions * self.action_scale + self.action_bias
            next_actions_normalized = (
                next_actions_scaled - self.action_bias
            ) / self.action_scale

            # Target Q-values
            target_q = self.critic_target(next_states, next_actions_normalized)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q = self.critic(states, actions_normalized)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ============ Update Actor ============
        # Actor loss: maximize Q-value
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============ Soft Update Target Networks ============
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        # Record metrics
        self.training_metrics["actor_loss"].append(actor_loss.item())
        self.training_metrics["critic_loss"].append(critic_loss.item())
        self.training_metrics["q_values"].append(current_q.mean().item())

    def soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def train(
        self,
        total_timesteps: int,
        warmup_steps: int = 10000,
        update_frequency: int = 1,
        callback=None,
        log_interval: int = 10,
    ):
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            warmup_steps: Random exploration steps before training
            update_frequency: Update network every N steps
            callback: Optional callback function
            log_interval: Logging interval
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_count = 0

        print(f"Training DDPG agent for {total_timesteps} timesteps...")
        print(f"Device: {self.device}")
        print(f"Warmup steps: {warmup_steps}")

        for timestep in range(1, total_timesteps + 1):
            # Select action
            if timestep < warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state, add_noise=True)

            # Take action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)

            episode_reward += reward

            # Update networks
            if timestep >= warmup_steps and timestep % update_frequency == 0:
                self.update()

            if done:
                self.training_metrics["episode_rewards"].append(episode_reward)
                episode_count += 1

                if episode_count % log_interval == 0:
                    avg_reward = np.mean(self.training_metrics["episode_rewards"][-10:])
                    avg_q = (
                        np.mean(self.training_metrics["q_values"][-100:])
                        if self.training_metrics["q_values"]
                        else 0
                    )
                    print(
                        f"Episode {episode_count}, Timestep {timestep}, "
                        f"Avg Reward: {avg_reward:.2f}, Avg Q: {avg_q:.2f}"
                    )

                state, _ = self.env.reset()
                episode_reward = 0
                self.noise.reset()
            else:
                state = next_state

            if callback:
                callback(self, timestep)

        print(f"Training completed! Total episodes: {episode_count}")

    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "training_metrics": self.training_metrics,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_metrics = checkpoint["training_metrics"]
        print(f"Model loaded from {path}")
