"""
Deep Q-Network (DQN) Agent implementation.

DQN is a value-based algorithm for discrete action spaces using
experience replay and target networks. Includes dueling architecture
and double DQN enhancements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
import gymnasium as gym
from collections import deque
import random
import os

from src.agents.policy_networks import QNetwork


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
        action: int,
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


class DQNAgent:
    """
    DQN Agent for discrete action spaces.

    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Optional dueling architecture
    - Optional double DQN

    Suitable for:
    - Factor Tracking (discrete portfolio construction)
    - Regime-based strategy selection
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_frequency: int = 1000,
        use_dueling: bool = True,
        use_double_dqn: bool = True,
        device: str = "auto",
    ):
        """
        Initialize DQN Agent.

        Args:
            env: Gymnasium environment with discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay steps
            buffer_size: Replay buffer size
            batch_size: Minibatch size
            target_update_frequency: Target network update frequency
            use_dueling: Use dueling architecture
            use_double_dqn: Use double DQN
            device: Device to use
        """
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.use_double_dqn = use_double_dqn

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Q-Networks
        self.q_network = QNetwork(
            self.state_dim, self.n_actions, hidden_dims, use_dueling=use_dueling
        ).to(self.device)

        self.target_network = QNetwork(
            self.state_dim, self.n_actions, hidden_dims, use_dueling=use_dueling
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training state
        self.steps = 0

        # Metrics
        self.training_metrics = {
            "loss": [],
            "q_values": [],
            "episode_rewards": [],
            "epsilon": [],
        }

    def get_epsilon(self) -> float:
        """Get current epsilon value with linear decay."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0, 1 - self.steps / self.epsilon_decay
        )
        return epsilon

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            deterministic: If True, always select greedy action

        Returns:
            Selected action
        """
        epsilon = 0.0 if deterministic else self.get_epsilon()

        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def update(self):
        """Update Q-network using a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select actions
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = (
                    self.target_network(next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(dim=1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Record metrics
        self.training_metrics["loss"].append(loss.item())
        self.training_metrics["q_values"].append(current_q_values.mean().item())

    def train(
        self,
        total_timesteps: int,
        warmup_steps: int = 10000,
        update_frequency: int = 4,
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

        print(f"Training DQN agent for {total_timesteps} timesteps...")
        print(f"Device: {self.device}")
        print(f"Warmup steps: {warmup_steps}")
        print(
            f"Dueling: {self.q_network.use_dueling}, Double DQN: {self.use_double_dqn}"
        )

        for timestep in range(1, total_timesteps + 1):
            self.steps = timestep

            # Select action
            if timestep < warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)

            episode_reward += reward

            # Update network
            if timestep >= warmup_steps and timestep % update_frequency == 0:
                self.update()

            if done:
                self.training_metrics["episode_rewards"].append(episode_reward)
                self.training_metrics["epsilon"].append(self.get_epsilon())
                episode_count += 1

                if episode_count % log_interval == 0:
                    avg_reward = np.mean(self.training_metrics["episode_rewards"][-10:])
                    avg_q = (
                        np.mean(self.training_metrics["q_values"][-100:])
                        if self.training_metrics["q_values"]
                        else 0
                    )
                    current_epsilon = self.get_epsilon()
                    print(
                        f"Episode {episode_count}, Timestep {timestep}, "
                        f"Avg Reward: {avg_reward:.2f}, Avg Q: {avg_q:.2f}, "
                        f"Epsilon: {current_epsilon:.3f}"
                    )

                state, _ = self.env.reset()
                episode_reward = 0
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
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "training_metrics": self.training_metrics,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.training_metrics = checkpoint["training_metrics"]
        print(f"Model loaded from {path}")
