"""
Base Trading Environment for Specialist Strategies

This module provides an abstract base class for all specialist trading
environments in the hierarchical DRL framework. It implements common
functionality for:
- Portfolio management (cash, positions, P&L tracking)
- Transaction cost modeling
- Risk metrics calculation
- Performance tracking
- State management

Each specialist strategy inherits from this base and implements
strategy-specific observation, action, and reward logic.
"""

import abc
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List, Any
from collections import deque
import logging


class BaseTradingEnv(gym.Env, abc.ABC):
    """
    Abstract base class for all specialist trading environments.

    This class provides the core structure for a trading environment, including
    data handling, step/reset logic, portfolio value tracking, and risk metrics.

    Specialist agents must inherit from this class and implement:
    1. `_get_obs()` - Define what the agent observes
    2. `_get_info()` - Define auxiliary information for logging
    3. `_take_action(action)` - Define how actions modify state
    4. `_calculate_reward(action)` - Define the learning signal

    Optionally override:
    - `_update_portfolio_value()` - For complex portfolio calculations
    - `_check_constraints()` - For strategy-specific constraints
    - `_get_done()` - For custom termination conditions
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_size: Optional[float] = None,
        max_leverage: float = 1.0,
        margin_requirement: float = 0.0,
        risk_free_rate: float = 0.02,
        lookback_window: int = 60,
        frame_bound: Optional[Tuple[int, int]] = None,
        render_mode: Optional[str] = "human",
    ):
        """
        Initialize the base trading environment.

        Args:
            df: DataFrame containing market data (OHLCV, features, etc.)
            initial_balance: Starting cash balance
            transaction_cost_pct: Transaction cost as % of trade value
            slippage_pct: Slippage as % of trade value
            max_position_size: Maximum position size (None = unlimited)
            max_leverage: Maximum leverage allowed
            margin_requirement: Margin requirement as % of position value
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            lookback_window: Number of historical steps for observation
            frame_bound: Optional (start, end) indices for data subset
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        # === Data Configuration ===
        self.df = df.copy()
        self.lookback_window = lookback_window

        # Set frame boundaries
        if frame_bound is None:
            self.frame_bound = (lookback_window, len(df) - 1)
        else:
            self.frame_bound = frame_bound

        # === Portfolio Configuration ===
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.margin_requirement = margin_requirement
        self.risk_free_rate = risk_free_rate

        # === Action and Observation Spaces (to be defined in child) ===
        # Placeholder - child classes MUST override these
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        # === Episode State Variables ===
        self._current_step = 0
        self._start_step = self.frame_bound[0]
        self._end_step = self.frame_bound[1]

        # Portfolio state
        self.balance = 0.0
        self.shares_held = 0.0
        self.net_worth = 0.0
        self.portfolio_value = 0.0
        self.total_pnl = 0.0
        self.total_trades = 0
        self.total_fees = 0.0

        # Position tracking (for multi-asset or complex positions)
        self.positions: Dict[str, float] = {}
        self.position_history: List[Dict[str, float]] = []

        # Performance tracking
        self.portfolio_history: List[float] = []
        self.action_history: List[Any] = []
        self.reward_history: List[float] = []
        self.returns_history: deque = deque(maxlen=252)  # One year

        # Risk metrics
        self.max_portfolio_value = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Rendering
        self.render_mode = render_mode

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validation
        self._validate_config()

    @property
    def current_step(self) -> int:
        """Expose current_step for compatibility with specialist environments."""
        return self._current_step

    @property
    def initial_capital(self) -> float:
        """Alias for initial_balance for compatibility."""
        return self.initial_balance

    @property
    def cash(self) -> float:
        """Alias for balance for compatibility."""
        return self.balance

    @cash.setter
    def cash(self, value: float):
        """Allow setting cash via the alias."""
        self.balance = value

    def _validate_config(self):
        """Validate environment configuration."""
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")

        if self.transaction_cost_pct < 0 or self.transaction_cost_pct > 0.1:
            raise ValueError("transaction_cost_pct must be between 0 and 0.1")

        if self.max_leverage < 0:
            raise ValueError("max_leverage must be non-negative")

        if len(self.df) == 0:
            raise ValueError("DataFrame cannot be empty")

        if self.frame_bound[0] >= self.frame_bound[1]:
            raise ValueError("Invalid frame_bound: start must be < end")

    # === Abstract Methods (MUST be implemented by child classes) ===

    @abc.abstractmethod
    def _get_obs(self) -> np.ndarray:
        """
        Return the current observation from the environment.
        This is what the agent "sees."

        MUST be implemented by the child class.

        Returns:
            Observation array matching observation_space shape

        Example:
            ```python
            # Get current market data
            current_data = self.df.iloc[self._current_step]

            # Extract features
            features = [
                current_data['price'],
                current_data['volume'],
                current_data['z_score'],
                self.shares_held / 1000,  # Normalized position
                self.balance / self.initial_balance
            ]

            return np.array(features, dtype=np.float32)
            ```
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """
        Return auxiliary diagnostic information.
        Not used for training, but useful for logging and analysis.

        MUST be implemented by the child class.

        Returns:
            Dictionary with diagnostic information

        Example:
            ```python
            return {
                'portfolio_value': self.portfolio_value,
                'pnl': self.total_pnl,
                'position': self.shares_held,
                'cash': self.balance,
                'total_trades': self.total_trades,
                'total_fees': self.total_fees,
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'max_drawdown': self.max_drawdown
            }
            ```
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _take_action(self, action) -> None:
        """
        Execute the given action and update environment state.
        This is where transaction costs and slippage are applied.

        MUST be implemented by the child class.

        Args:
            action: The action to execute (format depends on action_space)

        Example (discrete actions):
            ```python
            current_price = self.df.iloc[self._current_step]['close']

            if action == 0:  # Buy
                max_shares = self.balance / (current_price * (1 + self.transaction_cost_pct))
                shares_to_buy = max_shares * 0.5  # Buy 50% of available

                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost_pct)
                    self.shares_held += shares_to_buy
                    self.balance -= cost
                    self.total_trades += 1
                    self.total_fees += cost * self.transaction_cost_pct

            elif action == 1:  # Sell
                if self.shares_held > 0:
                    shares_to_sell = self.shares_held * 0.5  # Sell 50%
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost_pct)
                    self.shares_held -= shares_to_sell
                    self.balance += proceeds
                    self.total_trades += 1
                    self.total_fees += proceeds * self.transaction_cost_pct
            ```
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_reward(self, action) -> float:
        """
        Calculate the reward for the current step.
        This is the learning signal for the agent.

        MUST be implemented by the child class.

        Args:
            action: The action that was taken

        Returns:
            Scalar reward value

        Common reward formulations:

        1. Simple P&L:
            ```python
            return self.portfolio_value - prev_portfolio_value
            ```

        2. Log returns:
            ```python
            return np.log(self.portfolio_value / prev_portfolio_value)
            ```

        3. Sharpe ratio:
            ```python
            if len(self.returns_history) > 30:
                returns = np.array(self.returns_history)
                sharpe = (returns.mean() - self.risk_free_rate/252) / (returns.std() + 1e-8)
                return sharpe
            return 0.0
            ```
        """
        raise NotImplementedError

    # === Optionally Override These Methods ===

    def _update_portfolio_value(self) -> None:
        """
        Update portfolio value based on current market prices.
        Override for complex portfolio calculations.

        Default implementation for single-asset trading:
        """
        if "close" in self.df.columns:
            current_price = self.df.iloc[self._current_step]["close"]
            self.portfolio_value = self.balance + (self.shares_held * current_price)
            self.net_worth = self.portfolio_value
        else:
            # Fallback if no price column
            self.portfolio_value = self.balance
            self.net_worth = self.balance

    def _check_constraints(self) -> bool:
        """
        Check if current state violates any constraints.
        Override for strategy-specific constraints.

        Returns:
            True if constraints are satisfied, False otherwise
        """
        # Position size constraint
        if self.max_position_size is not None:
            if abs(self.shares_held) > self.max_position_size:
                return False

        # Leverage constraint
        if "close" in self.df.columns:
            current_price = self.df.iloc[self._current_step]["close"]
            position_value = abs(self.shares_held * current_price)
            leverage = position_value / (self.net_worth + 1e-8)

            if leverage > self.max_leverage:
                return False

        # Margin requirement
        if self.margin_requirement > 0:
            if "close" in self.df.columns:
                current_price = self.df.iloc[self._current_step]["close"]
                position_value = abs(self.shares_held * current_price)
                required_margin = position_value * self.margin_requirement

                if self.balance < required_margin:
                    return False

        return True

    def _get_done(self) -> Tuple[bool, bool]:
        """
        Determine if episode should terminate.
        Override for custom termination logic.

        Returns:
            (terminated, truncated) tuple
        """
        # Portfolio blown up
        if self.portfolio_value <= 0:
            return True, False

        # Reached end of data
        if self._current_step >= self._end_step:
            return True, False

        # Constraint violation
        if not self._check_constraints():
            return True, False

        return False, False

    # === Core Gym Methods ===

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value

        # 1. Take the action (apply transaction costs)
        self._take_action(action)

        # 2. Move to next time step
        self._current_step += 1

        # 3. Update portfolio value based on new market data
        self._update_portfolio_value()

        # 4. Calculate reward
        reward = self._calculate_reward(action)

        # 5. Update tracking
        self.total_pnl = self.portfolio_value - self.initial_balance
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)
        self.reward_history.append(reward)

        # Calculate returns
        if prev_portfolio_value > 0:
            period_return = (
                self.portfolio_value - prev_portfolio_value
            ) / prev_portfolio_value
            self.returns_history.append(period_return)

        # Update risk metrics
        self._update_risk_metrics()

        # 6. Check for termination
        terminated, truncated = self._get_done()

        # 7. Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Reset episode state
        self._current_step = self._start_step

        # Reset portfolio state
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.net_worth = self.initial_balance
        self.total_pnl = 0.0
        self.total_trades = 0
        self.total_fees = 0.0

        # Reset positions (but preserve if child class uses different type)
        if isinstance(self.positions, dict):
            self.positions = {}
        elif isinstance(self.positions, np.ndarray):
            self.positions = np.zeros_like(self.positions)
        self.position_history = []

        # Reset tracking
        self.portfolio_history = [self.initial_balance]
        self.action_history = []
        self.reward_history = []
        self.returns_history.clear()

        # Reset risk metrics
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Get initial observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    # === Performance Metrics ===

    def _update_risk_metrics(self) -> None:
        """Update risk metrics (drawdown, etc.)."""
        # Update max portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

        # Calculate current drawdown
        if self.max_portfolio_value > 0:
            self.current_drawdown = (
                self.portfolio_value - self.max_portfolio_value
            ) / self.max_portfolio_value

            # Update max drawdown
            if self.current_drawdown < self.max_drawdown:
                self.max_drawdown = self.current_drawdown

    def calculate_sharpe_ratio(self, periods: int = 252) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            periods: Number of periods for annualization (252 for daily)

        Returns:
            Sharpe ratio
        """
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history)
        mean_return = returns.mean() * periods
        std_return = returns.std() * np.sqrt(periods)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - self.risk_free_rate) / std_return
        return sharpe

    def calculate_sortino_ratio(self, periods: int = 252) -> float:
        """
        Calculate Sortino ratio (downside deviation).

        Args:
            periods: Number of periods for annualization

        Returns:
            Sortino ratio
        """
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(self.returns_history)
        mean_return = returns.mean() * periods

        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf

        downside_std = downside_returns.std() * np.sqrt(periods)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - self.risk_free_rate) / downside_std
        return sortino

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Returns:
            Calmar ratio
        """
        if len(self.returns_history) < 2:
            return 0.0

        total_return = (
            self.portfolio_value - self.initial_balance
        ) / self.initial_balance

        if self.max_drawdown == 0:
            return np.inf if total_return > 0 else 0.0

        calmar = total_return / abs(self.max_drawdown)
        return calmar

    def calculate_win_rate(self) -> float:
        """
        Calculate win rate (% of positive returns).

        Returns:
            Win rate between 0 and 1
        """
        if len(self.returns_history) == 0:
            return 0.0

        returns = np.array(self.returns_history)
        win_rate = np.sum(returns > 0) / len(returns)
        return win_rate

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Returns:
            Profit factor
        """
        if len(self.returns_history) == 0:
            return 0.0

        returns = np.array(self.returns_history)
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = abs(np.sum(returns[returns < 0]))

        if gross_losses == 0:
            return np.inf if gross_profits > 0 else 0.0

        profit_factor = gross_profits / gross_losses
        return profit_factor

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        total_return = (
            self.portfolio_value - self.initial_balance
        ) / self.initial_balance

        metrics = {
            "total_return": total_return,
            "total_pnl": self.total_pnl,
            "portfolio_value": self.portfolio_value,
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "calmar_ratio": self.calculate_calmar_ratio(),
            "max_drawdown": self.max_drawdown,
            "win_rate": self.calculate_win_rate(),
            "profit_factor": self.calculate_profit_factor(),
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
        }

        return metrics

    # === Rendering ===
    def render(self) -> None:
        """
        Render the environment state.
        For trading, this prints portfolio statistics.
        """
        if self.render_mode == "human":
            print(f"\n{'=' * 60}")
            print(f"Step: {self._current_step}/{self._end_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash Balance: ${self.balance:,.2f}")
            print(f"Position: {self.shares_held:,.2f} shares")
            print(
                f"Total P&L: ${self.total_pnl:,.2f} ({self.total_pnl / self.initial_balance * 100:.2f}%)"
            )
            print(f"Max Drawdown: {self.max_drawdown * 100:.2f}%")

            if len(self.returns_history) > 20:
                print(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}")
                print(f"Win Rate: {self.calculate_win_rate() * 100:.2f}%")

            print(f"Total Trades: {self.total_trades}")
            print(f"Total Fees: ${self.total_fees:,.2f}")
            print(f"{'=' * 60}\n")

    def close(self) -> None:
        """
        Perform cleanup when environment is closed.
        Override to add custom cleanup logic.
        """
        pass
