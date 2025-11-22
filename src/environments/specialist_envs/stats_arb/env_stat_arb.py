"""
Statistical Arbitrage Trading Environment

Implements pair trading and mean reversion strategies on cointegrated stocks.
Agent learns to trade spreads between pairs using continuous position sizing.

Strategy: Long-short positions based on z-score of price spread
Agent: PPO (continuous action space)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.environments.specialist_envs.base_trading_env import BaseTradingEnv


class StatisticalArbitrageEnv(BaseTradingEnv):
    """
    Environment for statistical arbitrage (pairs trading).

    Observation Space:
        - Spread z-score
        - Spread return
        - Half-life of mean reversion
        - Hedge ratio
        - Current position in pair
        - Portfolio metrics (cash, P&L)
        - Technical indicators on spread

    Action Space:
        - Continuous: [-1, 1] representing position in the spread
          -1 = full short spread, 0 = no position, +1 = full long spread
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        transaction_cost_pct: float = 0.001,
        max_position_size: float = 100_000,
        lookback_window: int = 60,
        **kwargs,
    ):
        """
        Initialize Statistical Arbitrage Environment.

        Args:
            df: DataFrame with columns ['asset1_price', 'asset2_price', 'spread', 'z_score', etc.]
            initial_balance: Starting capital
            transaction_cost_pct: Transaction cost
            max_position_size: Maximum position per asset
            lookback_window: Window for observations
        """
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            transaction_cost_pct=transaction_cost_pct,
            max_position_size=max_position_size,
            lookback_window=lookback_window,
            **kwargs,
        )

        # Pair trading specific state
        self.asset1_shares = 0.0
        self.asset2_shares = 0.0
        self.spread_position = 0.0  # -1 to 1

        # Define action space: continuous position in spread
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define observation space
        # 7 spread features + 5 position/portfolio features + 8 technical features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        current_data = self.df.iloc[self._current_step]

        # Spread features
        z_score = current_data.get("z_score", 0.0)
        spread = current_data.get("spread", 0.0)
        spread_return = current_data.get("spread_return", 0.0)
        half_life = current_data.get("half_life", 20.0)
        hedge_ratio = current_data.get("hedge_ratio", 1.0)
        spread_volatility = current_data.get("spread_volatility", 0.01)
        correlation = current_data.get("correlation", 0.8)

        # Position features
        position_norm = self.spread_position  # Already -1 to 1
        asset1_price = current_data.get("asset1_price", 100.0)
        asset2_price = current_data.get("asset2_price", 100.0)

        position_value = (
            self.asset1_shares * asset1_price + self.asset2_shares * asset2_price
        )
        position_pct = position_value / (self.net_worth + 1e-8)

        # Portfolio features
        cash_pct = self.balance / self.initial_balance
        pnl_pct = self.total_pnl / self.initial_balance

        # Technical indicators on spread
        if self._current_step >= 20:
            recent_spread = (
                self.df["spread"]
                .iloc[self._current_step - 20 : self._current_step]
                .values
            )
            sma_20 = np.mean(recent_spread)
            std_20 = np.std(recent_spread)
            spread_momentum = spread - recent_spread[0]

            if self._current_step >= 60:
                longer_spread = (
                    self.df["spread"]
                    .iloc[self._current_step - 60 : self._current_step]
                    .values
                )
                sma_60 = np.mean(longer_spread)
                std_60 = np.std(longer_spread)
            else:
                sma_60 = sma_20
                std_60 = std_20
        else:
            sma_20 = spread
            sma_60 = spread
            std_20 = spread_volatility
            std_60 = spread_volatility
            spread_momentum = 0.0

        # Mean reversion signal
        mean_reversion_signal = -z_score  # Negative z-score means buy spread

        # Construct observation
        obs = np.array(
            [
                # Spread features (7)
                z_score,
                spread,
                spread_return,
                half_life / 100.0,  # Normalize
                hedge_ratio,
                spread_volatility * 100,  # Scale up
                correlation,
                # Position features (5)
                position_norm,
                position_pct,
                cash_pct,
                pnl_pct,
                self.total_trades / 100.0,  # Normalize
                # Technical features (8)
                (spread - sma_20) / (std_20 + 1e-8),
                (spread - sma_60) / (std_60 + 1e-8),
                (sma_20 - sma_60) / (std_60 + 1e-8),
                spread_momentum / (std_20 + 1e-8),
                mean_reversion_signal,
                std_20 / (std_60 + 1e-8),
                asset1_price / 100.0,  # Normalize
                asset2_price / 100.0,  # Normalize
            ],
            dtype=np.float32,
        )

        # Handle NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    def _get_info(self) -> Dict:
        """Get diagnostic information."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.balance,
            "spread_position": self.spread_position,
            "asset1_shares": self.asset1_shares,
            "asset2_shares": self.asset2_shares,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "total_fees": self.total_fees,
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "max_drawdown": self.max_drawdown,
        }

    def _take_action(self, action) -> None:
        """
        Execute trading action.

        Action is continuous [-1, 1]:
        - Positive: Long spread (long asset1, short asset2)
        - Negative: Short spread (short asset1, long asset2)
        - Magnitude: Position size
        """
        target_position = float(action[0])
        target_position = np.clip(target_position, -1.0, 1.0)

        current_data = self.df.iloc[self._current_step]
        asset1_price = current_data.get("asset1_price", 100.0)
        asset2_price = current_data.get("asset2_price", 100.0)
        hedge_ratio = current_data.get("hedge_ratio", 1.0)

        # Calculate target shares for each asset
        # Position = 1 means: long $X of asset1, short $X*hedge_ratio of asset2
        position_value = self.max_position_size * abs(target_position)

        if target_position > 0:
            # Long spread: long asset1, short asset2
            target_asset1_shares = position_value / asset1_price
            target_asset2_shares = -(position_value * hedge_ratio) / asset2_price
        elif target_position < 0:
            # Short spread: short asset1, long asset2
            target_asset1_shares = -position_value / asset1_price
            target_asset2_shares = (position_value * hedge_ratio) / asset2_price
        else:
            # Close position
            target_asset1_shares = 0.0
            target_asset2_shares = 0.0

        # Calculate trades needed
        asset1_trade = target_asset1_shares - self.asset1_shares
        asset2_trade = target_asset2_shares - self.asset2_shares

        # Execute trades with transaction costs
        if abs(asset1_trade) > 0:
            trade_value = abs(asset1_trade * asset1_price)
            cost = trade_value * (1 + self.transaction_cost_pct)

            if asset1_trade > 0:  # Buying
                if self.balance >= cost:
                    self.asset1_shares = target_asset1_shares
                    self.balance -= cost
                    self.total_fees += cost * self.transaction_cost_pct
                    self.total_trades += 1
            else:  # Selling
                proceeds = trade_value * (1 - self.transaction_cost_pct)
                self.asset1_shares = target_asset1_shares
                self.balance += proceeds
                self.total_fees += trade_value * self.transaction_cost_pct
                self.total_trades += 1

        if abs(asset2_trade) > 0:
            trade_value = abs(asset2_trade * asset2_price)
            cost = trade_value * (1 + self.transaction_cost_pct)

            if asset2_trade > 0:  # Buying
                if self.balance >= cost:
                    self.asset2_shares = target_asset2_shares
                    self.balance -= cost
                    self.total_fees += cost * self.transaction_cost_pct
                    self.total_trades += 1
            else:  # Selling
                proceeds = trade_value * (1 - self.transaction_cost_pct)
                self.asset2_shares = target_asset2_shares
                self.balance += proceeds
                self.total_fees += trade_value * self.transaction_cost_pct
                self.total_trades += 1

        # Update spread position
        self.spread_position = target_position

    def _calculate_reward(self, action) -> float:
        """
        Calculate reward based on:
        - Spread P&L
        - Mean reversion capture
        - Transaction costs
        """
        # P&L-based reward
        if len(self.portfolio_history) > 1:
            pnl = self.portfolio_value - self.portfolio_history[-2]
            reward = pnl / self.initial_balance * 1000  # Scale up
        else:
            reward = 0.0

        # Penalty for holding position when spread is not extreme
        current_data = self.df.iloc[self._current_step]
        z_score = current_data.get("z_score", 0.0)

        # Reward for mean reversion trades (entering when |z| > 2)
        if abs(z_score) > 2.0 and abs(self.spread_position) > 0.5:
            # Check if position direction matches mean reversion expectation
            if (z_score > 2.0 and self.spread_position < 0) or (
                z_score < -2.0 and self.spread_position > 0
            ):
                reward += 0.5  # Bonus for good timing

        # Penalty for holding large position when spread is neutral
        if abs(z_score) < 0.5 and abs(self.spread_position) > 0.5:
            reward -= 0.1 * abs(self.spread_position)

        return reward

    def _update_portfolio_value(self) -> None:
        """Update portfolio value including spread position."""
        current_data = self.df.iloc[self._current_step]
        asset1_price = current_data.get("asset1_price", 100.0)
        asset2_price = current_data.get("asset2_price", 100.0)

        position_value = (
            self.asset1_shares * asset1_price + self.asset2_shares * asset2_price
        )

        self.portfolio_value = self.balance + position_value
        self.net_worth = self.portfolio_value

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset pair-specific state
        self.asset1_shares = 0.0
        self.asset2_shares = 0.0
        self.spread_position = 0.0

        return obs, info


if __name__ == "__main__":
    # Test environment
    print("Testing Statistical Arbitrage Environment...")

    # Create dummy data
    n_steps = 500
    np.random.seed(42)

    # Simulate cointegrated pair
    asset1_price = 100 + np.cumsum(np.random.randn(n_steps) * 0.5)
    asset2_price = 100 + np.cumsum(np.random.randn(n_steps) * 0.5)

    # Create spread
    hedge_ratio = 1.0
    spread = asset1_price - hedge_ratio * asset2_price
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std

    df = pd.DataFrame(
        {
            "asset1_price": asset1_price,
            "asset2_price": asset2_price,
            "spread": spread,
            "z_score": z_score,
            "spread_return": np.r_[0, np.diff(spread)] / (spread + 1e-8),
            "half_life": np.full(n_steps, 20.0),
            "hedge_ratio": np.full(n_steps, hedge_ratio),
            "spread_volatility": np.full(n_steps, spread_std),
            "correlation": np.full(n_steps, 0.8),
        }
    )

    # Create environment
    env = StatisticalArbitrageEnv(df, initial_balance=1_000_000)

    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Run a few steps
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Total reward after 10 steps: {total_reward:.2f}")
    print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")
    print("\n✓ Statistical Arbitrage Environment working correctly!")
