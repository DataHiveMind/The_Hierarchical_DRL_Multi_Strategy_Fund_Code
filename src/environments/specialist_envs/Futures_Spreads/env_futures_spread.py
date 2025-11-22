"""
Futures Spreads Environment

Implements calendar and inter-commodity spread trading.
Compatible with PPO agent (continuous spread positions).

Action Space:
- Spread position: [-1, 1] (long/short the spread)

Observation Space:
- Spread level and z-score
- Seasonality patterns
- Roll yield and basis
- Correlation dynamics
- Mean reversion indicators
"""

from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ..base_trading_env import BaseTradingEnv


class FuturesSpreadsEnv(BaseTradingEnv):
    """
    Futures spreads trading environment for PPO agent.
    Trades calendar spreads (near vs far contracts) or inter-commodity spreads.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        spread_type: str = "calendar",  # "calendar" or "inter_commodity"
        contract_size: float = 1000.0,
        transaction_cost: float = 0.0002,
        margin_requirement: float = 0.05,
        carry_cost: float = 0.0001,
        window_size: int = 60,
    ):
        """
        Initialize Futures Spreads Environment.

        Args:
            data: DataFrame with 'near' and 'far' contract prices
            initial_capital: Starting capital
            spread_type: Type of spread to trade
            contract_size: Notional per contract
            transaction_cost: Transaction cost rate
            margin_requirement: Margin as fraction of notional
            carry_cost: Daily carry cost
            window_size: Lookback for statistics
        """
        super().__init__(data, initial_capital)

        self.spread_type = spread_type
        self.contract_size = contract_size
        self.transaction_cost = transaction_cost
        self.margin_requirement = margin_requirement
        self.carry_cost = carry_cost
        self.window_size = window_size

        # Spread trading state
        self.spread_position = 0.0  # Net spread position
        self.near_contracts = 0.0
        self.far_contracts = 0.0
        self.margin_used = 0.0

        # Action space: spread position in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space (24 features):
        # - Spread level (1)
        # - Spread z-score (1)
        # - Spread returns (1)
        # - Near contract price (1)
        # - Far contract price (1)
        # - Basis (far - near) (1)
        # - Roll yield (1)
        # - Spread volatility (1)
        # - Mean reversion half-life (1)
        # - Correlation (1)
        # - Seasonality indicators (3)
        # - Momentum indicators (4)
        # - Position and portfolio state (6)
        # - Market regime (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

    def _calculate_spread(self, near_price: float, far_price: float) -> float:
        """Calculate spread (far - near for calendar, or ratio for inter-commodity)."""
        if self.spread_type == "calendar":
            return far_price - near_price
        else:  # inter_commodity
            return far_price / near_price if near_price > 0 else 1.0

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get prices
        if "near" in self.data.columns and "far" in self.data.columns:
            near_price = self.data.iloc[self.current_step]["near"]
            far_price = self.data.iloc[self.current_step]["far"]
        else:
            # Simulate near and far from close
            near_price = self.data.iloc[self.current_step]["close"]
            far_price = near_price * (
                1 + np.random.normal(0.02, 0.01)
            )  # Simulate contango

        # Spread features
        spread = self._calculate_spread(near_price, far_price)

        # Calculate spread statistics
        if self.current_step >= self.window_size:
            # Historical spreads
            if "near" in self.data.columns:
                hist_near = self.data.iloc[
                    self.current_step - self.window_size : self.current_step
                ]["near"].values
                hist_far = self.data.iloc[
                    self.current_step - self.window_size : self.current_step
                ]["far"].values
            else:
                hist_near = self.data.iloc[
                    self.current_step - self.window_size : self.current_step
                ]["close"].values
                hist_far = hist_near * (
                    1 + np.random.normal(0.02, 0.01, len(hist_near))
                )

            hist_spreads = [
                self._calculate_spread(n, f) for n, f in zip(hist_near, hist_far)
            ]

            spread_mean = np.mean(hist_spreads)
            spread_std = np.std(hist_spreads)
            spread_z = (spread - spread_mean) / spread_std if spread_std > 0 else 0

            # Spread returns
            spread_return = (
                (spread - hist_spreads[-2]) / hist_spreads[-2]
                if len(hist_spreads) > 1
                else 0
            )

            # Volatility
            spread_vol = spread_std

            # Mean reversion (half-life estimation)
            from statsmodels.tsa.stattools import adfuller

            try:
                adf_result = adfuller(hist_spreads)
                half_life = -np.log(2) / adf_result[0] if adf_result[0] < 0 else 100
            except:
                half_life = 50  # Default

            # Correlation
            correlation = np.corrcoef(hist_near, hist_far)[0, 1]
        else:
            spread_mean = spread
            spread_std = abs(spread) * 0.1
            spread_z = 0
            spread_return = 0
            spread_vol = spread_std
            half_life = 50
            correlation = 0.95

        # Basis and roll yield
        basis = far_price - near_price
        roll_yield = basis / near_price if near_price > 0 else 0

        # Seasonality (day of year, month, quarter)
        day_of_year = self.current_step % 252
        month_indicator = np.sin(2 * np.pi * (self.current_step % 21) / 21)
        quarter_indicator = np.cos(2 * np.pi * (self.current_step % 63) / 63)

        # Momentum indicators
        if self.current_step >= 20:
            spread_sma_20 = (
                np.mean(hist_spreads[-20:])
                if self.current_step >= self.window_size
                else spread
            )
            spread_momentum = spread - spread_sma_20

            # Spread trend
            if self.current_step >= 40:
                recent_spreads = hist_spreads[-20:]
                older_spreads = hist_spreads[-40:-20]
                spread_trend = np.mean(recent_spreads) - np.mean(older_spreads)
            else:
                spread_trend = 0
        else:
            spread_momentum = 0
            spread_trend = 0

        # RSI-like indicator for spread
        if self.current_step >= 14:
            changes = np.diff(hist_spreads[-14:])
            gains = changes[changes > 0].sum()
            losses = -changes[changes < 0].sum()
            rs = gains / losses if losses > 0 else 1
            spread_rsi = 100 - (100 / (1 + rs))
        else:
            spread_rsi = 50

        # Position state
        position_pct = self.spread_position / 100.0
        margin_pct = (
            self.margin_used / self.initial_capital if self.initial_capital > 0 else 0
        )
        cash_pct = self.cash / self.initial_capital

        # P&L
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        unrealized_pct = self.unrealized_pnl / self.initial_capital

        # Market regime
        contango = 1 if basis > 0 else 0  # Contango vs backwardation
        high_vol_regime = 1 if spread_vol > spread_std * 1.5 else 0

        observation = np.array(
            [
                spread / 10.0,
                spread_z,
                spread_return,
                near_price / 100.0,
                far_price / 100.0,
                basis / 10.0,
                roll_yield,
                spread_vol,
                half_life / 100.0,
                correlation,
                day_of_year / 252.0,
                month_indicator,
                quarter_indicator,
                spread_momentum / 10.0,
                spread_trend / 10.0,
                spread_rsi / 100.0,
                (spread - spread_mean) / 10.0,
                position_pct,
                margin_pct,
                cash_pct,
                pnl_pct,
                unrealized_pct,
                float(contango),
                float(high_vol_regime),
            ],
            dtype=np.float32,
        )

        return observation

    def _take_action(self, action: np.ndarray):
        """
        Execute spread trading action.

        Args:
            action: Target spread position in [-1, 1]
        """
        # Get prices
        if "near" in self.data.columns and "far" in self.data.columns:
            near_price = self.data.iloc[self.current_step]["near"]
            far_price = self.data.iloc[self.current_step]["far"]
        else:
            near_price = self.data.iloc[self.current_step]["close"]
            far_price = near_price * (1 + np.random.normal(0.02, 0.01))

        # Scale action to position size
        target_position = action[0] * 100.0  # Max 100 spread contracts

        # Calculate position change
        position_change = target_position - self.spread_position

        if abs(position_change) > 1.0:  # Minimum trade size
            # Calculate costs
            near_notional = abs(position_change) * self.contract_size * near_price
            far_notional = abs(position_change) * self.contract_size * far_price
            total_notional = near_notional + far_notional

            transaction_cost = total_notional * self.transaction_cost
            margin_change = total_notional * self.margin_requirement

            # Check if we have enough capital
            if self.cash >= transaction_cost + margin_change:
                # Execute trade
                self.spread_position = target_position
                self.near_contracts = -target_position  # Short near for calendar spread
                self.far_contracts = target_position  # Long far

                self.cash -= transaction_cost
                self.margin_used = (
                    abs(target_position)
                    * self.contract_size
                    * (near_price + far_price)
                    * self.margin_requirement
                )

                self.total_trades += 1

        # Calculate P&L
        if self.current_step > 0:
            # Get previous prices
            if "near" in self.data.columns:
                prev_near = self.data.iloc[self.current_step - 1]["near"]
                prev_far = self.data.iloc[self.current_step - 1]["far"]
            else:
                prev_near = self.data.iloc[self.current_step - 1]["close"]
                prev_far = prev_near * (1 + np.random.normal(0.02, 0.01))

            # Spread P&L
            prev_spread = self._calculate_spread(prev_near, prev_far)
            current_spread = self._calculate_spread(near_price, far_price)

            if self.spread_type == "calendar":
                spread_pnl = (
                    self.spread_position
                    * self.contract_size
                    * (current_spread - prev_spread)
                )
            else:
                spread_pnl = (
                    self.spread_position
                    * self.contract_size
                    * far_price
                    * ((current_spread - prev_spread) / prev_spread)
                )

            # Carry cost
            carry = (
                -abs(self.spread_position)
                * self.contract_size
                * (near_price + far_price)
                * self.carry_cost
            )

            self.unrealized_pnl = spread_pnl + carry

        self.portfolio_value = self.cash + self.margin_used + self.unrealized_pnl

    def _calculate_reward(self) -> float:
        """
        Calculate reward for futures spread trading.

        Components:
        - Spread P&L
        - Roll yield capture
        - Mean reversion bonus
        - Carry cost penalty
        """
        # P&L change
        pnl = self.portfolio_value - self.previous_portfolio_value

        # Mean reversion bonus (reward trading on extreme z-scores)
        obs = self._get_observation()
        spread_z = obs[1]

        mean_reversion_bonus = 0
        if abs(spread_z) > 2.0 and self.spread_position != 0:
            # Reward taking positions on extreme spreads
            if (spread_z > 2 and self.spread_position < 0) or (
                spread_z < -2 and self.spread_position > 0
            ):
                mean_reversion_bonus = 0.5

        # Penalize inactivity
        if abs(self.spread_position) < 5:
            inactivity_penalty = -0.1
        else:
            inactivity_penalty = 0

        reward = pnl + mean_reversion_bonus + inactivity_penalty

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.spread_position = 0.0
        self.near_contracts = 0.0
        self.far_contracts = 0.0
        self.margin_used = 0.0

        return self._get_observation(), info


# Test the environment
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    n_steps = 500

    # Simulate near and far futures with mean-reverting spread
    near_returns = np.random.normal(0, 0.01, n_steps)
    near_prices = 100 * np.exp(np.cumsum(near_returns))

    # Far prices with mean-reverting basis
    basis = [2.0]  # Start in contango
    for i in range(n_steps - 1):
        # Mean-reverting basis
        basis.append(basis[-1] + 0.1 * (2.0 - basis[-1]) + np.random.normal(0, 0.5))

    far_prices = near_prices + np.array(basis)

    data = pd.DataFrame(
        {
            "near": near_prices,
            "far": far_prices,
            "volume": np.random.randint(10000, 100000, n_steps),
        }
    )

    env = FuturesSpreadsEnv(data, spread_type="calendar")

    print("Futures Spreads Environment Test")
    print("=" * 50)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Spread type: {env.spread_type}")

    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        near = data.iloc[env.current_step - 1]["near"]
        far = data.iloc[env.current_step - 1]["far"]
        spread = far - near

        print(f"\nStep {i + 1}:")
        print(f"  Spread: {spread:.2f}")
        print(f"  Position: {env.spread_position:.1f}")
        print(f"  Portfolio value: ${env.portfolio_value:,.2f}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nFinal Stats:")
    print(f"  Total trades: {env.total_trades}")
    print(f"  Final P&L: ${env.portfolio_value - env.initial_capital:,.2f}")
