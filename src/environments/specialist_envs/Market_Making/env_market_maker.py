"""
Market Making Environment

Implements a market making strategy where the agent posts bid/ask quotes
and manages inventory risk. Compatible with DDPG agent (continuous actions).

Action Space:
- Bid spread offset: [-1, 1] (distance below mid)
- Ask spread offset: [-1, 1] (distance above mid)

Observation Space:
- Order book features (bid-ask spread, depth, imbalance)
- Current inventory level and risk
- Recent fill rates and adverse selection
- Market microstructure indicators
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from ..base_trading_env import BaseTradingEnv


class MarketMakingEnv(BaseTradingEnv):
    """
    Market making environment for DDPG agent.
    Agent learns to post competitive quotes while managing inventory risk.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_inventory: int = 100,
        tick_size: float = 0.01,
        spread_range: Tuple[float, float] = (0.0005, 0.02),
        maker_rebate: float = 0.0002,
        taker_fee: float = 0.0005,
        inventory_penalty: float = 0.01,
        window_size: int = 20,
    ):
        """
        Initialize Market Making Environment.

        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            initial_capital: Starting capital
            max_inventory: Maximum inventory (shares)
            tick_size: Minimum price increment
            spread_range: (min_spread, max_spread) in fraction of price
            maker_rebate: Rebate for providing liquidity
            taker_fee: Fee for taking liquidity
            inventory_penalty: Penalty coefficient for inventory risk
            window_size: Lookback period for features
        """
        super().__init__(data, initial_capital)
        self.data = self.df  # Alias for compatibility

        self.max_inventory = max_inventory
        self.tick_size = tick_size
        self.min_spread, self.max_spread = spread_range
        self.maker_rebate = maker_rebate
        self.taker_fee = taker_fee
        self.inventory_penalty = inventory_penalty
        self.window_size = window_size

        # Market making state
        self.inventory = 0
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.bid_filled = False
        self.ask_filled = False
        self.total_fills = 0
        self.adverse_selection_count = 0
        self.unrealized_pnl = 0.0  # Track unrealized P&L from inventory
        self.previous_portfolio_value = initial_capital  # Track for reward calculation

        # Action space: [bid_spread_offset, ask_spread_offset] in [-1, 1]
        # Will be scaled to actual spread range
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space (25 features):
        # - Mid price (1)
        # - Bid-ask spread (1)
        # - Order book imbalance (1)
        # - Inventory level (1)
        # - Inventory percentage (1)
        # - Recent volatility (1)
        # - Fill rate (bid and ask) (2)
        # - Adverse selection rate (1)
        # - Price momentum indicators (5)
        # - Volume features (3)
        # - P&L and portfolio state (5)
        # - Market microstructure (4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Get current observation (required by base class)."""
        return self._get_observation()

    def _get_info(self) -> Dict:
        """Get auxiliary diagnostic information."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "inventory": self.inventory,
            "total_trades": self.total_trades,
            "total_fills": self.total_fills,
            "adverse_selection_rate": self.adverse_selection_count
            / max(self.total_fills, 1),
        }

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_price = self.data.iloc[self.current_step]["close"]

        # Order book features (simulated)
        bid_ask_spread = current_price * (
            self.min_spread + (self.max_spread - self.min_spread) * 0.5
        )
        order_imbalance = np.random.normal(0, 0.1)  # Simulated imbalance

        # Inventory features
        inventory_level = float(self.inventory)
        inventory_pct = (
            inventory_level / self.max_inventory if self.max_inventory > 0 else 0
        )

        # Volatility
        if self.current_step >= self.window_size:
            recent_returns = (
                self.data.iloc[
                    self.current_step - self.window_size : self.current_step
                ]["close"]
                .pct_change()
                .dropna()
            )
            volatility = recent_returns.std()
        else:
            volatility = 0.02  # Default

        # Fill rates
        bid_fill_rate = 1 if self.bid_filled else 0
        ask_fill_rate = 1 if self.ask_filled else 0

        # Adverse selection
        adverse_selection_rate = self.adverse_selection_count / max(self.total_fills, 1)

        # Price momentum
        if self.current_step >= 5:
            ret_1 = (
                self.data.iloc[self.current_step]["close"]
                / self.data.iloc[self.current_step - 1]["close"]
                - 1
            )
            ret_5 = (
                self.data.iloc[self.current_step]["close"]
                / self.data.iloc[self.current_step - 5]["close"]
                - 1
            )
            ret_10 = (
                self.data.iloc[self.current_step]["close"]
                / self.data.iloc[self.current_step - 10]["close"]
                - 1
                if self.current_step >= 10
                else 0
            )

            sma_5 = self.data.iloc[self.current_step - 5 : self.current_step][
                "close"
            ].mean()
            sma_10 = (
                self.data.iloc[self.current_step - 10 : self.current_step][
                    "close"
                ].mean()
                if self.current_step >= 10
                else current_price
            )
        else:
            ret_1 = ret_5 = ret_10 = 0
            sma_5 = sma_10 = current_price

        # Volume features
        current_volume = self.data.iloc[self.current_step]["volume"]
        avg_volume = self.data.iloc[max(0, self.current_step - 20) : self.current_step][
            "volume"
        ].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        volume_trend = (
            self.data.iloc[max(0, self.current_step - 10) : self.current_step]["volume"]
            .diff()
            .mean()
            if self.current_step >= 10
            else 0
        )

        # P&L and portfolio
        cash_pct = self.cash / self.initial_capital
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        total_trades = float(self.total_trades)

        # Market microstructure
        effective_spread = (
            abs(self.ask_price - self.bid_price) / current_price
            if self.ask_price > 0
            else 0
        )
        mid_price = (
            (self.bid_price + self.ask_price) / 2
            if self.bid_price > 0
            else current_price
        )
        price_impact = (
            abs(current_price - mid_price) / current_price if mid_price > 0 else 0
        )
        realized_spread = effective_spread / 2  # Simplified

        observation = np.array(
            [
                current_price / 1000.0,  # Normalize
                bid_ask_spread / current_price,
                order_imbalance,
                inventory_level / 100.0,
                inventory_pct,
                volatility,
                bid_fill_rate,
                ask_fill_rate,
                adverse_selection_rate,
                ret_1,
                ret_5,
                ret_10,
                (current_price - sma_5) / current_price,
                (current_price - sma_10) / current_price,
                volume_ratio,
                volume_trend / 1000.0,
                current_volume / 1000000.0,
                cash_pct,
                pnl_pct,
                total_trades / 100.0,
                self.unrealized_pnl / 10000.0,
                effective_spread,
                price_impact,
                realized_spread,
                (self.portfolio_value / self.initial_capital - 1),
            ],
            dtype=np.float32,
        )

        return observation

    def _take_action(self, action: np.ndarray):
        """
        Execute market making action.

        Args:
            action: [bid_spread_offset, ask_spread_offset] in [-1, 1]
        """
        current_price = self.data.iloc[self.current_step]["close"]

        # Scale actions to spread range
        bid_offset = (action[0] * 0.5 + 0.5) * (
            self.max_spread - self.min_spread
        ) + self.min_spread
        ask_offset = (action[1] * 0.5 + 0.5) * (
            self.max_spread - self.min_spread
        ) + self.min_spread

        # Set quotes
        self.bid_price = current_price * (1 - bid_offset)
        self.ask_price = current_price * (1 + ask_offset)

        # Round to tick size
        self.bid_price = round(self.bid_price / self.tick_size) * self.tick_size
        self.ask_price = round(self.ask_price / self.tick_size) * self.tick_size

        # Reset fill flags
        self.bid_filled = False
        self.ask_filled = False

        # Simulate order fills (probability based on spread competitiveness)
        # Tighter spreads = higher fill probability
        bid_fill_prob = max(0.1, 1.0 - bid_offset / self.max_spread) * 0.5
        ask_fill_prob = max(0.1, 1.0 - ask_offset / self.max_spread) * 0.5

        # Only fill if we have inventory room
        if np.random.random() < bid_fill_prob and self.inventory < self.max_inventory:
            # Bid filled - we buy
            self.bid_filled = True
            shares = min(10, self.max_inventory - self.inventory)  # Fill size
            cost = shares * self.bid_price
            rebate = shares * self.bid_price * self.maker_rebate

            if self.cash >= cost - rebate:
                self.inventory += shares
                self.cash -= cost - rebate
                self.total_trades += 1
                self.total_fills += 1

                # Check for adverse selection
                next_price = self.data.iloc[
                    min(self.current_step + 1, len(self.data) - 1)
                ]["close"]
                if next_price < self.bid_price * 0.999:  # Price moved against us
                    self.adverse_selection_count += 1

        if np.random.random() < ask_fill_prob and self.inventory > -self.max_inventory:
            # Ask filled - we sell
            self.ask_filled = True
            shares = min(10, self.inventory + self.max_inventory)  # Fill size

            if shares > 0:
                proceeds = shares * self.ask_price
                rebate = shares * self.ask_price * self.maker_rebate

                self.inventory -= shares
                self.cash += proceeds + rebate
                self.total_trades += 1
                self.total_fills += 1

                # Check for adverse selection
                next_price = self.data.iloc[
                    min(self.current_step + 1, len(self.data) - 1)
                ]["close"]
                if next_price > self.ask_price * 1.001:  # Price moved against us
                    self.adverse_selection_count += 1

        # Update portfolio value
        self.unrealized_pnl = self.inventory * current_price
        self.portfolio_value = self.cash + self.unrealized_pnl

    def _calculate_reward(self, action) -> float:
        """
        Calculate reward for market making.

        Components:
        - Realized P&L from spreads captured
        - Maker rebates
        - Inventory penalty (penalize large positions)
        - Fill rate bonus
        - Adverse selection penalty
        """
        # Realized P&L (change in portfolio value)
        pnl = self.portfolio_value - self.previous_portfolio_value

        # Inventory penalty (quadratic in inventory)
        inventory_penalty = (
            -self.inventory_penalty * (self.inventory / self.max_inventory) ** 2
        )

        # Fill rate bonus
        fill_bonus = 0.1 * (int(self.bid_filled) + int(self.ask_filled))

        # Adverse selection penalty
        adverse_penalty = (
            -0.5
            if (self.bid_filled or self.ask_filled)
            and (self.adverse_selection_count > self.total_fills * 0.3)
            else 0
        )

        # Spread capture bonus
        spread_bonus = 0
        if self.bid_filled and self.ask_filled:
            spread_bonus = 1.0  # Bonus for both sides filled

        reward = pnl + inventory_penalty + fill_bonus + adverse_penalty + spread_bonus

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.inventory = 0
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.bid_filled = False
        self.ask_filled = False
        self.total_fills = 0
        self.adverse_selection_count = 0
        self.unrealized_pnl = 0.0
        self.previous_portfolio_value = self.initial_capital

        return self._get_observation(), info


# Test the environment
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    n_steps = 500

    # GBM for price
    returns = np.random.normal(0.0001, 0.02, n_steps)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.001, n_steps)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.002, n_steps))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.002, n_steps))),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_steps),
        }
    )

    env = MarketMakingEnv(data)

    print("Market Making Environment Test")
    print("=" * 50)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${env.portfolio_value:,.2f}")

    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i + 1}:")
        print(f"  Inventory: {env.inventory}")
        print(f"  Bid filled: {env.bid_filled}, Ask filled: {env.ask_filled}")
        print(f"  Portfolio value: ${env.portfolio_value:,.2f}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nFinal Stats:")
    print(f"  Total trades: {env.total_trades}")
    print(f"  Total fills: {env.total_fills}")
    print(
        f"  Adverse selection rate: {env.adverse_selection_count / max(env.total_fills, 1):.2%}"
    )
    print(f"  Final P&L: ${env.portfolio_value - env.initial_capital:,.2f}")
