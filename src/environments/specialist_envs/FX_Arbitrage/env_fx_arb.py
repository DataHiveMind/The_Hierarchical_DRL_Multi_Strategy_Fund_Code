"""
FX Arbitrage Environment

Implements FX arbitrage strategies including triangular arbitrage and carry trades.
Compatible with DDPG agent (continuous position sizing).

Action Space:
- Position sizes for currency pairs: [-1, 1] for each pair

Observation Space:
- Spot rates and crosses
- Forward points and interest differentials
- Triangular arbitrage deviations
- Carry trade metrics
- Volatility and correlations
"""

from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from ..base_trading_env import BaseTradingEnv


class FXArbitrageEnv(BaseTradingEnv):
    """
    FX arbitrage environment for DDPG agent.
    Trades FX triangles and carry opportunities.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        currency_pairs: List[str] = ["EUR/USD", "USD/JPY", "EUR/JPY"],
        leverage: float = 10.0,
        transaction_cost: float = 0.0001,  # 1 pip
        funding_cost: float = 0.00001,  # Daily funding
        window_size: int = 20,
    ):
        """
        Initialize FX Arbitrage Environment.

        Args:
            data: DataFrame with FX rate columns and interest rates
            initial_capital: Starting capital
            currency_pairs: List of currency pairs to trade
            leverage: Maximum leverage
            transaction_cost: Transaction cost (spread)
            funding_cost: Daily funding cost
            window_size: Lookback for statistics
        """
        super().__init__(data, initial_capital)

        self.currency_pairs = currency_pairs
        self.num_pairs = len(currency_pairs)
        self.leverage = leverage
        self.transaction_cost = transaction_cost
        self.funding_cost = funding_cost
        self.window_size = window_size

        # FX trading state
        self.positions = np.zeros(self.num_pairs)  # Position in each pair
        self.entry_rates = np.zeros(self.num_pairs)
        self.total_funding_cost = 0.0

        # Action space: position size for each pair in [-1, 1]
        # Will be scaled by leverage and capital
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_pairs,), dtype=np.float32
        )

        # Observation space (dynamic based on num_pairs):
        # Per pair: rate, return, volatility, carry, position (5 * num_pairs)
        # Cross-pair: correlations (num_pairs * (num_pairs-1) / 2)
        # Triangular arb: deviations (1 for triangle)
        # Portfolio: (4 metrics)
        # Interest differentials: (num_pairs)

        obs_dim = (
            5 * self.num_pairs
            + int(self.num_pairs * (self.num_pairs - 1) / 2)
            + 1  # triangle deviation
            + 4  # portfolio metrics
            + self.num_pairs  # interest differentials
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_fx_rate(self, pair: str) -> float:
        """Get current FX rate for pair."""
        pair_clean = pair.replace("/", "_")

        if pair_clean in self.data.columns:
            return self.data.iloc[self.current_step][pair_clean]
        elif pair in self.data.columns:
            return self.data.iloc[self.current_step][pair]
        else:
            # Simulate rate
            if pair == "EUR/USD":
                return 1.10 + np.random.normal(0, 0.01)
            elif pair == "USD/JPY":
                return 110.0 + np.random.normal(0, 1.0)
            elif pair == "EUR/JPY":
                return 121.0 + np.random.normal(0, 1.2)
            else:
                return 1.0 + np.random.normal(0, 0.05)

    def _get_interest_diff(self, pair: str) -> float:
        """Get interest rate differential for pair."""
        pair_clean = pair.replace("/", "_")
        col_name = f"{pair_clean}_interest"

        if col_name in self.data.columns:
            return self.data.iloc[self.current_step][col_name]
        else:
            # Simulate interest differential
            return np.random.normal(0.02, 0.01)  # 2% average carry

    def _calculate_triangle_deviation(self) -> float:
        """
        Calculate triangular arbitrage deviation.
        EUR/USD * USD/JPY should equal EUR/JPY
        """
        if (
            "EUR/USD" in self.currency_pairs
            and "USD/JPY" in self.currency_pairs
            and "EUR/JPY" in self.currency_pairs
        ):
            eur_usd = self._get_fx_rate("EUR/USD")
            usd_jpy = self._get_fx_rate("USD/JPY")
            eur_jpy = self._get_fx_rate("EUR/JPY")

            synthetic_eur_jpy = eur_usd * usd_jpy
            deviation = (eur_jpy - synthetic_eur_jpy) / synthetic_eur_jpy

            return deviation
        else:
            return 0.0

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        observation_parts = []

        # Per-pair features
        for i, pair in enumerate(self.currency_pairs):
            # Current rate
            rate = self._get_fx_rate(pair)

            # Return
            if self.current_step > 0:
                prev_rate = (
                    self._get_fx_rate(pair)
                    if self.current_step == 1
                    else self.entry_rates[i]
                )
                fx_return = (rate - prev_rate) / prev_rate if prev_rate > 0 else 0
            else:
                fx_return = 0

            # Volatility
            pair_clean = pair.replace("/", "_")
            if (
                self.current_step >= self.window_size
                and pair_clean in self.data.columns
            ):
                rates = self.data.iloc[
                    self.current_step - self.window_size : self.current_step
                ][pair_clean].values
                returns = np.diff(rates) / rates[:-1]
                volatility = np.std(returns) * np.sqrt(252)
            else:
                volatility = 0.10  # Default 10% vol

            # Carry (interest differential)
            carry = self._get_interest_diff(pair)

            # Position
            position = self.positions[i]

            observation_parts.extend(
                [rate / 100.0, fx_return, volatility, carry, position / 100.0]
            )

        # Cross-pair correlations
        if self.current_step >= self.window_size:
            for i in range(self.num_pairs):
                for j in range(i + 1, self.num_pairs):
                    pair_i = self.currency_pairs[i].replace("/", "_")
                    pair_j = self.currency_pairs[j].replace("/", "_")

                    if pair_i in self.data.columns and pair_j in self.data.columns:
                        rates_i = self.data.iloc[
                            self.current_step - self.window_size : self.current_step
                        ][pair_i].values
                        rates_j = self.data.iloc[
                            self.current_step - self.window_size : self.current_step
                        ][pair_j].values
                        returns_i = np.diff(rates_i) / rates_i[:-1]
                        returns_j = np.diff(rates_j) / rates_j[:-1]
                        corr = (
                            np.corrcoef(returns_i, returns_j)[0, 1]
                            if len(returns_i) > 1
                            else 0
                        )
                    else:
                        corr = np.random.normal(0, 0.3)

                    observation_parts.append(corr)
        else:
            # Default correlations
            num_corrs = int(self.num_pairs * (self.num_pairs - 1) / 2)
            observation_parts.extend([0.0] * num_corrs)

        # Triangular arbitrage deviation
        triangle_dev = self._calculate_triangle_deviation()
        observation_parts.append(triangle_dev)

        # Portfolio metrics
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        cash_pct = self.cash / self.initial_capital
        total_exposure = np.abs(self.positions).sum() / 100.0
        funding_cost_pct = self.total_funding_cost / self.initial_capital

        observation_parts.extend([pnl_pct, cash_pct, total_exposure, funding_cost_pct])

        # Interest differentials
        for pair in self.currency_pairs:
            interest_diff = self._get_interest_diff(pair)
            observation_parts.append(interest_diff)

        return np.array(observation_parts, dtype=np.float32)

    def _take_action(self, action: np.ndarray):
        """
        Execute FX arbitrage action.

        Args:
            action: Target position sizes [-1, 1] for each pair
        """
        # Scale actions to actual position sizes
        max_position = self.leverage * self.initial_capital
        target_positions = action * max_position / self.num_pairs

        # Calculate position changes
        position_changes = target_positions - self.positions

        # Execute trades
        for i, pair in enumerate(self.currency_pairs):
            if abs(position_changes[i]) > max_position * 0.01:  # Min trade threshold
                # Get current rate
                rate = self._get_fx_rate(pair)

                # Transaction cost
                trade_cost = abs(position_changes[i]) * self.transaction_cost

                if self.cash >= trade_cost:
                    # Update position
                    self.positions[i] = target_positions[i]
                    self.entry_rates[i] = rate

                    # Deduct transaction cost
                    self.cash -= trade_cost
                    self.total_trades += 1

        # Calculate P&L
        total_pnl = 0

        for i, pair in enumerate(self.currency_pairs):
            if abs(self.positions[i]) > 0:
                # Current rate
                current_rate = self._get_fx_rate(pair)

                # P&L from rate change
                if self.entry_rates[i] > 0:
                    rate_pnl = (
                        self.positions[i]
                        * (current_rate - self.entry_rates[i])
                        / self.entry_rates[i]
                    )
                else:
                    rate_pnl = 0

                # Carry P&L (interest differential)
                carry = self._get_interest_diff(pair)
                carry_pnl = self.positions[i] * carry / 252  # Daily carry

                # Funding cost
                funding = abs(self.positions[i]) * self.funding_cost
                self.total_funding_cost += funding

                total_pnl += rate_pnl + carry_pnl - funding

        self.unrealized_pnl = total_pnl
        self.portfolio_value = self.cash + total_pnl

    def _calculate_reward(self) -> float:
        """
        Calculate reward for FX arbitrage.

        Components:
        - P&L from rate changes
        - Carry profit
        - Funding cost penalty
        - Triangular arbitrage bonus
        """
        # P&L change
        pnl = self.portfolio_value - self.previous_portfolio_value

        # Triangular arbitrage bonus
        triangle_dev = self._calculate_triangle_deviation()
        if abs(triangle_dev) > 0.001:  # Meaningful deviation
            # Reward taking advantage of mispricing
            triangle_bonus = 1.0 * abs(triangle_dev) * 100
        else:
            triangle_bonus = 0

        # Carry bonus (reward positive carry positions)
        carry_bonus = 0
        for i, pair in enumerate(self.currency_pairs):
            if abs(self.positions[i]) > 0:
                carry = self._get_interest_diff(pair)
                if self.positions[i] * carry > 0:  # Position aligned with carry
                    carry_bonus += 0.1

        # Funding penalty
        funding_penalty = -self.total_funding_cost / 1000.0

        reward = pnl + triangle_bonus + carry_bonus + funding_penalty

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.positions = np.zeros(self.num_pairs)
        self.entry_rates = np.zeros(self.num_pairs)
        self.total_funding_cost = 0.0

        return self._get_observation(), info


# Test the environment
if __name__ == "__main__":
    # Create synthetic FX data
    np.random.seed(42)
    n_steps = 500

    # Simulate FX rates with realistic dynamics
    eur_usd = 1.10 + np.cumsum(np.random.normal(0, 0.001, n_steps))
    usd_jpy = 110.0 + np.cumsum(np.random.normal(0, 0.1, n_steps))

    # EUR/JPY should approximately equal EUR/USD * USD/JPY (with some noise)
    eur_jpy = eur_usd * usd_jpy + np.random.normal(0, 0.5, n_steps)

    # Interest rate differentials
    eur_usd_int = np.random.normal(0.02, 0.005, n_steps)
    usd_jpy_int = np.random.normal(-0.01, 0.005, n_steps)
    eur_jpy_int = eur_usd_int + usd_jpy_int

    data = pd.DataFrame(
        {
            "EUR_USD": eur_usd,
            "USD_JPY": usd_jpy,
            "EUR_JPY": eur_jpy,
            "EUR_USD_interest": eur_usd_int,
            "USD_JPY_interest": usd_jpy_int,
            "EUR_JPY_interest": eur_jpy_int,
            "volume": np.random.randint(1000000, 10000000, n_steps),
        }
    )

    env = FXArbitrageEnv(data)

    print("FX Arbitrage Environment Test")
    print("=" * 50)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Currency pairs: {env.currency_pairs}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")

    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Calculate triangle deviation
        triangle_dev = env._calculate_triangle_deviation()

        print(f"\nStep {i + 1}:")
        print(f"  Positions: {env.positions}")
        print(f"  Triangle deviation: {triangle_dev:.6f}")
        print(f"  Portfolio value: ${env.portfolio_value:,.2f}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nFinal Stats:")
    print(f"  Total trades: {env.total_trades}")
    print(f"  Total funding cost: ${env.total_funding_cost:,.2f}")
    print(f"  Final P&L: ${env.portfolio_value - env.initial_capital:,.2f}")
