"""
Volatility Trading Environment

Implements delta-hedged volatility trading where the agent trades options
and dynamically hedges delta exposure. Compatible with PPO agent.

Action Space:
- Vega position size: [-1, 1] (long/short volatility)

Observation Space:
- Implied volatility levels and term structure
- Realized volatility
- Vol-of-vol
- Greeks (delta, gamma, vega, theta)
- Hedge ratio and hedging costs
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ..base_trading_env import BaseTradingEnv


class VolatilityTradingEnv(BaseTradingEnv):
    """
    Volatility trading environment for PPO agent.
    Agent trades volatility through delta-hedged option positions.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        option_multiplier: float = 100.0,
        rehedge_threshold: float = 0.1,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02,
        window_size: int = 20,
    ):
        """
        Initialize Volatility Trading Environment.

        Args:
            data: OHLCV data with 'close', 'volume', and optionally 'implied_vol'
            initial_capital: Starting capital
            option_multiplier: Contract size multiplier
            rehedge_threshold: Delta threshold to trigger rehedge
            transaction_cost: Hedging transaction cost
            risk_free_rate: Risk-free rate for Greeks calculation
            window_size: Lookback for realized vol calculation
        """
        super().__init__(data, initial_capital)

        self.option_multiplier = option_multiplier
        self.rehedge_threshold = rehedge_threshold
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size

        # Vol trading state
        self.vega_position = 0.0  # Net vega exposure
        self.delta_position = 0.0  # Net delta from options
        self.hedge_position = 0.0  # Underlying shares for delta hedge
        self.gamma = 0.0
        self.theta = 0.0
        self.total_hedge_cost = 0.0

        # Action space: vega position in [-1, 1]
        # Will be scaled to actual position size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space (22 features):
        # - Implied vol (current) (1)
        # - Realized vol (1)
        # - Vol-of-vol (1)
        # - IV term structure (3: 1m, 3m, 6m)
        # - Greeks: delta, gamma, vega, theta (4)
        # - Hedge ratio (1)
        # - Vega position (1)
        # - Vol spread (IV - RV) (1)
        # - Price momentum (3)
        # - Volatility regime indicators (3)
        # - Portfolio state (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

    def _calculate_realized_vol(self) -> float:
        """Calculate realized volatility."""
        if self.current_step < self.window_size:
            return 0.20  # Default 20% vol

        returns = (
            self.data.iloc[self.current_step - self.window_size : self.current_step][
                "close"
            ]
            .pct_change()
            .dropna()
        )

        return returns.std() * np.sqrt(252)

    def _calculate_vol_of_vol(self) -> float:
        """Calculate volatility of volatility."""
        if self.current_step < self.window_size * 2:
            return 0.5

        # Calculate rolling realized vol
        vols = []
        for i in range(self.window_size):
            step = self.current_step - self.window_size + i
            if step >= self.window_size:
                rets = (
                    self.data.iloc[step - self.window_size : step]["close"]
                    .pct_change()
                    .dropna()
                )
                vols.append(rets.std())

        if len(vols) > 1:
            return np.std(vols) / np.mean(vols) if np.mean(vols) > 0 else 0.5
        return 0.5

    def _get_implied_vol(self) -> float:
        """Get implied volatility (simulated if not in data)."""
        if "implied_vol" in self.data.columns:
            return self.data.iloc[self.current_step]["implied_vol"]

        # Simulate IV as RV + noise
        rv = self._calculate_realized_vol()
        iv_noise = np.random.normal(0.05, 0.03)  # IV tends to be higher than RV
        return max(0.05, rv + iv_noise)

    def _calculate_greeks(
        self, S: float, K: float, T: float, sigma: float
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks (simplified).

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            sigma: Implied volatility

        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}

        # Simplified Black-Scholes approximations
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )
        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal PDF and CDF
        from scipy.stats import norm

        pdf = norm.pdf(d1)
        cdf = norm.cdf(d1)

        delta = cdf
        gamma = pdf / (S * sigma * np.sqrt(T))
        vega = S * pdf * np.sqrt(T) / 100  # Divide by 100 for per-point
        theta = -(S * pdf * sigma) / (
            2 * np.sqrt(T)
        ) - self.risk_free_rate * K * norm.cdf(d2)
        theta = theta / 365  # Per day

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_price = self.data.iloc[self.current_step]["close"]

        # Volatility features
        iv = self._get_implied_vol()
        rv = self._calculate_realized_vol()
        vov = self._calculate_vol_of_vol()
        vol_spread = iv - rv

        # Term structure (simulated)
        iv_1m = iv * (1 + np.random.normal(0, 0.05))
        iv_3m = iv * (1 + np.random.normal(0.02, 0.05))
        iv_6m = iv * (1 + np.random.normal(0.03, 0.05))

        # Greeks (using ATM options)
        K = current_price
        T = 30 / 365  # 30-day options
        greeks = self._calculate_greeks(current_price, K, T, iv)

        # Position features
        hedge_ratio = self.hedge_position / (abs(self.delta_position) + 1e-8)

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
            ret_20 = (
                self.data.iloc[self.current_step]["close"]
                / self.data.iloc[self.current_step - 20]["close"]
                - 1
                if self.current_step >= 20
                else 0
            )
        else:
            ret_1 = ret_5 = ret_20 = 0

        # Volatility regime
        vol_regime = "high" if rv > 0.25 else "medium" if rv > 0.15 else "low"
        vol_trending = (
            1 if self.current_step >= 10 and rv > self._calculate_realized_vol() else -1
        )
        vol_percentile = np.clip(rv / 0.5, 0, 1)  # Normalize to 0-1

        # Portfolio state
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        cash_pct = self.cash / self.initial_capital

        observation = np.array(
            [
                iv,
                rv,
                vov,
                iv_1m,
                iv_3m,
                iv_6m,
                greeks["delta"],
                greeks["gamma"],
                greeks["vega"],
                greeks["theta"] / 100.0,  # Scale down
                hedge_ratio,
                self.vega_position / 100.0,
                vol_spread,
                ret_1,
                ret_5,
                ret_20,
                1.0 if vol_regime == "high" else 0.0,
                1.0 if vol_regime == "medium" else 0.0,
                float(vol_trending),
                vol_percentile,
                pnl_pct,
                cash_pct,
            ],
            dtype=np.float32,
        )

        return observation

    def _take_action(self, action: np.ndarray):
        """
        Execute volatility trading action.

        Args:
            action: Target vega position in [-1, 1]
        """
        current_price = self.data.iloc[self.current_step]["close"]

        # Scale action to actual vega position
        target_vega = action[0] * 100.0  # Max 100 vega

        # Calculate position change
        vega_change = target_vega - self.vega_position

        if abs(vega_change) > 1.0:  # Only trade if meaningful change
            # Option cost (simplified: vega * IV)
            iv = self._get_implied_vol()
            option_cost = abs(vega_change) * iv * self.option_multiplier

            if self.cash >= option_cost:
                self.vega_position = target_vega
                self.cash -= option_cost * self.transaction_cost  # Transaction cost
                self.total_trades += 1

                # Update delta from options (simplified)
                K = current_price
                T = 30 / 365
                greeks = self._calculate_greeks(current_price, K, T, iv)
                self.delta_position = self.vega_position * greeks["delta"]
                self.gamma = self.vega_position * greeks["gamma"]
                self.theta = self.vega_position * greeks["theta"]

        # Delta hedging
        delta_to_hedge = self.delta_position + self.hedge_position

        if abs(delta_to_hedge) > self.rehedge_threshold * abs(self.delta_position):
            # Rehedge by trading underlying
            shares_to_trade = -delta_to_hedge
            hedge_cost = abs(shares_to_trade) * current_price * self.transaction_cost

            if self.cash >= hedge_cost:
                self.hedge_position += shares_to_trade
                self.cash -= hedge_cost
                self.total_hedge_cost += hedge_cost
                self.total_trades += 1

        # Update portfolio value
        # Simplified P&L: vega P&L + theta decay + delta hedged
        iv = self._get_implied_vol()
        rv = self._calculate_realized_vol()

        # Vega P&L (from change in IV)
        if hasattr(self, "previous_iv"):
            iv_change = iv - self.previous_iv
            vega_pnl = self.vega_position * iv_change * self.option_multiplier
        else:
            vega_pnl = 0

        self.previous_iv = iv

        # Theta decay
        theta_pnl = self.theta * self.option_multiplier

        # Gamma P&L (realized vol trading)
        gamma_pnl = 0.5 * self.gamma * (rv - iv) * self.option_multiplier

        # Delta hedge P&L
        hedge_pnl = self.hedge_position * (
            current_price - self.data.iloc[self.current_step - 1]["close"]
            if self.current_step > 0
            else 0
        )

        self.unrealized_pnl = vega_pnl + theta_pnl + gamma_pnl + hedge_pnl
        self.portfolio_value = self.cash + self.unrealized_pnl

    def _calculate_reward(self) -> float:
        """
        Calculate reward for volatility trading.

        Components:
        - Vega P&L (profit from vol changes)
        - Gamma P&L (profit from realized vol vs implied)
        - Theta decay (negative carry)
        - Hedging costs (penalty)
        """
        # P&L change
        pnl = self.portfolio_value - self.previous_portfolio_value

        # Hedging cost penalty
        hedge_penalty = -self.total_hedge_cost / 1000.0

        # Position size penalty (encourage active trading)
        if abs(self.vega_position) < 10:
            size_penalty = -0.1
        else:
            size_penalty = 0

        reward = pnl + hedge_penalty + size_penalty

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.vega_position = 0.0
        self.delta_position = 0.0
        self.hedge_position = 0.0
        self.gamma = 0.0
        self.theta = 0.0
        self.total_hedge_cost = 0.0
        self.previous_iv = self._get_implied_vol()

        return self._get_observation(), info


# Test the environment
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    n_steps = 500

    # GBM with stochastic vol
    returns = np.random.normal(0, 0.02, n_steps)
    prices = 100 * np.exp(np.cumsum(returns))

    # Simulate implied vol (mean-reverting)
    iv = [0.25]
    for _ in range(n_steps - 1):
        iv.append(iv[-1] + 0.3 * (0.25 - iv[-1]) + np.random.normal(0, 0.02))

    data = pd.DataFrame(
        {
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_steps),
            "implied_vol": np.array(iv),
        }
    )

    env = VolatilityTradingEnv(data)

    print("Volatility Trading Environment Test")
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
        print(f"  Vega position: {env.vega_position:.2f}")
        print(f"  Delta position: {env.delta_position:.2f}")
        print(f"  Hedge position: {env.hedge_position:.2f}")
        print(f"  Portfolio value: ${env.portfolio_value:,.2f}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nFinal Stats:")
    print(f"  Total trades: {env.total_trades}")
    print(f"  Total hedge cost: ${env.total_hedge_cost:,.2f}")
    print(f"  Final P&L: ${env.portfolio_value - env.initial_capital:,.2f}")
