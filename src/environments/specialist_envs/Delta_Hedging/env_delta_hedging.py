"""
Delta Hedging Environment

Implements dynamic delta hedging of option positions.
Compatible with DDPG agent (continuous hedge ratio control).

Action Space:
- Hedge ratio: [-1.5, 1.5] (multiple of theoretical delta)

Observation Space:
- Greeks (delta, gamma, vega, theta, rho)
- Underlying price and dynamics
- Time to expiry
- Hedge effectiveness metrics
- Transaction costs incurred
"""

from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ..base_trading_env import BaseTradingEnv


class DeltaHedgingEnv(BaseTradingEnv):
    """
    Delta hedging environment for DDPG agent.
    Agent learns optimal hedge ratios to minimize delta risk.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        option_position: float = 100.0,  # Fixed long option position
        strike_price: float = 100.0,
        time_to_expiry_days: int = 30,
        volatility: float = 0.25,
        risk_free_rate: float = 0.02,
        transaction_cost: float = 0.0005,
        rehedge_frequency: int = 1,  # Every step
    ):
        """
        Initialize Delta Hedging Environment.

        Args:
            data: OHLCV data
            initial_capital: Starting capital
            option_position: Fixed option position (vega units)
            strike_price: Option strike
            time_to_expiry_days: Days to expiration
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            transaction_cost: Cost per trade
            rehedge_frequency: Steps between rehedges
        """
        super().__init__(data, initial_capital)
        self.data = self.df  # Alias for compatibility

        self.option_position = option_position
        self.strike_price = strike_price
        self.initial_tte = time_to_expiry_days / 365.0
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.rehedge_frequency = rehedge_frequency

        # Hedging state
        self.hedge_shares = 0.0
        self.time_to_expiry = self.initial_tte
        self.theoretical_delta = 0.0
        self.gamma = 0.0
        self.vega = 0.0
        self.theta = 0.0
        self.total_hedge_cost = 0.0
        self.hedge_errors = []
        self.previous_portfolio_value = initial_capital  # Track for reward calculation

        # Action space: hedge ratio multiplier [-1.5, 1.5]
        # 1.0 = theoretical delta, <1 = under-hedge, >1 = over-hedge
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float32)

        # Observation space (18 features):
        # - Spot price (1)
        # - Moneyness (S/K) (1)
        # - Time to expiry (1)
        # - Greeks: delta, gamma, vega, theta, rho (5)
        # - Current hedge ratio (1)
        # - Hedge error (actual - theoretical) (1)
        # - Price change (1)
        # - Realized vol (1)
        # - Implied vol (1)
        # - Cumulative hedge cost (1)
        # - Portfolio state (3)
        # - Hedge effectiveness (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

    def _calculate_greeks(
        self, S: float, K: float, T: float, sigma: float, r: float
    ) -> Dict[str, float]:
        """Calculate Black-Scholes Greeks."""
        if T <= 0:
            # At expiry
            delta = 1.0 if S > K else 0.0
            return {"delta": delta, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

        from scipy.stats import norm

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        delta = cdf_d1
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T) / 100
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
        ) / 365
        rho = K * T * np.exp(-r * T) * cdf_d2 / 100

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    def _get_obs(self) -> np.ndarray:
        """Get current observation (required by base class)."""
        return self._get_observation()

    def _get_info(self) -> Dict:
        """Get auxiliary diagnostic information."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "hedge_shares": self.hedge_shares,
            "theoretical_delta": self.theoretical_delta,
            "total_trades": self.total_trades,
            "total_hedge_cost": self.total_hedge_cost,
            "time_to_expiry": self.time_to_expiry,
        }

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_price = self.data.iloc[self.current_step]["close"]

        # Update time to expiry (decay)
        days_passed = self.current_step
        self.time_to_expiry = max(0, self.initial_tte - days_passed / 365.0)

        # Calculate Greeks
        greeks = self._calculate_greeks(
            current_price,
            self.strike_price,
            self.time_to_expiry,
            self.volatility,
            self.risk_free_rate,
        )

        self.theoretical_delta = greeks["delta"] * self.option_position
        self.gamma = greeks["gamma"] * self.option_position
        self.vega = greeks["vega"] * self.option_position
        self.theta = greeks["theta"] * self.option_position

        # Hedge metrics
        current_hedge_ratio = (
            self.hedge_shares / abs(self.theoretical_delta)
            if abs(self.theoretical_delta) > 0
            else 0
        )
        hedge_error = self.hedge_shares - self.theoretical_delta

        # Price dynamics
        if self.current_step > 0:
            price_change = (
                current_price - self.data.iloc[self.current_step - 1]["close"]
            ) / current_price
        else:
            price_change = 0

        # Realized volatility
        if self.current_step >= 20:
            returns = (
                self.data.iloc[self.current_step - 20 : self.current_step]["close"]
                .pct_change()
                .dropna()
            )
            realized_vol = returns.std() * np.sqrt(252)
        else:
            realized_vol = self.volatility

        # Hedge effectiveness
        if len(self.hedge_errors) > 0:
            hedge_error_std = np.std(self.hedge_errors)
            hedge_error_mean = np.mean(self.hedge_errors)
        else:
            hedge_error_std = 0
            hedge_error_mean = 0

        # Portfolio state
        moneyness = current_price / self.strike_price
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        cash_pct = self.cash / self.initial_capital

        observation = np.array(
            [
                current_price / 100.0,
                moneyness,
                self.time_to_expiry,
                greeks["delta"],
                greeks["gamma"] * 100,  # Scale
                greeks["vega"],
                greeks["theta"],
                greeks["rho"],
                current_hedge_ratio,
                hedge_error / 100.0,
                price_change,
                realized_vol,
                self.volatility,
                self.total_hedge_cost / 10000.0,
                pnl_pct,
                cash_pct,
                hedge_error_std / 100.0,
                hedge_error_mean / 100.0,
            ],
            dtype=np.float32,
        )

        return observation

    def _take_action(self, action: np.ndarray):
        """
        Execute delta hedging action.

        Args:
            action: Hedge ratio multiplier in [-1.5, 1.5]
        """
        current_price = self.data.iloc[self.current_step]["close"]

        # Only rehedge at specified frequency
        if self.current_step % self.rehedge_frequency != 0:
            return

        # Calculate target hedge
        hedge_multiplier = action[0]
        target_hedge = self.theoretical_delta * hedge_multiplier

        # Calculate shares to trade
        shares_to_trade = target_hedge - self.hedge_shares

        if abs(shares_to_trade) > 0.1:  # Minimum trade size
            # Transaction cost
            trade_cost = abs(shares_to_trade) * current_price * self.transaction_cost

            if self.cash >= trade_cost:
                self.hedge_shares += shares_to_trade
                self.cash -= trade_cost
                self.total_hedge_cost += trade_cost
                self.total_trades += 1

                # Track hedge error
                hedge_error = self.hedge_shares - self.theoretical_delta
                self.hedge_errors.append(hedge_error)

                # Keep only recent errors
                if len(self.hedge_errors) > 100:
                    self.hedge_errors = self.hedge_errors[-100:]

        # Calculate P&L components

        # 1. Option P&L (from price move and Greeks)
        if self.current_step > 0:
            price_change = (
                current_price - self.data.iloc[self.current_step - 1]["close"]
            )

            # Delta P&L
            option_delta_pnl = self.theoretical_delta * price_change

            # Gamma P&L (second-order)
            gamma_pnl = 0.5 * self.gamma * (price_change**2)

            # Theta decay
            theta_pnl = self.theta
        else:
            option_delta_pnl = 0
            gamma_pnl = 0
            theta_pnl = 0

        # 2. Hedge P&L
        if self.current_step > 0:
            hedge_pnl = self.hedge_shares * price_change
        else:
            hedge_pnl = 0

        # Net P&L
        self.unrealized_pnl = option_delta_pnl + gamma_pnl + theta_pnl + hedge_pnl
        self.portfolio_value = self.cash + self.unrealized_pnl

    def _calculate_reward(self, action) -> float:
        """
        Calculate reward for delta hedging.

        Reward components:
        - Hedge effectiveness (minimize delta exposure)
        - P&L (gamma profits)
        - Transaction cost penalty
        """
        # P&L change
        pnl = self.portfolio_value - self.previous_portfolio_value

        # Hedge error penalty (want to minimize residual delta)
        hedge_error = abs(self.hedge_shares - self.theoretical_delta)
        hedge_penalty = -hedge_error / 10.0

        # Transaction cost penalty
        cost_penalty = -self.total_hedge_cost / 1000.0

        # Gamma profit bonus (successful hedging should capture gamma)
        gamma_bonus = 0
        if self.current_step > 0:
            price_change = abs(
                self.data.iloc[self.current_step]["close"]
                - self.data.iloc[self.current_step - 1]["close"]
            )
            if price_change > 0:
                # More price movement = more gamma profit opportunity
                gamma_bonus = 0.1 * price_change

        reward = pnl + hedge_penalty + cost_penalty + gamma_bonus

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.hedge_shares = 0.0
        self.time_to_expiry = self.initial_tte
        self.total_hedge_cost = 0.0
        self.hedge_errors = []
        self.previous_portfolio_value = self.initial_capital

        return self._get_observation(), info


# Test the environment
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    n_steps = 100

    # GBM
    returns = np.random.normal(0, 0.02, n_steps)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {"close": prices, "volume": np.random.randint(100000, 1000000, n_steps)}
    )

    env = DeltaHedgingEnv(data, strike_price=100.0, time_to_expiry_days=30)

    print("Delta Hedging Environment Test")
    print("=" * 50)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Strike: {env.strike_price}")
    print(f"Option position: {env.option_position}")

    # Run a few steps with different hedge ratios
    for i in range(10):
        # Try different strategies: under-hedge, perfect, over-hedge
        if i < 3:
            action = np.array([0.8])  # Under-hedge
        elif i < 6:
            action = np.array([1.0])  # Perfect hedge
        else:
            action = np.array([1.2])  # Over-hedge

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i + 1}:")
        print(f"  Theoretical delta: {env.theoretical_delta:.2f}")
        print(f"  Hedge shares: {env.hedge_shares:.2f}")
        print(f"  Hedge error: {env.hedge_shares - env.theoretical_delta:.2f}")
        print(f"  Portfolio value: ${env.portfolio_value:,.2f}")
        print(f"  Reward: {reward:.4f}")

        if terminated or truncated:
            break

    print(f"\nFinal Stats:")
    print(f"  Total trades: {env.total_trades}")
    print(f"  Total hedge cost: ${env.total_hedge_cost:,.2f}")
    print(f"  Final P&L: ${env.portfolio_value - env.initial_capital:,.2f}")
