"""
Factor Tracking Environment

Implements factor-based portfolio construction with discrete factor tilts.
Compatible with DQN agent (discrete actions for factor exposures).

Action Space:
- Discrete actions: [0, 1, 2] for each factor = [Short, Neutral, Long]
- With 3 factors: 27 possible action combinations (3^3)

Observation Space:
- Factor returns and momentum
- Factor valuations
- Factor correlations
- Portfolio factor exposures
- Risk metrics
"""

from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ..base_trading_env import BaseTradingEnv


class FactorTrackingEnv(BaseTradingEnv):
    """
    Factor tracking environment for DQN agent.
    Agent selects discrete factor tilts to construct optimal portfolio.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        num_factors: int = 3,  # Value, Momentum, Quality
        rebalance_frequency: int = 5,  # Rebalance every N steps
        transaction_cost: float = 0.001,
        target_volatility: float = 0.15,
        window_size: int = 60,
    ):
        """
        Initialize Factor Tracking Environment.

        Args:
            data: DataFrame with factor returns columns: 'value_ret', 'momentum_ret', 'quality_ret', 'market_ret'
            initial_capital: Starting capital
            num_factors: Number of factors to track
            rebalance_frequency: Steps between rebalancing
            transaction_cost: Transaction cost rate
            target_volatility: Target portfolio volatility
            window_size: Lookback for statistics
        """
        super().__init__(data, initial_capital)
        self.data = self.df  # Alias for compatibility

        self.num_factors = num_factors
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.target_volatility = target_volatility
        self.window_size = window_size

        # Factor names
        self.factor_names = ["value", "momentum", "quality"][:num_factors]

        # Portfolio state
        self.factor_weights = np.zeros(num_factors)
        self.factor_exposures = np.zeros(num_factors)
        self.last_rebalance = 0
        self.turnover = 0.0
        self.previous_portfolio_value = initial_capital  # Track for reward calculation

        # Action space: Discrete actions for each factor exposure
        # 0 = Short (-1), 1 = Neutral (0), 2 = Long (+1)
        # Total actions = 3^num_factors
        self.action_space = spaces.Discrete(3**num_factors)

        # Observation space (variable based on num_factors):
        # Base features (12):
        # - Factor returns (num_factors)
        # - Factor momentum (num_factors)
        # - Factor valuations (num_factors)
        # - Current factor weights (num_factors)
        # - Factor volatilities (num_factors)
        # - Factor correlations (num_factors * (num_factors-1) / 2)
        # - Portfolio metrics (6)
        # - Market features (3)
        obs_dim = (
            num_factors * 5  # returns, momentum, valuations, weights, volatilities
            + int(num_factors * (num_factors - 1) / 2)  # correlations
            + 9  # portfolio + market metrics
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _decode_action(self, action: int) -> np.ndarray:
        """
        Decode discrete action to factor exposures.

        Args:
            action: Integer action [0, 3^num_factors - 1]

        Returns:
            Array of factor exposures [-1, 0, 1]
        """
        exposures = []
        remaining = action

        for _ in range(self.num_factors):
            exposure = remaining % 3
            exposures.append(exposure - 1)  # Convert 0,1,2 to -1,0,1
            remaining //= 3

        return np.array(exposures, dtype=np.float32)

    def _get_factor_returns(self) -> np.ndarray:
        """Get current factor returns."""
        returns = []
        for factor in self.factor_names:
            col_name = f"{factor}_ret"
            if col_name in self.data.columns:
                returns.append(self.data.iloc[self.current_step][col_name])
            else:
                # Simulate factor returns
                returns.append(np.random.normal(0.0005, 0.01))

        return np.array(returns)

    def _get_obs(self) -> np.ndarray:
        """Get current observation (required by base class)."""
        return self._get_observation()

    def _get_info(self) -> Dict:
        """Get auxiliary diagnostic information."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "factor_weights": self.factor_weights.tolist(),
            "factor_exposures": self.factor_exposures.tolist(),
            "total_trades": self.total_trades,
            "turnover": self.turnover,
        }

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Factor returns
        factor_returns = self._get_factor_returns()

        # Factor momentum (trailing returns)
        factor_momentum = []
        for i, factor in enumerate(self.factor_names):
            col_name = f"{factor}_ret"
            if self.current_step >= 20:
                if col_name in self.data.columns:
                    momentum = self.data.iloc[
                        self.current_step - 20 : self.current_step
                    ][col_name].sum()
                else:
                    momentum = np.random.normal(0, 0.05)
            else:
                momentum = 0
            factor_momentum.append(momentum)

        factor_momentum = np.array(factor_momentum)

        # Factor valuations (z-score of cumulative returns)
        factor_valuations = []
        for i, factor in enumerate(self.factor_names):
            col_name = f"{factor}_ret"
            if self.current_step >= self.window_size:
                if col_name in self.data.columns:
                    cum_ret = self.data.iloc[
                        self.current_step - self.window_size : self.current_step
                    ][col_name].sum()
                    hist_rets = [
                        self.data.iloc[j - self.window_size : j][col_name].sum()
                        for j in range(self.window_size, self.current_step)
                        if j >= self.window_size
                    ]
                    if len(hist_rets) > 0:
                        z_score = (cum_ret - np.mean(hist_rets)) / (
                            np.std(hist_rets) + 1e-8
                        )
                    else:
                        z_score = 0
                else:
                    z_score = np.random.normal(0, 1)
            else:
                z_score = 0
            factor_valuations.append(z_score)

        factor_valuations = np.array(factor_valuations)

        # Factor volatilities
        factor_vols = []
        for i, factor in enumerate(self.factor_names):
            col_name = f"{factor}_ret"
            if self.current_step >= 20:
                if col_name in self.data.columns:
                    vol = self.data.iloc[self.current_step - 20 : self.current_step][
                        col_name
                    ].std() * np.sqrt(252)
                else:
                    vol = 0.15
            else:
                vol = 0.15
            factor_vols.append(vol)

        factor_vols = np.array(factor_vols)

        # Factor correlations (pairwise)
        factor_corrs = []
        if self.current_step >= self.window_size:
            for i in range(self.num_factors):
                for j in range(i + 1, self.num_factors):
                    col_i = f"{self.factor_names[i]}_ret"
                    col_j = f"{self.factor_names[j]}_ret"

                    if col_i in self.data.columns and col_j in self.data.columns:
                        ret_i = self.data.iloc[
                            self.current_step - self.window_size : self.current_step
                        ][col_i].values
                        ret_j = self.data.iloc[
                            self.current_step - self.window_size : self.current_step
                        ][col_j].values
                        corr = np.corrcoef(ret_i, ret_j)[0, 1]
                    else:
                        corr = np.random.normal(0, 0.3)

                    factor_corrs.append(corr)
        else:
            factor_corrs = [0.0] * int(self.num_factors * (self.num_factors - 1) / 2)

        factor_corrs = np.array(factor_corrs)

        # Portfolio metrics
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital
        cash_pct = self.cash / self.initial_capital

        # Portfolio volatility
        if self.current_step >= 20:
            portfolio_rets = []
            for step in range(max(0, self.current_step - 20), self.current_step):
                step_ret = 0
                for i, factor in enumerate(self.factor_names):
                    col_name = f"{factor}_ret"
                    if col_name in self.data.columns:
                        step_ret += (
                            self.factor_weights[i] * self.data.iloc[step][col_name]
                        )
                portfolio_rets.append(step_ret)

            portfolio_vol = (
                np.std(portfolio_rets) * np.sqrt(252)
                if len(portfolio_rets) > 0
                else 0.15
            )
        else:
            portfolio_vol = 0.15

        # Sharpe ratio
        if portfolio_vol > 0 and self.current_step >= 20:
            avg_ret = np.mean(portfolio_rets) if len(portfolio_rets) > 0 else 0
            sharpe = (avg_ret * 252) / portfolio_vol
        else:
            sharpe = 0

        # Turnover
        turnover_pct = self.turnover / max(self.initial_capital, 1)

        # Steps since rebalance
        steps_since_rebal = (
            self.current_step - self.last_rebalance
        ) / self.rebalance_frequency

        # Market features
        market_ret = (
            self.data.iloc[self.current_step]["market_ret"]
            if "market_ret" in self.data.columns
            else 0
        )

        if self.current_step >= 20:
            market_vol = (
                self.data.iloc[self.current_step - 20 : self.current_step][
                    "market_ret"
                ].std()
                * np.sqrt(252)
                if "market_ret" in self.data.columns
                else 0.20
            )
            market_trend = (
                self.data.iloc[self.current_step - 20 : self.current_step][
                    "market_ret"
                ].sum()
                if "market_ret" in self.data.columns
                else 0
            )
        else:
            market_vol = 0.20
            market_trend = 0

        # Combine all features
        observation = np.concatenate(
            [
                factor_returns,
                factor_momentum,
                factor_valuations,
                self.factor_weights,
                factor_vols,
                factor_corrs,
                np.array(
                    [
                        pnl_pct,
                        cash_pct,
                        portfolio_vol,
                        sharpe,
                        turnover_pct,
                        steps_since_rebal,
                        market_ret,
                        market_vol,
                        market_trend,
                    ]
                ),
            ]
        ).astype(np.float32)

        return observation

    def _take_action(self, action: int):
        """
        Execute factor allocation action.

        Args:
            action: Discrete action encoding factor exposures
        """
        # Only rebalance at specified frequency
        if (self.current_step - self.last_rebalance) < self.rebalance_frequency:
            # Just collect returns from current positions
            self._update_portfolio_value()
            return

        # Decode action to factor exposures
        new_exposures = self._decode_action(action)

        # Calculate target weights (equal risk weighting)
        # Scale exposures by target volatility / factor volatility
        target_weights = np.zeros(self.num_factors)

        for i in range(self.num_factors):
            col_name = f"{self.factor_names[i]}_ret"
            if self.current_step >= 20 and col_name in self.data.columns:
                factor_vol = self.data.iloc[self.current_step - 20 : self.current_step][
                    col_name
                ].std() * np.sqrt(252)
            else:
                factor_vol = 0.15

            if factor_vol > 0 and new_exposures[i] != 0:
                target_weights[i] = (
                    new_exposures[i]
                    * (self.target_volatility / factor_vol)
                    / self.num_factors
                )

        # Normalize weights
        total_weight = np.abs(target_weights).sum()
        if total_weight > 1.0:
            target_weights /= total_weight

        # Calculate turnover
        weight_change = np.abs(target_weights - self.factor_weights).sum()
        turnover_cost = weight_change * self.portfolio_value * self.transaction_cost

        if self.cash >= turnover_cost:
            # Update weights
            self.factor_weights = target_weights
            self.factor_exposures = new_exposures

            # Deduct transaction cost
            self.cash -= turnover_cost
            self.turnover += turnover_cost

            # Mark rebalance
            self.last_rebalance = self.current_step
            self.total_trades += 1

        # Update portfolio value
        self._update_portfolio_value()

    def _update_portfolio_value(self):
        """Update portfolio value based on factor returns."""
        # Get factor returns
        factor_returns = self._get_factor_returns()

        # Calculate portfolio return
        portfolio_ret = np.dot(self.factor_weights, factor_returns)

        # Update portfolio value
        pnl = self.portfolio_value * portfolio_ret
        self.unrealized_pnl = pnl
        self.portfolio_value = self.cash + pnl

    def _calculate_reward(self, action) -> float:
        """
        Calculate reward for factor tracking.

        Components:
        - Portfolio return
        - Risk-adjusted return (Sharpe)
        - Diversification bonus
        - Turnover penalty
        """
        # P&L change
        pnl = self.portfolio_value - self.previous_portfolio_value

        # Diversification bonus (reward using multiple factors)
        active_factors = np.count_nonzero(self.factor_weights)
        diversification_bonus = 0.1 * active_factors / self.num_factors

        # Concentration penalty
        if np.max(np.abs(self.factor_weights)) > 0.5:
            concentration_penalty = -0.2
        else:
            concentration_penalty = 0

        # Turnover penalty
        turnover_penalty = -self.turnover / 10000.0

        reward = pnl + diversification_bonus + concentration_penalty + turnover_penalty

        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.factor_weights = np.zeros(self.num_factors)
        self.factor_exposures = np.zeros(self.num_factors)
        self.last_rebalance = 0
        self.turnover = 0.0
        self.previous_portfolio_value = self.initial_capital

        return self._get_observation(), info


# Test the environment
if __name__ == "__main__":
    # Create synthetic data with factor returns
    np.random.seed(42)
    n_steps = 500

    # Simulate factor returns
    value_rets = np.random.normal(0.0003, 0.008, n_steps)
    momentum_rets = np.random.normal(0.0005, 0.012, n_steps)
    quality_rets = np.random.normal(0.0002, 0.006, n_steps)
    market_rets = np.random.normal(0.0004, 0.01, n_steps)

    data = pd.DataFrame(
        {
            "value_ret": value_rets,
            "momentum_ret": momentum_rets,
            "quality_ret": quality_rets,
            "market_ret": market_rets,
            "volume": np.random.randint(100000, 1000000, n_steps),
        }
    )

    env = FactorTrackingEnv(data, num_factors=3)

    print("Factor Tracking Environment Test")
    print("=" * 50)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n} discrete actions")
    print(f"Factors: {env.factor_names}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")

    # Test specific actions
    print("\nTesting specific factor combinations:")

    # Action examples:
    # 0 = All short: (-1, -1, -1)
    # 13 = All long: (1, 1, 1)
    # 4 = Neutral: (0, 0, 0)

    test_actions = [
        (13, "All Long"),
        (0, "All Short"),
        (4, "All Neutral"),
        (12, "Value+Mom Short, Quality Long"),
    ]

    for action, description in test_actions:
        obs, info = env.reset()
        exposures = env._decode_action(action)

        # Run for several steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n{description} (Action {action}):")
        print(f"  Exposures: {exposures}")
        print(f"  Weights: {env.factor_weights}")
        print(f"  Portfolio value: ${env.portfolio_value:,.2f}")
        print(f"  P&L: ${env.portfolio_value - env.initial_capital:,.2f}")
