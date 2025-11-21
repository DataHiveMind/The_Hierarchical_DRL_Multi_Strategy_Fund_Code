"""
CIO Allocator Environment - Master Agent for Capital Allocation

This Gymnasium environment implements the master agent in the hierarchical
DRL framework. The CIO allocator dynamically allocates capital across 7
specialist trading strategies based on:
- Individual strategy performance metrics
- Market regime indicators
- Portfolio-level risk constraints
- Correlation structure between strategies
- Current portfolio composition
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import deque


class CIOAllocatorEnv(gym.Env):
    """
    Gymnasium environment for the master CIO allocator agent.

    The agent receives observations about specialist strategies and market
    conditions, then outputs capital allocation weights across strategies.

    Observation Space:
        - Per-strategy metrics (7 strategies × N metrics)
        - Market regime indicators
        - Portfolio metrics
        - Correlation matrix features

    Action Space:
        - Continuous allocation weights [0, 1] for each of 7 strategies
        - Weights automatically normalized to sum to 1

    Reward:
        - Portfolio-level risk-adjusted returns (Sharpe ratio)
        - Penalties for excessive turnover
        - Bonuses for diversification
    """

    metadata = {"render_modes": ["human"]}

    # Strategy names
    STRATEGIES = [
        "statistical_arbitrage",
        "market_making",
        "volatility_trading",
        "delta_hedging",
        "futures_spreads",
        "factor_tracking",
        "fx_arbitrage",
    ]

    def __init__(
        self,
        specialist_data: pd.DataFrame,
        market_data: pd.DataFrame,
        initial_capital: float = 10_000_000,
        rebalance_frequency: int = 5,  # days
        lookback_window: int = 60,  # days
        transaction_cost: float = 0.001,
        max_single_allocation: float = 0.40,
        min_single_allocation: float = 0.05,
        max_turnover: float = 0.50,
        target_volatility: float = 0.12,
        risk_free_rate: float = 0.02,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize CIO Allocator Environment.

        Args:
            specialist_data: DataFrame with specialist strategy returns/metrics
            market_data: DataFrame with market indicators
            initial_capital: Starting portfolio value
            rebalance_frequency: Days between rebalancing
            lookback_window: Days of historical data for observation
            transaction_cost: Cost per dollar of turnover
            max_single_allocation: Maximum weight to any strategy
            min_single_allocation: Minimum weight to any strategy
            max_turnover: Maximum portfolio turnover per rebalance
            target_volatility: Target portfolio volatility (annualized)
            risk_free_rate: Risk-free rate for Sharpe calculation
            render_mode: Rendering mode
        """
        super().__init__()

        self.specialist_data = specialist_data
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_single_allocation = max_single_allocation
        self.min_single_allocation = min_single_allocation
        self.max_turnover = max_turnover
        self.target_volatility = target_volatility
        self.risk_free_rate = risk_free_rate
        self.render_mode = render_mode

        # Validate data
        self._validate_data()

        # Number of strategies
        self.n_strategies = len(self.STRATEGIES)

        # Define observation space
        # Per-strategy features: returns, volatility, Sharpe, drawdown, etc.
        self.n_strategy_features = 12
        # Market features: VIX, regime, trend indicators, etc.
        self.n_market_features = 15
        # Portfolio features: current weights, total value, etc.
        self.n_portfolio_features = 10
        # Correlation features: flattened upper triangle of correlation matrix
        self.n_correlation_features = (self.n_strategies * (self.n_strategies - 1)) // 2

        self.observation_dim = (
            self.n_strategies * self.n_strategy_features
            + self.n_market_features
            + self.n_portfolio_features
            + self.n_correlation_features
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # Define action space: allocation weights for each strategy
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_strategies,), dtype=np.float32
        )

        # Initialize state
        self.current_step = 0
        self.max_steps = len(self.specialist_data) - self.lookback_window
        self.portfolio_value = initial_capital
        self.current_weights = np.ones(self.n_strategies) / self.n_strategies

        # Performance tracking
        self.portfolio_history = []
        self.weights_history = []
        self.returns_history = []
        self.turnover_history = []

        # Logger
        self.logger = logging.getLogger(__name__)

    def _validate_data(self):
        """Validate input data."""
        required_cols = [f"{s}_return" for s in self.STRATEGIES]
        missing_cols = set(required_cols) - set(self.specialist_data.columns)
        if missing_cols:
            raise ValueError(f"Missing specialist data columns: {missing_cols}")

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.

        Returns:
            Observation array with strategy metrics, market indicators,
            portfolio state, and correlation features.
        """
        obs_components = []

        # Get current time window
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window

        # ============ Per-Strategy Features ============
        for strategy in self.STRATEGIES:
            returns_col = f"{strategy}_return"
            returns = self.specialist_data[returns_col].iloc[start_idx:end_idx].values

            # Calculate strategy metrics
            mean_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe = (mean_return * 252 - self.risk_free_rate) / (volatility + 1e-8)

            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Recent performance
            recent_1w_return = np.mean(returns[-5:])
            recent_1m_return = np.mean(returns[-20:])

            # Momentum and trend
            momentum = returns[-1] - returns[-20] if len(returns) >= 20 else 0

            # Skewness and kurtosis
            skewness = pd.Series(returns).skew()

            # Win rate
            win_rate = np.sum(returns > 0) / len(returns)

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = (
                np.std(downside_returns) * np.sqrt(252)
                if len(downside_returns) > 0
                else 1e-8
            )
            sortino = (mean_return * 252 - self.risk_free_rate) / (downside_std + 1e-8)

            # Calmar ratio
            calmar = (mean_return * 252) / (abs(max_drawdown) + 1e-8)

            # Rolling volatility change
            vol_recent = np.std(returns[-20:]) if len(returns) >= 20 else volatility
            vol_change = (vol_recent - volatility) / (volatility + 1e-8)

            strategy_features = [
                mean_return,
                volatility,
                sharpe,
                max_drawdown,
                recent_1w_return,
                recent_1m_return,
                momentum,
                skewness,
                win_rate,
                sortino,
                calmar,
                vol_change,
            ]

            obs_components.extend(strategy_features)

        # ============ Market Features ============
        market_window = self.market_data.iloc[start_idx:end_idx]

        # Market returns and volatility
        if "market_return" in market_window.columns:
            market_returns = market_window["market_return"].values
            market_return = np.mean(market_returns)
            market_vol = np.std(market_returns) * np.sqrt(252)
        else:
            market_return = 0
            market_vol = 0.15

        # VIX or volatility proxy
        vix_level = (
            market_window["vix"].iloc[-1] if "vix" in market_window.columns else 20.0
        )
        vix_change = (
            (vix_level - market_window["vix"].iloc[0])
            if "vix" in market_window.columns
            else 0
        )

        # Regime indicators
        regime_bull = 1.0 if market_return > 0.001 else 0.0
        regime_high_vol = 1.0 if market_vol > 0.20 else 0.0

        # Trend indicators
        sma_20 = (
            market_window["close"].rolling(20).mean().iloc[-1]
            if "close" in market_window.columns
            else 100
        )
        sma_60 = (
            market_window["close"].rolling(60).mean().iloc[-1]
            if "close" in market_window.columns
            else 100
        )
        trend_signal = 1.0 if sma_20 > sma_60 else -1.0

        # Credit spreads (if available)
        credit_spread = (
            market_window["credit_spread"].iloc[-1]
            if "credit_spread" in market_window.columns
            else 0.02
        )

        # Correlation to market
        portfolio_returns = self._calculate_portfolio_returns(start_idx, end_idx)
        if len(portfolio_returns) > 0 and "market_return" in market_window.columns:
            correlation_to_market = np.corrcoef(portfolio_returns, market_returns)[0, 1]
        else:
            correlation_to_market = 0.0

        # Economic indicators
        gdp_growth = (
            market_window["gdp_growth"].iloc[-1]
            if "gdp_growth" in market_window.columns
            else 0.02
        )
        inflation = (
            market_window["inflation"].iloc[-1]
            if "inflation" in market_window.columns
            else 0.02
        )
        unemployment = (
            market_window["unemployment"].iloc[-1]
            if "unemployment" in market_window.columns
            else 0.04
        )

        # Liquidity indicators
        liquidity_score = (
            market_window["liquidity"].iloc[-1]
            if "liquidity" in market_window.columns
            else 0.5
        )

        # Time features (cyclical encoding)
        month_sin = np.sin(2 * np.pi * end_idx / 252)
        month_cos = np.cos(2 * np.pi * end_idx / 252)

        market_features = [
            market_return,
            market_vol,
            vix_level / 100,  # Normalize
            vix_change / 100,
            regime_bull,
            regime_high_vol,
            trend_signal,
            credit_spread,
            correlation_to_market,
            gdp_growth,
            inflation,
            unemployment,
            liquidity_score,
            month_sin,
            month_cos,
        ]

        obs_components.extend(market_features)

        # ============ Portfolio Features ============
        # Current allocation weights
        weight_entropy = -np.sum(
            self.current_weights * np.log(self.current_weights + 1e-8)
        )
        weight_concentration = np.max(self.current_weights)
        weight_std = np.std(self.current_weights)

        # Portfolio value metrics
        portfolio_return = (
            self.portfolio_value - self.initial_capital
        ) / self.initial_capital

        # Historical portfolio volatility
        if len(self.returns_history) > 20:
            portfolio_vol = np.std(self.returns_history[-60:]) * np.sqrt(252)
            portfolio_sharpe = (
                np.mean(self.returns_history[-60:]) * 252 - self.risk_free_rate
            ) / (portfolio_vol + 1e-8)
        else:
            portfolio_vol = 0.10
            portfolio_sharpe = 0.0

        # Drawdown
        if len(self.portfolio_history) > 1:
            portfolio_values = np.array(self.portfolio_history)
            running_max = np.maximum.accumulate(portfolio_values)
            current_drawdown = (self.portfolio_value - running_max[-1]) / running_max[
                -1
            ]
        else:
            current_drawdown = 0.0

        # Recent turnover
        recent_turnover = (
            np.mean(self.turnover_history[-5:])
            if len(self.turnover_history) > 0
            else 0.0
        )

        # Days since inception
        days_elapsed = self.current_step / 252  # Years

        portfolio_features = [
            self.current_weights[0],  # Weight to strategy 1
            self.current_weights[1],  # Weight to strategy 2
            weight_entropy,
            weight_concentration,
            weight_std,
            portfolio_return,
            portfolio_vol,
            portfolio_sharpe,
            current_drawdown,
            recent_turnover,
        ]

        obs_components.extend(portfolio_features)

        # ============ Correlation Features ============
        # Calculate correlation matrix between strategies
        strategy_returns = []
        for strategy in self.STRATEGIES:
            returns_col = f"{strategy}_return"
            strategy_returns.append(
                self.specialist_data[returns_col].iloc[start_idx:end_idx].values
            )

        strategy_returns = np.array(strategy_returns)
        corr_matrix = np.corrcoef(strategy_returns)

        # Extract upper triangle (excluding diagonal)
        corr_features = corr_matrix[np.triu_indices(self.n_strategies, k=1)]

        obs_components.extend(corr_features)

        # Combine all features
        observation = np.array(obs_components, dtype=np.float32)

        # Handle NaN/Inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _calculate_portfolio_returns(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Calculate portfolio returns for a given window."""
        returns = []
        for i in range(start_idx, min(end_idx, len(self.specialist_data))):
            strategy_returns = []
            for strategy in self.STRATEGIES:
                returns_col = f"{strategy}_return"
                strategy_returns.append(self.specialist_data[returns_col].iloc[i])

            # Weighted return
            portfolio_return = np.dot(self.current_weights, strategy_returns)
            returns.append(portfolio_return)

        return np.array(returns)

    def _calculate_reward(
        self, new_weights: np.ndarray, portfolio_return: float, turnover: float
    ) -> float:
        """
        Calculate reward for the allocation decision.

        Reward components:
        1. Risk-adjusted portfolio return (Sharpe-based)
        2. Turnover penalty
        3. Diversification bonus
        4. Constraint violation penalties

        Args:
            new_weights: New allocation weights
            portfolio_return: Portfolio return for this period
            turnover: Portfolio turnover

        Returns:
            Scalar reward
        """
        # Base reward: portfolio return
        reward = portfolio_return * 100  # Scale up

        # Risk adjustment
        if len(self.returns_history) > 20:
            portfolio_vol = np.std(self.returns_history[-60:]) * np.sqrt(252)
            sharpe = (
                np.mean(self.returns_history[-60:]) * 252 - self.risk_free_rate
            ) / (portfolio_vol + 1e-8)
            reward += sharpe * 0.1  # Sharpe bonus

        # Turnover penalty
        turnover_cost = turnover * self.transaction_cost * self.portfolio_value
        reward -= turnover_cost / self.initial_capital * 100

        # Diversification bonus (entropy-based)
        entropy = -np.sum(new_weights * np.log(new_weights + 1e-8))
        max_entropy = np.log(self.n_strategies)
        diversification_score = entropy / max_entropy
        reward += diversification_score * 0.05

        # Constraint penalties
        if np.any(new_weights > self.max_single_allocation):
            reward -= 1.0  # Penalty for exceeding max allocation

        if np.any(new_weights < self.min_single_allocation):
            reward -= 0.5  # Penalty for too-small allocations

        if turnover > self.max_turnover:
            reward -= 2.0  # Large penalty for excessive turnover

        # Volatility targeting bonus/penalty
        if len(self.returns_history) > 60:
            portfolio_vol = np.std(self.returns_history[-60:]) * np.sqrt(252)
            vol_diff = abs(portfolio_vol - self.target_volatility)
            reward -= vol_diff * 2.0  # Penalty for deviation from target vol

        return reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Raw allocation weights (will be normalized)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0, 1)
        new_weights = action / (np.sum(action) + 1e-8)

        # Apply allocation constraints
        new_weights = np.clip(
            new_weights, self.min_single_allocation, self.max_single_allocation
        )
        new_weights = new_weights / np.sum(new_weights)  # Re-normalize

        # Calculate turnover
        turnover = np.sum(np.abs(new_weights - self.current_weights))

        # Get specialist returns for this period
        current_idx = self.current_step + self.lookback_window
        strategy_returns = []
        for strategy in self.STRATEGIES:
            returns_col = f"{strategy}_return"
            strategy_returns.append(self.specialist_data[returns_col].iloc[current_idx])

        strategy_returns = np.array(strategy_returns)

        # Calculate portfolio return
        portfolio_return = np.dot(new_weights, strategy_returns)

        # Update portfolio value
        self.portfolio_value *= 1 + portfolio_return

        # Apply transaction costs
        transaction_costs = turnover * self.transaction_cost * self.portfolio_value
        self.portfolio_value -= transaction_costs

        # Update weights
        self.current_weights = new_weights

        # Record history
        self.portfolio_history.append(self.portfolio_value)
        self.weights_history.append(new_weights.copy())
        self.returns_history.append(portfolio_return)
        self.turnover_history.append(turnover)

        # Calculate reward
        reward = self._calculate_reward(new_weights, portfolio_return, turnover)

        # Advance time
        self.current_step += self.rebalance_frequency

        # Check if episode is done
        terminated = self.current_step >= self.max_steps - self.lookback_window
        truncated = False

        # Get next observation
        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_dim, dtype=np.float32)

        # Info dict
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "weights": new_weights,
            "sharpe_ratio": self._calculate_sharpe(),
            "max_drawdown": self._calculate_max_drawdown(),
            "transaction_costs": transaction_costs,
        }

        return observation, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_strategies) / self.n_strategies

        self.portfolio_history = [self.initial_capital]
        self.weights_history = [self.current_weights.copy()]
        self.returns_history = []
        self.turnover_history = []

        observation = self._get_observation()

        info = {
            "portfolio_value": self.portfolio_value,
            "initial_weights": self.current_weights,
        }

        return observation, info

    def _calculate_sharpe(self) -> float:
        """Calculate portfolio Sharpe ratio."""
        if len(self.returns_history) < 20:
            return 0.0

        returns = np.array(self.returns_history)
        mean_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (mean_return - self.risk_free_rate) / (volatility + 1e-8)

        return sharpe

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.portfolio_history) < 2:
            return 0.0

        portfolio_values = np.array(self.portfolio_history)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        return max_drawdown

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            print(f"\n{'=' * 60}")
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(
                f"Return: {(self.portfolio_value / self.initial_capital - 1) * 100:.2f}%"
            )
            print(f"Sharpe Ratio: {self._calculate_sharpe():.2f}")
            print(f"Max Drawdown: {self._calculate_max_drawdown() * 100:.2f}%")
            print(f"\nCurrent Allocations:")
            for i, strategy in enumerate(self.STRATEGIES):
                print(f"  {strategy:25s}: {self.current_weights[i] * 100:5.2f}%")
            print(f"{'=' * 60}\n")

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics.

        Returns:
            Dictionary of performance metrics
        """
        if len(self.returns_history) < 20:
            return {}

        returns = np.array(self.returns_history)

        # Annualized metrics
        total_return = (self.portfolio_value / self.initial_capital) - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_vol = np.std(returns) * np.sqrt(252)

        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / (
            annualized_vol + 1e-8
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = (
            np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0
            else 1e-8
        )
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_vol

        # Drawdown metrics
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-8)

        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)

        # Profit factor
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profits / (gross_losses + 1e-8)

        # Average turnover
        avg_turnover = np.mean(self.turnover_history) if self.turnover_history else 0

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_turnover": avg_turnover,
            "final_value": self.portfolio_value,
        }

        return metrics


def main():
    """Example usage of CIO Allocator Environment."""
    # Create dummy data for demonstration
    np.random.seed(42)
    n_days = 500

    # Generate synthetic specialist returns
    strategies = CIOAllocatorEnv.STRATEGIES
    specialist_data = pd.DataFrame()

    for strategy in strategies:
        # Generate correlated returns with different characteristics
        base_return = np.random.randn(n_days) * 0.01
        specialist_data[f"{strategy}_return"] = base_return

    # Generate market data
    market_data = pd.DataFrame(
        {
            "market_return": np.random.randn(n_days) * 0.012,
            "vix": 15 + np.random.randn(n_days) * 3,
            "close": 100 * (1 + np.random.randn(n_days) * 0.01).cumprod(),
            "credit_spread": 0.02 + np.random.randn(n_days) * 0.002,
            "gdp_growth": np.full(n_days, 0.02),
            "inflation": np.full(n_days, 0.02),
            "unemployment": np.full(n_days, 0.04),
            "liquidity": np.full(n_days, 0.5),
        }
    )

    # Create environment
    env = CIOAllocatorEnv(
        specialist_data=specialist_data,
        market_data=market_data,
        initial_capital=10_000_000,
        render_mode="human",
    )

    # Run random episode
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dimension: {env.observation_dim}")

    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()  # Random allocation
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            break

    # Print final metrics
    metrics = env.get_portfolio_metrics()
    print("\nFinal Portfolio Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
