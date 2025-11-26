"""
Benchmark strategies for comparison with DRL agents.
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy.optimize import minimize


class EqualWeightBenchmark:
    """Benchmark 1: Static equal-weight allocation to all specialists."""

    def __init__(self, num_specialists: int = 7):
        self.num_specialists = num_specialists
        self.weights = np.ones(num_specialists) / num_specialists

    def get_allocations(self, step: int = 0) -> np.ndarray:
        """Return equal weights at every step."""
        return self.weights.copy()

    def backtest(
        self, specialist_returns: pd.DataFrame, initial_capital: float = 1000000
    ) -> pd.Series:
        """
        Backtest equal-weight strategy.

        Args:
            specialist_returns: DataFrame with returns for each specialist
            initial_capital: Starting capital

        Returns:
            Equity curve
        """
        # Equal weight allocation
        portfolio_returns = specialist_returns.mean(axis=1)
        equity_curve = initial_capital * (1 + portfolio_returns).cumprod()
        
        # Prepend initial capital
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            initial_index = equity_curve.index[0] - pd.Timedelta(days=1)
        else:
            initial_index = equity_curve.index[0] - 1
            
        equity_curve = pd.concat(
            [
                pd.Series(
                    [initial_capital],
                    index=[initial_index],
                ),
                equity_curve,
            ]
        )

        return equity_curve


class MeanVarianceBenchmark:
    """Benchmark 2: Mean-Variance optimization with quarterly rebalancing."""

    def __init__(
        self,
        lookback_days: int = 252,
        rebalance_freq: int = 63,
        risk_free_rate: float = 0.02,
    ):
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.risk_free_rate = risk_free_rate
        self.weights_history = []

    def optimize_portfolio(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance optimization.

        Args:
            returns_df: Historical returns for lookback period

        Returns:
            Optimal weights
        """
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        num_assets = len(mean_returns)

        # Objective: minimize portfolio variance
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        # Bounds: no short selling, no leverage
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess: equal weight
        initial_weights = np.ones(num_assets) / num_assets

        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return result.x
        else:
            # Fallback to equal weight if optimization fails
            return initial_weights

    def backtest(
        self, specialist_returns: pd.DataFrame, initial_capital: float = 1000000
    ) -> pd.Series:
        """
        Backtest mean-variance strategy with quarterly rebalancing.

        Args:
            specialist_returns: DataFrame with returns for each specialist
            initial_capital: Starting capital

        Returns:
            Equity curve
        """
        # Vectorized approach - calculate all rebalancing points upfront
        n_periods = len(specialist_returns)
        equity_values = np.zeros(n_periods + 1)
        equity_values[0] = initial_capital
        
        # Initialize weights
        current_weights = np.ones(specialist_returns.shape[1]) / specialist_returns.shape[1]
        
        # Pre-calculate rebalancing indices
        rebalance_indices = [i for i in range(self.lookback_days, n_periods, self.rebalance_freq)]
        
        # Convert returns to numpy for faster access
        returns_array = specialist_returns.values
        
        # Track next rebalance index
        rebalance_idx = 0
        
        for i in range(n_periods):
            # Check if we need to rebalance
            if rebalance_idx < len(rebalance_indices) and i == rebalance_indices[rebalance_idx]:
                lookback_returns = specialist_returns.iloc[i - self.lookback_days : i]
                current_weights = self.optimize_portfolio(lookback_returns)
                self.weights_history.append(current_weights)
                rebalance_idx += 1
            
            # Calculate portfolio return using vectorized dot product
            portfolio_return = np.dot(current_weights, returns_array[i])
            equity_values[i + 1] = equity_values[i] * (1 + portfolio_return)
        
        # Convert to Series (skip initial capital)
        equity_series = pd.Series(equity_values[1:], index=specialist_returns.index)
        
        # Prepend initial capital
        if isinstance(equity_series.index, pd.DatetimeIndex):
            initial_index = equity_series.index[0] - pd.Timedelta(days=1)
        else:
            initial_index = equity_series.index[0] - 1
            
        equity_series = pd.concat(
            [
                pd.Series(
                    [initial_capital],
                    index=[initial_index],
                ),
                equity_series,
            ]
        )

        return equity_series


class RiskParityBenchmark:
    """Benchmark 2 Alternative: Risk Parity allocation."""

    def __init__(self, lookback_days: int = 252, rebalance_freq: int = 63):
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.weights_history = []

    def risk_parity_weights(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk parity weights (inverse volatility weighting).

        Args:
            returns_df: Historical returns for lookback period

        Returns:
            Risk parity weights
        """
        volatilities = returns_df.std()

        # Avoid division by zero
        volatilities = volatilities.replace(0, 1e-6)

        # Inverse volatility weights
        inv_vol_weights = 1 / volatilities
        weights = inv_vol_weights / inv_vol_weights.sum()

        return weights.values

    def backtest(
        self, specialist_returns: pd.DataFrame, initial_capital: float = 1000000
    ) -> pd.Series:
        """
        Backtest risk parity strategy with quarterly rebalancing.

        Args:
            specialist_returns: DataFrame with returns for each specialist
            initial_capital: Starting capital

        Returns:
            Equity curve
        """
        equity_curve = [initial_capital]
        current_weights = (
            np.ones(specialist_returns.shape[1]) / specialist_returns.shape[1]
        )

        for i in range(len(specialist_returns)):
            # Rebalance quarterly
            if i > self.lookback_days and i % self.rebalance_freq == 0:
                lookback_returns = specialist_returns.iloc[i - self.lookback_days : i]
                current_weights = self.risk_parity_weights(lookback_returns)
                self.weights_history.append(current_weights)

            # Calculate portfolio return
            period_returns = specialist_returns.iloc[i].values
            portfolio_return = np.dot(current_weights, period_returns)

            # Update equity
            new_equity = equity_curve[-1] * (1 + portfolio_return)
            equity_curve.append(new_equity)

        # Convert to Series
        equity_series = pd.Series(equity_curve[1:], index=specialist_returns.index)
        
        # Prepend initial capital
        if isinstance(equity_series.index, pd.DatetimeIndex):
            initial_index = equity_series.index[0] - pd.Timedelta(days=1)
        else:
            initial_index = equity_series.index[0] - 1
            
        equity_series = pd.concat(
            [
                pd.Series(
                    [initial_capital],
                    index=[initial_index],
                ),
                equity_series,
            ]
        )

        return equity_series


class FullEnsembleBenchmark:
    """Benchmark 3: Run all specialists independently with full capital and sum P&L."""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital

    def backtest(
        self, specialist_returns: pd.DataFrame, initial_capital: float = 1000000
    ) -> pd.Series:
        """
        Backtest ensemble by running all specialists with full capital.

        Args:
            specialist_returns: DataFrame with returns for each specialist
            initial_capital: Starting capital for EACH specialist

        Returns:
            Combined equity curve (sum of all specialists' P&L)
        """
        # Each specialist gets full capital
        specialist_equities = {}

        for col in specialist_returns.columns:
            specialist_equity = (
                initial_capital * (1 + specialist_returns[col]).cumprod()
            )
            specialist_equities[col] = specialist_equity

        # Sum the P&L from all specialists
        total_pnl = sum(
            (equity - initial_capital) for equity in specialist_equities.values()
        )

        # Combined equity = initial capital (from first specialist) + total P&L
        combined_equity = initial_capital + total_pnl

        # Prepend initial value
        if isinstance(combined_equity.index, pd.DatetimeIndex):
            initial_index = combined_equity.index[0] - pd.Timedelta(days=1)
        else:
            initial_index = combined_equity.index[0] - 1
            
        combined_equity = pd.concat(
            [
                pd.Series(
                    [initial_capital],
                    index=[initial_index],
                ),
                combined_equity,
            ]
        )

        return combined_equity


def run_all_benchmarks(
    specialist_returns: pd.DataFrame, initial_capital: float = 1000000
) -> Dict[str, pd.Series]:
    """
    Run all benchmark strategies.

    Args:
        specialist_returns: DataFrame with returns for each specialist
        initial_capital: Starting capital

    Returns:
        Dictionary mapping benchmark names to equity curves
    """
    benchmarks = {}

    # Benchmark 1: Equal Weight
    print("Running Benchmark 1: Equal Weight (1/N)...")
    eq_weight = EqualWeightBenchmark(num_specialists=specialist_returns.shape[1])
    benchmarks["Equal_Weight_1/N"] = eq_weight.backtest(
        specialist_returns, initial_capital
    )

    # Benchmark 2a: Mean-Variance
    print("Running Benchmark 2a: Mean-Variance Optimization...")
    mv_opt = MeanVarianceBenchmark()
    benchmarks["Mean_Variance_Opt"] = mv_opt.backtest(
        specialist_returns, initial_capital
    )

    # Benchmark 2b: Risk Parity
    print("Running Benchmark 2b: Risk Parity...")
    rp = RiskParityBenchmark()
    benchmarks["Risk_Parity"] = rp.backtest(specialist_returns, initial_capital)

    # Benchmark 3: Ensemble (sum of all P&L)
    print("Running Benchmark 3: Ensemble (Full Capital to All)...")
    ensemble = FullEnsembleBenchmark()
    benchmarks["Ensemble_Full_Capital"] = ensemble.backtest(
        specialist_returns, initial_capital
    )

    print(f"\n✅ All {len(benchmarks)} benchmarks computed")

    return benchmarks
