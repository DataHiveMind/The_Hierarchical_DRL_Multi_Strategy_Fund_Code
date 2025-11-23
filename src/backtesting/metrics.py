"""
Performance metrics for backtesting trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class PerformanceMetrics:
    """Calculate comprehensive performance metrics for trading strategies."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate returns from equity curve.

        Args:
            equity_curve: Time series of portfolio values

        Returns:
            Time series of returns
        """
        return equity_curve.pct_change().fillna(0)

    def sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year (252 for daily)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

    def sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation).

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

    def max_drawdown(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            equity_curve: Time series of portfolio values

        Returns:
            Dictionary with max_drawdown, max_drawdown_duration, current_drawdown
        """
        if len(equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "current_drawdown": 0.0,
            }

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Maximum drawdown
        max_dd = drawdown.min()

        # Current drawdown
        current_dd = drawdown.iloc[-1]

        # Maximum drawdown duration
        is_in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for in_dd in is_in_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0

        return {
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "current_drawdown": current_dd,
        }

    def calmar_ratio(
        self, returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Series of returns
            equity_curve: Time series of portfolio values
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        max_dd = abs(self.max_drawdown(equity_curve)["max_drawdown"])

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    def value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            VaR value
        """
        if len(returns) < 2:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Series of returns
            confidence: Confidence level

        Returns:
            CVaR value
        """
        if len(returns) < 2:
            return 0.0

        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns).

        Args:
            returns: Series of returns

        Returns:
            Win rate as a percentage
        """
        if len(returns) == 0:
            return 0.0

        return (returns > 0).sum() / len(returns)

    def profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Series of returns

        Returns:
            Profit factor
        """
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return np.inf if wins > 0 else 0.0

        return wins / losses

    def calculate_all_metrics(
        self, equity_curve: pd.Series, periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Time series of portfolio values
            periods_per_year: Number of periods in a year

        Returns:
            Dictionary of all metrics
        """
        returns = self.calculate_returns(equity_curve)

        # Basic statistics
        total_return = (
            (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
            if len(equity_curve) > 0
            else 0.0
        )
        annual_return = returns.mean() * periods_per_year
        annual_volatility = returns.std() * np.sqrt(periods_per_year)

        # Drawdown metrics
        dd_metrics = self.max_drawdown(equity_curve)

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": self.sharpe_ratio(returns, periods_per_year),
            "sortino_ratio": self.sortino_ratio(returns, periods_per_year),
            "calmar_ratio": self.calmar_ratio(returns, equity_curve, periods_per_year),
            "max_drawdown": dd_metrics["max_drawdown"],
            "max_drawdown_duration": dd_metrics["max_drawdown_duration"],
            "current_drawdown": dd_metrics["current_drawdown"],
            "var_95": self.value_at_risk(returns, 0.95),
            "cvar_95": self.conditional_var(returns, 0.95),
            "win_rate": self.win_rate(returns),
            "profit_factor": self.profit_factor(returns),
            "num_periods": len(equity_curve),
        }

        return metrics


class StrategyComparison:
    """Compare multiple strategies."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize strategy comparison.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.metrics_calc = PerformanceMetrics(risk_free_rate)

    def compare_strategies(
        self, strategies: Dict[str, pd.Series], periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Args:
            strategies: Dictionary mapping strategy names to equity curves
            periods_per_year: Number of periods in a year

        Returns:
            DataFrame with metrics for each strategy
        """
        comparison = {}

        for name, equity_curve in strategies.items():
            metrics = self.metrics_calc.calculate_all_metrics(
                equity_curve, periods_per_year
            )
            comparison[name] = metrics

        df = pd.DataFrame(comparison).T
        return df

    def rank_strategies(
        self,
        strategies: Dict[str, pd.Series],
        metric: str = "sharpe_ratio",
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        Rank strategies by a specific metric.

        Args:
            strategies: Dictionary mapping strategy names to equity curves
            metric: Metric to rank by
            periods_per_year: Number of periods in a year

        Returns:
            DataFrame sorted by the specified metric
        """
        comparison = self.compare_strategies(strategies, periods_per_year)
        return comparison.sort_values(metric, ascending=False)


class RollingMetrics:
    """Calculate rolling performance metrics."""

    def __init__(self, window: int = 252, risk_free_rate: float = 0.02):
        """
        Initialize rolling metrics calculator.

        Args:
            window: Rolling window size
            risk_free_rate: Annual risk-free rate
        """
        self.window = window
        self.metrics_calc = PerformanceMetrics(risk_free_rate)

    def rolling_sharpe(
        self, returns: pd.Series, periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year

        Returns:
            Series of rolling Sharpe ratios
        """
        excess_returns = returns - self.metrics_calc.risk_free_rate / periods_per_year
        rolling_mean = excess_returns.rolling(window=self.window).mean()
        rolling_std = returns.rolling(window=self.window).std()

        return np.sqrt(periods_per_year) * rolling_mean / rolling_std

    def rolling_volatility(
        self, returns: pd.Series, periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year

        Returns:
            Series of rolling annualized volatility
        """
        return returns.rolling(window=self.window).std() * np.sqrt(periods_per_year)

    def rolling_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate rolling drawdown.

        Args:
            equity_curve: Time series of portfolio values

        Returns:
            Series of rolling drawdowns
        """
        rolling_max = equity_curve.rolling(window=self.window, min_periods=1).max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown
