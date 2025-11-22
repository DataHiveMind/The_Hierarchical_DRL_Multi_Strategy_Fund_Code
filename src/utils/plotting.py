"""
Plotting Utilities for Hierarchical DRL Multi-Strategy Fund

This module provides comprehensive visualization tools for:
- Training progress (rewards, losses, metrics)
- Portfolio performance (equity curves, drawdowns)
- Strategy comparison and allocation
- Risk metrics and attribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class PerformancePlotter:
    """Visualization tools for trading strategy performance."""

    def __init__(self, save_dir: str = "reports/figures"):
        """
        Initialize plotter.

        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            "profit": "#2ecc71",
            "loss": "#e74c3c",
            "neutral": "#95a5a6",
            "primary": "#3498db",
            "secondary": "#9b59b6",
            "warning": "#f39c12",
        }

    def plot_training_progress(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Progress",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot training metrics over time.

        Args:
            metrics: Dictionary of metric names to values
            title: Plot title
            save_name: Filename to save (without extension)
            show: Whether to display plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]

            # Plot raw values
            ax.plot(values, alpha=0.3, color=self.colors["primary"], linewidth=0.5)

            # Plot smoothed values
            if len(values) > 10:
                window = min(50, len(values) // 10)
                smoothed = pd.Series(values).rolling(window=window, center=True).mean()
                ax.plot(
                    smoothed,
                    color=self.colors["primary"],
                    linewidth=2,
                    label="Smoothed",
                )

            ax.set_title(
                f"{metric_name.replace('_', ' ').title()}",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Episode / Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_equity_curve(
        self,
        portfolio_values: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        benchmark: Optional[np.ndarray] = None,
        title: str = "Portfolio Equity Curve",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot portfolio equity curve with drawdown.

        Args:
            portfolio_values: Array of portfolio values
            dates: Optional datetime index
            benchmark: Optional benchmark values
            title: Plot title
            save_name: Filename to save
            show: Whether to display
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Use dates if provided, otherwise use index
        x_axis = dates if dates is not None else np.arange(len(portfolio_values))

        # Equity curve
        returns = (portfolio_values / portfolio_values[0] - 1) * 100
        ax1.plot(
            x_axis, returns, color=self.colors["primary"], linewidth=2, label="Strategy"
        )

        if benchmark is not None:
            bench_returns = (benchmark / benchmark[0] - 1) * 100
            ax1.plot(
                x_axis,
                bench_returns,
                color=self.colors["secondary"],
                linewidth=2,
                alpha=0.7,
                label="Benchmark",
            )

        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.set_ylabel("Returns (%)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100

        ax2.fill_between(x_axis, drawdown, 0, color=self.colors["loss"], alpha=0.3)
        ax2.plot(x_axis, drawdown, color=self.colors["loss"], linewidth=1.5)
        ax2.set_xlabel("Time" if dates is not None else "Steps", fontsize=12)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        title: str = "Returns Distribution",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot returns distribution with statistics.

        Args:
            returns: Array of returns
            title: Plot title
            save_name: Filename to save
            show: Whether to display
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram with KDE
        ax1 = axes[0]
        ax1.hist(
            returns,
            bins=50,
            density=True,
            alpha=0.6,
            color=self.colors["primary"],
            edgecolor="black",
        )

        # Add normal distribution overlay
        mu, std = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax1.plot(
            x,
            1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std) ** 2),
            "r-",
            linewidth=2,
            label="Normal Distribution",
        )

        ax1.axvline(
            mu, color="green", linestyle="--", linewidth=2, label=f"Mean: {mu:.4f}"
        )
        ax1.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax1.set_title("Returns Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Returns")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        ax2 = axes[1]
        from scipy import stats

        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normality Test)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        skew = pd.Series(returns).skew()
        kurt = pd.Series(returns).kurtosis()
        stats_text = f"Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}"
        ax2.text(
            0.05,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_strategy_comparison(
        self,
        strategy_returns: Dict[str, np.ndarray],
        title: str = "Strategy Comparison",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Compare multiple strategies.

        Args:
            strategy_returns: Dict of strategy names to return arrays
            title: Plot title
            save_name: Filename to save
            show: Whether to display
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Cumulative returns
        ax1 = axes[0, 0]
        for name, returns in strategy_returns.items():
            cum_returns = (1 + returns).cumprod() - 1
            ax1.plot(cum_returns * 100, label=name, linewidth=2)

        ax1.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Risk-Return scatter
        ax2 = axes[0, 1]
        for name, returns in strategy_returns.items():
            annual_return = returns.mean() * 252 * 100
            annual_vol = returns.std() * np.sqrt(252) * 100
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            ax2.scatter(annual_vol, annual_return, s=200, alpha=0.6, label=name)
            ax2.annotate(
                f"SR:{sharpe:.2f}",
                xy=(annual_vol, annual_return),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax2.set_title("Risk-Return Profile", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Annualized Volatility (%)")
        ax2.set_ylabel("Annualized Return (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Rolling Sharpe ratios
        ax3 = axes[1, 0]
        window = 60
        for name, returns in strategy_returns.items():
            if len(returns) > window:
                rolling_sharpe = (
                    returns.rolling(window=window).mean()
                    * 252
                    / (returns.rolling(window=window).std() * np.sqrt(252))
                )
                ax3.plot(rolling_sharpe, label=name, linewidth=2, alpha=0.7)

        ax3.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax3.set_title(
            f"Rolling Sharpe Ratio ({window}-day)", fontsize=12, fontweight="bold"
        )
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Sharpe Ratio")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Correlation matrix
        ax4 = axes[1, 1]
        returns_df = pd.DataFrame(strategy_returns)
        corr_matrix = returns_df.corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax4,
            cbar_kws={"label": "Correlation"},
        )
        ax4.set_title("Strategy Correlation Matrix", fontsize=12, fontweight="bold")

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_allocation_weights(
        self,
        weights_history: np.ndarray,
        strategy_names: List[str],
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "CIO Allocation Weights",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot allocation weights over time.

        Args:
            weights_history: Array of shape (time, n_strategies)
            strategy_names: List of strategy names
            dates: Optional datetime index
            title: Plot title
            save_name: Filename to save
            show: Whether to display
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x_axis = dates if dates is not None else np.arange(len(weights_history))

        # Stacked area chart
        ax1.stackplot(x_axis, weights_history.T, labels=strategy_names, alpha=0.7)
        ax1.set_title("Allocation Weights (Stacked)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Weight")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)

        # Individual weight lines
        for i, name in enumerate(strategy_names):
            ax2.plot(x_axis, weights_history[:, i], label=name, linewidth=2, alpha=0.7)

        ax2.set_title("Individual Allocation Weights", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Time" if dates is not None else "Steps")
        ax2.set_ylabel("Weight")
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_performance_metrics(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Metrics",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot performance metrics as bar chart.

        Args:
            metrics: Dictionary of metric names to values
            title: Plot title
            save_name: Filename to save
            show: Whether to display
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        metric_names = list(metrics.keys())
        values = list(metrics.values())

        colors = [
            self.colors["profit"] if v > 0 else self.colors["loss"] for v in values
        ]

        bars = ax.barh(metric_names, values, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                value, i, f" {value:.3f}", va="center", fontsize=10, fontweight="bold"
            )

        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def create_performance_dashboard(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        metrics: Dict[str, float],
        strategy_name: str,
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        Create comprehensive performance dashboard.

        Args:
            portfolio_values: Portfolio values over time
            returns: Period returns
            metrics: Performance metrics dict
            strategy_name: Name of strategy
            save_name: Filename to save
            show: Whether to display
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        cum_returns = (portfolio_values / portfolio_values[0] - 1) * 100
        ax1.plot(cum_returns, color=self.colors["primary"], linewidth=2)
        ax1.fill_between(
            range(len(cum_returns)),
            cum_returns,
            0,
            alpha=0.3,
            color=self.colors["primary"],
        )
        ax1.set_title(f"{strategy_name} - Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100
        ax2.fill_between(
            range(len(drawdown)), drawdown, 0, color=self.colors["loss"], alpha=0.3
        )
        ax2.plot(drawdown, color=self.colors["loss"], linewidth=1.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # Returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(
            returns * 100,
            bins=50,
            color=self.colors["primary"],
            alpha=0.6,
            edgecolor="black",
        )
        ax3.axvline(
            returns.mean() * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {returns.mean() * 100:.3f}%",
        )
        ax3.set_title("Returns Distribution", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Returns (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Monthly returns heatmap
        ax4 = fig.add_subplot(gs[2, 1])
        if len(returns) >= 252:
            # Reshape to approximate monthly returns
            monthly_returns = returns.reshape(-1, 21).mean(axis=1) * 100
            monthly_grid = (
                monthly_returns.reshape(-1, 12)
                if len(monthly_returns) >= 12
                else monthly_returns.reshape(1, -1)
            )

            sns.heatmap(
                monthly_grid,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                ax=ax4,
                cbar_kws={"label": "Return (%)"},
            )
            ax4.set_title("Monthly Returns Heatmap", fontsize=10, fontweight="bold")
            ax4.set_xlabel("Month")
            ax4.set_ylabel("Year")
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient data\nfor monthly heatmap",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Monthly Returns Heatmap", fontsize=10, fontweight="bold")

        # Metrics table
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis("off")

        metrics_text = "\n".join(
            [f"{k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()]
        )
        ax5.text(
            0.1,
            0.9,
            metrics_text,
            transform=ax5.transAxes,
            verticalalignment="top",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax5.set_title("Performance Metrics", fontsize=10, fontweight="bold")

        plt.suptitle(
            f"{strategy_name} - Performance Dashboard", fontsize=16, fontweight="bold"
        )

        if save_name:
            plt.savefig(
                self.save_dir / f"{save_name}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        else:
            plt.close()


def save_training_log(
    strategy_name: str, metrics: Dict[str, Any], save_dir: str = "logs"
):
    """
    Save training metrics to JSON log file.

    Args:
        strategy_name: Name of strategy
        metrics: Dictionary of metrics
        save_dir: Directory to save logs
    """
    save_path = Path(save_dir) / f"{strategy_name}_training_log.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["strategy_name"] = strategy_name

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Training log saved to {save_path}")


def load_training_log(strategy_name: str, save_dir: str = "logs") -> Dict[str, Any]:
    """
    Load training metrics from JSON log file.

    Args:
        strategy_name: Name of strategy
        save_dir: Directory containing logs

    Returns:
        Dictionary of metrics
    """
    save_path = Path(save_dir) / f"{strategy_name}_training_log.json"

    if not save_path.exists():
        raise FileNotFoundError(f"Log file not found: {save_path}")

    with open(save_path, "r") as f:
        metrics = json.load(f)

    return metrics


if __name__ == "__main__":
    # Example usage
    plotter = PerformancePlotter(save_dir="reports/figures")

    # Generate sample data
    np.random.seed(42)
    n_steps = 1000

    # Training progress
    metrics = {
        "episode_rewards": np.cumsum(np.random.randn(n_steps) * 10 + 0.1),
        "policy_loss": np.abs(np.random.randn(n_steps)) * 0.5,
        "value_loss": np.abs(np.random.randn(n_steps)) * 1.0,
    }

    plotter.plot_training_progress(
        metrics, title="PPO Training Progress", save_name="example_training", show=False
    )

    # Equity curve
    portfolio_values = 1000000 * (1 + np.random.randn(n_steps).cumsum() * 0.01)
    plotter.plot_equity_curve(
        portfolio_values,
        title="Example Equity Curve",
        save_name="example_equity",
        show=False,
    )

    print("Example plots generated successfully!")
