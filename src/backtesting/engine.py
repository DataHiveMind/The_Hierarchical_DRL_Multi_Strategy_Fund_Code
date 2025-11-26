"""
Backtesting engine for evaluating trained RL agents on historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
from datetime import datetime

from .metrics import PerformanceMetrics, StrategyComparison


class BacktestEngine:
    """
    Backtesting engine for testing trained agents on historical data.
    Supports both specialist and master-level agents.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital for backtest
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
            slippage: Price slippage as fraction
            risk_free_rate: Annual risk-free rate for metrics
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

        self.metrics_calc = PerformanceMetrics(risk_free_rate)
        self.results = {}

    def run_specialist_backtest(
        self,
        agent,
        env,
        test_data: pd.DataFrame,
        strategy_name: str,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Run backtest for a single specialist agent.

        Args:
            agent: Trained specialist agent (DDPG or DQN)
            env: Trading environment for the specialist
            test_data: Historical test data
            strategy_name: Name of the strategy
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary containing backtest results
        """
        # Reset environment with test data
        # Handle different attribute names (data vs df)
        if hasattr(env, "df"):
            env.df = test_data
        elif hasattr(env, "data"):
            env.data = test_data
        else:
            raise AttributeError(
                f"Environment {type(env).__name__} has no 'data' or 'df' attribute"
            )

        reset_result = env.reset()
        
        # Handle different reset() return formats (Gym 0.26+ returns tuple)
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # (observation, info)
        else:
            state = reset_result  # Just observation

        # Storage for tracking
        equity_curve = [self.initial_capital]
        positions = []
        actions_taken = []
        rewards = []
        timestamps = []

        done = False
        step = 0

        while not done and step < len(test_data) - 1:
            # Select action
            with torch.no_grad():
                # Check agent type by class name
                agent_class = agent.__class__.__name__
                if agent_class == "DDPGAgent":
                    action = agent.select_action(state, add_noise=not deterministic)
                elif agent_class == "DQNAgent":
                    action = agent.select_action(state, deterministic=deterministic)
                else:  # PPO or other agents
                    action = agent.select_action(state, deterministic=deterministic)

            # Take action in environment
            try:
                step_result = env.step(action)
            except IndexError:
                # Environment ran out of data
                print(f"  Info: Environment reached end of data at step {step}")
                done = True
                break
            
            # Handle different step() return formats
            if len(step_result) == 5:
                # Gym 0.26+: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # Gym <0.26: (obs, reward, done, info)
                next_state, reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected step() return format: {len(step_result)} values")
            
            # Ensure next_state is observation only (not a tuple)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Record data
            equity = info.get(
                "portfolio_value", info.get("total_value", self.initial_capital)
            )
            equity_curve.append(equity)
            positions.append(info.get("position", 0))
            actions_taken.append(action)
            rewards.append(reward)

            if step < len(test_data):
                timestamps.append(
                    test_data.index[step] if hasattr(test_data, "index") else step
                )

            state = next_state
            step += 1

        # Create results dataframe
        results_df = pd.DataFrame(
            {
                "equity": equity_curve[1:],  # Skip initial capital
                "position": positions,
                "action": actions_taken,
                "reward": rewards,
            }
        )

        if timestamps:
            results_df.index = timestamps

        # Calculate metrics
        equity_series = pd.Series(equity_curve)
        metrics = self.metrics_calc.calculate_all_metrics(equity_series)

        # Store results
        backtest_results = {
            "strategy_name": strategy_name,
            "equity_curve": equity_series,
            "results_df": results_df,
            "metrics": metrics,
            "final_value": equity_curve[-1],
            "total_return": (equity_curve[-1] / equity_curve[0] - 1),
            "num_trades": len(positions),
            "test_data": test_data,
        }

        self.results[strategy_name] = backtest_results
        return backtest_results

    def run_master_backtest(
        self,
        master_agent,
        specialist_agents: Dict[str, Any],
        env,
        test_data: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Run backtest for master agent allocating to specialists.

        Args:
            master_agent: Trained master allocation agent (PPO)
            specialist_agents: Dictionary of specialist agents
            env: Master environment
            test_data: Historical test data
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary containing backtest results
        """
        # Reset environment
        reset_result = env.reset()
        
        # Handle different reset() return formats (Gym 0.26+ returns tuple)
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # (observation, info)
        else:
            state = reset_result  # Just observation

        # Storage for tracking
        equity_curve = [self.initial_capital]
        allocations = []
        specialist_returns = []
        actions_taken = []
        rewards = []
        timestamps = []

        done = False
        step = 0

        while not done and step < len(test_data) - 1:
            # Select allocation action
            with torch.no_grad():
                action_result = master_agent.select_action(state, deterministic=deterministic)
                
                # PPO returns (action, log_prob, value), extract just action
                if isinstance(action_result, tuple):
                    action = action_result[0]
                else:
                    action = action_result
                
                # Ensure action is a 1D numpy array
                action = np.array(action).flatten()

            # Take action in environment
            try:
                step_result = env.step(action)
            except IndexError:
                # Environment ran out of data
                print(f"  Info: Environment reached end of data at step {step}")
                done = True
                break
            
            # Handle different step() return formats
            if len(step_result) == 5:
                # Gym 0.26+: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # Gym <0.26: (obs, reward, done, info)
                next_state, reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected step() return format: {len(step_result)} values")
            
            # Ensure next_state is observation only (not a tuple)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Record data
            equity = info.get("portfolio_value", self.initial_capital)
            equity_curve.append(equity)

            # Extract allocations (typically 3 weights that sum to 1)
            if isinstance(action, (list, np.ndarray)):
                allocations.append(action)
            else:
                allocations.append([action])

            actions_taken.append(action)
            rewards.append(reward)

            # Record specialist returns if available
            spec_returns = info.get("specialist_returns", [])
            specialist_returns.append(spec_returns)

            if step < len(test_data):
                timestamps.append(
                    test_data.index[step] if hasattr(test_data, "index") else step
                )

            state = next_state
            step += 1

        # Create results dataframe
        results_df = pd.DataFrame({"equity": equity_curve[1:], "reward": rewards})

        # Add allocation columns
        if allocations:
            alloc_array = np.array(allocations)
            for i in range(alloc_array.shape[1]):
                results_df[f"allocation_{i}"] = alloc_array[:, i]

        if timestamps:
            results_df.index = timestamps

        # Calculate metrics
        equity_series = pd.Series(equity_curve)
        metrics = self.metrics_calc.calculate_all_metrics(equity_series)

        # Store results
        backtest_results = {
            "strategy_name": "Master_CIO_Allocator",
            "equity_curve": equity_series,
            "results_df": results_df,
            "metrics": metrics,
            "final_value": equity_curve[-1],
            "total_return": (equity_curve[-1] / equity_curve[0] - 1),
            "allocations": allocations,
            "specialist_returns": specialist_returns,
            "test_data": test_data,
        }

        self.results["Master_CIO_Allocator"] = backtest_results
        return backtest_results

    def run_multiple_specialists(
        self,
        agents_dict: Dict[str, Tuple[Any, Any]],
        test_data: pd.DataFrame,
        deterministic: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run backtests for multiple specialist agents.

        Args:
            agents_dict: Dictionary mapping strategy names to (agent, env) tuples
            test_data: Historical test data
            deterministic: Whether to use deterministic policies

        Returns:
            Dictionary of results for each strategy
        """
        all_results = {}

        for strategy_name, (agent, env) in agents_dict.items():
            print(f"Running backtest for {strategy_name}...")

            results = self.run_specialist_backtest(
                agent=agent,
                env=env,
                test_data=test_data,
                strategy_name=strategy_name,
                deterministic=deterministic,
            )

            all_results[strategy_name] = results

            # Print summary
            print(f"  Total Return: {results['total_return']:.2%}")
            print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
            print()

        return all_results

    def compare_strategies(
        self, strategy_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare performance of multiple strategies.

        Args:
            strategy_names: List of strategy names to compare (None = all)

        Returns:
            DataFrame comparing metrics across strategies
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())

        equity_curves = {
            name: self.results[name]["equity_curve"]
            for name in strategy_names
            if name in self.results
        }

        comparison = StrategyComparison(self.risk_free_rate)
        return comparison.compare_strategies(equity_curves)

    def get_summary(self, strategy_name: str) -> str:
        """
        Get formatted summary of backtest results.

        Args:
            strategy_name: Name of strategy to summarize

        Returns:
            Formatted string summary
        """
        if strategy_name not in self.results:
            return f"No results found for {strategy_name}"

        results = self.results[strategy_name]
        metrics = results["metrics"]

        summary = f"""
        ═══════════════════════════════════════════════════
        Backtest Summary: {strategy_name}
        ═══════════════════════════════════════════════════
        
        Returns:
          Total Return:        {results["total_return"]:>10.2%}
          Annual Return:       {metrics["annual_return"]:>10.2%}
          Annual Volatility:   {metrics["annual_volatility"]:>10.2%}
        
        Risk Metrics:
          Sharpe Ratio:        {metrics["sharpe_ratio"]:>10.2f}
          Sortino Ratio:       {metrics["sortino_ratio"]:>10.2f}
          Calmar Ratio:        {metrics["calmar_ratio"]:>10.2f}
          Max Drawdown:        {metrics["max_drawdown"]:>10.2%}
          Current Drawdown:    {metrics["current_drawdown"]:>10.2%}
        
        Risk Measures:
          VaR (95%):           {metrics["var_95"]:>10.4f}
          CVaR (95%):          {metrics["cvar_95"]:>10.4f}
        
        Trading Statistics:
          Win Rate:            {metrics["win_rate"]:>10.2%}
          Profit Factor:       {metrics["profit_factor"]:>10.2f}
          Number of Periods:   {metrics["num_periods"]:>10.0f}
        
        Capital:
          Initial Capital:     ${self.initial_capital:>10,.2f}
          Final Value:         ${results["final_value"]:>10,.2f}
          P&L:                 ${results["final_value"] - self.initial_capital:>10,.2f}
        ═══════════════════════════════════════════════════
        """

        return summary

    def export_results(
        self, strategy_name: str, output_dir: str = "reports/backtest_results"
    ) -> None:
        """
        Export backtest results to CSV and summary text file.

        Args:
            strategy_name: Name of strategy to export
            output_dir: Directory to save results
        """
        if strategy_name not in self.results:
            print(f"No results found for {strategy_name}")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = self.results[strategy_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export results dataframe
        results_file = output_path / f"{strategy_name}_results_{timestamp}.csv"
        results["results_df"].to_csv(results_file)

        # Export equity curve
        equity_file = output_path / f"{strategy_name}_equity_{timestamp}.csv"
        results["equity_curve"].to_csv(equity_file)

        # Export summary
        summary_file = output_path / f"{strategy_name}_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(self.get_summary(strategy_name))

        print(f"Results exported to {output_path}")
        print(f"  - {results_file.name}")
        print(f"  - {equity_file.name}")
        print(f"  - {summary_file.name}")


class WalkForwardAnalysis:
    """
    Walk-forward analysis for out-of-sample testing.
    Trains on rolling windows and tests on subsequent periods.
    """

    def __init__(
        self, train_window: int = 252, test_window: int = 63, step_size: int = 63
    ):
        """
        Initialize walk-forward analysis.

        Args:
            train_window: Size of training window in periods
            test_window: Size of test window in periods
            step_size: Step size for rolling window
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def generate_splits(
        self, data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train-test splits for walk-forward analysis.

        Args:
            data: Full dataset

        Returns:
            List of (train, test) dataframe tuples
        """
        splits = []
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= len(data):
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]

            splits.append((train_data, test_data))
            start_idx += self.step_size

        return splits

    def run_walk_forward(
        self,
        agent_factory,
        env_factory,
        data: pd.DataFrame,
        train_timesteps: int = 10000,
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.

        Args:
            agent_factory: Function that creates a new agent instance
            env_factory: Function that creates a new environment instance
            data: Full dataset
            train_timesteps: Training timesteps per window

        Returns:
            Dictionary containing walk-forward results
        """
        splits = self.generate_splits(data)
        all_test_results = []

        for i, (train_data, test_data) in enumerate(splits):
            print(f"Walk-forward iteration {i + 1}/{len(splits)}")

            # Create fresh agent and environment
            env = env_factory(train_data)
            agent = agent_factory(env)

            # Train on training window
            print(f"  Training on {len(train_data)} periods...")
            agent.train(total_timesteps=train_timesteps, log_interval=0)

            # Test on test window
            print(f"  Testing on {len(test_data)} periods...")
            backtest_engine = BacktestEngine()
            test_env = env_factory(test_data)

            results = backtest_engine.run_specialist_backtest(
                agent=agent,
                env=test_env,
                test_data=test_data,
                strategy_name=f"WF_Period_{i}",
                deterministic=True,
            )

            all_test_results.append(results)
            print(f"  Return: {results['total_return']:.2%}")

        # Aggregate results
        total_equity = pd.concat([r["equity_curve"] for r in all_test_results])
        metrics_calc = PerformanceMetrics()
        overall_metrics = metrics_calc.calculate_all_metrics(total_equity)

        return {
            "splits": splits,
            "individual_results": all_test_results,
            "combined_equity": total_equity,
            "overall_metrics": overall_metrics,
        }
