"""
Utility functions for training specialists and master agent on real market data.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Any, List
from pathlib import Path


def prepare_specialist_data(
    equities_train,
    equities_val,
    equities_test,
    fx_train,
    fx_val,
    fx_test,
    futures_train,
    futures_val,
    futures_test,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Prepare data for each specialist strategy.

    Returns:
        Dictionary mapping specialist names to their train/val/test data
    """
    specialist_datasets = {}

    # 1. Statistical Arbitrage - uses pairs of equities
    if len(equities_train) >= 2:
        # Select 2 correlated stocks
        symbols = list(equities_train.keys())[:2]
        stats_arb_train = pd.concat(
            [equities_train[symbols[0]]["Close"], equities_train[symbols[1]]["Close"]],
            axis=1,
        )
        stats_arb_train.columns = ["asset1_price", "asset2_price"]

        stats_arb_val = pd.concat(
            [equities_val[symbols[0]]["Close"], equities_val[symbols[1]]["Close"]],
            axis=1,
        )
        stats_arb_val.columns = ["asset1_price", "asset2_price"]

        stats_arb_test = pd.concat(
            [equities_test[symbols[0]]["Close"], equities_test[symbols[1]]["Close"]],
            axis=1,
        )
        stats_arb_test.columns = ["asset1_price", "asset2_price"]

        specialist_datasets["statistical_arbitrage"] = {
            "train": stats_arb_train.dropna(),
            "val": stats_arb_val.dropna(),
            "test": stats_arb_test.dropna(),
        }

    # 2. Market Making - uses single equity with high liquidity
    if len(equities_train) >= 1:
        symbol = list(equities_train.keys())[0]
        mm_train = equities_train[symbol][["Close", "High", "Low", "Volume"]].copy()
        mm_val = equities_val[symbol][["Close", "High", "Low", "Volume"]].copy()
        mm_test = equities_test[symbol][["Close", "High", "Low", "Volume"]].copy()

        # Rename for environment compatibility
        mm_train.columns = ["close", "high", "low", "volume"]
        mm_val.columns = ["close", "high", "low", "volume"]
        mm_test.columns = ["close", "high", "low", "volume"]

        specialist_datasets["market_making"] = {
            "train": mm_train.dropna(),
            "val": mm_val.dropna(),
            "test": mm_test.dropna(),
        }

    # 3. Factor Tracking - uses equity returns to create factor exposures
    if len(equities_train) >= 10:
        symbols = list(equities_train.keys())[:10]

        def create_factor_data(equity_dict, symbols_list):
            returns_df = pd.DataFrame()
            for sym in symbols_list:
                if "Close" in equity_dict[sym].columns:
                    returns_df[sym] = equity_dict[sym]["Close"].pct_change()

            # Create synthetic factors
            factor_df = pd.DataFrame(index=returns_df.index)
            factor_df["value_ret"] = returns_df.iloc[:, :3].mean(axis=1)  # Value factor
            factor_df["momentum_ret"] = returns_df.iloc[:, 3:6].mean(axis=1)  # Momentum
            factor_df["quality_ret"] = returns_df.iloc[:, 6:9].mean(axis=1)  # Quality
            factor_df["market_ret"] = returns_df.mean(axis=1)  # Market

            return factor_df.dropna()

        specialist_datasets["factor_tracking"] = {
            "train": create_factor_data(equities_train, symbols),
            "val": create_factor_data(equities_val, symbols),
            "test": create_factor_data(equities_test, symbols),
        }

    # 4. Volatility Trading - uses equity with vol features
    if len(equities_train) >= 1:
        symbol = list(equities_train.keys())[0]

        def prep_vol_data(equity_dict, symbol):
            df = equity_dict[symbol][["Close"]].copy()
            df.columns = ["close"]

            # Add implied vol (use historical vol as proxy)
            df["implied_vol"] = df["close"].pct_change().rolling(20).std() * np.sqrt(
                252
            )
            df["volume"] = (
                equity_dict[symbol]["Volume"]
                if "Volume" in equity_dict[symbol].columns
                else 1000000
            )

            return df.dropna()

        specialist_datasets["volatility_trading"] = {
            "train": prep_vol_data(equities_train, symbol),
            "val": prep_vol_data(equities_val, symbol),
            "test": prep_vol_data(equities_test, symbol),
        }

    # 5. Delta Hedging - similar to volatility trading
    if len(equities_train) >= 1:
        symbol = list(equities_train.keys())[0]
        specialist_datasets["delta_hedging"] = {
            "train": prep_vol_data(equities_train, symbol),
            "val": prep_vol_data(equities_val, symbol),
            "test": prep_vol_data(equities_test, symbol),
        }

    # 6. Futures Spreads - uses futures contracts
    if len(futures_train) >= 2:
        symbols = list(futures_train.keys())[:2]

        def prep_futures_spread(futures_dict, symbols_list):
            df = pd.DataFrame()
            df["near"] = futures_dict[symbols_list[0]]["Close"]
            df["far"] = futures_dict[symbols_list[1]]["Close"]
            return df.dropna()

        specialist_datasets["futures_spreads"] = {
            "train": prep_futures_spread(futures_train, symbols),
            "val": prep_futures_spread(futures_val, symbols),
            "test": prep_futures_spread(futures_test, symbols),
        }

    # 7. FX Arbitrage - uses FX pairs
    if len(fx_train) >= 3:
        symbols = list(fx_train.keys())[:3]

        def prep_fx_data(fx_dict, symbols_list):
            df = pd.DataFrame()
            for i, sym in enumerate(symbols_list):
                df[f"rate_{i}"] = fx_dict[sym]["Close"]
            return df.dropna()

        specialist_datasets["fx_arbitrage"] = {
            "train": prep_fx_data(fx_train, symbols),
            "val": prep_fx_data(fx_val, symbols),
            "test": prep_fx_data(fx_test, symbols),
        }

    return specialist_datasets


def train_all_specialists(
    specialist_datasets: Dict[str, Dict[str, pd.DataFrame]],
    initial_capital: float = 100000,
    train_timesteps: int = 50000,
) -> Dict[str, Tuple[Any, Any, str]]:
    """
    Train all 7 specialist agents.

    Returns:
        Dictionary mapping specialist names to (agent, env, algorithm) tuples
    """
    from src.agents.ddpg import DDPGAgent
    from src.agents.dqn import DQNAgent
    from src.environments.specialist_envs.stats_arb.env_stat_arb import (
        StatisticalArbitrageEnv,
    )
    from src.environments.specialist_envs.Market_Making.env_market_maker import (
        MarketMakingEnv,
    )
    from src.environments.specialist_envs.Factor_Tracking.env_factor_tracker import (
        FactorTrackingEnv,
    )
    from src.environments.specialist_envs.Volatility_Trading.env_vol_trading import (
        VolatilityTradingEnv,
    )
    from src.environments.specialist_envs.Delta_Hedging.env_delta_hedging import (
        DeltaHedgingEnv,
    )
    from src.environments.specialist_envs.Futures_Spreads.env_futures_spread import (
        FuturesSpreadsEnv,
    )
    from src.environments.specialist_envs.FX_Arbitrage.env_fx_arb import FXArbitrageEnv

    trained_specialists = {}

    # 1. Statistical Arbitrage (DDPG)
    if "statistical_arbitrage" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training Statistical Arbitrage Agent (DDPG)")
        print("=" * 80)

        env = StatisticalArbitrageEnv(
            df=specialist_datasets["statistical_arbitrage"]["train"],
            initial_balance=initial_capital,
            transaction_cost_pct=0.001,
        )
        agent = DDPGAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["statistical_arbitrage"] = (agent, env, "ddpg")
        print("✅ Statistical Arbitrage training complete")

    # 2. Market Making (DDPG)
    if "market_making" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training Market Making Agent (DDPG)")
        print("=" * 80)

        env = MarketMakingEnv(
            data=specialist_datasets["market_making"]["train"],
            initial_capital=initial_capital,
        )
        agent = DDPGAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["market_making"] = (agent, env, "ddpg")
        print("✅ Market Making training complete")

    # 3. Factor Tracking (DQN)
    if "factor_tracking" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training Factor Tracking Agent (DQN)")
        print("=" * 80)

        env = FactorTrackingEnv(
            data=specialist_datasets["factor_tracking"]["train"],
            initial_capital=initial_capital,
            num_factors=3,
        )
        agent = DQNAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["factor_tracking"] = (agent, env, "dqn")
        print("✅ Factor Tracking training complete")

    # 4. Volatility Trading (DDPG)
    if "volatility_trading" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training Volatility Trading Agent (DDPG)")
        print("=" * 80)

        env = VolatilityTradingEnv(
            data=specialist_datasets["volatility_trading"]["train"],
            initial_capital=initial_capital,
        )
        agent = DDPGAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["volatility_trading"] = (agent, env, "ddpg")
        print("✅ Volatility Trading training complete")

    # 5. Delta Hedging (DDPG)
    if "delta_hedging" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training Delta Hedging Agent (DDPG)")
        print("=" * 80)

        env = DeltaHedgingEnv(
            data=specialist_datasets["delta_hedging"]["train"],
            initial_capital=initial_capital,
        )
        agent = DDPGAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["delta_hedging"] = (agent, env, "ddpg")
        print("✅ Delta Hedging training complete")

    # 6. Futures Spreads (DDPG)
    if "futures_spreads" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training Futures Spreads Agent (DDPG)")
        print("=" * 80)

        env = FuturesSpreadsEnv(
            data=specialist_datasets["futures_spreads"]["train"],
            initial_capital=initial_capital,
        )
        agent = DDPGAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["futures_spreads"] = (agent, env, "ddpg")
        print("✅ Futures Spreads training complete")

    # 7. FX Arbitrage (DDPG)
    if "fx_arbitrage" in specialist_datasets:
        print("\n" + "=" * 80)
        print("Training FX Arbitrage Agent (DDPG)")
        print("=" * 80)

        env = FXArbitrageEnv(
            data=specialist_datasets["fx_arbitrage"]["train"],
            initial_capital=initial_capital,
        )
        agent = DDPGAgent(env=env)
        agent.train(total_timesteps=train_timesteps, log_interval=5000)

        trained_specialists["fx_arbitrage"] = (agent, env, "ddpg")
        print("✅ FX Arbitrage training complete")

    print("\n" + "=" * 80)
    print(f"✅ ALL SPECIALISTS TRAINED: {len(trained_specialists)}/7")
    print("=" * 80)

    return trained_specialists


def save_trained_models(
    trained_specialists: Dict[str, Tuple[Any, Any, str]],
    models_dir: str = "models/specialists",
) -> None:
    """Save all trained specialist models."""
    models_path = Path(models_dir)

    print("\n" + "=" * 80)
    print("SAVING TRAINED MODELS")
    print("=" * 80)

    for strategy_name, (agent, env, algo) in trained_specialists.items():
        specialist_dir = models_path / strategy_name
        specialist_dir.mkdir(parents=True, exist_ok=True)

        model_path = specialist_dir / f"{strategy_name}_{algo}.pt"
        agent.save(str(model_path))

        print(f"✅ {strategy_name}: {model_path}")

    # Save metadata
    metadata_path = models_path / "training_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"Training completed: {pd.Timestamp.now()}\n")
        f.write(f"Total specialists: {len(trained_specialists)}\n\n")

        for strategy_name, (agent, env, algo) in trained_specialists.items():
            f.write(f"{strategy_name}:\n")
            f.write(f"  Algorithm: {algo.upper()}\n")
            f.write(f"  Model: {strategy_name}/{strategy_name}_{algo}.pt\n\n")

    print(f"\n✅ Metadata saved: {metadata_path}")
    print("=" * 80)
