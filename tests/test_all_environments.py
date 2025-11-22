"""
Test All Specialist Environments

Validates that all 7 specialist environments are working correctly.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.specialist_envs import (
    StatisticalArbitrageEnv,
    MarketMakingEnv,
    VolatilityTradingEnv,
    DeltaHedgingEnv,
    FuturesSpreadsEnv,
    FactorTrackingEnv,
    FXArbitrageEnv,
    SPECIALIST_ENVS,
    list_specialist_envs,
)


def create_test_data(n_steps=100):
    """Create synthetic test data."""
    np.random.seed(42)

    # Basic price data
    returns = np.random.normal(0.0001, 0.01, n_steps)
    prices = 100 * np.exp(np.cumsum(returns))

    # Pairs for stat arb
    asset1 = prices
    asset2 = prices * 0.95 + np.random.normal(0, 1, n_steps)

    # Futures
    near = prices
    far = near + np.random.normal(2, 0.5, n_steps)

    # FX rates
    eur_usd = 1.10 + np.cumsum(np.random.normal(0, 0.001, n_steps))
    usd_jpy = 110.0 + np.cumsum(np.random.normal(0, 0.1, n_steps))
    eur_jpy = eur_usd * usd_jpy + np.random.normal(0, 0.5, n_steps)

    # Factor returns
    value_ret = np.random.normal(0.0003, 0.008, n_steps)
    momentum_ret = np.random.normal(0.0005, 0.012, n_steps)
    quality_ret = np.random.normal(0.0002, 0.006, n_steps)
    market_ret = np.random.normal(0.0004, 0.01, n_steps)

    # Implied vol
    iv = [0.25]
    for _ in range(n_steps - 1):
        iv.append(iv[-1] + 0.3 * (0.25 - iv[-1]) + np.random.normal(0, 0.02))

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.001, n_steps)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.002, n_steps))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.002, n_steps))),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_steps),
            # Stat arb
            "asset1": asset1,
            "asset2": asset2,
            # Futures
            "near": near,
            "far": far,
            # Vol
            "implied_vol": np.array(iv),
            # FX
            "EUR_USD": eur_usd,
            "USD_JPY": usd_jpy,
            "EUR_JPY": eur_jpy,
            "EUR_USD_interest": np.random.normal(0.02, 0.005, n_steps),
            "USD_JPY_interest": np.random.normal(-0.01, 0.005, n_steps),
            "EUR_JPY_interest": np.random.normal(0.01, 0.005, n_steps),
            # Factors
            "value_ret": value_ret,
            "momentum_ret": momentum_ret,
            "quality_ret": quality_ret,
            "market_ret": market_ret,
        }
    )

    return data


def test_environment(env_name, env_class, data, **kwargs):
    """Test a single environment."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {env_name.upper().replace('_', ' ')}")
    print(f"{'=' * 80}")

    try:
        # Create environment
        env = env_class(data, **kwargs)

        print(f"✓ Environment created")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space}")

        # Reset
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial portfolio: ${env.portfolio_value:,.2f}")

        # Run 5 steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"✓ Stepped through {step + 1} episodes")
        print(f"  Final portfolio: ${env.portfolio_value:,.2f}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Total trades: {env.total_trades}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all environment tests."""
    print("\n" + "=" * 80)
    print("SPECIALIST ENVIRONMENTS TEST SUITE")
    print("=" * 80)

    # Create test data
    print("\nGenerating synthetic test data...")
    data = create_test_data(n_steps=100)
    print(f"✓ Created {len(data)} timesteps of data")
    print(f"  Columns: {', '.join(data.columns[:5])}... ({len(data.columns)} total)")

    # List all environments
    list_specialist_envs()

    # Test each environment
    results = {}

    # 1. Statistical Arbitrage
    results["stat_arb"] = test_environment(
        "statistical_arbitrage", StatisticalArbitrageEnv, data
    )

    # 2. Market Making
    results["market_making"] = test_environment("market_making", MarketMakingEnv, data)

    # 3. Volatility Trading
    results["vol_trading"] = test_environment(
        "volatility_trading", VolatilityTradingEnv, data
    )

    # 4. Delta Hedging
    results["delta_hedging"] = test_environment(
        "delta_hedging", DeltaHedgingEnv, data, strike_price=100.0
    )

    # 5. Futures Spreads
    results["futures_spreads"] = test_environment(
        "futures_spreads", FuturesSpreadsEnv, data
    )

    # 6. Factor Tracking
    results["factor_tracking"] = test_environment(
        "factor_tracking", FactorTrackingEnv, data
    )

    # 7. FX Arbitrage
    results["fx_arbitrage"] = test_environment("fx_arbitrage", FXArbitrageEnv, data)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}  {name.upper().replace('_', ' ')}")

    print("\n" + "-" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! All environments are ready for training.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")

    print("=" * 80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
