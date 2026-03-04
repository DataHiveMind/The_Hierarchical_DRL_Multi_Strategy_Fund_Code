from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.experiment_spec import get_default_spec_path, load_experiment_spec


def parse_readme_mapping(readme_text: str) -> Dict[str, str]:
    table_pattern = re.compile(r"\|\s*\*\*?([A-Za-z\s]+?)\*\*?\s*\|\s*(PPO|DDPG|DQN)\s*\|")
    mapping = {}
    name_norm = {
        "Statistical Arbitrage": "statistical_arbitrage",
        "Market Making": "market_making",
        "Volatility Trading": "volatility_trading",
        "Delta Hedging": "delta_hedging",
        "Futures Spreads": "futures_spreads",
        "Factor Tracking": "factor_tracking",
        "FX Arbitrage": "fx_arbitrage",
    }
    for strategy_name, algo in table_pattern.findall(readme_text):
        key = name_norm.get(strategy_name.strip())
        if key:
            mapping[key] = algo
    return mapping


def parse_strategy_config_mapping(strategy_config_text: str) -> Dict[str, str]:
    block_pattern = re.compile(
        r'"([a-z_]+)"\s*:\s*StrategyConfig\(.*?agent_type="(PPO|DDPG|DQN)"',
        re.DOTALL,
    )
    return {strategy: algo for strategy, algo in block_pattern.findall(strategy_config_text)}


def extract_all_date_windows(text: str) -> List[Tuple[str, str]]:
    return re.findall(r"(20\d{2}-\d{2}-\d{2}).*?(20\d{2}-\d{2}-\d{2})", text)


def validate_algorithm_consistency(readme_text: str, strategy_config_text: str) -> List[str]:
    issues = []
    readme_map = parse_readme_mapping(readme_text)
    config_map = parse_strategy_config_mapping(strategy_config_text)

    for strategy, config_algo in config_map.items():
        readme_algo = readme_map.get(strategy)
        if readme_algo and readme_algo != config_algo:
            issues.append(
                f"Algorithm mismatch for '{strategy}': README={readme_algo}, strategy_config={config_algo}"
            )
    return issues


def validate_split_consistency(spec: dict, notebook_text: str, readme_text: str) -> List[str]:
    issues = []
    split = spec["splits"]
    canonical = [
        split["train_start"],
        split["train_end"],
        split["validation_start"],
        split["validation_end"],
        split["test_start"],
        split["test_end"],
    ]

    for source_name, text in [("README", readme_text), ("notebook_03", notebook_text)]:
        dates = re.findall(r"20\d{2}-\d{2}-\d{2}", text)
        for date_value in canonical:
            if date_value not in dates:
                issues.append(f"{source_name} does not reference canonical split date: {date_value}")
                break
    return issues


def main() -> int:
    spec = load_experiment_spec(get_default_spec_path())
    readme_path = ROOT / "README.md"
    strategy_config_path = ROOT / "src" / "utils" / "strategy_config.py"
    notebook_path = ROOT / "notebooks" / "03_results_and_visualization.ipynb"

    readme_text = readme_path.read_text(encoding="utf-8")
    strategy_config_text = strategy_config_path.read_text(encoding="utf-8")
    notebook_text = notebook_path.read_text(encoding="utf-8")

    issues = []
    issues.extend(validate_algorithm_consistency(readme_text, strategy_config_text))
    issues.extend(validate_split_consistency(spec, notebook_text, readme_text))

    if issues:
        print("Research consistency check: FAILED")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("Research consistency check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
