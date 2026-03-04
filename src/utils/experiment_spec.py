from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import datetime as dt

import yaml


@dataclass(frozen=True)
class DateSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


def _parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def validate_date_splits(splits: DateSplit) -> None:
    train_start = _parse_date(splits.train_start)
    train_end = _parse_date(splits.train_end)
    validation_start = _parse_date(splits.validation_start)
    validation_end = _parse_date(splits.validation_end)
    test_start = _parse_date(splits.test_start)
    test_end = _parse_date(splits.test_end)

    if not (train_start <= train_end < validation_start <= validation_end < test_start <= test_end):
        raise ValueError(
            "Invalid split ordering. Expected: train <= validation <= test with non-overlapping chronological windows."
        )


def load_experiment_spec(spec_path: str | Path) -> Dict[str, Any]:
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment spec not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    required_top_level = [
        "project",
        "data",
        "splits",
        "market_frictions",
        "risk",
        "evaluation",
        "reporting",
    ]
    missing = [key for key in required_top_level if key not in spec]
    if missing:
        raise ValueError(f"Missing required spec sections: {missing}")

    split = DateSplit(**spec["splits"])
    validate_date_splits(split)
    return spec


def get_default_spec_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "experiment_spec.yaml"
