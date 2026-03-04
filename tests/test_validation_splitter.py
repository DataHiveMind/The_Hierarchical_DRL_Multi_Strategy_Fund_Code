import sys
from pathlib import Path
import importlib.util

sys.path.insert(0, str(Path(__file__).parent.parent))

module_path = Path(__file__).parent.parent / "src" / "backtesting" / "validation.py"
spec = importlib.util.spec_from_file_location("validation_module", module_path)
validation_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(validation_module)

SplitConfig = validation_module.SplitConfig
PurgedWalkForwardSplitter = validation_module.PurgedWalkForwardSplitter


def test_purged_splitter_non_overlapping_train_test():
    cfg = SplitConfig(train_size=50, test_size=10, step_size=20, purge_size=5, embargo_size=5)
    splitter = PurgedWalkForwardSplitter(cfg)

    for train_idx, test_idx in splitter.split(n_samples=200):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert max(train_idx) < min(test_idx)


def test_purged_splitter_respects_purge_gap():
    cfg = SplitConfig(train_size=30, test_size=10, step_size=10, purge_size=3, embargo_size=0)
    splitter = PurgedWalkForwardSplitter(cfg)

    train_idx, test_idx = next(splitter.split(n_samples=100))
    assert min(test_idx) - max(train_idx) - 1 == cfg.purge_size


def test_get_n_splits_positive():
    cfg = SplitConfig(train_size=40, test_size=20, step_size=20, purge_size=2, embargo_size=2)
    splitter = PurgedWalkForwardSplitter(cfg)
    assert splitter.get_n_splits(220) > 0
