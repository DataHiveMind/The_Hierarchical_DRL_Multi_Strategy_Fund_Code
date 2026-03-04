from dataclasses import dataclass
from typing import Iterator, Tuple, List


@dataclass(frozen=True)
class SplitConfig:
    train_size: int
    test_size: int
    step_size: int
    purge_size: int = 0
    embargo_size: int = 0


class PurgedWalkForwardSplitter:
    """
    Purged walk-forward splitter for leakage-safe time-series validation.

    Index layout for each fold:
        [train ...][purge][test][embargo]
    """

    def __init__(self, config: SplitConfig):
        if config.train_size <= 0 or config.test_size <= 0 or config.step_size <= 0:
            raise ValueError("train_size, test_size, and step_size must be positive")
        if config.purge_size < 0 or config.embargo_size < 0:
            raise ValueError("purge_size and embargo_size must be non-negative")
        self.config = config

    def split(self, n_samples: int) -> Iterator[Tuple[List[int], List[int]]]:
        cfg = self.config
        start = 0

        while True:
            train_start = start
            train_end = train_start + cfg.train_size
            test_start = train_end + cfg.purge_size
            test_end = test_start + cfg.test_size
            embargo_end = test_end + cfg.embargo_size

            if test_end > n_samples:
                break

            train_idx = list(range(train_start, train_end))
            test_idx = list(range(test_start, test_end))
            yield train_idx, test_idx

            if embargo_end >= n_samples:
                break

            start += cfg.step_size

    def get_n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self.split(n_samples))
