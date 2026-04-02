"""Tests for stratified split behavior."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.splits.stratified_split import SplitConfig, stratified_train_valid_test_split


class StratifiedSplitTests(unittest.TestCase):
    def test_stratified_split_preserves_class_distribution(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        y = np.array([0] * 210 + [1] * 90)
        rng.shuffle(y)
        df = pd.DataFrame(
            {
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
                "target": y,
            }
        )
        cfg = SplitConfig(test_size=0.2, validation_size=0.2, random_state=42, stratify_column="target")
        splits = stratified_train_valid_test_split(df, cfg)

        base_rate = df["target"].mean()
        for key in ("train", "valid", "test"):
            self.assertFalse(splits[key].empty)
            self.assertAlmostEqual(splits[key]["target"].mean(), base_rate, delta=0.06)


if __name__ == "__main__":
    unittest.main()
