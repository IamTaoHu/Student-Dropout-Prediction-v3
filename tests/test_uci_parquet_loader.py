from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.loaders.uct_student_loader import load_uct_student_predefined_splits


class UciParquetLoaderTests(unittest.TestCase):
    def test_predefined_parquet_loader_validates_and_reorders_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.parquet"
            test_path = root / "test.parquet"
            train_path.write_text("stub", encoding="utf-8")
            test_path.write_text("stub", encoding="utf-8")
            train_df = pd.DataFrame(
                {
                    "FeatureA": [1, 2],
                    "FeatureB": [0.1, 0.2],
                    "Target": ["Graduate", "Dropout"],
                }
            )
            test_df = pd.DataFrame(
                {
                    "FeatureB": [0.3],
                    "Target": ["Enrolled"],
                    "FeatureA": [3],
                }
            )
            dataset_cfg = {
                "data_source": {
                    "format": "parquet",
                    "split_mode": "predefined",
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                },
                "schema": {"outcome_column": "Target"},
            }

            with patch("src.data.loaders.uct_student_loader.pd.read_parquet", side_effect=[train_df, test_df]):
                loaded = load_uct_student_predefined_splits(dataset_cfg)

            self.assertListEqual(list(loaded["train"].columns), ["FeatureA", "FeatureB", "Target"])
            self.assertListEqual(list(loaded["test"].columns), ["FeatureA", "FeatureB", "Target"])
            self.assertTrue(bool(loaded["schema_report"]["schema_validation_passed"]))

    def test_predefined_parquet_loader_fails_on_unsafe_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_path = root / "train.parquet"
            test_path = root / "test.parquet"
            train_path.write_text("stub", encoding="utf-8")
            test_path.write_text("stub", encoding="utf-8")
            train_df = pd.DataFrame(
                {
                    "FeatureA": [1],
                    "Target": ["Graduate"],
                }
            )
            test_df = pd.DataFrame(
                {
                    "FeatureB": [2],
                    "Target": ["Enrolled"],
                }
            )
            dataset_cfg = {
                "data_source": {
                    "format": "parquet",
                    "split_mode": "predefined",
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                },
                "schema": {"outcome_column": "Target"},
            }

            with patch("src.data.loaders.uct_student_loader.pd.read_parquet", side_effect=[train_df, test_df]):
                with self.assertRaises(ValueError):
                    load_uct_student_predefined_splits(dataset_cfg)


if __name__ == "__main__":
    unittest.main()
