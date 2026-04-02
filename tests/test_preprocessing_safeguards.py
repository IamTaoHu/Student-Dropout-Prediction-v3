"""Tests for train-only preprocessing safeguards."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_experiment import _prepare_preprocessing_config
from src.preprocessing.balancing import apply_balancing
from src.preprocessing.outlier import apply_outlier_filter
from src.preprocessing.tabular_pipeline import _extract_features_target


class PreprocessingSafeguardTests(unittest.TestCase):
    def test_extract_features_target_drops_final_result_leakage_column(self) -> None:
        df = pd.DataFrame(
            {
                "id_student": [101, 102],
                "feature_1": [0.2, 0.8],
                "final_result": ["Pass", "Fail"],
                "target": [0, 1],
            }
        )
        X, y = _extract_features_target(df, target_column="target", id_columns=["id_student"])

        self.assertListEqual(y.tolist(), [0, 1])
        self.assertNotIn("id_student", X.columns)
        self.assertNotIn("target", X.columns)
        self.assertNotIn("final_result", X.columns)
        self.assertIn("feature_1", X.columns)

    def test_isolation_forest_filters_training_only_inputs(self) -> None:
        rng = np.random.default_rng(7)
        X_train = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"x{i}" for i in range(5)])
        y_train = pd.Series([0] * 90 + [1] * 30, name="target")
        X_f, y_f, meta = apply_outlier_filter(
            X_train,
            y_train,
            {"enabled": True, "method": "isolation_forest", "contamination": 0.1, "random_state": 7},
        )
        self.assertLess(len(X_f), len(X_train))
        self.assertEqual(len(X_f), len(y_f))
        self.assertEqual(meta["n_original"], len(X_train))
        self.assertEqual(meta["n_remaining"], len(X_f))

    def test_smote_balancing_resamples_train_data(self) -> None:
        try:
            import imblearn  # noqa: F401
        except Exception:
            self.skipTest("imbalanced-learn is not installed in this environment.")

        rng = np.random.default_rng(9)
        X_train = pd.DataFrame(rng.normal(size=(80, 4)), columns=[f"x{i}" for i in range(4)])
        y_train = pd.Series([0] * 60 + [1] * 20, name="target")
        X_b, y_b, meta = apply_balancing(
            X_train,
            y_train,
            {"enabled": True, "method": "smote", "random_state": 9, "k_neighbors": 3},
        )
        counts = y_b.value_counts().to_dict()
        self.assertEqual(counts.get(0), counts.get(1))
        self.assertEqual(len(X_b), len(y_b))
        self.assertIn("class_distribution_before", meta)
        self.assertIn("class_distribution_after", meta)

    def test_smote_balancing_fails_loudly_on_incompatible_features(self) -> None:
        try:
            import imblearn  # noqa: F401
        except Exception:
            self.skipTest("imbalanced-learn is not installed in this environment.")
        X_train = pd.DataFrame(
            {
                "x0": ["a"] * 6 + ["b"] * 6,
                "x1": ["u"] * 6 + ["v"] * 6,
            }
        )
        y_train = pd.Series([0] * 8 + [1] * 4, name="target")
        with self.assertRaisesRegex(ValueError, "Balancing failed during fit_resample"):
            _ = apply_balancing(
                X_train,
                y_train,
                {"enabled": True, "method": "smote", "random_state": 9, "k_neighbors": 3},
            )

    def test_dataset_level_forbidden_columns_are_merged_in_runner_config(self) -> None:
        preprocessing_cfg = _prepare_preprocessing_config(
            exp_cfg={"preprocessing": {"forbidden_feature_columns": ["source_grade"]}},
            dataset_cfg={"preprocessing": {"forbidden_feature_columns": ["assessment_score"]}},
            id_column="id_student",
            source_target_col="final_result",
        )
        forbidden = preprocessing_cfg["forbidden_feature_columns"]
        self.assertIn("final_result", forbidden)
        self.assertIn("source_grade", forbidden)
        self.assertIn("assessment_score", forbidden)


if __name__ == "__main__":
    unittest.main()
