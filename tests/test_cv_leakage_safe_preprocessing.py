from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.models.train_eval import run_leakage_safe_stratified_cv


class CvLeakageSafePreprocessingTests(unittest.TestCase):
    def test_fold_level_preprocessing_uses_train_fold_only(self) -> None:
        train_df = pd.DataFrame(
            {
                "id": list(range(12)),
                "feature_num": [float(i) for i in range(12)],
                "feature_cat": ["a", "b", "c"] * 4,
                "target": [0, 1, 2] * 4,
            }
        )
        observed_calls: list[dict[str, int]] = []

        def fake_run_tabular_preprocessing(split_payload: dict[str, pd.DataFrame], _config: dict[str, object]):
            observed_calls.append(
                {
                    "train_rows": int(len(split_payload["train"])),
                    "valid_rows": int(len(split_payload["valid"])),
                    "test_rows": int(len(split_payload["test"])),
                }
            )
            self.assertEqual(len(split_payload["valid"]), len(split_payload["test"]))
            self.assertListEqual(
                split_payload["valid"]["id"].tolist(),
                split_payload["test"]["id"].tolist(),
            )
            self.assertLess(len(split_payload["train"]), len(train_df))
            self.assertFalse(set(split_payload["train"]["id"]).intersection(set(split_payload["valid"]["id"])))

            fold_train = split_payload["train"].reset_index(drop=True)
            fold_valid = split_payload["valid"].reset_index(drop=True)
            return type(
                "Artifacts",
                (),
                {
                    "X_train": fold_train[["feature_num"]].copy(),
                    "y_train": fold_train["target"].copy(),
                    "X_valid": pd.DataFrame(columns=["feature_num"]),
                    "y_valid": pd.Series(dtype=int),
                    "X_test": fold_valid[["feature_num"]].copy(),
                    "y_test": fold_valid["target"].copy(),
                    "metadata": {"train_only_outlier": {"enabled": False}},
                },
            )()

        def fake_apply_outlier_filter(X_train: pd.DataFrame, y_train: pd.Series, _config: dict[str, object]):
            self.assertEqual(len(X_train), len(y_train))
            return X_train, y_train, {"enabled": True, "reverted": False}

        def fake_apply_balancing(X_train: pd.DataFrame, y_train: pd.Series, _config: dict[str, object]):
            self.assertEqual(len(X_train), len(y_train))
            return X_train, y_train, {"enabled": True, "skipped": False}

        def fake_train_and_evaluate(**kwargs):
            self.assertEqual(len(kwargs["X_train"]), len(kwargs["y_train"]))
            self.assertEqual(len(kwargs["X_test"]), len(kwargs["y_test"]))
            return type(
                "TrainEvalResult",
                (),
                {
                    "metrics": {
                        "test_macro_f1": 0.5,
                        "test_accuracy": 0.5,
                        "test_balanced_accuracy": 0.5,
                        "test_macro_precision": 0.5,
                        "test_macro_recall": 0.5,
                    },
                    "artifacts": {
                        "per_class_metrics_test": {
                            "0": {"precision": 0.4, "recall": 0.4, "f1": 0.4, "support": 1.0},
                            "1": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1.0},
                            "2": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "support": 1.0},
                        }
                    },
                },
            )()

        with (
            patch("src.models.train_eval.run_tabular_preprocessing", side_effect=fake_run_tabular_preprocessing),
            patch("src.models.train_eval.apply_outlier_filter", side_effect=fake_apply_outlier_filter),
            patch("src.models.train_eval.apply_balancing", side_effect=fake_apply_balancing),
            patch("src.models.train_eval.train_and_evaluate", side_effect=fake_train_and_evaluate),
        ):
            result = run_leakage_safe_stratified_cv(
                model_name="decision_tree",
                params={},
                train_df=train_df,
                preprocess_config={"target_column": "target", "id_columns": ["id"]},
                outlier_config={"enabled": True},
                balancing_config={"enabled": True},
                cv_config={"n_splits": 3, "shuffle": True, "random_state": 42},
                eval_config={"seed": 42, "label_order": [0, 1, 2], "decision_rule": "argmax"},
            )

        self.assertEqual(len(observed_calls), 3)
        self.assertEqual(len(result["folds"]), 3)
        self.assertIn("cv_macro_f1", result["aggregate_metrics"])


if __name__ == "__main__":
    unittest.main()
