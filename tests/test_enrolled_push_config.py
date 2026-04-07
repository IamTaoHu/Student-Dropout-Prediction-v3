"""Regression tests for enrolled-push UCT benchmark config and ranking."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from scripts.run_experiment import _resolve_model_selection_config, _sort_leaderboard_with_tiebreak


class EnrolledPushConfigTests(unittest.TestCase):
    def test_new_bundled_config_exists_and_matches_expected_bundle_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_enrolled_push_v1.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_enrolled_push_v1")
        self.assertEqual(payload["experiment"]["dataset_config"], "configs/datasets/uct_student.yaml")
        self.assertEqual(payload["experiment"]["target_formulation"], "three_class")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_enrolled_push_v1")
        self.assertEqual(payload["outputs"]["runtime_artifact_format"], "csv")
        self.assertTrue(bool(payload["outputs"]["mirror_benchmark_outputs_to_runtime"]))
        self.assertListEqual(
            payload["models"]["candidates"],
            ["svm", "lightgbm", "catboost", "xgboost", "random_forest", "gradient_boosting", "mlp", "decision_tree"],
        )
        self.assertEqual(payload["selection"]["primary"], "macro_f1")
        self.assertListEqual(
            payload["selection"]["tie_breakers"],
            ["enrolled_f1", "enrolled_recall", "balanced_accuracy"],
        )
        self.assertEqual(payload["inference"]["multiclass_decision"]["strategy"], "enrolled_push")

    def test_selection_config_accepts_ordered_tie_breakers(self) -> None:
        resolved = _resolve_model_selection_config(
            {
                "selection": {
                    "primary": "macro_f1",
                    "tie_breakers": ["enrolled_f1", "enrolled_recall", "balanced_accuracy"],
                }
            }
        )
        self.assertListEqual(
            resolved["ranking_metrics"],
            ["macro_f1", "enrolled_f1", "enrolled_recall", "balanced_accuracy"],
        )

    def test_selection_config_accepts_enrolled_first_priority(self) -> None:
        resolved = _resolve_model_selection_config(
            {
                "selection": {
                    "primary": "enrolled_f1",
                    "tie_breakers": ["macro_f1", "enrolled_recall", "balanced_accuracy", "accuracy"],
                }
            }
        )
        self.assertListEqual(
            resolved["ranking_metrics"],
            ["enrolled_f1", "macro_f1", "enrolled_recall", "balanced_accuracy", "accuracy"],
        )

    def test_leaderboard_sort_uses_enrolled_alias_metrics(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {"model": "svm", "test_macro_f1": 0.70, "test_f1_enrolled": 0.62, "test_recall_enrolled": 0.66, "test_balanced_accuracy": 0.71},
                {"model": "xgboost", "test_macro_f1": 0.70, "test_f1_enrolled": 0.64, "test_recall_enrolled": 0.61, "test_balanced_accuracy": 0.72},
            ]
        )
        ranked, best_model, ranking_columns = _sort_leaderboard_with_tiebreak(
            leaderboard_df=leaderboard,
            selection_cfg={
                "ranking_metrics": ["macro_f1", "enrolled_f1", "enrolled_recall", "balanced_accuracy"],
            },
            source="test",
        )
        self.assertEqual(best_model, "xgboost")
        self.assertListEqual(
            ranking_columns,
            ["test_macro_f1", "test_f1_enrolled", "test_recall_enrolled", "test_balanced_accuracy"],
        )
        self.assertEqual(str(ranked.iloc[0]["model"]), "xgboost")

    def test_v2_config_exists_and_uses_enrolled_first_selection_and_sweep(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_enrolled_push_v2.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_enrolled_push_v2")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_enrolled_push_v2")
        self.assertEqual(payload["selection"]["primary"], "enrolled_f1")
        self.assertListEqual(
            payload["selection"]["tie_breakers"],
            ["macro_f1", "enrolled_recall", "balanced_accuracy", "accuracy"],
        )
        self.assertListEqual(
            payload["models"]["class_weight"]["enrolled_weight_sweep"],
            [1.0, 1.15, 1.35, 1.6, 1.85],
        )
        self.assertTrue(bool(payload["feature_engineering"]["dataset_features"]["derive_enrolled_focus_features"]))
        self.assertEqual(
            payload["inference"]["multiclass_decision"]["per_model"]["xgboost"]["strategy"],
            "enrolled_push",
        )

    def test_v3_config_exists_and_uses_locked_weight_and_conservative_policy(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_enrolled_push_v3.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_enrolled_push_v3")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_enrolled_push_v3")
        self.assertEqual(payload["training"]["class_weight"]["mode"], "explicit")
        self.assertEqual(
            payload["training"]["class_weight"]["values"],
            {"Dropout": 1.0, "Enrolled": 1.35, "Graduate": 1.0},
        )
        self.assertEqual(payload["selection"]["primary"], "enrolled_f1")
        self.assertListEqual(
            payload["selection"]["tie_breakers"],
            ["macro_f1", "enrolled_recall", "balanced_accuracy", "accuracy"],
        )
        self.assertEqual(payload["models"]["tuning"]["objective_metric"], "enrolled_f1")
        self.assertEqual(
            payload["models"]["tuning"]["per_model_n_trials"],
            {
                "svm": 15,
                "lightgbm": 40,
                "catboost": 40,
                "xgboost": 30,
                "random_forest": 12,
                "gradient_boosting": 12,
                "mlp": 15,
                "decision_tree": 8,
            },
        )
        self.assertTrue(bool(payload["feature_engineering"]["dataset_features"]["derive_enrolled_focus_features"]))
        self.assertFalse(bool(payload["inference"]["multiclass_decision"]["per_model"]["svm"]["enabled"]))
        self.assertFalse(bool(payload["inference"]["multiclass_decision"]["per_model"]["xgboost"]["auto_tune"]["enabled"]))


if __name__ == "__main__":
    unittest.main()
