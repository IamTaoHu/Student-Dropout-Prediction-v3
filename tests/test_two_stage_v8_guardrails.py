from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scripts.run_experiment import (
    _apply_global_balance_guard,
    _ensure_explainability_compatible_artifact_paths,
    _write_benchmark_failure_summary,
    align_feature_schema,
    validate_feature_schema,
    validate_and_sanitize_feature_matrix,
)
from src.data.feature_builders.uct_stage2_advanced_features import (
    build_stage2_prototype_distance_features,
    build_stage2_selective_interaction_split_data,
)


class TwoStageV8GuardrailTests(unittest.TestCase):
    def test_explainability_artifact_paths_use_best_model_fallback_and_validate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            best_model = root / "best_model.joblib"
            x_train = root / "X_train_preprocessed.csv"
            x_test = root / "X_test_preprocessed.csv"
            y_train = root / "y_train.csv"
            for path in (best_model, x_train, x_test, y_train):
                path.write_text("ok", encoding="utf-8")

            summary = {
                "artifact_paths": {
                    "best_model_copy": str(best_model),
                    "X_train_preprocessed": str(x_train),
                    "X_test_preprocessed": str(x_test),
                    "y_train": str(y_train),
                }
            }

            _ensure_explainability_compatible_artifact_paths(summary)
            self.assertEqual(summary["artifact_paths"]["best_model"], str(best_model))

    def test_failure_summary_is_written_when_no_candidates_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = _write_benchmark_failure_summary(
                Path(tmp_dir),
                experiment_id="exp_test",
                requested_models=["catboost", "svm"],
                model_results={
                    "catboost": {"error": "Training/evaluation failed: catboost broke"},
                    "svm": {"error": "Training/evaluation failed: svm broke"},
                },
            )
            self.assertTrue(Path(artifacts["benchmark_failure_summary_json"]).exists())
            self.assertTrue(Path(artifacts["benchmark_failure_summary_md"]).exists())

    def test_selective_interactions_only_create_allowlisted_available_features(self) -> None:
        df = pd.DataFrame(
            {
                "approved_1st_sem": [3, 0],
                "approved_2nd_sem": [4, 1],
                "enrolled_1st_sem": [5, 0],
                "enrolled_2nd_sem": [5, 2],
                "grade_1st_sem": [12.0, 10.0],
                "grade_2nd_sem": [13.0, 11.0],
                "target": [1, 0],
            }
        )
        splits, report = build_stage2_selective_interaction_split_data(
            {"train": df, "valid": df.iloc[[0]].copy(), "test": df.iloc[[1]].copy()},
            feature_cfg={
                "enabled": True,
                "feature_allowlist": ["sem1_approval_rate", "grade_delta", "completion_balance", "load_pressure_sem2"],
            },
        )

        self.assertEqual(
            report["created_features"],
            ["sem1_approval_rate", "grade_delta", "completion_balance", "load_pressure_sem2"],
        )
        train_df = splits["train"]
        self.assertTrue(np.isfinite(train_df.drop(columns=["target"]).to_numpy(dtype=float)).all())
        self.assertAlmostEqual(float(train_df.loc[1, "sem1_approval_rate"]), 0.0)

    def test_selective_interactions_skip_existing_feature_names(self) -> None:
        df = pd.DataFrame(
            {
                "approved_1st_sem": [3],
                "approved_2nd_sem": [4],
                "enrolled_1st_sem": [5],
                "enrolled_2nd_sem": [5],
                "approval_rate_delta": [0.25],
                "target": [1],
            }
        )
        splits, report = build_stage2_selective_interaction_split_data(
            {"train": df, "valid": df.copy(), "test": df.copy()},
            feature_cfg={"enabled": True, "feature_allowlist": ["approval_rate_delta"]},
        )
        self.assertEqual(report["created_features"], [])
        self.assertEqual(report["skipped_existing_features"], ["approval_rate_delta"])
        self.assertEqual(list(splits["train"].columns), ["target"])

    def test_align_feature_schema_preserves_reference_order(self) -> None:
        reference = pd.DataFrame(columns=["a", "b", "c"])
        target = pd.DataFrame({"c": [3.0], "a": [1.0], "d": [9.0]})
        aligned = align_feature_schema(reference, target, fill_value=0.0)
        self.assertEqual(list(aligned.columns), ["a", "b", "c"])
        self.assertEqual(float(aligned.loc[0, "b"]), 0.0)
        validate_feature_schema(reference, aligned, context="unit_test")

    def test_robust_prototypes_are_finite_and_fold_safe(self) -> None:
        X_train = pd.DataFrame(
            {
                "f1": [0.0, 0.2, 1.0, 1.2],
                "f2": [1.0, np.nan, 0.0, 0.1],
            }
        )
        y_train = pd.Series([1, 1, 0, 0], name="target")
        X_valid = pd.DataFrame({"f1": [0.1, 1.1], "f2": [1.1, np.nan]})
        X_test = pd.DataFrame({"f1": [0.3, 0.9], "f2": [0.8, 0.2]})

        splits, report = build_stage2_prototype_distance_features(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            X_test=X_test,
            feature_cfg={
                "enabled": True,
                "add_distance_features": True,
                "add_ratio_features": True,
                "add_margin_feature": True,
                "eps": 1.0e-8,
                "on_failure": "disable_and_continue",
            },
            enrolled_positive_label=1,
        )

        self.assertTrue(report["enabled"])
        self.assertEqual(
            list(splits["train"].columns),
            ["dist_to_enrolled_proto", "dist_to_graduate_proto", "proto_margin", "proto_ratio"],
        )
        self.assertTrue(np.isfinite(splits["train"].to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(splits["valid"].to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(splits["test"].to_numpy(dtype=float)).all())

    def test_validate_and_sanitize_feature_matrix_imputes_non_finite_values(self) -> None:
        X_train = pd.DataFrame({"a": [1.0, np.nan], "b": [np.inf, 3.0]})
        X_valid = pd.DataFrame({"a": [2.0], "b": [np.nan]})
        X_test = pd.DataFrame({"a": [np.nan], "b": [4.0]})
        extra = {"valid_full": pd.DataFrame({"a": [np.nan], "b": [np.inf]})}

        sanitized, report = validate_and_sanitize_feature_matrix(
            X_train,
            X_valid,
            X_test,
            model_name="svm",
            feature_stage="stage2_test",
            sanitation_cfg={
                "enabled": True,
                "replace_inf": True,
                "impute_missing": True,
                "fail_if_non_finite_after_impute": True,
                "strategy": "median",
            },
            extra_frames=extra,
        )

        self.assertTrue(report["imputation_applied"])
        for frame in sanitized.values():
            self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())

    def test_global_balance_guard_penalizes_graduate_collapse(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "model": "catboost",
                    "test_macro_f1": 0.71,
                    "test_balanced_accuracy": 0.73,
                    "test_accuracy": 0.76,
                    "test_f1_enrolled": 0.50,
                    "test_f1_graduate": 0.86,
                    "test_f1_dropout": 0.79,
                },
                {
                    "model": "svm",
                    "test_macro_f1": 0.705,
                    "test_balanced_accuracy": 0.72,
                    "test_accuracy": 0.75,
                    "test_f1_enrolled": 0.54,
                    "test_f1_graduate": 0.80,
                    "test_f1_dropout": 0.78,
                },
            ]
        )

        guarded, report = _apply_global_balance_guard(
            leaderboard,
            guard_cfg={
                "enabled": True,
                "reference_source": "baseline_stage2",
                "max_graduate_f1_drop": 0.03,
                "min_macro_f1": None,
                "min_graduate_f1": None,
                "penalty_weight": 0.5,
                "fallback_to_plain_macro_f1_if_no_candidate_passes": True,
            },
        )

        self.assertFalse(report["fallback_used"])
        self.assertEqual(str(guarded.iloc[0]["model"]), "catboost")
        svm_decision = next(row for row in report["candidate_decisions"] if row["model"] == "svm")
        self.assertFalse(svm_decision["guard_pass"])


if __name__ == "__main__":
    unittest.main()
