"""Synthetic smoke test for benchmark execution."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.run_experiment import _map_target, run_experiment


class SmokeBenchmarkTests(unittest.TestCase):
    def test_target_mapping_from_final_result_then_source_is_dropped(self) -> None:
        feature_df = pd.DataFrame(
            {
                "id_student": [1, 2, 3, 4],
                "feature_num": [0.1, 0.2, 0.3, 0.4],
                "final_result": ["Pass", "Fail", "Withdrawn", "Distinction"],
            }
        )

        mapped = feature_df.copy()
        mapped["target"] = _map_target(
            mapped,
            dataset_name="oulad",
            source_target_col="final_result",
            formulation="binary",
            mapping=None,
        )
        mapped = mapped.drop(columns=["final_result"], errors="ignore")

        self.assertListEqual(mapped["target"].tolist(), [0, 1, 1, 0])
        self.assertNotIn("final_result", mapped.columns)

    def test_tiny_uct_benchmark_runs_end_to_end(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_csv = tmp_dir / "uct_student.csv"
            results_dir = tmp_dir / "results"

            rows = 120
            df = pd.DataFrame(
                {
                    "student_id": np.arange(rows),
                    "feature_num": rng.normal(size=rows),
                    "feature_cat": np.where(rng.random(rows) > 0.5, "A", "B"),
                    "academic_outcome": np.where(rng.random(rows) > 0.35, "persisted", "dropout"),
                }
            )
            df.to_csv(raw_csv, index=False)

            dataset_cfg = {
                "dataset": {"name": "uct_student", "source_format": "csv"},
                "paths": {"raw_file": str(raw_csv)},
                "source": {"delimiter": ",", "encoding": "utf-8"},
                "schema": {"entity_id": "student_id", "outcome_column": "academic_outcome"},
                "features": {"derive_safe_features": True},
                "target_mappings": {"binary": {"dropout": 1, "persisted": 0}},
            }
            dataset_cfg_path = tmp_dir / "dataset.yaml"
            dataset_cfg_path.write_text(yaml.safe_dump(dataset_cfg), encoding="utf-8")

            exp_cfg = {
                "experiment": {
                    "id": "smoke_exp",
                    "seed": 42,
                    "dataset_config": str(dataset_cfg_path),
                    "target_formulation": "binary",
                },
                "splits": {"test_size": 0.2, "validation_size": 0.2, "stratify_column": "target"},
                "preprocessing": {
                    "imputation": "median_mode",
                    "encoding": "onehot",
                    "scaling": "standard",
                    "outlier": {"enabled": False},
                    "balancing": {"enabled": False},
                },
                "models": {
                    "candidates": ["decision_tree"],
                    "tuning": {"backend": "none", "n_trials": 0},
                },
                "metrics": {"primary": "macro_f1"},
                "outputs": {"results_dir": str(results_dir)},
            }
            exp_cfg_path = tmp_dir / "experiment.yaml"
            exp_cfg_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")

            summary = run_experiment(exp_cfg_path)
            self.assertEqual(summary["experiment_id"], "smoke_exp")
            self.assertEqual(summary["best_model"], "decision_tree")
            self.assertTrue((results_dir / "benchmark_summary.json").exists())
            self.assertIn("artifact_paths", summary)
            best_payload = summary["model_results"]["decision_tree"]
            self.assertIn("test_macro_f1", best_payload["metrics"])
            self.assertIn("per_class_metrics_test", best_payload["artifacts"])
            self.assertTrue(Path(summary["artifact_paths"]["best_model"]).exists())
            runtime_metadata_path = Path(summary["artifact_paths"]["runtime_metadata"])
            runtime_metadata = yaml.safe_load(runtime_metadata_path.read_text(encoding="utf-8"))
            output_feature_names = runtime_metadata.get("feature_names", [])
            self.assertFalse(any(name == "final_result" or name.startswith("final_result_") for name in output_feature_names))

    def test_runner_drops_source_target_column_before_preprocessing(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")
        rng = np.random.default_rng(456)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            results_dir = tmp_dir / "results"

            dataset_cfg = {"dataset": {"name": "oulad"}, "schema": {"outcome_column": "final_result"}}
            dataset_cfg_path = tmp_dir / "dataset.yaml"
            dataset_cfg_path.write_text(yaml.safe_dump(dataset_cfg), encoding="utf-8")

            exp_cfg = {
                "experiment": {
                    "id": "smoke_exp_oulad_guard",
                    "seed": 42,
                    "dataset_config": str(dataset_cfg_path),
                    "target_formulation": "binary",
                },
                "splits": {"test_size": 0.2, "validation_size": 0.2, "stratify_column": "target"},
                "preprocessing": {
                    "imputation": "median_mode",
                    "encoding": "onehot",
                    "scaling": "standard",
                    "outlier": {"enabled": False},
                    "balancing": {"enabled": False},
                },
                "models": {
                    "candidates": ["decision_tree"],
                    "tuning": {"backend": "none", "n_trials": 0},
                },
                "metrics": {"primary": "macro_f1"},
                "outputs": {"results_dir": str(results_dir)},
            }
            exp_cfg_path = tmp_dir / "experiment.yaml"
            exp_cfg_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")

            rows = 120
            leaked_feature_df = pd.DataFrame(
                {
                    "id_student": np.arange(rows),
                    "feature_num": rng.normal(size=rows),
                    "feature_cat": np.where(rng.random(rows) > 0.5, "A", "B"),
                    "final_result": np.where(rng.random(rows) > 0.4, "Pass", "Fail"),
                }
            )

            with patch(
                "scripts.run_experiment._build_feature_table",
                return_value=(leaked_feature_df, "oulad", "id_student", "final_result"),
            ):
                summary = run_experiment(exp_cfg_path)

            runtime_metadata_path = Path(summary["artifact_paths"]["runtime_metadata"])
            runtime_metadata = yaml.safe_load(runtime_metadata_path.read_text(encoding="utf-8"))
            output_feature_names = runtime_metadata.get("feature_names", [])
            self.assertTrue(output_feature_names)
            self.assertFalse(any(name == "final_result" or name.startswith("final_result_") for name in output_feature_names))

    def test_tiny_uct_three_class_summary_has_named_metrics_and_runtime_mirror(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")
        rng = np.random.default_rng(789)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_csv = tmp_dir / "uct_student_3class.csv"
            results_dir = tmp_dir / "results"

            rows = 180
            outcomes = np.array(["Dropout", "Enrolled", "Graduate"])
            df = pd.DataFrame(
                {
                    "student_id": np.arange(rows),
                    "feature_num": rng.normal(size=rows),
                    "feature_cat": np.where(rng.random(rows) > 0.5, "A", "B"),
                    "Target": outcomes[np.arange(rows) % 3],
                }
            )
            df.to_csv(raw_csv, sep=";", index=False)

            dataset_cfg = {
                "dataset": {"name": "uct_student", "source_format": "csv"},
                "paths": {"raw_file": str(raw_csv)},
                "source": {"delimiter": ";", "encoding": "utf-8"},
                "schema": {"entity_id": "student_id", "outcome_column": "Target"},
                "features": {"derive_safe_features": True},
                "target_mappings": {
                    "three_class": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                },
            }
            dataset_cfg_path = tmp_dir / "dataset_3class.yaml"
            dataset_cfg_path.write_text(yaml.safe_dump(dataset_cfg), encoding="utf-8")

            exp_cfg = {
                "experiment": {
                    "id": "smoke_exp_uct_3class",
                    "seed": 42,
                    "dataset_config": str(dataset_cfg_path),
                    "target_formulation": "three_class",
                },
                "target": {"class_order": ["Dropout", "Enrolled", "Graduate"]},
                "target_mapping_override": {
                    "three_class": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                },
                "splits": {"test_size": 0.2, "validation_size": 0.2, "stratify_column": "target"},
                "preprocessing": {
                    "drop_missing_rows": True,
                    "missing_values": {"drop_rows": True},
                    "imputation": "median_mode",
                    "encoding": "onehot",
                    "scaling": "standard",
                    "outlier": {"enabled": False},
                    "balancing": {"enabled": False},
                },
                "models": {
                    "candidates": ["decision_tree"],
                    "tuning": {"backend": "none", "n_trials": 0},
                },
                "metrics": {"primary": "macro_f1"},
                "outputs": {
                    "results_dir": str(results_dir),
                    "runtime_artifact_format": "csv",
                    "mirror_benchmark_outputs_to_runtime": True,
                },
            }
            exp_cfg_path = tmp_dir / "experiment_3class.yaml"
            exp_cfg_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")

            summary = run_experiment(exp_cfg_path)
            best_metrics = summary["model_results"]["decision_tree"]["metrics"]
            for metric_key in (
                "test_macro_f1",
                "test_accuracy",
                "test_weighted_f1",
                "test_balanced_accuracy",
                "precision_dropout",
                "recall_dropout",
                "f1_dropout",
                "precision_enrolled",
                "recall_enrolled",
                "f1_enrolled",
                "precision_graduate",
                "recall_graduate",
                "f1_graduate",
            ):
                self.assertIn(metric_key, best_metrics)
            self.assertEqual(summary.get("seed"), 42)
            self.assertIn("split_sizes", summary)
            self.assertIn("class_distribution_train_before_outlier", summary["preprocessing"])
            self.assertIn("class_distribution_train_after_outlier", summary["preprocessing"])

            runtime_dir = results_dir / "runtime_artifacts"
            self.assertTrue((runtime_dir / "benchmark_summary.json").exists())
            self.assertTrue((runtime_dir / "benchmark_summary.md").exists())
            self.assertTrue((runtime_dir / "leaderboard.csv").exists())
            self.assertTrue((runtime_dir / "metrics.json").exists())
            self.assertTrue((runtime_dir / "predictions.csv").exists())
            self.assertTrue((runtime_dir / "artifact_manifest.json").exists())
            self.assertTrue((runtime_dir / "confusion_matrix_decision_tree.png").exists())

    def test_multi_model_single_folder_outputs_are_aggregated(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")
        rng = np.random.default_rng(999)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_csv = tmp_dir / "uct_student_agg.csv"
            results_dir = tmp_dir / "results_single_bundle"

            rows = 150
            outcomes = np.array(["Dropout", "Enrolled", "Graduate"])
            df = pd.DataFrame(
                {
                    "student_id": np.arange(rows),
                    "feature_num": rng.normal(size=rows),
                    "feature_cat": np.where(rng.random(rows) > 0.5, "A", "B"),
                    "Target": outcomes[np.arange(rows) % 3],
                }
            )
            df.to_csv(raw_csv, sep=";", index=False)

            dataset_cfg = {
                "dataset": {"name": "uct_student", "source_format": "csv"},
                "paths": {"raw_file": str(raw_csv)},
                "source": {"delimiter": ";", "encoding": "utf-8"},
                "schema": {"entity_id": "student_id", "outcome_column": "Target"},
                "features": {"derive_safe_features": True},
                "target_mappings": {
                    "three_class": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                },
            }
            dataset_cfg_path = tmp_dir / "dataset_agg.yaml"
            dataset_cfg_path.write_text(yaml.safe_dump(dataset_cfg), encoding="utf-8")

            exp_cfg = {
                "experiment": {
                    "id": "smoke_exp_uct_agg",
                    "seed": 42,
                    "dataset_config": str(dataset_cfg_path),
                    "target_formulation": "three_class",
                },
                "target": {"class_order": ["Dropout", "Enrolled", "Graduate"]},
                "target_mapping_override": {
                    "three_class": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                },
                "splits": {"test_size": 0.2, "validation_size": 0.2, "stratify_column": "target"},
                "preprocessing": {
                    "drop_missing_rows": True,
                    "missing_values": {"drop_rows": True},
                    "imputation": "median_mode",
                    "encoding": "onehot",
                    "scaling": "standard",
                    "outlier": {"enabled": False},
                    "balancing": {"enabled": False},
                },
                "models": {
                    "candidates": ["decision_tree", "gradient_boosting"],
                    "tuning": {"backend": "none", "n_trials": 0},
                },
                "metrics": {"primary": "macro_f1"},
                "outputs": {
                    "results_dir": str(results_dir),
                    "runtime_artifact_format": "csv",
                    "mirror_benchmark_outputs_to_runtime": True,
                },
            }
            exp_cfg_path = tmp_dir / "experiment_agg.yaml"
            exp_cfg_path.write_text(yaml.safe_dump(exp_cfg), encoding="utf-8")

            summary = run_experiment(exp_cfg_path)
            self.assertTrue((results_dir / "benchmark_summary.json").exists())
            self.assertTrue((results_dir / "benchmark_summary.md").exists())
            self.assertTrue((results_dir / "leaderboard.csv").exists())
            self.assertTrue((results_dir / "metrics.json").exists())
            self.assertTrue((results_dir / "predictions.csv").exists())
            self.assertTrue((results_dir / "artifact_manifest.json").exists())

            leaderboard = pd.read_csv(results_dir / "leaderboard.csv")
            self.assertSetEqual(set(leaderboard["model"].astype(str).tolist()), {"decision_tree", "gradient_boosting"})
            self.assertEqual(len(summary.get("leaderboard", [])), 2)
            self.assertTrue((results_dir / "confusion_matrix_decision_tree.png").exists())
            self.assertTrue((results_dir / "confusion_matrix_gradient_boosting.png").exists())


if __name__ == "__main__":
    unittest.main()
