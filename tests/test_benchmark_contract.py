"""Tests for benchmark summary contract validation and compact persistence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.reporting.benchmark_contract import BENCHMARK_SUMMARY_VERSION, validate_benchmark_summary_for_explainability
from src.reporting.benchmark_summary import save_benchmark_summary


class BenchmarkContractTests(unittest.TestCase):
    def test_validate_benchmark_summary_contract_success(self) -> None:
        summary = {
            "benchmark_summary_version": BENCHMARK_SUMMARY_VERSION,
            "best_model": "decision_tree",
            "artifact_paths": {
                "best_model": "results/model.joblib",
                "X_train_preprocessed": "results/X_train.parquet",
                "X_test_preprocessed": "results/X_test.parquet",
                "y_train": "results/y_train.parquet",
            },
        }
        validated = validate_benchmark_summary_for_explainability(summary)
        self.assertEqual(validated["best_model_name"], "decision_tree")
        self.assertEqual(validated["benchmark_summary_version"], BENCHMARK_SUMMARY_VERSION)

    def test_validate_benchmark_summary_contract_missing_keys_fails_cleanly(self) -> None:
        summary = {
            "best_model": "decision_tree",
            "artifact_paths": {
                "best_model": "results/model.joblib",
                "X_train_preprocessed": "results/X_train.parquet",
            },
        }
        with self.assertRaisesRegex(ValueError, "Missing artifact_paths keys"):
            validate_benchmark_summary_for_explainability(summary)

    def test_compact_benchmark_summary_omits_heavy_arrays(self) -> None:
        summary = {
            "experiment_id": "compact_test",
            "leaderboard": [{"model": "decision_tree", "test_macro_f1": 0.8}],
            "model_results": {
                "decision_tree": {
                    "metrics": {"test_macro_f1": 0.8},
                    "artifacts": {
                        "y_pred_test": [0, 1, 0, 1],
                        "y_proba_test": [[0.7, 0.3], [0.2, 0.8]],
                        "confusion_matrix": [[2, 0], [0, 2]],
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            save_benchmark_summary(summary, output_dir, compact=True)
            payload = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
            artifacts = payload["model_results"]["decision_tree"]["artifacts"]
            self.assertNotIn("y_pred_test", artifacts)
            self.assertNotIn("y_proba_test", artifacts)
            self.assertIn("compact_omitted_artifacts", payload["model_results"]["decision_tree"])
            self.assertEqual(payload["summary_mode"], "compact")


if __name__ == "__main__":
    unittest.main()

