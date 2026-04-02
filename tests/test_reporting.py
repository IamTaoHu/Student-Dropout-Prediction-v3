"""Tests for benchmark summary output generation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.reporting.benchmark_summary import save_benchmark_summary


class ReportingTests(unittest.TestCase):
    def test_benchmark_summary_writes_expected_files(self) -> None:
        summary = {
            "experiment_id": "unit_test_exp",
            "primary_metric": "test_macro_f1",
            "best_model": "decision_tree",
            "leaderboard": [{"model": "decision_tree", "test_macro_f1": 0.5}],
            "model_results": {
                "decision_tree": {
                    "metrics": {"test_macro_f1": 0.5},
                    "artifacts": {"confusion_matrix": [[8, 1], [2, 5]]},
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            json_path = save_benchmark_summary(summary, out_dir)
            self.assertTrue(json_path.exists())
            self.assertTrue((out_dir / "leaderboard.csv").exists())
            self.assertTrue((out_dir / "benchmark_summary.md").exists())
            self.assertTrue((out_dir / "confusion_matrix_decision_tree.png").exists())


if __name__ == "__main__":
    unittest.main()
