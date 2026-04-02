"""Tests for normalized explainability report status contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.reporting.explanation_report import save_explanation_report


class ExplanationReportTests(unittest.TestCase):
    def test_methods_are_normalized_to_consistent_status_shape(self) -> None:
        artifacts = {
            "shap": {
                "status": "ok",
                "global_importance": [{"feature": "f1", "mean_abs_shap": 0.5}],
                "local_explanations": [{"instance_index": 0, "top_features": [{"feature": "f1", "shap_value": 0.2}]}],
            },
            "lime": {
                "status": "skipped",
                "reason": "dependency_missing: lime",
                "results": [],
            },
            "aime": {
                "global_importance": [{"feature": "f1", "importance": 0.3}],
                "per_class_importance": [{"class": 0, "feature": "f1", "importance": 0.3}],
                "local_importance": [{"instance_index": 0, "feature": "f1", "importance": 0.1}],
                "representative_instances": [{"instance_index": 0}],
                "similarity_plot_path": "aime_similarity.png",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report_path = save_explanation_report(artifacts, output_dir)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

            for method in ("shap", "lime", "aime"):
                self.assertIn("status", payload[method])
                self.assertIn("reason", payload[method])
                self.assertIn("error_message", payload[method])
                self.assertIn("artifacts", payload[method])

            self.assertEqual(payload["shap"]["status"], "generated")
            self.assertEqual(payload["lime"]["status"], "skipped")
            self.assertEqual(payload["lime"]["reason"], "dependency_missing: lime")
            self.assertEqual(payload["aime"]["status"], "generated")

            shap_files = payload["shap"]["artifacts"].get("saved_files", [])
            self.assertTrue(any(path.endswith("shap_global_importance.csv") for path in shap_files))
            self.assertTrue(any(path.endswith("shap_local_importance.csv") for path in shap_files))


if __name__ == "__main__":
    unittest.main()

