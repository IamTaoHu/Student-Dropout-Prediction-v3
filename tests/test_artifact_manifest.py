"""Regression tests for artifact manifest contract and merge behavior."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.reporting.artifact_manifest import load_or_initialize_manifest, update_artifact_manifest


ALLOWED_STATUSES = {"generated", "created", "inherited", "skipped", "unavailable", "failed"}


def _read_manifest(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "artifact_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _assert_status_contract(testcase: unittest.TestCase, payload: dict[str, object]) -> None:
    for section in ("mandatory", "optional"):
        entries = payload.get(section, {})
        testcase.assertIsInstance(entries, dict)
        for name, entry in entries.items():
            testcase.assertIsInstance(entry, dict, msg=f"{section}.{name} must be a dict")
            testcase.assertIn("status", entry, msg=f"{section}.{name} missing status")
            testcase.assertIn(entry["status"], ALLOWED_STATUSES, msg=f"{section}.{name} has invalid status")


class ArtifactManifestTests(unittest.TestCase):
    def test_benchmark_only_manifest_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            update_artifact_manifest(
                output_dir=output_dir,
                mandatory_updates={
                    "benchmark_summary_json": {"status": "generated", "path": output_dir / "benchmark_summary.json"},
                    "metrics_json": {"status": "failed", "path": output_dir / "metrics.json", "reason": "missing_expected_output"},
                    "predictions_csv": {"status": "unavailable", "reason": "not_produced_in_test"},
                    "leaderboard_csv": {"status": "skipped", "reason": "not_generated_by_reporting"},
                },
                optional_updates={
                    "shap_outputs": {"status": "skipped", "reason": "explainability_not_run_yet"},
                    "explainability_dir": {
                        "status": "skipped",
                        "path": output_dir / "explainability",
                        "reason": "explainability_not_run_yet",
                    },
                },
                metadata_updates={
                    "experiment_id": "unit_manifest_benchmark",
                    "dataset_name": "uct_student",
                    "manifest_scope": "benchmark",
                },
            )

            payload = _read_manifest(output_dir)
            self.assertEqual(payload["contract_version"], "1.0")
            self.assertEqual(payload["manifest_path"], str(output_dir / "artifact_manifest.json"))
            self.assertIn("updated_at", payload)
            self.assertIn("metadata", payload)
            self.assertIn("mandatory", payload)
            self.assertIn("optional", payload)
            self.assertIn("benchmark_summary_json", payload["mandatory"])
            self.assertIn("shap_outputs", payload["optional"])
            _assert_status_contract(self, payload)

    def test_explainability_merge_preserves_benchmark_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            update_artifact_manifest(
                output_dir=output_dir,
                mandatory_updates={
                    "benchmark_summary_json": {"status": "generated", "path": output_dir / "benchmark_summary.json"},
                    "metrics_json": {"status": "generated", "path": output_dir / "metrics.json"},
                },
                optional_updates={
                    "shap_outputs": {"status": "skipped", "reason": "explainability_not_run_yet"},
                    "shap_beeswarm_png": {"status": "skipped", "reason": "best_model_not_shap_compatible"},
                },
                metadata_updates={"experiment_id": "unit_manifest_merge", "manifest_scope": "benchmark"},
            )
            benchmark_payload = _read_manifest(output_dir)

            update_artifact_manifest(
                output_dir=output_dir,
                optional_updates={
                    "explainability_dir": {"status": "generated", "path": output_dir / "explainability"},
                    "explainability_report_json": {
                        "status": "generated",
                        "path": output_dir / "explainability" / "explanation_report.json",
                    },
                    "shap_outputs": {
                        "status": "failed",
                        "reason": "dependency_missing: shap",
                    },
                    "lime_outputs": {"status": "generated", "paths": [output_dir / "explainability" / "lime_local_importance.csv"]},
                    "aime_outputs": {"status": "skipped", "reason": "not_applicable_for_model_type"},
                },
                metadata_updates={"explainability_last_model": "logistic_regression", "manifest_scope": "benchmark+explainability"},
            )
            merged_payload = _read_manifest(output_dir)

            self.assertEqual(
                merged_payload["mandatory"]["benchmark_summary_json"],
                benchmark_payload["mandatory"]["benchmark_summary_json"],
            )
            self.assertEqual(
                merged_payload["mandatory"]["metrics_json"],
                benchmark_payload["mandatory"]["metrics_json"],
            )
            self.assertEqual(merged_payload["metadata"]["experiment_id"], "unit_manifest_merge")
            self.assertEqual(merged_payload["metadata"]["explainability_last_model"], "logistic_regression")
            self.assertEqual(merged_payload["metadata"]["manifest_scope"], "benchmark+explainability")
            self.assertEqual(merged_payload["optional"]["explainability_dir"]["status"], "generated")
            self.assertEqual(merged_payload["optional"]["shap_beeswarm_png"]["status"], "skipped")
            _assert_status_contract(self, merged_payload)

    def test_optional_artifact_skip_reason_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            update_artifact_manifest(
                output_dir=output_dir,
                optional_updates={
                    "shap_outputs": {"status": "skipped", "reason": "best_model_not_shap_compatible"},
                    "lime_outputs": {"status": "failed", "reason": "dependency_missing: lime"},
                    "aime_outputs": {"status": "skipped", "reason": "not_applicable_for_model_type"},
                    "explainability_dir": {"status": "skipped", "reason": "explainability_not_run_yet"},
                    "shap_beeswarm_png": {"status": "failed", "reason": "generation_error"},
                },
            )
            payload = _read_manifest(output_dir)

            accepted_reason_values = {
                "best_model_not_shap_compatible",
                "explainability_not_run_yet",
                "not_applicable_for_model_type",
                "generation_error",
            }
            for name, entry in payload["optional"].items():
                self.assertIn(entry["status"], ALLOWED_STATUSES, msg=f"{name} has invalid status")
                self.assertIn("reason", entry, msg=f"{name} must include reason when no artifact exists")
                reason = entry["reason"]
                self.assertIsInstance(reason, str)
                self.assertTrue(reason.strip(), msg=f"{name} reason must be non-empty")
                self.assertTrue(
                    reason in accepted_reason_values or reason.startswith("dependency_missing:"),
                    msg=f"{name} reason '{reason}' is not a recognized form",
                )

    def test_backward_compatible_minimal_manifest_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest_path = output_dir / "artifact_manifest.json"
            legacy_payload = {
                "contract_version": "0.9",
                "mandatory": {"benchmark_summary_json": {"status": "generated", "path": "legacy.json"}},
            }
            manifest_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

            payload, _ = load_or_initialize_manifest(output_dir)
            self.assertIn("mandatory", payload)
            self.assertIn("optional", payload)
            self.assertIn("metadata", payload)
            self.assertEqual(payload["mandatory"]["benchmark_summary_json"]["status"], "generated")
            self.assertEqual(payload["optional"], {})
            self.assertEqual(payload["metadata"], {})


if __name__ == "__main__":
    unittest.main()
