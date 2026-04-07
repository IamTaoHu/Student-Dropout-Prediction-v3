"""Config-level regression tests for UCT 3-class paper-push benchmark track."""

from __future__ import annotations

from pathlib import Path
import unittest


class UctPaperPushConfigTests(unittest.TestCase):
    def test_only_bundled_paper_push_config_exists_and_is_valid(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        expected_all_file = "exp_bm_uct_3class_paper_push_all.yaml"
        expected_all_models = ["svm", "gradient_boosting", "lightgbm", "catboost", "xgboost"]
        deleted_files = [
            "exp_bm_uct_3class_paper_push_svm.yaml",
            "exp_bm_uct_3class_paper_push_gb.yaml",
            "exp_bm_uct_3class_paper_push_lgbm.yaml",
            "exp_bm_uct_3class_paper_push_catboost.yaml",
            "exp_bm_uct_3class_paper_push_xgb.yaml",
        ]

        exp_dir = Path("configs/experiments")
        all_path = exp_dir / expected_all_file
        self.assertTrue(all_path.exists(), msg=f"Missing required config: {all_path}")
        payload = yaml.safe_load(all_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_bm_uct_3class_paper_push_all")
        self.assertEqual(payload["experiment"]["dataset_config"], "configs/datasets/uct_student.yaml")
        self.assertEqual(payload["experiment"]["target_formulation"], "three_class")
        self.assertEqual(payload["metrics"]["primary"], "macro_f1")
        self.assertEqual(payload["outputs"]["runtime_artifact_format"], "csv")
        self.assertTrue(bool(payload["outputs"]["mirror_benchmark_outputs_to_runtime"]))
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_bm_uct_3class_paper_push_all")

        override = payload.get("target_mapping_override", {}).get("three_class", {})
        self.assertDictEqual(override, {"Dropout": 0, "Enrolled": 1, "Graduate": 2})

        candidates = payload.get("models", {}).get("candidates", [])
        self.assertListEqual(candidates, expected_all_models)

        for deleted_name in deleted_files:
            self.assertFalse((exp_dir / deleted_name).exists(), msg=f"Deprecated config should be removed: {deleted_name}")


if __name__ == "__main__":
    unittest.main()
