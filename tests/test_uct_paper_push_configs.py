"""Config-level regression tests for UCT 3-class paper-push benchmark track."""

from __future__ import annotations

from pathlib import Path
import unittest


class UctPaperPushConfigTests(unittest.TestCase):
    def test_all_required_paper_push_configs_exist_and_are_valid(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        expected_files = [
            "exp_bm_uct_3class_paper_push_svm.yaml",
            "exp_bm_uct_3class_paper_push_gb.yaml",
            "exp_bm_uct_3class_paper_push_lgbm.yaml",
            "exp_bm_uct_3class_paper_push_catboost.yaml",
            "exp_bm_uct_3class_paper_push_xgb.yaml",
            "exp_bm_uct_3class_paper_push_all.yaml",
        ]
        expected_single_model = {
            "exp_bm_uct_3class_paper_push_svm.yaml": ["svm"],
            "exp_bm_uct_3class_paper_push_gb.yaml": ["gradient_boosting"],
            "exp_bm_uct_3class_paper_push_lgbm.yaml": ["lightgbm"],
            "exp_bm_uct_3class_paper_push_catboost.yaml": ["catboost"],
            "exp_bm_uct_3class_paper_push_xgb.yaml": ["xgboost"],
        }
        expected_all_models = ["svm", "gradient_boosting", "lightgbm", "catboost", "xgboost"]

        exp_dir = Path("configs/experiments")
        for filename in expected_files:
            path = exp_dir / filename
            self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["experiment"]["dataset_config"], "configs/datasets/uct_student.yaml")
            self.assertEqual(payload["experiment"]["target_formulation"], "three_class")
            self.assertEqual(payload["metrics"]["primary"], "macro_f1")
            self.assertEqual(payload["outputs"]["runtime_artifact_format"], "csv")
            self.assertTrue(bool(payload["outputs"]["mirror_benchmark_outputs_to_runtime"]))

            override = payload.get("target_mapping_override", {}).get("three_class", {})
            self.assertDictEqual(override, {"Dropout": 0, "Enrolled": 1, "Graduate": 2})

            candidates = payload.get("models", {}).get("candidates", [])
            if filename in expected_single_model:
                self.assertListEqual(candidates, expected_single_model[filename])
            else:
                self.assertListEqual(candidates, expected_all_models)


if __name__ == "__main__":
    unittest.main()
