from __future__ import annotations

from pathlib import Path
import unittest


class PaperReproExactCvConfigTests(unittest.TestCase):
    def test_config_exists_and_matches_paper_repro_requirements(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uci_3class_paper_repro_exact_cv.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uci_3class_paper_repro_exact_cv")
        self.assertEqual(payload["experiment"]["target_formulation"], "three_class")
        self.assertEqual(payload["training"]["mode"], "paper_reproduction_cv")
        self.assertEqual(payload["evaluation"]["primary_selection_metric"], "cv_macro_f1")
        self.assertListEqual(payload["target"]["class_order"], ["Dropout", "Enrolled", "Graduate"])
        self.assertIn("mlp", payload["models"]["candidates"])
        self.assertTrue(bool(payload["preprocessing"]["outlier"]["enabled"]))
        self.assertEqual(payload["preprocessing"]["outlier"]["method"], "isolation_forest")
        self.assertTrue(bool(payload["preprocessing"]["balancing"]["enabled"]))
        self.assertEqual(payload["preprocessing"]["balancing"]["method"], "smote")
        self.assertTrue(bool(payload["preprocessing"]["missing_values"]["drop_rows"]))
        self.assertEqual(payload["models"]["tuning"]["n_trials"], 10)
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uci_3class_paper_repro_exact_cv")


if __name__ == "__main__":
    unittest.main()
