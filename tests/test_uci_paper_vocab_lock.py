from __future__ import annotations

from pathlib import Path
import unittest


class UciPaperVocabLockConfigTests(unittest.TestCase):
    def test_feature_exact_config_uses_locked_parquet_vocab_mode(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uci_3class_paper_repro_feature_exact.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["experiment"]["dataset_config"],
            "configs/datasets/uci_student_presplit_parquet.yaml",
        )
        self.assertEqual(payload["experiment"]["target_formulation"], "three_class")
        self.assertEqual(
            payload["feature_engineering"]["dataset_features"]["builder"],
            "uci_student_paper_style_features",
        )
        categorical_encoding = payload["preprocessing"]["categorical_encoding"]
        self.assertEqual(categorical_encoding["mode"], "onehot")
        self.assertTrue(bool(categorical_encoding["lock_category_vocabulary_from_pre_split_train"]))
        self.assertEqual(
            categorical_encoding["vocabulary_source"],
            "categorical_dtype_or_full_pre_split_train",
        )
        self.assertEqual(categorical_encoding["drop"], "none")
        self.assertEqual(categorical_encoding["handle_unknown"], "ignore")
        self.assertEqual(
            payload["outputs"]["results_dir"],
            "results/exp_uci_3class_paper_repro_feature_exact",
        )


if __name__ == "__main__":
    unittest.main()
