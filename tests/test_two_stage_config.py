"""Regression tests for two-stage UCT benchmark integration."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.run_experiment import (
    _resolve_two_stage_stage2_positive_target_label,
    _resolve_two_stage_stage_class_weights,
)
from src.models.two_stage_uct import TwoStageUct3ClassClassifier


class _DummyBinaryModel:
    def __init__(self, probabilities: list[list[float]]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._probabilities[: len(X)]


class TwoStageConfigTests(unittest.TestCase):
    def test_two_stage_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v1.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v1")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["experiment"]["dataset_config"], "configs/datasets/uct_student.yaml")
        self.assertEqual(payload["experiment"]["target_formulation"], "three_class")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_two_stage_v1")
        self.assertEqual(payload["selection"]["primary"], "enrolled_f1")
        self.assertListEqual(payload["selection"]["tie_breakers"], ["macro_f1", "balanced_accuracy"])
        self.assertEqual(payload["two_stage"]["stage2_positive_class"], "enrolled")
        self.assertEqual(
            payload["two_stage"]["class_weight"]["stage2"]["values"],
            {"Graduate": 1.0, "Enrolled": 1.35},
        )
        self.assertListEqual(
            payload["models"]["candidates"],
            ["svm", "lightgbm", "catboost", "xgboost", "random_forest", "gradient_boosting", "mlp", "decision_tree"],
        )

    def test_wrapper_supports_enrolled_as_stage2_positive_class(self) -> None:
        stage1_model = _DummyBinaryModel([[0.2, 0.8], [0.8, 0.2]])
        stage2_model = _DummyBinaryModel([[0.3, 0.7], [0.3, 0.7]])
        model = TwoStageUct3ClassClassifier(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            dropout_label=0,
            enrolled_label=1,
            graduate_label=2,
            decision_mode="hard_stage1",
            threshold_stage1=0.5,
            stage1_positive_label=1,
            stage2_positive_label=1,
            stage2_positive_target_label=1,
        )

        X = pd.DataFrame({"f1": [1.0, 2.0]})
        proba = model.predict_proba(X)
        pred = model.predict(X)

        self.assertTrue(np.allclose(proba[0], np.array([0.8, 0.14, 0.06])))
        self.assertTrue(np.allclose(proba[1], np.array([0.2, 0.56, 0.24])))
        self.assertListEqual(pred.tolist(), [0, 1])

    def test_two_stage_class_weight_resolution_uses_balanced_stage1_and_explicit_stage2(self) -> None:
        stage1_cfg, stage2_cfg = _resolve_two_stage_stage_class_weights(
            two_stage_cfg={
                "class_weight": {
                    "stage2": {
                        "enabled": True,
                        "mode": "explicit",
                        "strategy": "explicit",
                        "values": {"Graduate": 1.0, "Enrolled": 1.35},
                    }
                }
            },
            class_weight_cfg={
                "enabled": True,
                "mode": "explicit",
                "strategy": "explicit",
                "values": {"Dropout": 1.0, "Enrolled": 1.35, "Graduate": 1.0},
                "class_weight_map": {"Dropout": 1.0, "Enrolled": 1.35, "Graduate": 1.0},
            },
            dropout_idx=0,
            enrolled_idx=1,
            graduate_idx=2,
        )

        self.assertEqual(stage1_cfg["strategy"], "balanced")
        self.assertEqual(stage1_cfg["class_label_to_index"], {"Non-Dropout": 0, "Dropout": 1})
        self.assertEqual(stage2_cfg["mode"], "explicit")
        self.assertEqual(stage2_cfg["class_label_to_index"], {"Graduate": 0, "Enrolled": 1})
        self.assertEqual(stage2_cfg["values"], {"Graduate": 1.0, "Enrolled": 1.35})

    def test_stage2_positive_target_label_resolution_accepts_enrolled(self) -> None:
        resolved = _resolve_two_stage_stage2_positive_target_label(
            two_stage_cfg={"stage2_positive_class": "enrolled"},
            enrolled_idx=1,
            graduate_idx=2,
        )
        self.assertEqual(resolved, 1)


if __name__ == "__main__":
    unittest.main()
