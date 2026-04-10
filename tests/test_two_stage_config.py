"""Regression tests for two-stage UCT benchmark integration."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.run_experiment import (
    _resolve_two_stage_decision_mode,
    _resolve_two_stage_stage2_decision_config,
    _resolve_two_stage_stage2_positive_target_label,
    _resolve_two_stage_stage1_dropout_threshold_config,
    _resolve_two_stage_stage_class_weights,
)
from src.models.two_stage_uct import Stage2PositiveProbabilityCalibrator, TwoStageUct3ClassClassifier


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

    def test_two_stage_class_weight_resolution_supports_new_stage_specific_schema(self) -> None:
        stage1_cfg, stage2_cfg = _resolve_two_stage_stage_class_weights(
            two_stage_cfg={
                "stage1": {
                    "class_weight_mode": "custom",
                    "class_weight_positive": 1.0,
                    "class_weight_negative": 1.15,
                },
                "stage2": {
                    "class_weight_mode": "custom",
                    "class_weight_map": {"enrolled": 1.35, "graduate": 1.0},
                },
            },
            class_weight_cfg={"enabled": True, "mode": "balanced"},
            dropout_idx=0,
            enrolled_idx=1,
            graduate_idx=2,
        )

        self.assertEqual(stage1_cfg["mode"], "explicit")
        self.assertEqual(stage1_cfg["values"], {"Non-Dropout": 1.15, "Dropout": 1.0})
        self.assertEqual(stage2_cfg["mode"], "explicit")
        self.assertEqual(stage2_cfg["values"], {"Graduate": 1.0, "Enrolled": 1.35})

    def test_soft_fusion_with_dropout_threshold_overrides_dropout_before_argmax(self) -> None:
        stage1_model = _DummyBinaryModel([[0.49, 0.51], [0.55, 0.45]])
        stage2_model = _DummyBinaryModel([[0.1, 0.9], [0.8, 0.2]])
        model = TwoStageUct3ClassClassifier(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            dropout_label=0,
            enrolled_label=1,
            graduate_label=2,
            decision_mode="soft_fusion_with_dropout_threshold",
            threshold_stage1=0.5,
            stage1_positive_label=1,
            stage2_positive_label=1,
            stage2_positive_target_label=1,
        )

        X = pd.DataFrame({"f1": [1.0, 2.0]})
        pred = model.predict(X)

        self.assertListEqual(pred.tolist(), [0, 2])

    def test_soft_fusion_with_middle_band_routes_ambiguous_cases_through_full_fused_argmax(self) -> None:
        stage1_model = _DummyBinaryModel([[0.45, 0.55], [0.48, 0.52], [0.70, 0.30]])
        stage2_model = _DummyBinaryModel([[0.1, 0.9], [0.9, 0.1], [0.7, 0.3]])
        model = TwoStageUct3ClassClassifier(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            dropout_label=0,
            enrolled_label=1,
            graduate_label=2,
            decision_mode="soft_fusion_with_middle_band",
            threshold_stage1=0.5,
            threshold_stage1_low=0.40,
            threshold_stage1_high=0.60,
            stage1_positive_label=1,
            stage2_positive_label=1,
            stage2_positive_target_label=1,
        )

        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        pred = model.predict(X)

        self.assertListEqual(pred.tolist(), [1, 0, 2])

    def test_stage2_guarded_threshold_can_select_enrolled_without_overriding_dropout(self) -> None:
        stage1_model = _DummyBinaryModel([[0.7, 0.3], [0.2, 0.8], [0.45, 0.55]])
        stage2_model = _DummyBinaryModel([[0.45, 0.55], [0.46, 0.54], [0.49, 0.51]])
        model = TwoStageUct3ClassClassifier(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            dropout_label=0,
            enrolled_label=1,
            graduate_label=2,
            decision_mode="soft_fusion_with_dropout_threshold",
            threshold_stage1=0.5,
            stage1_positive_label=1,
            stage2_positive_label=1,
            stage2_positive_target_label=1,
            stage2_decision_config={
                "enabled": True,
                "strategy": "enrolled_guarded_threshold",
                "enrolled_probability_threshold": 0.50,
                "graduate_margin_guard": 0.10,
            },
        )

        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        pred = model.predict(X)

        self.assertListEqual(pred.tolist(), [2, 0, 1])

    def test_stage2_decision_config_resolver_builds_validation_search_grid(self) -> None:
        resolved = _resolve_two_stage_stage2_decision_config(
            {
                "stage2_decision": {
                    "enabled": True,
                    "strategy": "enrolled_guarded_threshold",
                    "enrolled_probability_threshold": 0.42,
                    "graduate_margin_guard": 0.06,
                    "tune_on_validation": True,
                    "search": {
                        "enrolled_probability_threshold": {"min": 0.30, "max": 0.34, "step": 0.02},
                        "graduate_margin_guard": {"min": 0.00, "max": 0.04, "step": 0.02},
                    },
                    "objective": {
                        "enrolled_f1_alpha": 0.4,
                        "graduate_f1_penalty_beta": 1.7,
                    },
                }
            }
        )

        self.assertTrue(resolved["enabled"])
        self.assertEqual(resolved["strategy"], "enrolled_guarded_threshold")
        self.assertListEqual(
            resolved["search"]["enrolled_probability_threshold_grid"],
            [0.3, 0.32, 0.34],
        )
        self.assertListEqual(resolved["search"]["graduate_margin_guard_grid"], [0.0, 0.02, 0.04])
        self.assertAlmostEqual(float(resolved["objective"]["enrolled_f1_alpha"]), 0.4)
        self.assertAlmostEqual(float(resolved["objective"]["graduate_f1_penalty_beta"]), 1.7)

    def test_stage2_decision_policy_config_resolver_supports_inner_split_and_calibration(self) -> None:
        resolved = _resolve_two_stage_stage2_decision_config(
            {
                "stage2": {
                    "decision_policy": {
                        "enabled": True,
                        "mode": "enrolled_aware_multi_objective",
                        "strategy": "enrolled_guarded_threshold",
                        "use_calibrated_proba": True,
                        "calibration_method": "temperature_scaling",
                        "enrolled_probability_threshold": 0.44,
                        "enrolled_margin": 0.03,
                        "search": {
                            "method": "grid",
                            "enrolled_probability_threshold": {"min": 0.40, "max": 0.44, "step": 0.02},
                            "enrolled_margin": {"min": 0.00, "max": 0.04, "step": 0.02},
                        },
                        "objective": {
                            "metric": "custom",
                            "alpha_enrolled_f1": 0.35,
                            "beta_graduate_drop_penalty": 0.25,
                            "gamma_macro_f1": 1.0,
                            "graduate_f1_tolerance_vs_baseline": 0.02,
                        },
                        "anti_overfit": {
                            "strategy": "stage2_train_inner_split",
                            "tuning_size": 0.25,
                            "min_tuning_samples": 16,
                        },
                    }
                }
            }
        )

        self.assertEqual(resolved["config_schema"], "decision_policy")
        self.assertTrue(bool(resolved["use_calibrated_proba"]))
        self.assertEqual(resolved["calibration_method"], "temperature_scaling")
        self.assertEqual(resolved["mode"], "enrolled_aware_multi_objective")
        self.assertListEqual(resolved["search"]["enrolled_margin_grid"], [0.0, 0.02, 0.04])
        self.assertEqual(resolved["anti_overfit"]["strategy"], "stage2_train_inner_split")

    def test_stage2_probability_calibrator_temperature_scaling_preserves_bounds(self) -> None:
        calibrator = Stage2PositiveProbabilityCalibrator(
            method="temperature_scaling",
            payload={"temperature": 2.0},
        )
        transformed = calibrator.transform(np.array([0.1, 0.5, 0.9], dtype=float))

        self.assertTrue(np.all(transformed > 0.0))
        self.assertTrue(np.all(transformed < 1.0))
        self.assertAlmostEqual(float(transformed[1]), 0.5, places=6)

    def test_new_soft_fusion_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v2_soft_fusion.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v2_soft_fusion")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_two_stage_v2_soft_fusion")
        self.assertEqual(payload["two_stage"]["final_decision"]["mode"], "soft_fusion_with_dropout_threshold")
        self.assertEqual(payload["two_stage"]["stage1"]["threshold_mode"], "tune")
        self.assertListEqual(
            payload["models"]["candidates"],
            ["decision_tree", "random_forest", "svm", "gradient_boosting", "xgboost", "lightgbm", "catboost"],
        )

    def test_new_two_stage_config_helpers_resolve_soft_fusion_threshold_mode(self) -> None:
        two_stage_cfg = {
            "stage1": {
                "threshold_mode": "tune",
                "dropout_threshold": 0.5,
                "threshold_grid": [0.4, 0.5, 0.6],
            },
            "final_decision": {"mode": "soft_fusion"},
        }
        threshold_cfg = _resolve_two_stage_stage1_dropout_threshold_config(two_stage_cfg)
        decision_mode = _resolve_two_stage_decision_mode("two_stage", two_stage_cfg)

        self.assertEqual(threshold_cfg["mode"], "tune")
        self.assertEqual(threshold_cfg["threshold_grid"], [0.4, 0.5, 0.6])
        self.assertEqual(decision_mode, "soft_fusion")

    def test_v3_two_stage_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v3_enrolled_push.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v3_enrolled_push")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_two_stage_v3_enrolled_push")
        self.assertEqual(payload["pipeline"]["task_type"], "two_stage_multiclass")
        self.assertEqual(payload["two_stage"]["final_decision"]["mode"], "soft_fusion_with_middle_band")
        self.assertTrue(bool(payload["two_stage"]["final_decision"]["middle_band_enabled"]))
        self.assertEqual(payload["two_stage"]["threshold_tuning"]["search_mode"], "band")
        self.assertEqual(payload["two_stage"]["threshold_tuning"]["objective"], "macro_f1_plus_enrolled_f1")
        self.assertAlmostEqual(float(payload["two_stage"]["threshold_tuning"]["enrolled_push_alpha"]), 0.35)
        self.assertListEqual(
            payload["models"]["candidates"],
            ["decision_tree", "random_forest", "svm", "gradient_boosting", "xgboost", "lightgbm", "catboost"],
        )

    def test_v4_auto_balance_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v4_balance_auto.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v4_balance_auto")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["pipeline"]["task_type"], "two_stage_multiclass")
        self.assertEqual(payload["two_stage"]["final_decision"]["mode"], "soft_fusion_with_middle_band")
        self.assertEqual(payload["two_stage"]["stage1"]["class_weight_mode"], "auto_search")
        self.assertEqual(payload["two_stage"]["stage2"]["class_weight_mode"], "auto_search")
        self.assertTrue(bool(payload["two_stage"]["auto_balance_search"]["enabled"]))
        self.assertEqual(payload["two_stage"]["selection"]["objective"], "constrained_macro_with_class_floors")
        self.assertEqual(payload["outputs"]["results_dir"], "results/exp_uct_3class_two_stage_v4_balance_auto")

    def test_v5_guarded_stage2_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v5_enrolled_guarded_threshold.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v5_enrolled_guarded_threshold")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["pipeline"]["task_type"], "two_stage_multiclass")
        self.assertEqual(payload["two_stage"]["stage1"]["threshold_mode"], "fixed")
        self.assertTrue(bool(payload["two_stage"]["stage2_decision"]["enabled"]))
        self.assertEqual(payload["two_stage"]["stage2_decision"]["strategy"], "enrolled_guarded_threshold")
        self.assertTrue(bool(payload["two_stage"]["auto_balance_search"]["enabled"]))
        self.assertListEqual(payload["two_stage"]["auto_balance_search"]["threshold_grid_low"], [0.30])
        self.assertListEqual(payload["two_stage"]["auto_balance_search"]["threshold_grid_high"], [0.55])
        self.assertEqual(
            payload["outputs"]["results_dir"],
            "results/exp_uct_3class_two_stage_v5_enrolled_guarded_threshold",
        )

    def test_v5_optuna_guarded_stage2_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v5_enrolled_optuna_guarded.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v5_enrolled_optuna_guarded")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["pipeline"]["task_type"], "two_stage_multiclass")
        self.assertEqual(payload["two_stage"]["stage1"]["threshold_mode"], "fixed")
        self.assertEqual(payload["two_stage"]["stage1"]["class_weight_mode"], "balanced")
        self.assertEqual(payload["two_stage"]["stage2"]["class_weight_mode"], "custom")
        self.assertTrue(bool(payload["two_stage"]["stage2"]["optuna_tuning"]["enabled"]))
        self.assertEqual(payload["two_stage"]["stage2"]["optuna_tuning"]["method"], "optuna")
        self.assertEqual(payload["two_stage"]["stage2"]["optuna_tuning"]["sampler"], "tpe")
        self.assertEqual(payload["two_stage"]["stage2"]["optuna_tuning"]["n_trials"], 30)
        self.assertTrue(bool(payload["two_stage"]["stage2_decision"]["enabled"]))
        self.assertEqual(payload["two_stage"]["stage2_decision"]["strategy"], "enrolled_guarded_threshold")
        self.assertFalse(bool(payload["two_stage"]["auto_balance_search"]["enabled"]))
        self.assertEqual(
            payload["outputs"]["results_dir"],
            "results/exp_uct_3class_two_stage_v5_enrolled_optuna_guarded",
        )

    def test_v7_interaction_prototype_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uct_3class_two_stage_v7_enrolled_interaction_prototype.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uct_3class_two_stage_v7_enrolled_interaction_prototype")
        self.assertEqual(payload["experiment"]["mode"], "two_stage")
        self.assertEqual(payload["pipeline"]["task_type"], "two_stage_multiclass")
        self.assertTrue(bool(payload["two_stage"]["stage2"]["feature_sharpening"]["enabled"]))
        self.assertTrue(bool(payload["two_stage"]["stage2"]["advanced_enrolled_separation"]["enabled"]))
        self.assertTrue(bool(payload["two_stage"]["stage2"]["advanced_enrolled_separation"]["interaction_features"]["enabled"]))
        self.assertTrue(bool(payload["two_stage"]["stage2"]["advanced_enrolled_separation"]["prototype_distance"]["enabled"]))
        self.assertEqual(
            payload["outputs"]["results_dir"],
            "results/exp_uct_3class_two_stage_v7_enrolled_interaction_prototype",
        )

    def test_v9_uci_decision_centric_config_exists_and_matches_expected_contract_shape(self) -> None:
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML is not installed in this environment.")

        path = Path("configs/experiments/exp_uci_3class_two_stage_v9_decision_centric.yaml")
        self.assertTrue(path.exists(), msg=f"Missing required config: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["experiment"]["id"], "exp_uci_3class_two_stage_v9_decision_centric")
        self.assertEqual(payload["experiment"]["dataset_config"], "configs/datasets/uci_student_presplit_parquet.yaml")
        self.assertListEqual(payload["models"]["candidates"], ["gradient_boosting", "xgboost", "lightgbm", "catboost"])
        self.assertFalse(bool(payload["two_stage"]["stage2"]["feature_sharpening"]["enabled"]))
        self.assertFalse(bool(payload["two_stage"]["stage2"]["robust_prototypes"]["enabled"]))
        self.assertTrue(bool(payload["two_stage"]["stage2"]["decision_policy"]["enabled"]))
        self.assertEqual(
            payload["outputs"]["results_dir"],
            "results/exp_uci_3class_two_stage_v9_decision_centric",
        )


if __name__ == "__main__":
    unittest.main()
