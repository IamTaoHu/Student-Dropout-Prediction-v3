"""Tests for run_experiment multiclass decision policy config resolution."""

from __future__ import annotations

import unittest

from scripts.run_experiment import _resolve_decision_rule_config, _resolve_model_decision_rule_config


class DecisionPolicyConfigTests(unittest.TestCase):
    def test_legacy_argmax_config_still_resolves(self) -> None:
        cfg = {
            "evaluation": {"decision_rule": "argmax"},
        }
        resolved = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        self.assertEqual(resolved["decision_rule"], "argmax")
        self.assertEqual(resolved["multiclass_decision"]["strategy"], "argmax")

    def test_enrolled_margin_requires_three_class_indices(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_margin",
                    "enrolled_margin_threshold": 0.10,
                }
            }
        }
        resolved = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        self.assertEqual(resolved["decision_rule"], "enrolled_margin")
        self.assertAlmostEqual(float(resolved["multiclass_decision"]["enrolled_margin_threshold"]), 0.10)

    def test_enrolled_middle_band_rejects_binary_formulation(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_middle_band",
                    "dropout_threshold": 0.55,
                    "graduate_threshold": 0.55,
                }
            }
        }
        with self.assertRaises(ValueError):
            _resolve_decision_rule_config(
                exp_cfg=cfg,
                formulation="binary",
                two_stage_enabled=False,
                class_metadata={"class_indices": [0, 1]},
            )

    def test_two_stage_overrides_probability_strategy_to_model_predict(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_middle_band",
                    "dropout_threshold": 0.55,
                    "graduate_threshold": 0.55,
                }
            }
        }
        resolved = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=True,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        self.assertEqual(resolved["decision_rule"], "model_predict")
        self.assertEqual(resolved["overridden_reason"], "two_stage_runner_controls_decision_logic")

    def test_autotune_margin_requires_non_empty_grid(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_margin",
                    "enrolled_margin_threshold": 0.10,
                    "auto_tune": {
                        "enabled": True,
                        "objective": "macro_f1",
                        "split": "validation",
                        "search": {"method": "grid", "enrolled_margin_thresholds": []},
                    },
                }
            }
        }
        with self.assertRaises(ValueError):
            _resolve_decision_rule_config(
                exp_cfg=cfg,
                formulation="three_class",
                two_stage_enabled=False,
                class_metadata={"class_indices": [0, 1, 2]},
            )

    def test_autotune_middle_band_is_exposed_in_resolved_policy(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_middle_band",
                    "dropout_threshold": 0.55,
                    "graduate_threshold": 0.55,
                    "auto_tune": {
                        "enabled": True,
                        "objective": "macro_f1",
                        "split": "validation",
                        "search": {
                            "method": "grid",
                            "dropout_thresholds": [0.50, 0.55],
                            "graduate_thresholds": [0.50, 0.55],
                        },
                    },
                }
            }
        }
        resolved = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        self.assertTrue(bool(resolved["multiclass_decision"]["auto_tune"]["enabled"]))
        self.assertEqual(resolved["multiclass_decision"]["auto_tune"]["search"]["method"], "grid")

    def test_enrolled_middle_band_guarded_tuning_resolves_named_labels(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_middle_band",
                    "dropout_threshold": 0.55,
                    "graduate_threshold": 0.55,
                    "enrolled_decision_tuning": {
                        "enabled": True,
                        "enrolled_label": "Enrolled",
                        "dropout_label": "Dropout",
                        "graduate_label": "Graduate",
                        "enrolled_min_proba": 0.30,
                        "enrolled_margin_gap": 0.08,
                        "ambiguity_max_gap": 0.12,
                        "graduate_guard_max": 0.62,
                        "dropout_guard_max": 0.62,
                        "require_enrolled_above_baseline": True,
                    },
                }
            }
        }
        resolved = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={
                "class_indices": [0, 1, 2],
                "class_label_to_index": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
            },
        )
        tuning = resolved["multiclass_decision"]["enrolled_decision_tuning"]
        self.assertTrue(bool(tuning["enabled"]))
        self.assertEqual(int(tuning["enrolled_label"]), 1)
        self.assertEqual(int(tuning["dropout_label"]), 0)
        self.assertEqual(int(tuning["graduate_label"]), 2)
        self.assertAlmostEqual(float(tuning["enrolled_margin_gap"]), 0.08)

    def test_enrolled_push_config_with_threshold_and_middle_band_resolves(self) -> None:
        cfg = {
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_push",
                    "enrolled_probability_threshold": {
                        "enabled": True,
                        "value": 0.40,
                    },
                    "enrolled_middle_band": {
                        "enabled": True,
                        "min_enrolled_prob": 0.30,
                        "max_top2_gap": 0.05,
                    },
                    "auto_tune": {
                        "enabled": True,
                        "objective": "enrolled_f1",
                        "split": "validation",
                        "search": {
                            "method": "grid",
                            "enrolled_probability_thresholds": [0.35, 0.40],
                            "min_enrolled_probs": [0.25, 0.30],
                            "max_top2_gaps": [0.03, 0.05],
                        },
                    },
                }
            }
        }
        resolved = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        self.assertEqual(resolved["decision_rule"], "enrolled_push")
        self.assertTrue(bool(resolved["multiclass_decision"]["enrolled_probability_threshold"]["enabled"]))
        self.assertAlmostEqual(float(resolved["multiclass_decision"]["enrolled_middle_band"]["max_top2_gap"]), 0.05)
        self.assertEqual(resolved["multiclass_decision"]["auto_tune"]["objective"], "enrolled_f1")

    def test_per_model_override_can_disable_enrolled_push_to_argmax(self) -> None:
        cfg = {
            "evaluation": {"decision_rule": "argmax"},
            "inference": {
                "multiclass_decision": {
                    "strategy": "enrolled_push",
                    "enrolled_probability_threshold": {"enabled": True, "value": 0.40},
                    "enrolled_middle_band": {"enabled": True, "min_enrolled_prob": 0.30, "max_top2_gap": 0.05},
                    "per_model": {
                        "svm": {"enabled": False},
                    },
                }
            },
        }
        base = _resolve_decision_rule_config(
            exp_cfg=cfg,
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        resolved = _resolve_model_decision_rule_config(
            exp_cfg=cfg,
            base_decision_rule_cfg=base,
            model_name="svm",
            formulation="three_class",
            two_stage_enabled=False,
            class_metadata={"class_indices": [0, 1, 2]},
        )
        self.assertEqual(resolved["decision_rule"], "argmax")
        self.assertEqual(resolved["multiclass_decision"]["strategy"], "argmax")


if __name__ == "__main__":
    unittest.main()
