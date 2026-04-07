"""Tests for class-weight config normalization in run_experiment."""

from __future__ import annotations

import unittest

from scripts.run_experiment import _resolve_class_weight_config


class ClassWeightConfigTests(unittest.TestCase):
    def test_training_explicit_class_weight_normalizes_case_insensitive_keys(self) -> None:
        exp_cfg = {
            "training": {
                "class_weight": {
                    "mode": "explicit",
                    "values": {"dropout": 1.0, "enrolled": 1.5, "graduate": 1.0},
                }
            },
            "models": {"class_weight": {}},
        }
        class_metadata = {
            "class_label_to_index": {
                "Dropout": 0,
                "Enrolled": 1,
                "Graduate": 2,
            }
        }
        resolved = _resolve_class_weight_config(exp_cfg=exp_cfg, class_metadata=class_metadata)
        self.assertTrue(bool(resolved.get("enabled", False)))
        self.assertEqual(str(resolved.get("mode", "")), "explicit")
        self.assertEqual(
            resolved.get("class_weight_map"),
            {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
        )

    def test_training_explicit_class_weight_rejects_unknown_class_keys(self) -> None:
        exp_cfg = {
            "training": {
                "class_weight": {
                    "mode": "explicit",
                    "values": {"dropout": 1.0, "enrolled": 1.5, "graduate": 1.0, "other": 2.0},
                }
            },
            "models": {"class_weight": {}},
        }
        class_metadata = {
            "class_label_to_index": {
                "Dropout": 0,
                "Enrolled": 1,
                "Graduate": 2,
            }
        }
        with self.assertRaises(ValueError):
            _resolve_class_weight_config(exp_cfg=exp_cfg, class_metadata=class_metadata)


if __name__ == "__main__":
    unittest.main()

