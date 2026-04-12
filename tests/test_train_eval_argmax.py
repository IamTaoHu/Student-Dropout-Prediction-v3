"""Regression tests for argmax decision rule and probability class-order alignment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.train_eval import (
    _validation_objective_metric_value,
    auto_tune_multiclass_decision_policy,
    run_leakage_safe_stratified_cv,
    train_and_evaluate,
)


class _DummyClassifier:
    def __init__(self) -> None:
        # Deliberately shuffled class order to validate explicit alignment.
        self.classes_ = np.array([2, 0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> "_DummyClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=2, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = np.array(
            [
                [0.05, 0.80, 0.15],
                [0.70, 0.20, 0.10],
            ],
            dtype=float,
        )
        if len(X) <= 2:
            return base[: len(X)]
        reps = int(np.ceil(len(X) / 2.0))
        return np.vstack([base for _ in range(reps)])[: len(X)]


class _AmbiguousThreeClassClassifier:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> "_AmbiguousThreeClassClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([0, 2, 0], dtype=int)[: len(X)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = np.array(
            [
                [0.42, 0.40, 0.18],
                [0.20, 0.30, 0.50],
                [0.60, 0.25, 0.15],
            ],
            dtype=float,
        )
        return base[: len(X)]


class _NoSampleWeightClassifier:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_NoSampleWeightClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([0] * len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(X), 3), dtype=float)
        probs[:, 0] = 1.0
        return probs


class _NoProbabilityClassifier:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_NoProbabilityClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([1] * len(X), dtype=int)


class TrainEvalArgmaxTests(unittest.TestCase):
    def test_validation_objective_metric_can_target_enrolled_f1(self) -> None:
        score = _validation_objective_metric_value(
            metrics={"macro_f1": 0.71, "balanced_accuracy": 0.74, "accuracy": 0.76},
            per_class={"1": {"f1": 0.63, "recall": 0.68, "precision": 0.59}},
            objective_metric="enrolled_f1",
        )
        self.assertAlmostEqual(score, 0.63, places=6)

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_DummyClassifier())
    def test_argmax_decision_rule_uses_aligned_probabilities(self, _mock_build: object, _mock_defaults: object) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4]})
        y_train = pd.Series([0, 1, 2, 1])
        X_valid = pd.DataFrame({"x": [0.5, 0.6]})
        y_valid = pd.Series([0, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8]})
        y_test = pd.Series([0, 2])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "argmax",
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["labels"], [0, 1, 2])
        self.assertListEqual(result.artifacts["y_pred_test"], [0, 2])
        self.assertTrue(result.artifacts["decision_rule_applied_on_probabilities"])

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_DummyClassifier())
    def test_model_predict_decision_rule_keeps_native_predictions(self, _mock_build: object, _mock_defaults: object) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4]})
        y_train = pd.Series([0, 1, 2, 1])
        X_valid = pd.DataFrame({"x": [0.5, 0.6]})
        y_valid = pd.Series([0, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8]})
        y_test = pd.Series([0, 2])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "model_predict",
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["y_pred_test"], [2, 2])
        self.assertFalse(result.artifacts["decision_rule_applied_on_probabilities"])

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_AmbiguousThreeClassClassifier())
    def test_enrolled_margin_decision_rule_overrides_low_margin_non_enrolled_predictions(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y_train = pd.Series([0, 1, 2, 0, 1, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8, 0.9]})
        y_test = pd.Series([1, 2, 0])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=pd.DataFrame(columns=["x"]),
            y_valid=pd.Series(dtype=int),
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "enrolled_margin",
                "multiclass_decision": {
                    "strategy": "enrolled_margin",
                    "enrolled_margin_threshold": 0.10,
                },
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["y_pred_test"], [1, 2, 0])
        self.assertTrue(result.artifacts["decision_rule_applied_on_probabilities"])

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_AmbiguousThreeClassClassifier())
    def test_enrolled_middle_band_decision_rule_routes_predictions_by_thresholds(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y_train = pd.Series([0, 1, 2, 0, 1, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8, 0.9]})
        y_test = pd.Series([1, 1, 0])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=pd.DataFrame(columns=["x"]),
            y_valid=pd.Series(dtype=int),
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "enrolled_middle_band",
                "multiclass_decision": {
                    "strategy": "enrolled_middle_band",
                    "dropout_threshold": 0.55,
                    "graduate_threshold": 0.55,
                },
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["y_pred_test"], [1, 1, 0])
        self.assertTrue(result.artifacts["decision_rule_applied_on_probabilities"])

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_AmbiguousThreeClassClassifier())
    def test_enrolled_middle_band_guarded_tuning_only_promotes_guarded_middle_cases(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y_train = pd.Series([0, 1, 2, 0, 1, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8, 0.9]})
        y_test = pd.Series([1, 2, 0])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=pd.DataFrame(columns=["x"]),
            y_valid=pd.Series(dtype=int),
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "enrolled_middle_band",
                "multiclass_decision": {
                    "strategy": "enrolled_middle_band",
                    "dropout_threshold": 0.55,
                    "graduate_threshold": 0.55,
                    "enrolled_decision_tuning": {
                        "enabled": True,
                        "enrolled_label": 1,
                        "dropout_label": 0,
                        "graduate_label": 2,
                        "enrolled_min_proba": 0.30,
                        "enrolled_margin_gap": 0.12,
                        "ambiguity_max_gap": 0.12,
                        "graduate_guard_max": 0.62,
                        "dropout_guard_max": 0.62,
                        "require_enrolled_above_baseline": True,
                    },
                },
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["y_pred_test"], [1, 2, 0])
        self.assertEqual(int(result.artifacts["decision_rule_audit_test"]["override_count"]), 1)
        self.assertEqual(int(result.artifacts["decision_rule_audit_test"]["enrolled_override_count"]), 1)

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_AmbiguousThreeClassClassifier())
    def test_enrolled_push_applies_threshold_then_middle_band(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y_train = pd.Series([0, 1, 2, 0, 1, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8, 0.9]})
        y_test = pd.Series([1, 2, 0])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=pd.DataFrame(columns=["x"]),
            y_valid=pd.Series(dtype=int),
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "enrolled_push",
                "multiclass_decision": {
                    "strategy": "enrolled_push",
                    "enrolled_probability_threshold": {"enabled": True, "value": 0.40},
                    "enrolled_middle_band": {"enabled": True, "min_enrolled_prob": 0.25, "max_top2_gap": 0.05},
                },
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["y_pred_test"], [1, 2, 0])
        self.assertTrue(result.artifacts["decision_rule_applied_on_probabilities"])

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_NoProbabilityClassifier())
    def test_probability_based_decision_rule_falls_back_to_native_prediction_when_probabilities_missing(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y_train = pd.Series([0, 1, 2, 0, 1, 2])
        X_test = pd.DataFrame({"x": [0.7, 0.8]})
        y_test = pd.Series([1, 1])

        result = train_and_evaluate(
            model_name="decision_tree",
            params={},
            X_train=X_train,
            y_train=y_train,
            X_valid=pd.DataFrame(columns=["x"]),
            y_valid=pd.Series(dtype=int),
            X_test=X_test,
            y_test=y_test,
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "enrolled_push",
                "multiclass_decision": {
                    "strategy": "enrolled_push",
                    "enrolled_probability_threshold": {"enabled": True, "value": 0.40},
                },
                "class_weight": {"enabled": False},
            },
        )

        self.assertListEqual(result.artifacts["y_pred_test"], [1, 1])
        self.assertFalse(result.artifacts["decision_rule_applied_on_probabilities"])

    def test_auto_tune_enrolled_margin_selects_best_validation_threshold(self) -> None:
        y_true_valid = pd.Series([1, 2, 0, 1], dtype=int)
        y_proba_valid = np.asarray(
            [
                [0.44, 0.43, 0.13],  # low margin => can flip to enrolled
                [0.10, 0.20, 0.70],  # confident graduate
                [0.80, 0.15, 0.05],  # confident dropout
                [0.46, 0.44, 0.10],  # low margin => can flip to enrolled
            ],
            dtype=float,
        )
        y_true_test = pd.Series([1, 2], dtype=int)
        y_proba_test = np.asarray([[0.45, 0.44, 0.11], [0.10, 0.15, 0.75]], dtype=float)

        tuned = auto_tune_multiclass_decision_policy(
            y_true_valid=y_true_valid,
            y_proba_valid=y_proba_valid,
            y_true_test=y_true_test,
            y_proba_test=y_proba_test,
            labels=[0, 1, 2],
            strategy="enrolled_margin",
            multiclass_decision_config={
                "strategy": "enrolled_margin",
                "enrolled_margin_threshold": 0.10,
                "auto_tune": {
                    "enabled": True,
                    "objective": "macro_f1",
                    "split": "validation",
                    "search": {
                        "method": "grid",
                        "enrolled_margin_thresholds": [0.02, 0.05, 0.08, 0.10],
                    },
                },
            },
        )

        self.assertEqual(tuned["status"], "applied")
        self.assertAlmostEqual(float(tuned["selected_parameters"]["enrolled_margin_threshold"]), 0.02)
        self.assertEqual(len(tuned["y_pred_test_tuned"]), 2)

    def test_auto_tune_middle_band_applies_deterministic_tie_break(self) -> None:
        y_true_valid = pd.Series([1, 1], dtype=int)
        y_proba_valid = np.asarray([[0.50, 0.30, 0.20], [0.20, 0.30, 0.50]], dtype=float)
        y_true_test = pd.Series([1], dtype=int)
        y_proba_test = np.asarray([[0.49, 0.31, 0.20]], dtype=float)

        tuned = auto_tune_multiclass_decision_policy(
            y_true_valid=y_true_valid,
            y_proba_valid=y_proba_valid,
            y_true_test=y_true_test,
            y_proba_test=y_proba_test,
            labels=[0, 1, 2],
            strategy="enrolled_middle_band",
            multiclass_decision_config={
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
            },
        )

        self.assertEqual(tuned["status"], "applied")
        self.assertAlmostEqual(float(tuned["selected_parameters"]["dropout_threshold"]), 0.55)
        self.assertAlmostEqual(float(tuned["selected_parameters"]["graduate_threshold"]), 0.55)

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_DummyClassifier())
    def test_explicit_class_weights_apply_native_class_weight_for_random_forest(
        self,
        mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y = pd.Series([0, 1, 2, 0, 1, 2], dtype=int)
        result = train_and_evaluate(
            model_name="random_forest",
            params={},
            X_train=X,
            y_train=y,
            X_valid=pd.DataFrame(columns=["x"]),
            y_valid=pd.Series(dtype=int),
            X_test=X.iloc[:2].reset_index(drop=True),
            y_test=pd.Series([0, 1], dtype=int),
            eval_config={
                "seed": 42,
                "label_order": [0, 1, 2],
                "decision_rule": "argmax",
                "class_weight": {
                    "enabled": True,
                    "mode": "explicit",
                    "values": {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
                    "class_label_to_index": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                },
            },
        )
        self.assertTrue(result.artifacts["class_weight_info"]["model_param_class_weight_applied"])
        self.assertEqual(result.artifacts["class_weight_info"]["class_weight_application_method"], "native_class_weight")
        _, kwargs = mock_build.call_args
        self.assertIn("class_weight", kwargs["params"])

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_DummyClassifier())
    def test_explicit_class_weights_require_all_mapped_classes(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y = pd.Series([0, 1, 2, 0, 1, 2], dtype=int)
        with self.assertRaises(ValueError):
            train_and_evaluate(
                model_name="decision_tree",
                params={},
                X_train=X,
                y_train=y,
                X_valid=pd.DataFrame(columns=["x"]),
                y_valid=pd.Series(dtype=int),
                X_test=X.iloc[:2].reset_index(drop=True),
                y_test=pd.Series([0, 1], dtype=int),
                eval_config={
                    "seed": 42,
                    "label_order": [0, 1, 2],
                    "decision_rule": "argmax",
                    "class_weight": {
                        "enabled": True,
                        "mode": "explicit",
                        "values": {"Dropout": 1.0, "Enrolled": 1.5},
                        "class_label_to_index": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                    },
                },
            )

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_NoSampleWeightClassifier())
    def test_class_weight_fails_when_no_sample_weight_fit_path(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        X = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
        y = pd.Series([0, 1, 2, 0, 1, 2], dtype=int)
        with self.assertRaises(ValueError):
            train_and_evaluate(
                model_name="gradient_boosting",
                params={},
                X_train=X,
                y_train=y,
                X_valid=pd.DataFrame(columns=["x"]),
                y_valid=pd.Series(dtype=int),
                X_test=X.iloc[:2].reset_index(drop=True),
                y_test=pd.Series([0, 1], dtype=int),
                eval_config={
                    "seed": 42,
                    "label_order": [0, 1, 2],
                    "decision_rule": "argmax",
                    "class_weight": {
                        "enabled": True,
                        "mode": "explicit",
                        "values": {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
                        "class_label_to_index": {"Dropout": 0, "Enrolled": 1, "Graduate": 2},
                    },
                },
            )

    @patch("src.models.train_eval.get_default_model_params", return_value={})
    @patch("src.models.train_eval.build_model", return_value=_DummyClassifier())
    def test_leakage_safe_cv_returns_fold_and_aggregate_metrics(
        self,
        _mock_build: object,
        _mock_defaults: object,
    ) -> None:
        rows = 15
        target = [0, 1, 2] * 5
        train_df = pd.DataFrame(
            {
                "id": list(range(rows)),
                "num": np.linspace(0.0, 1.0, rows),
                "cat": ["a", "b", "c"] * 5,
                "target": target,
            }
        )
        cv_results = run_leakage_safe_stratified_cv(
            model_name="decision_tree",
            params={},
            train_df=train_df,
            preprocess_config={
                "target_column": "target",
                "id_columns": ["id"],
                "forbidden_feature_columns": [],
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "scaling": True,
                "onehot": True,
            },
            outlier_config={"enabled": False},
            balancing_config={"enabled": False},
            cv_config={"n_splits": 3, "shuffle": True, "random_state": 42},
            eval_config={"seed": 42, "label_order": [0, 1, 2], "decision_rule": "argmax", "class_weight": {"enabled": False}},
        )
        self.assertIn("folds", cv_results)
        self.assertEqual(len(cv_results["folds"]), 3)
        agg = cv_results.get("aggregate_metrics", {})
        self.assertIn("cv_macro_f1_mean", agg)
        self.assertIn("cv_accuracy_mean", agg)
        self.assertEqual(int(agg.get("cv_num_folds", 0)), 3)


if __name__ == "__main__":
    unittest.main()
