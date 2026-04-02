"""Training, prediction, and metric evaluation utilities for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.registry import build_model
from src.models.search_spaces import get_default_model_params, suggest_optuna_params


@dataclass(frozen=True)
class TrainEvalResult:
    """Structured model run result with metrics and artifacts."""

    metrics: dict[str, float]
    artifacts: dict[str, Any]


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Return class probabilities when possible."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1 - probs_pos, probs_pos])
        return _softmax(scores)
    return None


def _compute_sample_weight(y: pd.Series, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    if y.empty:
        return None
    counts = y.value_counts(dropna=False)
    n_classes = len(counts)
    total = float(len(y))
    class_weights = {cls: total / (n_classes * float(count)) for cls, count in counts.items() if count > 0}
    return y.map(class_weights).to_numpy(dtype=float)


def _fit_model_with_optional_weights(model: Any, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common benchmark metrics for binary or multiclass targets."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_class_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision/recall/f1 metrics keyed by class label."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics_by_class: dict[str, dict[str, float]] = {}
    for key, payload in report.items():
        if not isinstance(payload, dict):
            continue
        if not key.lstrip("-").isdigit():
            continue
        metrics_by_class[str(key)] = {
            "precision": float(payload.get("precision", 0.0)),
            "recall": float(payload.get("recall", 0.0)),
            "f1": float(payload.get("f1-score", 0.0)),
            "support": float(payload.get("support", 0.0)),
        }
    return metrics_by_class


def tune_model_with_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_cfg: dict[str, Any],
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Tune hyperparameters using Optuna and return best params, score, and tuning details."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for tuning backend='optuna'.") from exc

    n_trials = int(tuning_cfg.get("n_trials", 30))
    random_state = int(tuning_cfg.get("seed", 42))
    cv_folds = int(tuning_cfg.get("cv_folds", 3))
    scoring = str(tuning_cfg.get("scoring", "f1_macro"))
    objective_source = str(tuning_cfg.get("objective_source", "cv")).strip().lower()
    use_class_weights = bool(tuning_cfg.get("use_class_weights", False))
    n_classes = int(pd.Series(y_train).nunique())

    if fixed_params is None:
        fixed_params = {}

    def objective(trial: Any) -> float:
        params = suggest_optuna_params(trial, model_name=model_name, n_classes=n_classes, random_state=random_state)
        if fixed_params:
            params.update(fixed_params)
        model = build_model(model_name, params)
        if objective_source == "validation":
            if X_valid is None or y_valid is None:
                raise ValueError("objective_source='validation' requires X_valid and y_valid.")
            sample_weight = _compute_sample_weight(y_train, enabled=use_class_weights)
            _fit_model_with_optional_weights(model, X_train, y_train, sample_weight=sample_weight)
            y_pred = model.predict(X_valid)
            metrics = compute_metrics(y_valid, y_pred)
            report = classification_report(y_valid, y_pred, output_dict=True, zero_division=0)
            per_class_f1 = {
                str(k): float(v.get("f1-score", 0.0))
                for k, v in report.items()
                if isinstance(v, dict) and "f1-score" in v
            }
            trial.set_user_attr("validation_metrics", metrics)
            trial.set_user_attr("per_class_f1", per_class_f1)
            return float(metrics["macro_f1"])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        score = float(np.mean(scores))
        trial.set_user_attr("cv_score", score)
        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials = []
    for trial in study.trials:
        trials.append(
            {
                "number": int(trial.number),
                "value": float(trial.value) if trial.value is not None else None,
                "state": str(trial.state),
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
            }
        )
    details = {
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "scoring": scoring,
        "cv_folds": cv_folds,
        "objective_source": objective_source,
        "best_validation_metrics": dict(study.best_trial.user_attrs.get("validation_metrics", {})),
        "best_per_class_f1": dict(study.best_trial.user_attrs.get("per_class_f1", {})),
        "trials": trials,
        "trials_dataframe": trials_df,
    }
    best_params = dict(study.best_params)
    if fixed_params:
        best_params.update(fixed_params)
    return best_params, float(study.best_value), details


def train_and_evaluate(
    model_name: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_config: dict[str, Any],
) -> TrainEvalResult:
    """Train one model and compute validation/test metrics."""
    random_state = int(eval_config.get("seed", 42))
    n_classes = int(pd.Series(y_train).nunique())
    default_params = get_default_model_params(model_name=model_name, random_state=random_state, n_classes=n_classes)
    full_params = {**default_params, **params}
    model = build_model(model_name=model_name, params=full_params)

    class_weight_cfg = eval_config.get("class_weight", {}) if isinstance(eval_config, dict) else {}
    sample_weight_train = _compute_sample_weight(y_train, enabled=bool(class_weight_cfg.get("enabled", False)))
    _fit_model_with_optional_weights(model, X_train, y_train, sample_weight=sample_weight_train)
    y_pred_valid = model.predict(X_valid) if not X_valid.empty else np.array([], dtype=int)
    y_proba_valid = predict_probabilities(model, X_valid) if not X_valid.empty else None
    y_pred_test = model.predict(X_test)
    y_proba_test = predict_probabilities(model, X_test)

    metrics: dict[str, float] = {}
    if not X_valid.empty:
        valid_metrics = compute_metrics(y_valid, y_pred_valid)
        metrics.update({f"valid_{k}": v for k, v in valid_metrics.items()})
    test_metrics = compute_metrics(y_test, y_pred_test)
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    configured_label_order = eval_config.get("label_order", []) if isinstance(eval_config, dict) else []
    labels = list(configured_label_order) if configured_label_order else sorted(pd.Series(y_train).unique().tolist())
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    per_class_metrics_test = compute_per_class_metrics(y_test, y_pred_test, labels=labels)
    classification_report_test = classification_report(
        y_test,
        y_pred_test,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    artifacts = {
        "model": model,
        "params": full_params,
        "labels": labels,
        "per_class_metrics_test": per_class_metrics_test,
        "classification_report_test": classification_report_test,
        "y_true_valid": y_valid.tolist() if not X_valid.empty else [],
        "y_pred_valid": y_pred_valid.tolist() if not X_valid.empty else [],
        "y_proba_valid": None if y_proba_valid is None else y_proba_valid.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_true_test": y_test.tolist(),
        "y_proba_test": None if y_proba_test is None else y_proba_test.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    return TrainEvalResult(metrics=metrics, artifacts=artifacts)


def retrain_on_full_train_and_evaluate_test(
    model_name: str,
    params: dict[str, Any],
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_config: dict[str, Any],
) -> TrainEvalResult:
    """Retrain a model on the full train split and evaluate on test."""
    random_state = int(eval_config.get("seed", 42))
    n_classes = int(pd.Series(y_train_full).nunique())
    default_params = get_default_model_params(model_name=model_name, random_state=random_state, n_classes=n_classes)
    full_params = {**default_params, **params}
    model = build_model(model_name=model_name, params=full_params)

    class_weight_cfg = eval_config.get("class_weight", {}) if isinstance(eval_config, dict) else {}
    sample_weight_train = _compute_sample_weight(y_train_full, enabled=bool(class_weight_cfg.get("enabled", False)))
    _fit_model_with_optional_weights(model, X_train_full, y_train_full, sample_weight=sample_weight_train)

    y_pred_test = model.predict(X_test)
    y_proba_test = predict_probabilities(model, X_test)
    test_metrics = compute_metrics(y_test, y_pred_test)
    metrics = {f"test_{k}": v for k, v in test_metrics.items()}

    configured_label_order = eval_config.get("label_order", []) if isinstance(eval_config, dict) else []
    labels = list(configured_label_order) if configured_label_order else sorted(pd.Series(y_train_full).unique().tolist())
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    per_class_metrics_test = compute_per_class_metrics(y_test, y_pred_test, labels=labels)
    classification_report_test = classification_report(
        y_test,
        y_pred_test,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    artifacts = {
        "model": model,
        "params": full_params,
        "labels": labels,
        "per_class_metrics_test": per_class_metrics_test,
        "classification_report_test": classification_report_test,
        "y_pred_test": y_pred_test.tolist(),
        "y_true_test": y_test.tolist(),
        "y_proba_test": None if y_proba_test is None else y_proba_test.tolist(),
        "confusion_matrix": cm.tolist(),
        "trained_on_full_train_split": True,
    }
    return TrainEvalResult(metrics=metrics, artifacts=artifacts)
