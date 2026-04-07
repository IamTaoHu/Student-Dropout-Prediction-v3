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
from src.preprocessing.balancing import apply_balancing
from src.preprocessing.outlier import apply_outlier_filter
from src.preprocessing.tabular_pipeline import run_tabular_preprocessing

NATIVE_CLASS_WEIGHT_SUPPORTED_MODELS = {"decision_tree", "random_forest", "svm"}
SAMPLE_WEIGHT_SUPPORTED_MODELS = {"decision_tree", "random_forest", "svm", "gradient_boosting", "xgboost", "lightgbm", "catboost"}


@dataclass(frozen=True)
class TrainEvalResult:
    """Structured model run result with metrics and artifacts."""

    metrics: dict[str, float]
    artifacts: dict[str, Any]


def run_leakage_safe_stratified_cv(
    model_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    preprocess_config: dict[str, Any],
    outlier_config: dict[str, Any],
    balancing_config: dict[str, Any],
    cv_config: dict[str, Any],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    """Run leakage-safe CV by fitting preprocessing/outlier/smote on each fold's train split only."""
    if "target" not in train_df.columns:
        raise KeyError("run_leakage_safe_stratified_cv requires a 'target' column in train_df.")
    if train_df.empty:
        raise ValueError("run_leakage_safe_stratified_cv received empty train_df.")

    n_splits = int(cv_config.get("n_splits", 5))
    if n_splits < 2:
        raise ValueError("cv n_splits must be >= 2.")
    shuffle = bool(cv_config.get("shuffle", True))
    random_state = int(cv_config.get("random_state", eval_config.get("seed", 42)))

    y_all = train_df["target"].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_results: list[dict[str, Any]] = []

    aggregate_keys = ("macro_f1", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall")

    for fold_idx, (fit_idx, val_idx) in enumerate(skf.split(train_df, y_all), start=1):
        fold_train_df = train_df.iloc[fit_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[val_idx].reset_index(drop=True)
        split_payload = {
            "train": fold_train_df,
            "valid": fold_valid_df,
            # `run_tabular_preprocessing` requires a `test` key; in CV we evaluate on fold-valid.
            "test": fold_valid_df,
        }
        fold_artifacts = run_tabular_preprocessing(split_payload, preprocess_config)
        X_fold_train_filtered, y_fold_train_filtered, outlier_meta = apply_outlier_filter(
            fold_artifacts.X_train,
            fold_artifacts.y_train,
            outlier_config,
        )
        X_fold_train_balanced, y_fold_train_balanced, balancing_meta = apply_balancing(
            X_fold_train_filtered,
            y_fold_train_filtered,
            balancing_config,
        )

        empty_valid_X = pd.DataFrame(columns=fold_artifacts.X_test.columns)
        empty_valid_y = pd.Series(dtype=fold_artifacts.y_test.dtype)
        fold_result = train_and_evaluate(
            model_name=model_name,
            params=params,
            X_train=X_fold_train_balanced,
            y_train=y_fold_train_balanced,
            X_valid=empty_valid_X,
            y_valid=empty_valid_y,
            X_test=fold_artifacts.X_test,
            y_test=fold_artifacts.y_test,
            eval_config=eval_config,
        )
        fold_metrics = {k: float(v) for k, v in fold_result.metrics.items() if k.startswith("test_")}
        fold_results.append(
            {
                "fold_index": int(fold_idx),
                "n_train": int(len(fold_train_df)),
                "n_valid": int(len(fold_valid_df)),
                "metrics": fold_metrics,
                "outlier": outlier_meta,
                "balancing": balancing_meta,
            }
        )

    aggregate: dict[str, float] = {}
    for metric_name in aggregate_keys:
        values = [float(f["metrics"].get(f"test_{metric_name}", np.nan)) for f in fold_results]
        arr = np.asarray(values, dtype=float)
        aggregate[f"cv_{metric_name}_mean"] = float(np.nanmean(arr))
        aggregate[f"cv_{metric_name}_std"] = float(np.nanstd(arr))
    aggregate["cv_num_folds"] = int(n_splits)

    return {
        "config": {
            "n_splits": int(n_splits),
            "shuffle": bool(shuffle),
            "random_state": int(random_state),
        },
        "folds": fold_results,
        "aggregate_metrics": aggregate,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _coerce_label_order(
    label_order: list[int] | tuple[int, ...] | np.ndarray | None,
    fallback: np.ndarray | None = None,
) -> list[int]:
    if label_order is None:
        if fallback is None:
            return []
        return [int(v) for v in np.asarray(fallback, dtype=int).tolist()]
    arr = np.asarray(label_order, dtype=int)
    if arr.ndim != 1:
        raise ValueError("label_order must be one-dimensional.")
    return [int(v) for v in arr.tolist()]


def _align_probabilities_to_label_order(
    model: Any,
    probabilities: np.ndarray,
    label_order: list[int] | tuple[int, ...] | np.ndarray | None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected probability matrix with 2 dimensions, got shape={probs.shape}.")

    model_classes_raw = getattr(model, "classes_", None)
    model_classes = np.asarray(model_classes_raw, dtype=int) if model_classes_raw is not None else None
    requested_labels = _coerce_label_order(label_order, fallback=model_classes)

    if not requested_labels:
        if model_classes is not None and probs.shape[1] != model_classes.size:
            raise ValueError(
                "Probability dimension mismatch with model.classes_. "
                f"shape={probs.shape}, n_classes={model_classes.size}."
            )
        return probs

    if probs.shape[1] != len(requested_labels):
        if model_classes is None:
            raise ValueError(
                "Probability columns do not match requested label order and model.classes_ is unavailable. "
                f"shape={probs.shape}, requested_labels={requested_labels}."
            )
        if probs.shape[1] != model_classes.size:
            raise ValueError(
                "Probability dimension mismatch with model.classes_. "
                f"shape={probs.shape}, model_classes={model_classes.tolist()}."
            )

    if model_classes is None:
        return probs

    class_to_col = {int(cls): idx for idx, cls in enumerate(model_classes.tolist())}
    missing = [int(lbl) for lbl in requested_labels if int(lbl) not in class_to_col]
    if missing:
        raise ValueError(
            f"Requested labels not found in model.classes_: missing={missing}, "
            f"model_classes={model_classes.tolist()}."
        )
    col_indices = [class_to_col[int(lbl)] for lbl in requested_labels]
    return probs[:, col_indices]


def _predict_labels_with_rule(
    model: Any,
    X: pd.DataFrame,
    decision_rule: str,
    label_order: list[int] | tuple[int, ...] | np.ndarray | None,
    multiclass_decision_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    normalized_rule = str(decision_rule or "model_predict").strip().lower()
    supported_rules = {"model_predict", "argmax", "enrolled_margin", "enrolled_middle_band", "enrolled_push"}
    if normalized_rule not in supported_rules:
        raise ValueError(
            f"Unsupported decision_rule='{decision_rule}'. Supported: {', '.join(sorted(supported_rules))}."
        )

    y_pred_native = np.asarray(model.predict(X), dtype=int)
    y_proba = predict_probabilities(model, X, label_order=label_order)

    if normalized_rule == "model_predict":
        return y_pred_native, y_proba
    if y_proba is None:
        return y_pred_native, y_proba

    labels = _coerce_label_order(label_order, fallback=getattr(model, "classes_", None))
    if not labels:
        labels = sorted(np.unique(y_pred_native).astype(int).tolist())
    if y_proba.shape[1] != len(labels):
        raise ValueError(
            "Probability columns do not align with labels for argmax decision rule. "
            f"shape={y_proba.shape}, labels={labels}."
        )
    y_pred = multiclass_predictions_from_probabilities(
        y_proba=y_proba,
        labels=labels,
        strategy=normalized_rule,
        config=multiclass_decision_config,
    )
    return y_pred.astype(int), y_proba


def multiclass_predictions_from_probabilities(
    y_proba: np.ndarray,
    labels: list[int] | tuple[int, ...] | np.ndarray,
    strategy: str,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert class probabilities into final predictions using a configured multiclass decision policy."""
    probs = np.asarray(y_proba, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected probability matrix with 2 dimensions, got shape={probs.shape}.")
    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if probs.shape[1] != labels_arr.size:
        raise ValueError(
            "Probability columns do not align with labels for multiclass decision policy. "
            f"shape={probs.shape}, labels={labels_arr.tolist()}."
        )

    normalized_strategy = str(strategy or "argmax").strip().lower()
    cfg = config if isinstance(config, dict) else {}
    argmax_cols = np.argmax(probs, axis=1)
    argmax_pred = labels_arr[argmax_cols].astype(int)

    if normalized_strategy == "argmax":
        return argmax_pred

    required_labels = {0, 1, 2}
    label_set = set(labels_arr.tolist())
    if labels_arr.size != 3 or label_set != required_labels:
        raise ValueError(
            f"multiclass decision strategy '{normalized_strategy}' requires label_order [0, 1, 2] "
            f"(dropout/enrolled/graduate), got {labels_arr.tolist()}."
        )

    class_to_col = {int(lbl): idx for idx, lbl in enumerate(labels_arr.tolist())}
    enrolled_label = int(cfg.get("enrolled_label", 1))
    dropout_label = int(cfg.get("dropout_label", 0))
    graduate_label = int(cfg.get("graduate_label", 2))

    if normalized_strategy == "enrolled_margin":
        threshold_raw = cfg.get("enrolled_margin_threshold", None)
        if threshold_raw is None:
            raise ValueError(
                "strategy='enrolled_margin' requires inference.multiclass_decision.enrolled_margin_threshold."
            )
        margin_threshold = float(threshold_raw)
        if margin_threshold < 0.0 or margin_threshold > 1.0:
            raise ValueError("enrolled_margin_threshold must be within [0.0, 1.0].")

        sorted_cols = np.argsort(-probs, axis=1)
        top1_cols = sorted_cols[:, 0]
        top2_cols = sorted_cols[:, 1]
        top1_prob = probs[np.arange(probs.shape[0]), top1_cols]
        top2_prob = probs[np.arange(probs.shape[0]), top2_cols]
        pred = argmax_pred.copy()
        override_mask = (pred != enrolled_label) & ((top1_prob - top2_prob) < margin_threshold)
        pred[override_mask] = enrolled_label
        return pred.astype(int)

    if normalized_strategy == "enrolled_middle_band":
        dropout_threshold_raw = cfg.get("dropout_threshold", None)
        graduate_threshold_raw = cfg.get("graduate_threshold", None)
        if dropout_threshold_raw is None or graduate_threshold_raw is None:
            raise ValueError(
                "strategy='enrolled_middle_band' requires both "
                "inference.multiclass_decision.dropout_threshold and graduate_threshold."
            )
        dropout_threshold = float(dropout_threshold_raw)
        graduate_threshold = float(graduate_threshold_raw)
        if dropout_threshold < 0.0 or dropout_threshold > 1.0:
            raise ValueError("dropout_threshold must be within [0.0, 1.0].")
        if graduate_threshold < 0.0 or graduate_threshold > 1.0:
            raise ValueError("graduate_threshold must be within [0.0, 1.0].")

        dropout_col = class_to_col[dropout_label]
        graduate_col = class_to_col[graduate_label]
        p_dropout = probs[:, dropout_col]
        p_graduate = probs[:, graduate_col]
        pred = np.full(shape=(probs.shape[0],), fill_value=enrolled_label, dtype=int)
        dropout_mask = p_dropout >= dropout_threshold
        graduate_mask = (~dropout_mask) & (p_graduate >= graduate_threshold)
        pred[dropout_mask] = dropout_label
        pred[graduate_mask] = graduate_label
        return pred.astype(int)

    if normalized_strategy == "enrolled_push":
        enrolled_col = class_to_col[enrolled_label]
        dropout_col = class_to_col[dropout_label]
        graduate_col = class_to_col[graduate_label]
        p_enrolled = probs[:, enrolled_col]

        pred = argmax_pred.copy()

        threshold_cfg = cfg.get("enrolled_probability_threshold", {})
        if isinstance(threshold_cfg, dict):
            threshold_enabled = bool(threshold_cfg.get("enabled", False))
            threshold_value = threshold_cfg.get("value", threshold_cfg.get("threshold"))
        else:
            threshold_enabled = threshold_cfg is not None
            threshold_value = threshold_cfg
        if threshold_enabled and threshold_value is not None:
            enrolled_threshold = float(threshold_value)
            if enrolled_threshold < 0.0 or enrolled_threshold > 1.0:
                raise ValueError("enrolled_probability_threshold must be within [0.0, 1.0].")
            pred[p_enrolled >= enrolled_threshold] = enrolled_label

        middle_band_cfg = cfg.get("enrolled_middle_band", {})
        if isinstance(middle_band_cfg, dict) and bool(middle_band_cfg.get("enabled", False)):
            min_prob = middle_band_cfg.get("min_enrolled_prob")
            max_gap = middle_band_cfg.get("max_top2_gap")
            if min_prob is None or max_gap is None:
                raise ValueError(
                    "strategy='enrolled_push' with enrolled_middle_band.enabled=true requires "
                    "min_enrolled_prob and max_top2_gap."
                )
            min_enrolled_prob = float(min_prob)
            max_top2_gap = float(max_gap)
            if min_enrolled_prob < 0.0 or min_enrolled_prob > 1.0:
                raise ValueError("enrolled_middle_band.min_enrolled_prob must be within [0.0, 1.0].")
            if max_top2_gap < 0.0 or max_top2_gap > 1.0:
                raise ValueError("enrolled_middle_band.max_top2_gap must be within [0.0, 1.0].")

            competing = np.maximum(probs[:, dropout_col], probs[:, graduate_col])
            gap = np.abs(probs[:, dropout_col] - probs[:, graduate_col])
            middle_band_mask = (
                (pred != enrolled_label)
                & (p_enrolled >= min_enrolled_prob)
                & (p_enrolled <= competing + 1e-12)
                & (gap <= max_top2_gap)
            )
            pred[middle_band_mask] = enrolled_label
        return pred.astype(int)

    raise ValueError(
        f"Unsupported multiclass decision strategy='{normalized_strategy}'. "
        "Supported: argmax, enrolled_margin, enrolled_middle_band, enrolled_push."
    )


def _decision_objective_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    objective: str,
    enrolled_label: int = 1,
) -> float:
    objective_name = str(objective or "macro_f1").strip().lower()
    metrics = compute_metrics(y_true, y_pred)
    if objective_name == "macro_f1":
        return float(metrics["macro_f1"])
    if objective_name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if objective_name == "enrolled_f1":
        per_class = compute_per_class_metrics(y_true, y_pred, labels=sorted(pd.Series(y_true).unique().tolist()))
        return float(per_class.get(str(int(enrolled_label)), {}).get("f1", 0.0))
    if objective_name == "enrolled_recall":
        per_class = compute_per_class_metrics(y_true, y_pred, labels=sorted(pd.Series(y_true).unique().tolist()))
        return float(per_class.get(str(int(enrolled_label)), {}).get("recall", 0.0))
    raise ValueError(
        "Unsupported multiclass decision auto_tune objective. Supported: "
        "macro_f1, enrolled_f1, enrolled_recall, balanced_accuracy."
    )


def _coerce_threshold_grid(values: Any, field_name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"auto_tune.search.{field_name} must be a non-empty list of thresholds.")
    parsed: list[float] = []
    for raw in values:
        val = float(raw)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"auto_tune.search.{field_name} values must be within [0.0, 1.0].")
        parsed.append(val)
    return sorted(set(parsed))


def auto_tune_multiclass_decision_policy(
    y_true_valid: pd.Series,
    y_proba_valid: np.ndarray,
    y_true_test: pd.Series,
    y_proba_test: np.ndarray,
    labels: list[int] | tuple[int, ...] | np.ndarray,
    strategy: str,
    multiclass_decision_config: dict[str, Any],
) -> dict[str, Any]:
    """Tune multiclass decision policy thresholds on validation and apply fixed selection to test."""
    cfg = multiclass_decision_config if isinstance(multiclass_decision_config, dict) else {}
    auto_tune_cfg = cfg.get("auto_tune", {}) if isinstance(cfg.get("auto_tune", {}), dict) else {}
    enabled = bool(auto_tune_cfg.get("enabled", False))
    strategy_name = str(strategy or "").strip().lower()
    objective = str(auto_tune_cfg.get("objective", "macro_f1")).strip().lower()
    split_name = str(auto_tune_cfg.get("split", "validation")).strip().lower()

    response_base = {
        "strategy": strategy_name,
        "auto_tune_enabled": enabled,
        "objective": objective,
        "selection_split": split_name,
    }
    if not enabled:
        return {**response_base, "status": "skipped", "reason": "disabled"}
    if split_name != "validation":
        raise ValueError("inference.multiclass_decision.auto_tune.split must be 'validation'.")
    if strategy_name not in {"enrolled_margin", "enrolled_middle_band", "enrolled_push"}:
        raise ValueError(
            "inference.multiclass_decision.auto_tune is only supported for strategies "
            "{'enrolled_margin', 'enrolled_middle_band', 'enrolled_push'}."
        )

    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.ndim != 1 or labels_arr.size != 3 or set(labels_arr.tolist()) != {0, 1, 2}:
        raise ValueError(
            "multiclass decision auto_tune requires label_order [0, 1, 2] "
            "(dropout/enrolled/graduate)."
        )
    p_valid = np.asarray(y_proba_valid, dtype=float)
    p_test = np.asarray(y_proba_test, dtype=float)
    if p_valid.ndim != 2 or p_test.ndim != 2:
        raise ValueError("multiclass decision auto_tune requires probability arrays with shape [n_samples, n_classes].")
    if p_valid.shape[1] != labels_arr.size or p_test.shape[1] != labels_arr.size:
        raise ValueError("Probability column count does not match labels for multiclass decision auto_tune.")
    if p_valid.shape[0] != len(y_true_valid):
        raise ValueError("Validation labels/probabilities length mismatch in multiclass decision auto_tune.")
    if p_test.shape[0] != len(y_true_test):
        raise ValueError("Test labels/probabilities length mismatch in multiclass decision auto_tune.")

    if objective not in {"macro_f1", "enrolled_f1", "enrolled_recall", "balanced_accuracy"}:
        raise ValueError(
            "inference.multiclass_decision.auto_tune.objective must be one of "
            "{'macro_f1','enrolled_f1','enrolled_recall','balanced_accuracy'}."
        )

    search_cfg = auto_tune_cfg.get("search", {}) if isinstance(auto_tune_cfg.get("search", {}), dict) else {}
    method = str(search_cfg.get("method", "grid")).strip().lower()
    if method != "grid":
        raise ValueError("inference.multiclass_decision.auto_tune.search.method must be 'grid'.")

    selected_cfg: dict[str, Any] = {"strategy": strategy_name}
    search_rows: list[dict[str, float]] = []

    if strategy_name == "enrolled_margin":
        candidates = _coerce_threshold_grid(search_cfg.get("enrolled_margin_thresholds"), "enrolled_margin_thresholds")
        default_threshold = float(cfg.get("enrolled_margin_threshold", 0.10))
        if default_threshold < 0.0 or default_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.enrolled_margin_threshold must be within [0.0, 1.0].")

        best_score = float("-inf")
        best_threshold = float(default_threshold)
        for threshold in candidates:
            candidate_cfg = {"strategy": strategy_name, "enrolled_margin_threshold": float(threshold)}
            pred_valid = multiclass_predictions_from_probabilities(
                y_proba=p_valid,
                labels=labels_arr,
                strategy=strategy_name,
                config=candidate_cfg,
            )
            score = _decision_objective_score(y_true_valid, pred_valid, objective=objective, enrolled_label=1)
            search_rows.append(
                {
                    "enrolled_margin_threshold": float(threshold),
                    "validation_objective": float(score),
                }
            )
            if (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and threshold < best_threshold):
                best_score = float(score)
                best_threshold = float(threshold)

        selected_cfg["enrolled_margin_threshold"] = float(best_threshold)
        selected_cfg["default_enrolled_margin_threshold"] = float(default_threshold)
        selected_cfg["validation_objective_score"] = float(best_score)
        selected_cfg["search_grid_size"] = int(len(candidates))

    elif strategy_name == "enrolled_middle_band":
        dropout_candidates = _coerce_threshold_grid(search_cfg.get("dropout_thresholds"), "dropout_thresholds")
        graduate_candidates = _coerce_threshold_grid(search_cfg.get("graduate_thresholds"), "graduate_thresholds")
        default_dropout = float(cfg.get("dropout_threshold", 0.55))
        default_graduate = float(cfg.get("graduate_threshold", 0.55))
        if default_dropout < 0.0 or default_dropout > 1.0:
            raise ValueError("inference.multiclass_decision.dropout_threshold must be within [0.0, 1.0].")
        if default_graduate < 0.0 or default_graduate > 1.0:
            raise ValueError("inference.multiclass_decision.graduate_threshold must be within [0.0, 1.0].")

        best_score = float("-inf")
        best_gap = float("inf")
        best_dropout = float(default_dropout)
        best_graduate = float(default_graduate)
        for dropout_threshold in dropout_candidates:
            for graduate_threshold in graduate_candidates:
                candidate_cfg = {
                    "strategy": strategy_name,
                    "dropout_threshold": float(dropout_threshold),
                    "graduate_threshold": float(graduate_threshold),
                }
                pred_valid = multiclass_predictions_from_probabilities(
                    y_proba=p_valid,
                    labels=labels_arr,
                    strategy=strategy_name,
                    config=candidate_cfg,
                )
                score = _decision_objective_score(y_true_valid, pred_valid, objective=objective, enrolled_label=1)
                gap = abs(dropout_threshold - default_dropout) + abs(graduate_threshold - default_graduate)
                search_rows.append(
                    {
                        "dropout_threshold": float(dropout_threshold),
                        "graduate_threshold": float(graduate_threshold),
                        "validation_objective": float(score),
                        "default_gap_l1": float(gap),
                    }
                )
                if (
                    (score > best_score + 1e-12)
                    or (
                        abs(score - best_score) <= 1e-12
                        and (
                            (gap < best_gap - 1e-12)
                            or (
                                abs(gap - best_gap) <= 1e-12
                                and (
                                    (dropout_threshold < best_dropout - 1e-12)
                                    or (
                                        abs(dropout_threshold - best_dropout) <= 1e-12
                                        and graduate_threshold < best_graduate - 1e-12
                                    )
                                )
                            )
                        )
                    )
                ):
                    best_score = float(score)
                    best_gap = float(gap)
                    best_dropout = float(dropout_threshold)
                    best_graduate = float(graduate_threshold)

        selected_cfg["dropout_threshold"] = float(best_dropout)
        selected_cfg["graduate_threshold"] = float(best_graduate)
        selected_cfg["default_dropout_threshold"] = float(default_dropout)
        selected_cfg["default_graduate_threshold"] = float(default_graduate)
        selected_cfg["validation_objective_score"] = float(best_score)
        selected_cfg["search_grid_size"] = int(len(dropout_candidates) * len(graduate_candidates))
    elif strategy_name == "enrolled_push":
        threshold_cfg = cfg.get("enrolled_probability_threshold", {})
        if not isinstance(threshold_cfg, dict):
            threshold_cfg = {"enabled": threshold_cfg is not None, "value": threshold_cfg}
        middle_band_cfg = cfg.get("enrolled_middle_band", {})
        if not isinstance(middle_band_cfg, dict):
            middle_band_cfg = {}

        threshold_enabled = bool(threshold_cfg.get("enabled", False))
        middle_band_enabled = bool(middle_band_cfg.get("enabled", False))
        threshold_candidates = (
            _coerce_threshold_grid(search_cfg.get("enrolled_probability_thresholds"), "enrolled_probability_thresholds")
            if threshold_enabled
            else [None]
        )
        min_prob_candidates = (
            _coerce_threshold_grid(search_cfg.get("min_enrolled_probs"), "min_enrolled_probs")
            if middle_band_enabled
            else [None]
        )
        max_gap_candidates = (
            _coerce_threshold_grid(search_cfg.get("max_top2_gaps"), "max_top2_gaps")
            if middle_band_enabled
            else [None]
        )

        default_threshold = float(threshold_cfg.get("value", threshold_cfg.get("threshold", 0.40))) if threshold_enabled else None
        default_min_prob = float(middle_band_cfg.get("min_enrolled_prob", 0.30)) if middle_band_enabled else None
        default_max_gap = float(middle_band_cfg.get("max_top2_gap", 0.05)) if middle_band_enabled else None
        best_score = float("-inf")
        best_gap = float("inf")
        best_threshold = default_threshold
        best_min_prob = default_min_prob
        best_max_gap = default_max_gap

        for threshold in threshold_candidates:
            for min_prob in min_prob_candidates:
                for max_gap in max_gap_candidates:
                    candidate_cfg: dict[str, Any] = {"strategy": strategy_name}
                    if threshold_enabled:
                        candidate_cfg["enrolled_probability_threshold"] = {
                            "enabled": True,
                            "value": float(threshold),
                        }
                    else:
                        candidate_cfg["enrolled_probability_threshold"] = {"enabled": False}
                    if middle_band_enabled:
                        candidate_cfg["enrolled_middle_band"] = {
                            "enabled": True,
                            "min_enrolled_prob": float(min_prob),
                            "max_top2_gap": float(max_gap),
                        }
                    else:
                        candidate_cfg["enrolled_middle_band"] = {"enabled": False}

                    pred_valid = multiclass_predictions_from_probabilities(
                        y_proba=p_valid,
                        labels=labels_arr,
                        strategy=strategy_name,
                        config=candidate_cfg,
                    )
                    score = _decision_objective_score(y_true_valid, pred_valid, objective=objective, enrolled_label=1)
                    default_gap = 0.0
                    if threshold_enabled and default_threshold is not None:
                        default_gap += abs(float(threshold) - float(default_threshold))
                    if middle_band_enabled and default_min_prob is not None and default_max_gap is not None:
                        default_gap += abs(float(min_prob) - float(default_min_prob))
                        default_gap += abs(float(max_gap) - float(default_max_gap))
                    search_rows.append(
                        {
                            "enrolled_probability_threshold": float(threshold) if threshold is not None else None,
                            "min_enrolled_prob": float(min_prob) if min_prob is not None else None,
                            "max_top2_gap": float(max_gap) if max_gap is not None else None,
                            "validation_objective": float(score),
                            "default_gap_l1": float(default_gap),
                        }
                    )
                    if (score > best_score + 1e-12) or (
                        abs(score - best_score) <= 1e-12 and default_gap < best_gap - 1e-12
                    ):
                        best_score = float(score)
                        best_gap = float(default_gap)
                        best_threshold = float(threshold) if threshold is not None else None
                        best_min_prob = float(min_prob) if min_prob is not None else None
                        best_max_gap = float(max_gap) if max_gap is not None else None

        if threshold_enabled:
            selected_cfg["enrolled_probability_threshold"] = {"enabled": True, "value": float(best_threshold)}
            selected_cfg["default_enrolled_probability_threshold"] = float(default_threshold)
        else:
            selected_cfg["enrolled_probability_threshold"] = {"enabled": False}
        if middle_band_enabled:
            selected_cfg["enrolled_middle_band"] = {
                "enabled": True,
                "min_enrolled_prob": float(best_min_prob),
                "max_top2_gap": float(best_max_gap),
            }
            selected_cfg["default_enrolled_middle_band"] = {
                "min_enrolled_prob": float(default_min_prob),
                "max_top2_gap": float(default_max_gap),
            }
        else:
            selected_cfg["enrolled_middle_band"] = {"enabled": False}
        selected_cfg["validation_objective_score"] = float(best_score)
        selected_cfg["search_grid_size"] = int(len(threshold_candidates) * len(min_prob_candidates) * len(max_gap_candidates))
    else:
        raise ValueError(f"Unsupported multiclass auto_tune strategy '{strategy_name}'.")

    pred_valid_selected = multiclass_predictions_from_probabilities(
        y_proba=p_valid,
        labels=labels_arr,
        strategy=strategy_name,
        config=selected_cfg,
    )
    pred_test_selected = multiclass_predictions_from_probabilities(
        y_proba=p_test,
        labels=labels_arr,
        strategy=strategy_name,
        config=selected_cfg,
    )

    return {
        **response_base,
        "status": "applied",
        "selected_parameters": selected_cfg,
        "search_results": search_rows,
        "y_pred_valid_tuned": pred_valid_selected.astype(int).tolist(),
        "y_pred_test_tuned": pred_test_selected.astype(int).tolist(),
        "valid_metrics_tuned": compute_metrics(y_true_valid, pred_valid_selected),
        "test_metrics_tuned": compute_metrics(y_true_test, pred_test_selected),
        "valid_per_class_tuned": compute_per_class_metrics(
            y_true_valid,
            pred_valid_selected,
            labels=labels_arr.tolist(),
        ),
        "test_per_class_tuned": compute_per_class_metrics(
            y_true_test,
            pred_test_selected,
            labels=labels_arr.tolist(),
        ),
        "validation_objective_score": float(selected_cfg.get("validation_objective_score", float("nan"))),
    }


def predict_probabilities(
    model: Any,
    X: pd.DataFrame,
    label_order: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> np.ndarray | None:
    """Return class probabilities when possible."""
    probs: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            probs = np.column_stack([1 - probs_pos, probs_pos])
        else:
            probs = _softmax(scores)
    if probs is None:
        return None
    return _align_probabilities_to_label_order(model=model, probabilities=probs, label_order=label_order)


def _resolve_class_weight_map(
    y: pd.Series,
    class_weight_cfg: dict[str, Any] | None,
) -> tuple[dict[int, float] | None, dict[str, Any]]:
    cfg = class_weight_cfg if isinstance(class_weight_cfg, dict) else {}
    mode = str(cfg.get("mode", cfg.get("strategy", "none"))).strip().lower()
    strategy = str(cfg.get("strategy", mode if mode else "none")).strip().lower()
    enabled = bool(cfg.get("enabled", False))
    if mode == "explicit":
        enabled = True
        strategy = "explicit"
    if not enabled or strategy in {"none", ""}:
        return None, {"enabled": False, "mode": mode or "none", "strategy": strategy or "none", "reason": "disabled"}
    if y.empty:
        return None, {"enabled": enabled, "mode": mode or strategy, "strategy": strategy, "reason": "empty_target"}

    unique_classes = sorted(pd.Series(y).dropna().unique().tolist())
    mapping = cfg.get("class_label_to_index", {})
    if not isinstance(mapping, dict):
        mapping = {}
    class_map: dict[int, float] = {}
    mapping_ci = {str(lbl).strip().lower(): int(idx) for lbl, idx in mapping.items()}
    expected_label_indices = sorted({int(v) for v in mapping.values()})

    def _resolve_class_idx(raw_key: Any) -> int | None:
        key_str = str(raw_key).strip()
        if key_str.lstrip("-").isdigit():
            return int(key_str)
        mapped = mapping.get(key_str)
        if mapped is not None:
            return int(mapped)
        mapped_ci = mapping_ci.get(key_str.lower())
        if mapped_ci is not None:
            return int(mapped_ci)
        return None

    explicit_map = cfg.get("values")
    if not isinstance(explicit_map, dict):
        explicit_map = cfg.get("class_weight_map")
    if isinstance(explicit_map, dict) and explicit_map:
        unresolved_keys: list[str] = []
        for key, value in explicit_map.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"class_weight value for key '{key}' must be numeric.")
            if weight <= 0.0:
                raise ValueError(f"class_weight value for key '{key}' must be > 0.")
            class_idx = _resolve_class_idx(key)
            if class_idx is None:
                unresolved_keys.append(str(key))
                continue
            class_map[int(class_idx)] = float(weight)

        if unresolved_keys:
            allowed_keys = sorted([str(k) for k in mapping.keys()])
            raise ValueError(
                "class_weight keys do not match current class mapping. "
                f"unresolved_keys={unresolved_keys}, allowed_label_keys={allowed_keys}, "
                f"allowed_index_keys={expected_label_indices}."
            )
        if strategy == "explicit":
            expected_indices = expected_label_indices if expected_label_indices else [int(v) for v in unique_classes]
            missing = [idx for idx in expected_indices if idx not in class_map]
            extra = sorted([idx for idx in class_map.keys() if idx not in expected_indices])
            if missing or extra:
                raise ValueError(
                    "explicit class_weight values must match the full current class mapping. "
                    f"missing_indices={missing}, extra_indices={extra}, expected_indices={expected_indices}."
                )
        if class_map:
            return class_map, {"enabled": True, "mode": mode or strategy, "strategy": strategy, "weight_map": class_map, "source": "explicit"}

    if strategy == "balanced":
        counts = y.value_counts(dropna=False)
        n_classes = len(counts)
        total = float(len(y))
        class_map = {int(cls): total / (n_classes * float(count)) for cls, count in counts.items() if count > 0}
        return class_map, {"enabled": True, "mode": mode or strategy, "strategy": strategy, "weight_map": class_map, "source": "balanced"}

    if strategy == "enrolled_boost":
        base_weight = float(cfg.get("base_weight", 1.0))
        enrolled_boost = float(cfg.get("enrolled_boost", 1.5))
        focus_label = str(cfg.get("focus_class_label", "Enrolled"))
        focus_idx = mapping.get(focus_label)
        if focus_idx is None:
            for lbl, idx in mapping.items():
                if str(lbl).strip().lower() == focus_label.strip().lower():
                    focus_idx = idx
                    break
        class_map = {int(cls): base_weight for cls in unique_classes}
        if focus_idx is not None and int(focus_idx) in class_map:
            class_map[int(focus_idx)] = enrolled_boost
            return class_map, {"enabled": True, "mode": mode or strategy, "strategy": strategy, "weight_map": class_map, "source": "enrolled_boost"}
        return class_map, {
            "enabled": True,
            "mode": mode or strategy,
            "strategy": strategy,
            "weight_map": class_map,
            "source": "enrolled_boost_no_focus_match",
            "focus_class_label": focus_label,
        }

    raise ValueError(
        "Unsupported class_weight configuration. "
        f"mode='{mode}', strategy='{strategy}'. "
        "Supported strategies: explicit, balanced, enrolled_boost."
    )


def _compute_sample_weight(
    y: pd.Series,
    class_weight_cfg: dict[str, Any] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    class_map, info = _resolve_class_weight_map(y, class_weight_cfg)
    if not class_map:
        return None, info
    mapped = y.map(class_map)
    if mapped.isna().any():
        missing_labels = sorted({int(v) for v in pd.Series(y[mapped.isna()]).astype(int).tolist()})
        raise ValueError(
            "Failed to build sample_weight because class_weight map is missing labels in y. "
            f"missing_labels={missing_labels}, class_weight_labels={sorted(class_map.keys())}."
        )
    sample_weight = mapped.fillna(1.0).to_numpy(dtype=float)
    info = {**info, "sample_weight_non_default_count": int(np.sum(sample_weight != 1.0))}
    return sample_weight, info


def _fit_model_with_optional_weights(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None,
    require_sample_weight_support: bool = False,
) -> bool:
    if sample_weight is None:
        model.fit(X, y)
        return False
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return True
    except TypeError as exc:
        if require_sample_weight_support:
            raise ValueError(
                "Class weights were requested but sample_weight is not supported by this model fit path."
            ) from exc
        model.fit(X, y)
        return False


def _trial_metric_alias_value(metrics: dict[str, Any], metric_name: str) -> float:
    token = str(metric_name or "").strip().lower()
    alias_map = {
        "macro_f1": "macro_f1",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "enrolled_f1": "f1_enrolled",
        "f1_enrolled": "f1_enrolled",
        "enrolled_recall": "recall_enrolled",
        "recall_enrolled": "recall_enrolled",
        "enrolled_precision": "precision_enrolled",
        "precision_enrolled": "precision_enrolled",
    }
    resolved = alias_map.get(token, token)
    raw = metrics.get(resolved, metrics.get(token))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def _validation_objective_metric_value(
    metrics: dict[str, Any],
    per_class: dict[str, Any],
    objective_metric: str,
    enrolled_label: int = 1,
) -> float:
    token = str(objective_metric or "macro_f1").strip().lower()
    if token in {"macro_f1", "balanced_accuracy", "accuracy"}:
        return float(metrics.get(token, float("-inf")))
    class_metrics = per_class.get(str(int(enrolled_label)), {}) if isinstance(per_class, dict) else {}
    if token == "enrolled_f1":
        return float(class_metrics.get("f1", float("-inf")))
    if token == "enrolled_recall":
        return float(class_metrics.get("recall", float("-inf")))
    if token == "enrolled_precision":
        return float(class_metrics.get("precision", float("-inf")))
    raise ValueError(
        "Unsupported tuning objective_metric for objective_source='validation'. "
        "Supported: macro_f1, balanced_accuracy, accuracy, enrolled_f1, enrolled_recall, enrolled_precision."
    )


def _collect_trial_ranking_metrics(trial: Any, objective_source: str) -> dict[str, float]:
    if objective_source == "validation":
        metrics = trial.user_attrs.get("validation_metrics", {}) if isinstance(trial.user_attrs.get("validation_metrics", {}), dict) else {}
        per_class = trial.user_attrs.get("per_class_metrics", {}) if isinstance(trial.user_attrs.get("per_class_metrics", {}), dict) else {}
        enrolled = per_class.get("1", {}) if isinstance(per_class.get("1", {}), dict) else {}
        return {
            "macro_f1": float(metrics.get("macro_f1", float("-inf"))),
            "balanced_accuracy": float(metrics.get("balanced_accuracy", float("-inf"))),
            "accuracy": float(metrics.get("accuracy", float("-inf"))),
            "f1_enrolled": float(enrolled.get("f1", float("-inf"))),
            "recall_enrolled": float(enrolled.get("recall", float("-inf"))),
            "precision_enrolled": float(enrolled.get("precision", float("-inf"))),
        }
    if objective_source == "paper_cv":
        metrics = trial.user_attrs.get("cv_aggregate_metrics", {}) if isinstance(trial.user_attrs.get("cv_aggregate_metrics", {}), dict) else {}
        return {
            "macro_f1": float(metrics.get("cv_macro_f1_mean", float("-inf"))),
            "balanced_accuracy": float(metrics.get("cv_balanced_accuracy_mean", float("-inf"))),
            "accuracy": float(metrics.get("cv_accuracy_mean", float("-inf"))),
        }
    return {
        "macro_f1": float(trial.value) if trial.value is not None else float("-inf"),
    }


def _select_best_optuna_trial(
    trials: list[Any],
    direction: str,
    objective_source: str,
    ranking_metrics: list[str],
) -> Any:
    completed = [trial for trial in trials if "COMPLETE" in str(getattr(trial, "state", "")).upper() and trial.value is not None]
    if not completed:
        raise ValueError("No completed Optuna trials were available for model selection.")
    descending = str(direction).strip().lower() != "minimize"

    def _sort_key(trial: Any) -> tuple[Any, ...]:
        objective_value = float(trial.value)
        metrics = _collect_trial_ranking_metrics(trial, objective_source=objective_source)
        metric_values = []
        for metric_name in ranking_metrics:
            val = _trial_metric_alias_value(metrics, metric_name)
            metric_values.append(-val if descending else val)
        return ((-objective_value if descending else objective_value), *metric_values, int(trial.number))

    return sorted(completed, key=_sort_key)[0]


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
    mode = str(tuning_cfg.get("mode", "optuna")).strip().lower()
    if mode not in {"optuna", ""}:
        raise ValueError(f"Unsupported tuning mode '{mode}'. Supported: optuna.")
    random_state = int(tuning_cfg.get("seed", 42))
    cv_folds = int(tuning_cfg.get("cv_folds", 3))
    scoring_raw = str(tuning_cfg.get("scoring", "f1_macro")).strip().lower()
    scoring_aliases = {
        "macro_f1": "f1_macro",
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    scoring = scoring_aliases.get(scoring_raw, scoring_raw)
    direction = str(tuning_cfg.get("direction", "maximize")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("tuning.direction must be either 'maximize' or 'minimize'.")
    objective_source = str(tuning_cfg.get("objective_source", "cv")).strip().lower()
    objective_metric = str(tuning_cfg.get("objective_metric", "macro_f1")).strip().lower()
    use_class_weights = bool(tuning_cfg.get("use_class_weights", False))
    class_weight_cfg = tuning_cfg.get("class_weight", {}) if isinstance(tuning_cfg.get("class_weight", {}), dict) else {}
    trial_ranking_cfg = tuning_cfg.get("trial_selection", {}) if isinstance(tuning_cfg.get("trial_selection", {}), dict) else {}
    ranking_metrics = trial_ranking_cfg.get("ranking_metrics", [])
    if not isinstance(ranking_metrics, list) or not ranking_metrics:
        ranking_metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    ranking_metrics = [str(metric).strip() for metric in ranking_metrics if str(metric).strip()]
    n_classes = int(pd.Series(y_train).nunique())
    cv_train_df = tuning_cfg.get("cv_train_df")
    cv_preprocess_config = tuning_cfg.get("cv_preprocess_config")
    cv_outlier_config = tuning_cfg.get("cv_outlier_config")
    cv_balancing_config = tuning_cfg.get("cv_balancing_config")
    cv_config = tuning_cfg.get("cv_config", {})
    search_space_overrides = (
        tuning_cfg.get("search_space_overrides", {})
        if isinstance(tuning_cfg.get("search_space_overrides", {}), dict)
        else {}
    )

    if fixed_params is None:
        fixed_params = {}

    def objective(trial: Any) -> float:
        params = suggest_optuna_params(
            trial,
            model_name=model_name,
            n_classes=n_classes,
            random_state=random_state,
            search_space_overrides=search_space_overrides,
        )
        if fixed_params:
            params.update(fixed_params)
        model = build_model(model_name, params)
        if objective_source == "validation":
            if X_valid is None or y_valid is None:
                raise ValueError("objective_source='validation' requires X_valid and y_valid.")
            effective_class_weight_cfg = class_weight_cfg if class_weight_cfg else {"enabled": use_class_weights, "strategy": "balanced"}
            if not effective_class_weight_cfg.get("enabled", False):
                effective_class_weight_cfg = {"enabled": use_class_weights, "strategy": "balanced"}
            sample_weight, _ = _compute_sample_weight(y_train, class_weight_cfg=effective_class_weight_cfg)
            _fit_model_with_optional_weights(model, X_train, y_train, sample_weight=sample_weight)
            y_pred = model.predict(X_valid)
            metrics = compute_metrics(y_valid, y_pred)
            per_class = compute_per_class_metrics(y_valid, y_pred, labels=sorted(pd.Series(y_valid).unique().tolist()))
            trial.set_user_attr("validation_metrics", metrics)
            trial.set_user_attr("per_class_metrics", per_class)
            trial.set_user_attr("objective_metric", objective_metric)
            return _validation_objective_metric_value(
                metrics=metrics,
                per_class=per_class,
                objective_metric=objective_metric,
            )
        if objective_source == "paper_cv":
            if not isinstance(cv_train_df, pd.DataFrame):
                raise ValueError("objective_source='paper_cv' requires tuning_cfg.cv_train_df DataFrame.")
            if not isinstance(cv_preprocess_config, dict):
                raise ValueError("objective_source='paper_cv' requires tuning_cfg.cv_preprocess_config.")
            if not isinstance(cv_outlier_config, dict):
                raise ValueError("objective_source='paper_cv' requires tuning_cfg.cv_outlier_config.")
            if not isinstance(cv_balancing_config, dict):
                raise ValueError("objective_source='paper_cv' requires tuning_cfg.cv_balancing_config.")
            cv_eval = run_leakage_safe_stratified_cv(
                model_name=model_name,
                params=params,
                train_df=cv_train_df,
                preprocess_config=cv_preprocess_config,
                outlier_config=cv_outlier_config,
                balancing_config=cv_balancing_config,
                cv_config={
                    **(cv_config if isinstance(cv_config, dict) else {}),
                    "n_splits": int(cv_folds),
                    "random_state": int(random_state),
                },
                eval_config={
                    "seed": int(random_state),
                    "class_weight": {
                        **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                    },
                    "label_order": tuning_cfg.get("label_order", []),
                    "decision_rule": tuning_cfg.get("decision_rule", "model_predict"),
                    "multiclass_decision": tuning_cfg.get("multiclass_decision", {}),
                },
            )
            agg = cv_eval.get("aggregate_metrics", {})
            cv_macro = float(agg.get("cv_macro_f1_mean", float("-inf")))
            trial.set_user_attr("cv_aggregate_metrics", agg)
            trial.set_user_attr("cv_num_folds", int(agg.get("cv_num_folds", cv_folds)))
            return cv_macro

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        score = float(np.mean(scores))
        trial.set_user_attr("cv_score", score)
        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    selected_trial = _select_best_optuna_trial(
        trials=list(study.trials),
        direction=direction,
        objective_source=objective_source,
        ranking_metrics=ranking_metrics,
    )
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
        "best_trial_number": int(selected_trial.number),
        "best_value": float(selected_trial.value),
        "scoring": scoring,
        "cv_folds": cv_folds,
        "objective_source": objective_source,
        "objective_metric": objective_metric,
        "trial_ranking_metrics": ranking_metrics,
        "best_validation_metrics": dict(selected_trial.user_attrs.get("validation_metrics", {})),
        "best_cv_aggregate_metrics": dict(selected_trial.user_attrs.get("cv_aggregate_metrics", {})),
        "best_per_class_metrics": dict(selected_trial.user_attrs.get("per_class_metrics", {})),
        "best_per_class_f1": {
            key: float(value.get("f1", 0.0))
            for key, value in dict(selected_trial.user_attrs.get("per_class_metrics", {})).items()
            if isinstance(value, dict)
        },
        "trials": trials,
        "trials_dataframe": trials_df,
    }
    best_params = dict(selected_trial.params)
    if fixed_params:
        best_params.update(fixed_params)
    return best_params, float(selected_trial.value), details


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
    class_weight_cfg = eval_config.get("class_weight", {}) if isinstance(eval_config, dict) else {}
    class_weight_requested = (
        bool(class_weight_cfg.get("enabled", False)) or str(class_weight_cfg.get("mode", "")).strip().lower() == "explicit"
    ) if isinstance(class_weight_cfg, dict) else False
    class_map, class_weight_info = _resolve_class_weight_map(y_train, class_weight_cfg)
    native_param_supported = model_name in NATIVE_CLASS_WEIGHT_SUPPORTED_MODELS
    sample_weight_supported = model_name in SAMPLE_WEIGHT_SUPPORTED_MODELS
    if class_map and class_weight_requested and (not native_param_supported and not sample_weight_supported):
        raise ValueError(
            f"Class weights requested for model='{model_name}', but no native class_weight or sample_weight path is supported."
        )
    if native_param_supported and class_map:
        full_params["class_weight"] = class_map
        class_weight_info = {**class_weight_info, "model_param_class_weight_applied": True}
    else:
        class_weight_info = {**class_weight_info, "model_param_class_weight_applied": False}
    model = build_model(model_name=model_name, params=full_params)

    sample_weight_train, sample_weight_info = _compute_sample_weight(y_train, class_weight_cfg=class_weight_cfg)
    if not sample_weight_supported:
        sample_weight_train = None
    sample_weight_applied = _fit_model_with_optional_weights(
        model,
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        require_sample_weight_support=bool(class_weight_requested and class_map and not native_param_supported),
    )
    class_weight_info = {
        **class_weight_info,
        **sample_weight_info,
        "model_name": model_name,
        "class_weight_requested": class_weight_requested,
        "sample_weight_applied": bool(sample_weight_applied),
        "effective_mechanism": (
            "model_param+sample_weight"
            if class_weight_info.get("model_param_class_weight_applied") and sample_weight_applied
            else "model_param_only"
            if class_weight_info.get("model_param_class_weight_applied")
            else "sample_weight"
            if sample_weight_applied
            else "none"
        ),
    }
    class_weight_info["class_weight_native_param_supported"] = native_param_supported
    class_weight_info["class_weight_sample_weight_supported"] = sample_weight_supported
    class_weight_info["class_weight_supported"] = bool(native_param_supported or sample_weight_supported)
    class_weight_info["class_weight_applied"] = bool(
        class_weight_info.get("model_param_class_weight_applied") or class_weight_info.get("sample_weight_applied")
    )
    class_weight_info["class_weight_application_method"] = (
        "native_class_weight"
        if class_weight_info.get("model_param_class_weight_applied")
        else "sample_weight"
        if class_weight_info.get("sample_weight_applied")
        else "none"
    )
    if native_param_supported and sample_weight_supported:
        class_weight_info["class_weight_backend_note"] = "Native class_weight and sample_weight paths are both available."
    elif sample_weight_supported:
        class_weight_info["class_weight_backend_note"] = "Sample_weight path used for class weighting."
    elif native_param_supported:
        class_weight_info["class_weight_backend_note"] = "Native class_weight parameter path used."
    else:
        class_weight_info["class_weight_backend_note"] = "No class_weight support path available."
    decision_rule = str(eval_config.get("decision_rule", "model_predict")).strip().lower()
    multiclass_decision_cfg = eval_config.get("multiclass_decision", {}) if isinstance(eval_config, dict) else {}
    configured_label_order = eval_config.get("label_order", []) if isinstance(eval_config, dict) else []
    labels = list(configured_label_order) if configured_label_order else sorted(pd.Series(y_train).unique().tolist())

    y_pred_valid = np.array([], dtype=int)
    y_proba_valid = None
    if not X_valid.empty:
        y_pred_valid, y_proba_valid = _predict_labels_with_rule(
            model=model,
            X=X_valid,
            decision_rule=decision_rule,
            label_order=labels,
            multiclass_decision_config=multiclass_decision_cfg,
        )
    y_pred_test, y_proba_test = _predict_labels_with_rule(
        model=model,
        X=X_test,
        decision_rule=decision_rule,
        label_order=labels,
        multiclass_decision_config=multiclass_decision_cfg,
    )

    metrics: dict[str, float] = {}
    per_class_metrics_valid: dict[str, dict[str, float]] = {}
    classification_report_valid: dict[str, Any] = {}
    if not X_valid.empty:
        valid_metrics = compute_metrics(y_valid, y_pred_valid)
        metrics.update({f"valid_{k}": v for k, v in valid_metrics.items()})
        per_class_metrics_valid = compute_per_class_metrics(y_valid, y_pred_valid, labels=labels)
        classification_report_valid = classification_report(
            y_valid,
            y_pred_valid,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    test_metrics = compute_metrics(y_test, y_pred_test)
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

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
        "per_class_metrics_valid": per_class_metrics_valid,
        "per_class_metrics_test": per_class_metrics_test,
        "classification_report_valid": classification_report_valid,
        "classification_report_test": classification_report_test,
        "y_true_valid": y_valid.tolist() if not X_valid.empty else [],
        "y_pred_valid": y_pred_valid.tolist() if not X_valid.empty else [],
        "y_proba_valid": None if y_proba_valid is None else y_proba_valid.tolist(),
        "y_pred_test": y_pred_test.tolist(),
        "y_true_test": y_test.tolist(),
        "y_proba_test": None if y_proba_test is None else y_proba_test.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_weight_info": class_weight_info,
        "decision_rule": decision_rule,
        "multiclass_decision": multiclass_decision_cfg if isinstance(multiclass_decision_cfg, dict) else {},
        "decision_rule_applied_on_probabilities": bool(decision_rule != "model_predict" and y_proba_test is not None),
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
    class_weight_cfg = eval_config.get("class_weight", {}) if isinstance(eval_config, dict) else {}
    class_weight_requested = (
        bool(class_weight_cfg.get("enabled", False)) or str(class_weight_cfg.get("mode", "")).strip().lower() == "explicit"
    ) if isinstance(class_weight_cfg, dict) else False
    class_map, class_weight_info = _resolve_class_weight_map(y_train_full, class_weight_cfg)
    native_param_supported = model_name in NATIVE_CLASS_WEIGHT_SUPPORTED_MODELS
    sample_weight_supported = model_name in SAMPLE_WEIGHT_SUPPORTED_MODELS
    if class_map and class_weight_requested and (not native_param_supported and not sample_weight_supported):
        raise ValueError(
            f"Class weights requested for model='{model_name}', but no native class_weight or sample_weight path is supported."
        )
    if native_param_supported and class_map:
        full_params["class_weight"] = class_map
        class_weight_info = {**class_weight_info, "model_param_class_weight_applied": True}
    else:
        class_weight_info = {**class_weight_info, "model_param_class_weight_applied": False}
    model = build_model(model_name=model_name, params=full_params)

    sample_weight_train, sample_weight_info = _compute_sample_weight(y_train_full, class_weight_cfg=class_weight_cfg)
    if not sample_weight_supported:
        sample_weight_train = None
    sample_weight_applied = _fit_model_with_optional_weights(
        model,
        X_train_full,
        y_train_full,
        sample_weight=sample_weight_train,
        require_sample_weight_support=bool(class_weight_requested and class_map and not native_param_supported),
    )
    class_weight_info = {
        **class_weight_info,
        **sample_weight_info,
        "model_name": model_name,
        "class_weight_requested": class_weight_requested,
        "sample_weight_applied": bool(sample_weight_applied),
        "effective_mechanism": (
            "model_param+sample_weight"
            if class_weight_info.get("model_param_class_weight_applied") and sample_weight_applied
            else "model_param_only"
            if class_weight_info.get("model_param_class_weight_applied")
            else "sample_weight"
            if sample_weight_applied
            else "none"
        ),
    }
    class_weight_info["class_weight_native_param_supported"] = native_param_supported
    class_weight_info["class_weight_sample_weight_supported"] = sample_weight_supported
    class_weight_info["class_weight_supported"] = bool(native_param_supported or sample_weight_supported)
    class_weight_info["class_weight_applied"] = bool(
        class_weight_info.get("model_param_class_weight_applied") or class_weight_info.get("sample_weight_applied")
    )
    class_weight_info["class_weight_application_method"] = (
        "native_class_weight"
        if class_weight_info.get("model_param_class_weight_applied")
        else "sample_weight"
        if class_weight_info.get("sample_weight_applied")
        else "none"
    )
    if native_param_supported and sample_weight_supported:
        class_weight_info["class_weight_backend_note"] = "Native class_weight and sample_weight paths are both available."
    elif sample_weight_supported:
        class_weight_info["class_weight_backend_note"] = "Sample_weight path used for class weighting."
    elif native_param_supported:
        class_weight_info["class_weight_backend_note"] = "Native class_weight parameter path used."
    else:
        class_weight_info["class_weight_backend_note"] = "No class_weight support path available."

    decision_rule = str(eval_config.get("decision_rule", "model_predict")).strip().lower()
    multiclass_decision_cfg = eval_config.get("multiclass_decision", {}) if isinstance(eval_config, dict) else {}
    configured_label_order = eval_config.get("label_order", []) if isinstance(eval_config, dict) else []
    labels = list(configured_label_order) if configured_label_order else sorted(pd.Series(y_train_full).unique().tolist())

    y_pred_test, y_proba_test = _predict_labels_with_rule(
        model=model,
        X=X_test,
        decision_rule=decision_rule,
        label_order=labels,
        multiclass_decision_config=multiclass_decision_cfg,
    )
    test_metrics = compute_metrics(y_test, y_pred_test)
    metrics = {f"test_{k}": v for k, v in test_metrics.items()}

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
        "class_weight_info": class_weight_info,
        "decision_rule": decision_rule,
        "multiclass_decision": multiclass_decision_cfg if isinstance(multiclass_decision_cfg, dict) else {},
        "decision_rule_applied_on_probabilities": bool(decision_rule != "model_predict" and y_proba_test is not None),
    }
    return TrainEvalResult(metrics=metrics, artifacts=artifacts)
