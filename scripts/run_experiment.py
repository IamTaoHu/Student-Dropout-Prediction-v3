"""Run config-driven benchmark experiments for UCT Student and OULAD."""

from __future__ import annotations

# Imports

import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.adapters.oulad_adapter import adapt_oulad_schema
from src.data.adapters.uct_student_adapter import adapt_uct_student_schema
from src.data.feature_builders.oulad_paper_features import build_oulad_paper_features
from src.data.feature_builders.uct_stage2_feature_sharpening import (
    DEFAULT_STAGE2_FEATURE_GROUPS,
    build_stage2_feature_sharpening_split_data,
)
from src.data.feature_builders.uct_stage2_feature_separation import (
    DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS,
    build_advanced_enrolled_feature_separation_split_data,
)
from src.data.feature_builders.uct_stage2_advanced_features import (
    DEFAULT_INTERACTION_GROUPS,
    DEFAULT_PROTOTYPE_METRIC_SET,
    build_stage2_interaction_split_data,
    build_stage2_prototype_distance_features,
    build_stage2_selective_interaction_split_data,
)
from src.data.feature_builders.uci_student_paper_style_features import (
    build_uci_student_paper_style_features,
)
from src.data.feature_builders.uct_student_features import build_uct_student_features
from src.data.loaders.oulad_loader import load_oulad_tables
from src.data.loaders.uct_student_loader import load_uct_student_dataframe, load_uct_student_predefined_splits
from src.data.splits.stratified_split import SplitConfig, stratified_train_valid_test_split
from src.data.target_mapping.binary import map_binary_target
from src.data.target_mapping.four_class import map_four_class_target
from src.data.target_mapping.three_class import map_three_class_target
from src.models.registry import list_available_models
from src.models.train_eval import (
    auto_tune_multiclass_decision_policy,
    compute_metrics,
    compute_per_class_metrics,
    retrain_on_full_train_and_evaluate_test,
    run_leakage_safe_stratified_cv,
    train_and_evaluate,
    tune_model_with_optuna,
)
from src.models.two_stage_uct import Stage2PositiveProbabilityCalibrator, TwoStageUct3ClassClassifier
from src.preprocessing.balancing import apply_balancing
from src.preprocessing.outlier import apply_outlier_filter
from src.preprocessing.tabular_pipeline import detect_feature_types, run_tabular_preprocessing
from src.reporting.artifact_manifest import update_artifact_manifest
from src.reporting.benchmark_contract import BENCHMARK_SUMMARY_VERSION, REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS
from src.reporting.benchmark_summary import save_benchmark_summary
from src.reporting.generate_all_figures import generate_all_figures
from src.reporting.standard_artifacts import (
    ensure_standard_output_layout,
    resolve_results_dir,
    write_skipped_explainability_report,
)
from src.experiment.runners import (
    run_benchmark_mode,
    run_error_audit_mode,
    run_threshold_tuning_mode,
    run_two_stage_mode,
)
from src.experiment.config_resolution import (
    _deep_merge_dicts,
    _normalize_dataset_name,
    _normalize_experiment_config_schema,
    _resolve_cv_reporting_config,
    _resolve_decision_rule_config,
    _resolve_experiment_feature_config,
    _resolve_global_balance_guard_config,
    _resolve_model_decision_rule_config,
    _resolve_model_selection_config,
    _resolve_per_model_trial_budgets,
    _resolve_two_stage_stage2_advanced_config,
    _resolve_two_stage_stage2_feature_separation_config,
    _resolve_two_stage_stage2_feature_sharpening_config,
    _resolve_two_stage_stage2_finite_sanitation_config,
    _resolve_two_stage_stage2_selective_interactions_config,
    load_yaml,
)
from src.experiment.eval_validation import (
    _assert_1d_label_vector,
    _assert_probability_payload,
    _assert_same_length_arrays,
    _debug_lengths,
    _validate_two_stage_eval_bundle,
)
from src.experiment.feature_sanitation import validate_and_sanitize_feature_matrix
from src.experiment.finalization import (
    finalize_benchmark_run,
    finalize_error_audit_run,
    finalize_threshold_tuning_run,
)
from src.experiment.finalization.shared_types import (
    BenchmarkExecutionResult,
    BenchmarkFinalizationContext,
)
from src.experiment.model_selection import _apply_global_balance_guard, _sort_leaderboard_with_tiebreak
from src.experiment.schema_validation import (
    _duplicate_columns,
    _log_duplicate_feature_check,
    _sanitize_lightgbm_feature_frames,
    _sanitize_lightgbm_feature_name,
    _sanitize_lightgbm_feature_names,
    align_feature_schema,
    validate_feature_schema,
)
from src.experiment.two_stage_feature_bundle import _prepare_two_stage_stage2_feature_bundle
from src.reporting.prediction_exports import (
    _add_named_cv_per_class_metrics,
    _add_named_per_class_metrics,
    _add_named_per_class_metrics_with_suffix,
    _add_named_validation_per_class_metrics,
    _build_prediction_export_dataframe,
    _metric_label_token,
    _resolve_metric_column,
    _safe_filename_token,
)
from src.reporting.runtime_persistence import (
    _ensure_explainability_compatible_artifact_paths,
    _mirror_root_artifacts_to_runtime,
    _persist_per_model_run_outputs,
    _persist_required_contract_outputs,
    _persist_runtime_artifacts,
    _save_dataframe,
    _save_series,
    _status_from_path,
    _write_benchmark_failure_summary,
)
SUPPORTED_DATASETS = {"uct_student", "oulad"}

# Orchestration helpers still local










































































def _resolve_onehot_metadata_and_validate(
    artifacts: Any,
    preprocess_cfg: dict[str, Any],
    preprocessing_exp_cfg: dict[str, Any],
) -> dict[str, Any]:
    metadata = artifacts.metadata if hasattr(artifacts, "metadata") and isinstance(artifacts.metadata, dict) else {}
    categorical_columns = metadata.get("categorical_columns", [])
    numeric_columns = metadata.get("numeric_columns", [])
    if not isinstance(categorical_columns, list):
        categorical_columns = []
    if not isinstance(numeric_columns, list):
        numeric_columns = []
    encoded_categorical_feature_count = int(metadata.get("encoded_categorical_feature_count", 0) or 0)
    per_column_counts = metadata.get("onehot_column_category_counts", {})
    if not isinstance(per_column_counts, dict):
        per_column_counts = {}
    per_column_labels = metadata.get("onehot_column_category_labels", {})
    if not isinstance(per_column_labels, dict):
        per_column_labels = {}
    locked_vocabulary_mode = bool(metadata.get("onehot_categories_locked", False))
    vocabulary_source = metadata.get("onehot_categories_source")

    require_onehot = bool(preprocessing_exp_cfg.get("require_onehot_encoding", False))
    if require_onehot and not bool(preprocess_cfg.get("onehot", False)):
        raise ValueError("preprocessing.require_onehot_encoding=true but categorical one-hot encoding is disabled.")
    if require_onehot and len(categorical_columns) > 0 and encoded_categorical_feature_count <= 0:
        raise ValueError(
            "preprocessing.require_onehot_encoding=true but encoded categorical feature count is zero."
        )

    return {
        "onehot_enabled": bool(preprocess_cfg.get("onehot", False)),
        "require_onehot_encoding": require_onehot,
        "input_numeric_feature_count": int(len(numeric_columns)),
        "input_categorical_feature_count": int(len(categorical_columns)),
        "encoded_categorical_feature_count": int(encoded_categorical_feature_count),
        "preprocessed_feature_count": int(artifacts.X_train.shape[1]),
        "stable_locked_vocabulary_mode": locked_vocabulary_mode,
        "vocabulary_source": str(vocabulary_source) if vocabulary_source else None,
        "per_column_category_counts": {
            str(column): int(count)
            for column, count in per_column_counts.items()
        },
        "per_column_category_labels": {
            str(column): [str(value) for value in values]
            for column, values in per_column_labels.items()
            if isinstance(values, list)
        },
    }


















def _persist_paper_style_cv_artifacts(
    *,
    output_dir: Path,
    model_results: dict[str, Any],
) -> dict[str, str]:
    runtime_dir = output_dir / "runtime_artifacts"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    cv_model_rows: list[dict[str, Any]] = []
    cv_fold_payload: dict[str, Any] = {}
    for model_name, payload in model_results.items():
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        cv_results = payload.get("cv_results", {}) if isinstance(payload.get("cv_results", {}), dict) else {}
        if not cv_results:
            continue
        row = {"model": model_name}
        for key, value in metrics.items():
            if str(key).startswith("cv_"):
                row[str(key)] = value
        cv_model_rows.append(row)
        cv_fold_payload[model_name] = cv_results

    artifact_paths: dict[str, str] = {}
    if cv_model_rows:
        cv_model_results_path = runtime_dir / "cv_model_results.csv"
        pd.DataFrame(cv_model_rows).to_csv(cv_model_results_path, index=False)
        artifact_paths["cv_model_results_csv"] = str(cv_model_results_path)

        preferred_columns = [
            "model",
            "cv_macro_f1",
            "cv_macro_precision",
            "cv_macro_recall",
            "cv_macro_f1_mean",
            "cv_macro_precision_mean",
            "cv_macro_recall_mean",
        ]
        cv_summary_df = pd.DataFrame(cv_model_rows)
        ordered = [col for col in preferred_columns if col in cv_summary_df.columns]
        ordered.extend([col for col in cv_summary_df.columns if col not in ordered])
        cv_summary_df = cv_summary_df[ordered]
        paper_style_cv_summary_path = runtime_dir / "paper_style_cv_summary.csv"
        cv_summary_df.to_csv(paper_style_cv_summary_path, index=False)
        artifact_paths["paper_style_cv_summary_csv"] = str(paper_style_cv_summary_path)

    if cv_fold_payload:
        cv_fold_summary_path = runtime_dir / "cv_fold_summary.json"
        cv_fold_summary_path.write_text(json.dumps(cv_fold_payload, indent=2, default=str), encoding="utf-8")
        artifact_paths["cv_fold_summary_json"] = str(cv_fold_summary_path)

    return artifact_paths


def _parse_threshold_grid(raw_grid: Any) -> list[float]:
    if raw_grid is None:
        return [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    values: list[float] = []
    if isinstance(raw_grid, (list, tuple)):
        for token in raw_grid:
            try:
                val = float(token)
            except (TypeError, ValueError):
                continue
            if val > 0:
                values.append(val)
    return sorted(set(values)) or [1.0]


def _resolve_threshold_tuning_config(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    raw_cfg = evaluation_cfg.get("threshold_tuning", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    enabled = bool(raw_cfg.get("enabled", False))
    objective = str(raw_cfg.get("objective", "macro_f1")).strip().lower()
    if objective not in {"macro_f1", "enrolled_f1"}:
        objective = "macro_f1"
    focus_class = str(raw_cfg.get("focus_class", "Enrolled")).strip() or "Enrolled"
    apply_on_test = bool(raw_cfg.get("apply_on_test", True))
    grid = _parse_threshold_grid(raw_cfg.get("grid"))
    return {
        "enabled": enabled,
        "objective": objective,
        "focus_class": focus_class,
        "apply_on_test": apply_on_test,
        "grid": grid,
    }


def _predict_with_thresholds(probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = probabilities / thresholds.reshape(1, -1)
    pred_idx = np.argmax(adjusted, axis=1)
    return labels[pred_idx]


def _run_validation_threshold_tuning(
    payload: dict[str, Any],
    class_metadata: dict[str, Any],
    threshold_cfg: dict[str, Any],
) -> dict[str, Any]:
    artifacts = payload.get("artifacts", {})
    labels = artifacts.get("labels") or []
    y_true_valid = artifacts.get("y_true_valid")
    y_proba_valid = artifacts.get("y_proba_valid")
    y_proba_test = artifacts.get("y_proba_test")
    y_true_test = artifacts.get("y_true_test")

    if y_true_valid is None or y_proba_valid is None or y_proba_test is None or y_true_test is None:
        return {
            "status": "skipped",
            "reason": "missing_probability_or_validation_artifacts",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }
    if len(y_true_valid) == 0:
        return {
            "status": "skipped",
            "reason": "validation_split_is_empty",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }

    labels_arr = np.asarray(labels, dtype=int)
    if labels_arr.ndim != 1 or labels_arr.size == 0:
        return {
            "status": "skipped",
            "reason": "missing_label_order",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }

    p_valid = np.asarray(y_proba_valid, dtype=float)
    p_test = np.asarray(y_proba_test, dtype=float)
    y_valid = np.asarray(y_true_valid, dtype=int)
    y_test = np.asarray(y_true_test, dtype=int)
    if p_valid.ndim != 2 or p_test.ndim != 2:
        return {
            "status": "skipped",
            "reason": "invalid_probability_shape",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }
    if p_valid.shape[1] != labels_arr.size or p_test.shape[1] != labels_arr.size:
        return {
            "status": "skipped",
            "reason": "label_probability_dimension_mismatch",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }

    class_label_to_index = class_metadata.get("class_label_to_index", {})
    focus_label = str(threshold_cfg.get("focus_class", "Enrolled"))
    focus_idx = class_label_to_index.get(focus_label)
    if focus_idx is None:
        for key, val in class_label_to_index.items():
            if str(key).strip().lower() == focus_label.strip().lower():
                focus_idx = val
                break
    if focus_idx is None:
        return {
            "status": "skipped",
            "reason": f"focus_class_not_found:{focus_label}",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }
    focus_idx = int(focus_idx)
    if focus_idx not in labels_arr.tolist():
        return {
            "status": "skipped",
            "reason": f"focus_class_not_in_model_labels:{focus_idx}",
            "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
            "threshold_tuning_supported": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
        }

    focus_pos = int(np.where(labels_arr == focus_idx)[0][0])
    grid_values = [float(v) for v in threshold_cfg.get("grid", [1.0]) if float(v) > 0]
    if not grid_values:
        grid_values = [1.0]

    baseline_thresholds = np.ones(labels_arr.size, dtype=float)
    baseline_valid_pred = _predict_with_thresholds(p_valid, labels_arr, baseline_thresholds)
    baseline_valid_metrics = compute_metrics(pd.Series(y_valid), baseline_valid_pred)
    baseline_valid_per_class = compute_per_class_metrics(
        pd.Series(y_valid),
        baseline_valid_pred,
        labels=labels_arr.tolist(),
    )

    baseline_test_pred = _predict_with_thresholds(p_test, labels_arr, baseline_thresholds)
    baseline_test_metrics = compute_metrics(pd.Series(y_test), baseline_test_pred)
    baseline_test_per_class = compute_per_class_metrics(
        pd.Series(y_test),
        baseline_test_pred,
        labels=labels_arr.tolist(),
    )

    objective_name = str(threshold_cfg.get("objective", "macro_f1"))
    best_score = float("-inf")
    best_macro = float("-inf")
    best_accuracy = float("-inf")
    best_thresholds = baseline_thresholds.copy()
    search_rows: list[dict[str, Any]] = []

    for candidate in grid_values:
        thresholds = np.ones(labels_arr.size, dtype=float)
        thresholds[focus_pos] = float(candidate)
        pred_valid = _predict_with_thresholds(p_valid, labels_arr, thresholds)
        valid_metrics = compute_metrics(pd.Series(y_valid), pred_valid)
        valid_per_class = compute_per_class_metrics(
            pd.Series(y_valid),
            pred_valid,
            labels=labels_arr.tolist(),
        )
        enrolled_key = str(focus_idx)
        enrolled_valid_f1 = float(valid_per_class.get(enrolled_key, {}).get("f1", 0.0))
        objective_score = enrolled_valid_f1 if objective_name == "enrolled_f1" else float(valid_metrics["macro_f1"])
        search_rows.append(
            {
                "focus_threshold": float(candidate),
                "objective_score": objective_score,
                "valid_macro_f1": float(valid_metrics["macro_f1"]),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_f1_focus_class": enrolled_valid_f1,
            }
        )
        current_macro = float(valid_metrics["macro_f1"])
        current_accuracy = float(valid_metrics["accuracy"])
        if (
            objective_score > best_score + 1e-12
            or (
                abs(objective_score - best_score) <= 1e-12
                and current_macro > best_macro + 1e-12
            )
            or (
                abs(objective_score - best_score) <= 1e-12
                and abs(current_macro - best_macro) <= 1e-12
                and current_accuracy > best_accuracy + 1e-12
            )
        ):
            best_score = objective_score
            best_macro = current_macro
            best_accuracy = current_accuracy
            best_thresholds = thresholds

    tuned_test_pred = _predict_with_thresholds(p_test, labels_arr, best_thresholds)
    tuned_test_metrics = compute_metrics(pd.Series(y_test), tuned_test_pred)
    tuned_test_per_class = compute_per_class_metrics(
        pd.Series(y_test),
        tuned_test_pred,
        labels=labels_arr.tolist(),
    )
    tuned_cm = confusion_matrix(y_test, tuned_test_pred, labels=labels_arr.tolist()).tolist()

    label_name_map = class_metadata.get("class_index_to_label", {})
    selected_thresholds = {
        str(label_name_map.get(str(int(labels_arr[i])), int(labels_arr[i]))): float(best_thresholds[i])
        for i in range(labels_arr.size)
    }
    apply_on_test = bool(threshold_cfg.get("apply_on_test", True))
    return {
        "status": "applied",
        "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
        "threshold_tuning_supported": True,
        "threshold_selection_split": "validation",
        "threshold_applied_to": "test" if apply_on_test else "none",
        "validation_only_selection_confirmed": True,
        "objective": objective_name,
        "focus_class": focus_label,
        "default_decision_rule": "argmax",
        "search_results": search_rows,
        "selected_thresholds": selected_thresholds,
        "validation_baseline_metrics": baseline_valid_metrics,
        "validation_baseline_per_class": baseline_valid_per_class,
        "test_baseline_metrics": baseline_test_metrics,
        "test_baseline_per_class": baseline_test_per_class,
        "test_tuned_metrics": tuned_test_metrics,
        "test_tuned_per_class": tuned_test_per_class,
        "y_pred_test_tuned": tuned_test_pred.tolist(),
        "confusion_matrix_tuned": tuned_cm,
    }


def _run_multiclass_decision_autotune(
    payload: dict[str, Any],
    decision_rule_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
) -> dict[str, Any]:
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {}
    strategy = str(decision_rule_cfg.get("decision_rule", "model_predict")).strip().lower()
    multiclass_cfg = (
        decision_rule_cfg.get("multiclass_decision", {})
        if isinstance(decision_rule_cfg.get("multiclass_decision", {}), dict)
        else {}
    )
    auto_tune_cfg = multiclass_cfg.get("auto_tune", {}) if isinstance(multiclass_cfg.get("auto_tune", {}), dict) else {}
    enabled = bool(auto_tune_cfg.get("enabled", False))

    result_base = {
        "status": "skipped",
        "reason": "disabled",
        "multiclass_decision_strategy": strategy,
        "multiclass_decision_auto_tuned": False,
        "tuning_objective": str(auto_tune_cfg.get("objective", "macro_f1")).strip().lower(),
    }
    if not enabled:
        return result_base

    y_true_valid_raw = artifacts.get("y_true_valid")
    y_proba_valid_raw = artifacts.get("y_proba_valid")
    y_true_test_raw = artifacts.get("y_true_test")
    y_proba_test_raw = artifacts.get("y_proba_test")
    labels = artifacts.get("labels") or []
    if y_true_valid_raw is None or y_proba_valid_raw is None or y_true_test_raw is None or y_proba_test_raw is None:
        raise ValueError(
            "multiclass decision auto_tune requires y_true_valid, y_proba_valid, y_true_test, y_proba_test artifacts."
        )
    if len(y_true_valid_raw) == 0:
        raise ValueError("multiclass decision auto_tune requires a non-empty validation split.")

    y_true_valid = pd.Series(np.asarray(y_true_valid_raw, dtype=int))
    y_true_test = pd.Series(np.asarray(y_true_test_raw, dtype=int))
    tuned = auto_tune_multiclass_decision_policy(
        y_true_valid=y_true_valid,
        y_proba_valid=np.asarray(y_proba_valid_raw, dtype=float),
        y_true_test=y_true_test,
        y_proba_test=np.asarray(y_proba_test_raw, dtype=float),
        labels=[int(v) for v in labels],
        strategy=strategy,
        multiclass_decision_config=multiclass_cfg,
    )
    if str(tuned.get("status", "")).lower() != "applied":
        return {**result_base, **tuned}

    tuned_cfg = tuned.get("selected_parameters", {})
    y_pred_valid_tuned = np.asarray(tuned.get("y_pred_valid_tuned", []), dtype=int)
    y_pred_test_tuned = np.asarray(tuned.get("y_pred_test_tuned", []), dtype=int)
    labels_list = [int(v) for v in labels]

    artifacts["multiclass_decision_tuned"] = tuned_cfg
    artifacts["y_pred_valid_default"] = artifacts.get("y_pred_valid")
    artifacts["y_pred_test_default"] = artifacts.get("y_pred_test")
    artifacts["per_class_metrics_test_default"] = artifacts.get("per_class_metrics_test")
    artifacts["confusion_matrix_default"] = artifacts.get("confusion_matrix")
    artifacts["y_pred_valid"] = y_pred_valid_tuned.tolist()
    artifacts["y_pred_test"] = y_pred_test_tuned.tolist()
    artifacts["per_class_metrics_valid"] = tuned.get("valid_per_class_tuned", {})
    artifacts["per_class_metrics_test"] = tuned.get("test_per_class_tuned", {})
    artifacts["classification_report_valid"] = classification_report(
        y_true_valid,
        y_pred_valid_tuned,
        labels=labels_list,
        output_dict=True,
        zero_division=0,
    )
    artifacts["classification_report_test"] = classification_report(
        y_true_test,
        y_pred_test_tuned,
        labels=labels_list,
        output_dict=True,
        zero_division=0,
    )
    artifacts["confusion_matrix"] = confusion_matrix(y_true_test, y_pred_test_tuned, labels=labels_list).tolist()

    baseline_metrics = {
        key: float(value)
        for key, value in payload.get("metrics", {}).items()
        if isinstance(value, (int, float)) and str(key).startswith("test_")
    }
    for key, value in baseline_metrics.items():
        payload["metrics"][f"{key}_default"] = float(value)
    payload["metrics"].update({f"test_{k}": float(v) for k, v in tuned.get("test_metrics_tuned", {}).items()})
    payload["metrics"].update({f"valid_{k}": float(v) for k, v in tuned.get("valid_metrics_tuned", {}).items()})
    _add_named_validation_per_class_metrics(
        payload["metrics"],
        artifacts.get("per_class_metrics_valid"),
        class_metadata.get("class_index_to_label", {}),
    )
    _add_named_per_class_metrics_with_suffix(
        payload["metrics"],
        artifacts.get("per_class_metrics_test_default"),
        class_metadata.get("class_index_to_label", {}),
        "default",
    )

    payload["metrics"]["multiclass_decision_auto_tuned"] = 1.0
    payload["metrics"]["validation_objective_score_at_selected_threshold"] = float(
        tuned.get("validation_objective_score", float("nan"))
    )
    if "enrolled_margin_threshold" in tuned_cfg:
        payload["metrics"]["tuned_enrolled_margin_threshold"] = float(tuned_cfg["enrolled_margin_threshold"])
    if "dropout_threshold" in tuned_cfg:
        payload["metrics"]["tuned_dropout_threshold"] = float(tuned_cfg["dropout_threshold"])
    if "graduate_threshold" in tuned_cfg:
        payload["metrics"]["tuned_graduate_threshold"] = float(tuned_cfg["graduate_threshold"])

    return {
        **result_base,
        "status": "applied",
        "reason": "validation_grid_search_selected",
        "multiclass_decision_auto_tuned": True,
        "multiclass_decision_strategy": strategy,
        "tuning_objective": str(tuned.get("objective", "macro_f1")),
        "search_grid_size": int(tuned_cfg.get("search_grid_size", 0)),
        "selected_parameters": tuned_cfg,
        "validation_objective_score_at_selected_threshold": float(
            tuned.get("validation_objective_score", float("nan"))
        ),
        "search_results": tuned.get("search_results", []),
    }


def _build_uct_feature_table(
    adapted: dict[str, Any] | pd.DataFrame,
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    builder_token = str(feature_cfg.get("builder", "uci_student_features")).strip().lower()
    print(f"[features][uci] builder={builder_token}")
    if builder_token == "uci_student_features":
        return build_uct_student_features(adapted, feature_cfg)
    if builder_token == "uci_student_paper_style_features":
        return build_uci_student_paper_style_features(adapted, feature_cfg)
    raise ValueError(
        "Unsupported UCT/UCI feature builder "
        f"'{builder_token}'. Supported builders: "
        "['uci_student_features', 'uci_student_paper_style_features']."
    )


def _build_feature_table(dataset_cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, str, str]:
    raw_dataset_name = str(dataset_cfg.get("dataset", {}).get("name", ""))
    dataset_name = _normalize_dataset_name(raw_dataset_name)
    if not dataset_name:
        raise ValueError(
            "Dataset name is missing in dataset config under dataset.name. "
            "Expected one of: uct_student, oulad."
        )
    schema_cfg = dataset_cfg.get("schema", {})
    if dataset_name == "uct_student":
        raw_df = load_uct_student_dataframe(dataset_cfg)
        adapted = adapt_uct_student_schema(raw_df, schema_cfg)
        features = _build_uct_feature_table(adapted, dataset_cfg.get("features", {}))
        return features, dataset_name, adapted["id_column"], adapted["target_column"]
    if dataset_name == "oulad":
        raw_tables = load_oulad_tables(dataset_cfg)
        adapted = adapt_oulad_schema(raw_tables, schema_cfg)
        features = build_oulad_paper_features(adapted, dataset_cfg.get("features", {}))
        return features, dataset_name, "id_student", schema_cfg.get("outcome_column", "final_result")
    raise ValueError(
        "Unsupported dataset name "
        f"'{raw_dataset_name}' (normalized: '{dataset_name}'). "
        f"Supported datasets: {sorted(SUPPORTED_DATASETS)}."
    )


def _resolve_dataset_source_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = dataset_cfg.get("data_source", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    return {
        "format": str(raw_cfg.get("format", "csv")).strip().lower(),
        "split_mode": str(raw_cfg.get("split_mode", "single_file")).strip().lower(),
        "train_path": raw_cfg.get("train_path"),
        "valid_path": raw_cfg.get("valid_path"),
        "test_path": raw_cfg.get("test_path"),
    }


def _prepare_feature_df_with_target(
    feature_df: pd.DataFrame,
    *,
    dataset_name: str,
    source_target_col: str,
    formulation: str,
    target_mapping: dict[str, int] | None,
) -> pd.DataFrame:
    prepared = feature_df.copy()
    mapped_target = _map_target(prepared, dataset_name, source_target_col, formulation, target_mapping)
    if mapped_target is None:
        raise ValueError(
            "Target mapping returned None for "
            f"dataset='{dataset_name}', formulation='{formulation}', "
            f"source_target_col='{source_target_col}'."
        )
    if not isinstance(mapped_target, pd.Series):
        raise ValueError(
            "Target mapping must return pandas Series, got "
            f"{type(mapped_target).__name__} for dataset='{dataset_name}', "
            f"source_target_col='{source_target_col}'."
        )
    if len(mapped_target) != len(prepared):
        raise ValueError(
            "Mapped target length mismatch: "
            f"len(mapped_target)={len(mapped_target)} vs len(feature_df)={len(prepared)} "
            f"for dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    prepared["target"] = mapped_target
    if "target" not in prepared.columns:
        raise ValueError(
            "Failed to create 'target' column after mapping for "
            f"dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    columns_to_drop = [col for col in [source_target_col] if col and col != "target"]
    if columns_to_drop:
        prepared = prepared.drop(columns=columns_to_drop, errors="ignore")
    if prepared["target"].isna().any():
        raise ValueError(
            "Target mapping produced null values for "
            f"dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    return prepared


def _build_predefined_uci_feature_splits(
    dataset_cfg: dict[str, Any],
    *,
    formulation: str,
    target_mapping: dict[str, int] | None,
) -> tuple[dict[str, pd.DataFrame], str, str, str, dict[str, Any]]:
    source_cfg = _resolve_dataset_source_config(dataset_cfg)
    if source_cfg["format"] != "parquet" or source_cfg["split_mode"] != "predefined":
        raise ValueError("Predefined UCI feature split builder requires parquet predefined split config.")

    schema_cfg = dataset_cfg.get("schema", {})
    loaded = load_uct_student_predefined_splits(dataset_cfg)
    raw_train = loaded["train"]
    raw_valid = loaded.get("valid")
    raw_test = loaded["test"]

    adapted_train = adapt_uct_student_schema(raw_train, schema_cfg)
    adapted_valid = adapt_uct_student_schema(raw_valid, schema_cfg) if isinstance(raw_valid, pd.DataFrame) else None
    adapted_test = adapt_uct_student_schema(raw_test, schema_cfg)
    source_target_col = str(adapted_train["target_column"])
    if adapted_valid is not None and str(adapted_valid["target_column"]) != source_target_col:
        raise ValueError(
            "Predefined UCI parquet train/valid target column mismatch after schema adaptation. "
            f"train={adapted_train['target_column']} valid={adapted_valid['target_column']}"
        )
    if str(adapted_test["target_column"]) != source_target_col:
        raise ValueError(
            "Predefined UCI parquet train/test target column mismatch after schema adaptation. "
            f"train={adapted_train['target_column']} test={adapted_test['target_column']}"
        )
    id_column = str(adapted_train["id_column"])
    if str(adapted_test["id_column"]) != id_column:
        print(
            "[dataset][uci] id_column differs between train/test after adaptation; "
            f"train={adapted_train['id_column']} test={adapted_test['id_column']}"
        )

    train_features = _build_uct_feature_table(adapted_train, dataset_cfg.get("features", {}))
    valid_features = (
        _build_uct_feature_table(adapted_valid, dataset_cfg.get("features", {}))
        if adapted_valid is not None
        else None
    )
    test_features = _build_uct_feature_table(adapted_test, dataset_cfg.get("features", {}))
    train_prepared = _prepare_feature_df_with_target(
        train_features,
        dataset_name="uct_student",
        source_target_col=source_target_col,
        formulation=formulation,
        target_mapping=target_mapping,
    )
    valid_prepared = (
        _prepare_feature_df_with_target(
            valid_features,
            dataset_name="uct_student",
            source_target_col=source_target_col,
            formulation=formulation,
            target_mapping=target_mapping,
        )
        if isinstance(valid_features, pd.DataFrame)
        else None
    )
    test_prepared = _prepare_feature_df_with_target(
        test_features,
        dataset_name="uct_student",
        source_target_col=source_target_col,
        formulation=formulation,
        target_mapping=target_mapping,
    )
    feature_reference = train_prepared.drop(columns=["target"], errors="ignore")
    if valid_prepared is not None:
        feature_valid = valid_prepared.drop(columns=["target"], errors="ignore")
        if set(feature_reference.columns) != set(feature_valid.columns):
            missing_in_valid = [col for col in feature_reference.columns if col not in feature_valid.columns]
            extra_in_valid = [col for col in feature_valid.columns if col not in feature_reference.columns]
            raise ValueError(
                "UCI predefined parquet valid feature schema mismatch after feature building. "
                f"missing_in_valid={missing_in_valid[:10]} extra_in_valid={extra_in_valid[:10]}"
            )
        aligned_valid_features = align_feature_schema(feature_reference, feature_valid, fill_value=np.nan)
        validate_feature_schema(feature_reference, aligned_valid_features, context="uci_predefined_valid_feature_alignment")
        valid_prepared = pd.concat(
            [aligned_valid_features.reset_index(drop=True), valid_prepared[["target"]].reset_index(drop=True)],
            axis=1,
        )
    feature_test = test_prepared.drop(columns=["target"], errors="ignore")
    if set(feature_reference.columns) != set(feature_test.columns):
        missing_in_test = [col for col in feature_reference.columns if col not in feature_test.columns]
        extra_in_test = [col for col in feature_test.columns if col not in feature_reference.columns]
        raise ValueError(
            "UCI predefined parquet feature schema mismatch after feature building. "
            f"missing_in_test={missing_in_test[:10]} extra_in_test={extra_in_test[:10]}"
        )
    aligned_test_features = align_feature_schema(feature_reference, feature_test, fill_value=np.nan)
    validate_feature_schema(feature_reference, aligned_test_features, context="uci_predefined_test_feature_alignment")
    test_prepared = pd.concat(
        [aligned_test_features.reset_index(drop=True), test_prepared[["target"]].reset_index(drop=True)],
        axis=1,
    )
    return {
        "train": train_prepared.reset_index(drop=True),
        "valid": valid_prepared.reset_index(drop=True) if isinstance(valid_prepared, pd.DataFrame) else None,
        "test": test_prepared.reset_index(drop=True),
    }, "uct_student", id_column, source_target_col, {
        "source_format": source_cfg["format"],
        "split_mode": source_cfg["split_mode"],
        "schema_report": loaded.get("schema_report", {}),
        "valid_schema_report": loaded.get("valid_schema_report", {}),
        "resolved_paths": loaded.get("resolved_paths", {}),
    }


def _map_target(
    df: pd.DataFrame,
    dataset_name: str,
    source_target_col: str,
    formulation: str,
    mapping: dict[str, int] | None,
) -> pd.Series:
    if formulation == "binary":
        if dataset_name == "uct_student":
            return map_binary_target(df, source_target_col, dataset_name=dataset_name, mapping=mapping)
        return map_binary_target(df, source_target_col, dataset_name=dataset_name, mapping=mapping)
    if formulation == "three_class":
        return map_three_class_target(
            df,
            source_column=source_target_col,
            dataset_name=dataset_name,
            mapping=mapping,
        )
    if formulation == "four_class":
        return map_four_class_target(
            df,
            source_column=source_target_col,
            dataset_name=dataset_name,
            mapping=mapping,
        )
    raise ValueError(f"Unsupported target formulation '{formulation}'.")


def _resolve_target_mapping(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    formulation: str,
) -> dict[str, int] | None:
    override_cfg = exp_cfg.get("target_mapping_override", {})
    if not isinstance(override_cfg, dict):
        override_cfg = {}
    ds_cfg = dataset_cfg.get("target_mappings", {})
    if not isinstance(ds_cfg, dict):
        ds_cfg = {}

    override_mapping = override_cfg.get(formulation)
    if isinstance(override_mapping, dict):
        return override_mapping
    mapping = ds_cfg.get(formulation)
    if isinstance(mapping, dict):
        return mapping
    return None


def _resolve_class_metadata(
    exp_cfg: dict[str, Any],
    mapping: dict[str, int] | None,
) -> dict[str, Any]:
    if not isinstance(mapping, dict) or not mapping:
        return {
            "class_label_to_index": {},
            "class_index_to_label": {},
            "class_indices": [],
            "class_order": [],
        }

    class_label_to_index = {str(k): int(v) for k, v in mapping.items()}
    sorted_pairs = sorted(class_label_to_index.items(), key=lambda kv: (kv[1], str(kv[0]).lower()))
    class_indices = [int(v) for _, v in sorted_pairs]
    class_index_to_label = {str(v): str(k) for k, v in sorted_pairs}

    configured_order = exp_cfg.get("target", {}).get("class_order", [])
    if isinstance(configured_order, list) and configured_order:
        class_order = [str(label) for label in configured_order]
    else:
        class_order = [str(label) for label, _ in sorted_pairs]

    return {
        "class_label_to_index": class_label_to_index,
        "class_index_to_label": class_index_to_label,
        "class_indices": class_indices,
        "class_order": class_order,
    }


def _normalize_explicit_class_weight_values(
    values_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
) -> dict[str, float]:
    if not isinstance(values_cfg, dict) or not values_cfg:
        raise ValueError("training.class_weight.values must be a non-empty mapping of class label to positive weight.")

    class_label_to_index = (
        class_metadata.get("class_label_to_index", {})
        if isinstance(class_metadata.get("class_label_to_index", {}), dict)
        else {}
    )
    if not class_label_to_index:
        raise ValueError(
            "training.class_weight.mode='explicit' requires class metadata mapping. "
            "Ensure target mapping is defined for this experiment."
        )

    canonical_by_lower = {str(label).strip().lower(): str(label) for label in class_label_to_index.keys()}
    expected_labels = sorted([str(label) for label in class_label_to_index.keys()])
    expected_label_set = set(expected_labels)
    resolved: dict[str, float] = {}
    unknown_keys: list[str] = []

    for raw_key, raw_weight in values_cfg.items():
        canonical = canonical_by_lower.get(str(raw_key).strip().lower())
        if canonical is None:
            unknown_keys.append(str(raw_key))
            continue
        if canonical in resolved:
            raise ValueError(f"Duplicate class weight key after case normalization: '{raw_key}'.")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class weight for '{raw_key}' must be numeric.") from exc
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"class weight for '{raw_key}' must be a positive finite number.")
        resolved[canonical] = float(weight)

    if unknown_keys:
        raise ValueError(
            "training.class_weight.values contains unknown class keys. "
            f"unknown_keys={unknown_keys}, expected_keys={expected_labels}."
        )
    missing = sorted(expected_label_set.difference(set(resolved.keys())))
    if missing:
        raise ValueError(
            "training.class_weight.values must provide weights for every mapped class. "
            f"missing_keys={missing}, expected_keys={expected_labels}."
        )
    return resolved


def _resolve_class_weight_config(
    exp_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
) -> dict[str, Any]:
    models_cfg = exp_cfg.get("models", {}) if isinstance(exp_cfg.get("models", {}), dict) else {}
    class_weight_cfg = models_cfg.get("class_weight", {}) if isinstance(models_cfg.get("class_weight", {}), dict) else {}
    training_cfg = exp_cfg.get("training", {}) if isinstance(exp_cfg.get("training", {}), dict) else {}
    training_class_weight_cfg = (
        training_cfg.get("class_weight", {})
        if isinstance(training_cfg.get("class_weight", {}), dict)
        else {}
    )

    resolved = dict(class_weight_cfg)
    if training_class_weight_cfg:
        mode = str(training_class_weight_cfg.get("mode", "")).strip().lower()
        if mode and mode != "explicit":
            raise ValueError("training.class_weight.mode currently supports only 'explicit'.")
        resolved_values = _normalize_explicit_class_weight_values(
            training_class_weight_cfg.get("values", {}),
            class_metadata=class_metadata,
        )
        resolved.update(
            {
                "enabled": True,
                "mode": "explicit",
                "strategy": "explicit",
                "values": resolved_values,
                "class_weight_map": resolved_values,
                "source": "training.class_weight",
            }
        )

    resolved["class_label_to_index"] = (
        class_metadata.get("class_label_to_index", {})
        if isinstance(class_metadata.get("class_label_to_index", {}), dict)
        else {}
    )
    return resolved


def _class_weight_requested(class_weight_cfg: dict[str, Any] | None) -> bool:
    cfg = class_weight_cfg if isinstance(class_weight_cfg, dict) else {}
    return bool(cfg.get("enabled", False) or str(cfg.get("mode", "")).strip().lower() == "explicit")


def _add_class_weight_metadata_metrics(
    metrics: dict[str, Any],
    class_weight_info: dict[str, Any],
    class_metadata: dict[str, Any],
) -> None:
    if not isinstance(metrics, dict) or not isinstance(class_weight_info, dict):
        return
    mode = str(class_weight_info.get("mode", class_weight_info.get("strategy", "none")))
    metrics["class_weight_mode"] = mode
    metrics["class_weight_application_method"] = str(class_weight_info.get("class_weight_application_method", "none"))
    metrics["class_weight_requested_flag"] = 1.0 if bool(class_weight_info.get("class_weight_requested", False)) else 0.0
    metrics["class_weight_applied_flag"] = 1.0 if bool(class_weight_info.get("class_weight_applied", False)) else 0.0

    weight_map = class_weight_info.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return
    index_to_weight: dict[int, float] = {}
    for key, value in weight_map.items():
        key_token = str(key).strip()
        if key_token.lstrip("-").isdigit():
            index_to_weight[int(key_token)] = float(value)
    class_label_to_index = class_metadata.get("class_label_to_index", {}) if isinstance(class_metadata, dict) else {}
    if not isinstance(class_label_to_index, dict):
        class_label_to_index = {}
    for class_label, class_index in class_label_to_index.items():
        idx = int(class_index)
        metric_key = f"class_weight_{_metric_label_token(str(class_label))}"
        if idx in index_to_weight:
            metrics[metric_key] = float(index_to_weight[idx])


def _drop_rows_with_missing_values(
    df: pd.DataFrame,
    preprocessing_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_cfg = preprocessing_cfg.get("missing_values", {})
    drop_rows = bool(preprocessing_cfg.get("drop_missing_rows", False))
    drop_rows = bool(preprocessing_cfg.get("drop_rows_with_missing", drop_rows))
    if isinstance(missing_cfg, dict):
        drop_rows = bool(missing_cfg.get("drop_rows", drop_rows))
    if not drop_rows:
        return df, {"enabled": False, "method": "imputation"}

    original_count = int(len(df))
    dropped_df = df.dropna(axis=0).reset_index(drop=True)
    removed = original_count - int(len(dropped_df))
    return dropped_df, {
        "enabled": True,
        "method": "drop_rows",
        "n_original": original_count,
        "n_removed": removed,
        "n_remaining": int(len(dropped_df)),
    }


def _resolve_categorical_encoding_config(preprocessing_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = preprocessing_cfg.get("categorical_encoding")
    if isinstance(raw_cfg, dict):
        mode = str(raw_cfg.get("mode", preprocessing_cfg.get("encoding", "onehot"))).strip().lower()
        return {
            "mode": mode,
            "handle_unknown": str(raw_cfg.get("handle_unknown", "ignore")).strip().lower(),
            "drop": raw_cfg.get("drop"),
            "lock_category_vocabulary_from_pre_split_train": bool(
                raw_cfg.get("lock_category_vocabulary_from_pre_split_train", False)
            ),
            "vocabulary_source": str(
                raw_cfg.get("vocabulary_source", "categorical_dtype_or_full_pre_split_train")
            ).strip(),
        }
    return {
        "mode": str(raw_cfg or preprocessing_cfg.get("encoding", "onehot")).strip().lower(),
        "handle_unknown": "ignore",
        "drop": None,
        "lock_category_vocabulary_from_pre_split_train": False,
        "vocabulary_source": "subset_fit_only",
    }


def _build_locked_onehot_vocabulary(
    feature_df: pd.DataFrame,
    preprocess_cfg: dict[str, Any],
) -> dict[str, Any]:
    target_column = str(preprocess_cfg.get("target_column", "target"))
    id_columns = list(preprocess_cfg.get("id_columns", []))
    forbidden_columns = set(preprocess_cfg.get("forbidden_feature_columns", []))
    source_features = feature_df.drop(
        columns=[c for c in [target_column, *id_columns, *sorted(forbidden_columns)] if c in feature_df.columns]
    ).copy()
    source_features = source_features.where(pd.notna(source_features), np.nan)

    numeric_cols, categorical_cols = detect_feature_types(source_features)
    vocabulary: dict[str, list[Any]] = {}
    counts: dict[str, int] = {}
    labels: dict[str, list[str]] = {}
    sources: dict[str, str] = {}

    for column in categorical_cols:
        series = source_features[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = list(series.cat.categories)
            source = "categorical_dtype"
        else:
            categories = pd.Index(series.dropna().astype(object).unique()).tolist()
            source = "full_pre_split_train_unique_values"
        vocabulary[column] = list(categories)
        counts[column] = int(len(categories))
        labels[column] = [str(value) for value in categories]
        sources[column] = source

    total_encoded = int(sum(counts.values()))
    return {
        "categories": vocabulary,
        "source": str(preprocess_cfg.get("onehot_categories_source", "categorical_dtype_or_full_pre_split_train")),
        "column_counts": counts,
        "column_labels": labels,
        "column_sources": sources,
        "categorical_column_count": int(len(categorical_cols)),
        "numeric_column_count": int(len(numeric_cols)),
        "encoded_categorical_feature_count": total_encoded,
        "preprocessed_feature_count": int(total_encoded + len(numeric_cols)),
    }


def _prepare_preprocessing_config(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    id_column: str,
    source_target_col: str,
) -> dict[str, Any]:
    p_cfg = exp_cfg.get("preprocessing", {})
    ds_p_cfg = dataset_cfg.get("preprocessing", {})
    categorical_encoding_cfg = _resolve_categorical_encoding_config(p_cfg)
    imputation = str(p_cfg.get("imputation", "median_mode")).lower()
    scaling_raw = str(p_cfg.get("numeric_scaling", p_cfg.get("scaling", "standard"))).strip().lower()
    encoding_raw = str(categorical_encoding_cfg.get("mode", p_cfg.get("encoding", "onehot"))).strip().lower()
    forbidden_columns = {"final_result"}
    forbidden_columns.update(list(p_cfg.get("forbidden_feature_columns", []) or []))
    forbidden_columns.update(list(p_cfg.get("drop_columns", []) or []))
    forbidden_columns.update(list(ds_p_cfg.get("forbidden_feature_columns", []) or []))
    if source_target_col and source_target_col != "target":
        forbidden_columns.add(source_target_col)
    return {
        "target_column": "target",
        "id_columns": [id_column],
        "forbidden_feature_columns": sorted(forbidden_columns),
        "numeric_imputation": "median" if "median" in imputation else "mean",
        "categorical_imputation": "most_frequent",
        "scaling": scaling_raw in {"standard", "true", "1", "enabled"},
        "onehot": encoding_raw == "onehot",
        "onehot_handle_unknown": str(categorical_encoding_cfg.get("handle_unknown", "ignore")).strip().lower(),
        "onehot_drop": categorical_encoding_cfg.get("drop"),
        "lock_category_vocabulary_from_pre_split_train": bool(
            categorical_encoding_cfg.get("lock_category_vocabulary_from_pre_split_train", False)
        ),
        "onehot_categories_source": str(categorical_encoding_cfg.get("vocabulary_source", "subset_fit_only")).strip(),
    }


def _resolve_outlier_config(
    exp_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    preprocessing_cfg = exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}
    outlier_cfg = preprocessing_cfg.get("outlier", {"enabled": False})
    if not isinstance(outlier_cfg, dict):
        outlier_cfg = {"enabled": False}
    isolation_cfg = preprocessing_cfg.get("isolation_forest", {})
    if not isinstance(isolation_cfg, dict):
        isolation_cfg = {}
    enabled = bool(
        outlier_cfg.get(
            "enabled",
            preprocessing_cfg.get("apply_isolation_forest", isolation_cfg.get("enabled", False)),
        )
    )
    resolved = {
        **outlier_cfg,
        **isolation_cfg,
        "enabled": enabled,
        "method": str(outlier_cfg.get("method", isolation_cfg.get("method", "isolation_forest"))).lower(),
    }
    resolved["random_state"] = int(resolved.get("random_state", seed))
    return resolved


def _resolve_balancing_config(
    exp_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    preprocessing_cfg = exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}
    balancing_cfg = preprocessing_cfg.get("balancing", {"enabled": False})
    if not isinstance(balancing_cfg, dict):
        balancing_cfg = {"enabled": False}
    smote_cfg = preprocessing_cfg.get("smote", {})
    if not isinstance(smote_cfg, dict):
        smote_cfg = {}
    enabled = bool(
        balancing_cfg.get(
            "enabled",
            preprocessing_cfg.get("apply_smote_on_train_only", smote_cfg.get("enabled", False)),
        )
    )
    resolved = {
        **balancing_cfg,
        **smote_cfg,
        "enabled": enabled,
        "method": str(balancing_cfg.get("method", smote_cfg.get("method", "smote"))).lower(),
    }
    resolved["random_state"] = int(resolved.get("random_state", seed))
    return resolved


def _resolve_and_validate_model_candidates(exp_cfg: dict[str, Any]) -> list[str]:
    raw_candidates = exp_cfg.get("models", {}).get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("models.candidates must be a list of model names.")

    normalized_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        token = str(candidate).strip().lower()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized_candidates.append(token)

    if not normalized_candidates:
        raise ValueError(
            "No model candidates were provided. "
            "Set models.candidates with at least one registered model."
        )

    available = set(list_available_models())
    unavailable = [name for name in normalized_candidates if name not in available]
    if unavailable:
        raise ValueError(
            "Experiment config requested unregistered model(s): "
            f"{unavailable}. Available models: {sorted(available)}"
        )
    return normalized_candidates


def _resolve_uct_three_class_indices(class_metadata: dict[str, Any]) -> tuple[int, int, int]:
    label_to_index = class_metadata.get("class_label_to_index", {})
    dropout_idx = label_to_index.get("Dropout")
    enrolled_idx = label_to_index.get("Enrolled")
    graduate_idx = label_to_index.get("Graduate")
    if dropout_idx is None or enrolled_idx is None or graduate_idx is None:
        raise ValueError(
            "two_stage_uct_3class requires class labels Dropout/Enrolled/Graduate in class metadata."
        )
    return int(dropout_idx), int(enrolled_idx), int(graduate_idx)


def _resolve_two_stage_soft_class_thresholds(
    two_stage_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
) -> dict[int, float]:
    raw_thresholds = two_stage_cfg.get("final_class_thresholds", {})
    if not isinstance(raw_thresholds, dict):
        return {}

    class_label_to_index = class_metadata.get("class_label_to_index", {})
    alias_map = {
        "dropout": dropout_idx,
        "enrolled": enrolled_idx,
        "graduate": graduate_idx,
        "non_dropout_enrolled": enrolled_idx,
        "non_dropout_graduate": graduate_idx,
    }
    resolved: dict[int, float] = {}
    for raw_key, raw_value in raw_thresholds.items():
        try:
            threshold = float(raw_value)
        except (TypeError, ValueError):
            continue
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("two_stage.final_class_thresholds values must be within [0.0, 1.0].")

        key = str(raw_key).strip()
        key_lower = key.lower()
        class_idx: int | None = None
        if key_lower in alias_map:
            class_idx = alias_map[key_lower]
        elif key.lstrip("-").isdigit():
            class_idx = int(key)
        elif key in class_label_to_index:
            class_idx = int(class_label_to_index[key])
        else:
            for class_label, mapped_idx in class_label_to_index.items():
                if str(class_label).strip().lower() == key_lower:
                    class_idx = int(mapped_idx)
                    break
        if class_idx is None:
            continue
        resolved[int(class_idx)] = threshold
    return resolved


def _resolve_two_stage_calibration_config(two_stage_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_cfg = two_stage_cfg.get("calibration", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    global_enabled = bool(raw_cfg.get("enabled", False))
    global_method = str(raw_cfg.get("method", "sigmoid")).strip().lower()
    if global_method not in {"sigmoid", "isotonic", "temperature_scaling"}:
        global_method = "sigmoid"

    def _resolve_stage(stage_name: str) -> dict[str, Any]:
        stage_raw = raw_cfg.get(stage_name, {})
        if not isinstance(stage_raw, dict):
            stage_raw = {}
        enabled = bool(stage_raw.get("enabled", global_enabled))
        method = str(stage_raw.get("method", global_method)).strip().lower()
        if method not in {"sigmoid", "isotonic", "temperature_scaling"}:
            method = "sigmoid"
        return {
            "enabled": enabled,
            "method": method,
            "stage_name": stage_name,
        }

    return {
        "stage1": _resolve_stage("stage1"),
        "stage2": _resolve_stage("stage2"),
    }


def _resolve_two_stage_stage2_positive_target_label(
    two_stage_cfg: dict[str, Any],
    enrolled_idx: int,
    graduate_idx: int,
) -> int:
    raw_value = str(two_stage_cfg.get("stage2_positive_class", "graduate")).strip().lower()
    if raw_value in {"enrolled", str(enrolled_idx)}:
        return int(enrolled_idx)
    if raw_value in {"graduate", str(graduate_idx)}:
        return int(graduate_idx)
    raise ValueError("two_stage.stage2_positive_class must be 'enrolled' or 'graduate'.")


def _resolve_two_stage_stage_class_weights(
    two_stage_cfg: dict[str, Any],
    class_weight_cfg: dict[str, Any],
    *,
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_cfg_raw = two_stage_cfg.get("class_weight", {})
    stage_cfg = stage_cfg_raw if isinstance(stage_cfg_raw, dict) else {}
    stage1_override = stage_cfg.get("stage1", {}) if isinstance(stage_cfg.get("stage1", {}), dict) else {}
    stage2_override = stage_cfg.get("stage2", {}) if isinstance(stage_cfg.get("stage2", {}), dict) else {}
    stage1_cfg_new = two_stage_cfg.get("stage1", {}) if isinstance(two_stage_cfg.get("stage1", {}), dict) else {}
    stage2_cfg_new = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}

    requested = _class_weight_requested(class_weight_cfg)
    base_mode = str(class_weight_cfg.get("mode", class_weight_cfg.get("strategy", "none"))).strip().lower()
    base_values = class_weight_cfg.get("values", class_weight_cfg.get("class_weight_map", {}))
    if not isinstance(base_values, dict):
        base_values = {}

    stage1_cfg: dict[str, Any] = {
        "enabled": requested,
        "strategy": "balanced" if requested else "none",
        "class_label_to_index": {"Non-Dropout": 0, "Dropout": 1},
    }
    if requested:
        stage1_cfg["mode"] = "balanced"
    if stage1_override:
        stage1_cfg.update(stage1_override)
    stage1_cfg["class_label_to_index"] = {"Non-Dropout": 0, "Dropout": 1}

    stage2_cfg: dict[str, Any] = {
        "enabled": requested,
        "strategy": "none",
        "class_label_to_index": {"Graduate": 0, "Enrolled": 1},
    }
    if requested:
        stage2_cfg["mode"] = "balanced"
        stage2_cfg["strategy"] = "balanced"

    enrolled_weight = float(
        stage2_override.get(
            "enrolled_weight",
            base_values.get("Enrolled", class_weight_cfg.get("enrolled_boost", 1.35)),
        )
    )
    graduate_weight = float(stage2_override.get("graduate_weight", base_values.get("Graduate", 1.0)))
    if enrolled_weight <= 0.0 or graduate_weight <= 0.0:
        raise ValueError("two_stage class weights must be positive.")

    if base_mode == "explicit" and requested:
        stage2_cfg.update(
            {
                "enabled": True,
                "mode": "explicit",
                "strategy": "explicit",
                "values": {"Graduate": graduate_weight, "Enrolled": enrolled_weight},
                "class_weight_map": {"Graduate": graduate_weight, "Enrolled": enrolled_weight},
            }
        )
    elif str(class_weight_cfg.get("strategy", "")).strip().lower() == "enrolled_boost" and requested:
        stage2_cfg.update(
            {
                "enabled": True,
                "mode": "explicit",
                "strategy": "explicit",
                "values": {"Graduate": graduate_weight, "Enrolled": enrolled_weight},
                "class_weight_map": {"Graduate": graduate_weight, "Enrolled": enrolled_weight},
            }
        )

    if stage2_override:
        stage2_cfg.update(stage2_override)
    stage1_mode_new = str(stage1_cfg_new.get("class_weight_mode", "")).strip().lower()
    if stage1_mode_new in {"none", "balanced", "custom", "auto_search"}:
        stage1_cfg["enabled"] = bool(stage1_mode_new != "none")
        if stage1_mode_new == "custom":
            positive_weight = float(stage1_cfg_new.get("class_weight_positive", 1.0))
            negative_weight = float(stage1_cfg_new.get("class_weight_negative", 1.0))
            if positive_weight <= 0.0 or negative_weight <= 0.0:
                raise ValueError("two_stage.stage1 custom class weights must be positive.")
            stage1_cfg.update(
                {
                    "mode": "explicit",
                    "strategy": "explicit",
                    "values": {"Non-Dropout": negative_weight, "Dropout": positive_weight},
                    "class_weight_map": {0: negative_weight, 1: positive_weight},
                }
            )
        elif stage1_mode_new == "balanced":
            stage1_cfg.update({"mode": "balanced", "strategy": "balanced"})
        elif stage1_mode_new == "auto_search":
            # Auto-search resolves explicit weights later during model-specific training.
            stage1_cfg.update({"mode": "auto_search", "strategy": "auto_search"})
        else:
            stage1_cfg.update({"mode": "none", "strategy": "none"})

    stage2_mode_new = str(stage2_cfg_new.get("class_weight_mode", "")).strip().lower()
    if stage2_mode_new in {"none", "balanced", "custom", "auto_search"}:
        stage2_cfg["enabled"] = bool(stage2_mode_new != "none")
        if stage2_mode_new == "custom":
            raw_map = stage2_cfg_new.get("class_weight_map", {})
            if not isinstance(raw_map, dict):
                raise ValueError("two_stage.stage2.class_weight_map must be a mapping when class_weight_mode='custom'.")
            enrolled_weight = raw_map.get("enrolled", raw_map.get("Enrolled"))
            graduate_weight = raw_map.get("graduate", raw_map.get("Graduate"))
            if enrolled_weight is None or graduate_weight is None:
                raise ValueError("two_stage.stage2.class_weight_map must define enrolled and graduate.")
            enrolled_weight = float(enrolled_weight)
            graduate_weight = float(graduate_weight)
            if enrolled_weight <= 0.0 or graduate_weight <= 0.0:
                raise ValueError("two_stage.stage2 custom class weights must be positive.")
            stage2_cfg.update(
                {
                    "mode": "explicit",
                    "strategy": "explicit",
                    "values": {"Graduate": graduate_weight, "Enrolled": enrolled_weight},
                    "class_weight_map": {"Graduate": graduate_weight, "Enrolled": enrolled_weight},
                }
            )
        elif stage2_mode_new == "balanced":
            stage2_cfg.update({"mode": "balanced", "strategy": "balanced"})
        elif stage2_mode_new == "auto_search":
            # Auto-search resolves explicit weights later during model-specific training.
            stage2_cfg.update({"mode": "auto_search", "strategy": "auto_search"})
        else:
            stage2_cfg.update({"mode": "none", "strategy": "none"})
    stage2_cfg["class_label_to_index"] = {"Graduate": 0, "Enrolled": 1}
    return stage1_cfg, stage2_cfg


def _resolve_two_stage_threshold_tuning_config(
    two_stage_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
) -> dict[str, Any]:
    raw_cfg = two_stage_cfg.get("threshold_tuning", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    enabled = bool(raw_cfg.get("enabled", raw_cfg.get("strategy", "fixed") == "tune"))
    objective = str(raw_cfg.get("objective", raw_cfg.get("metric", "macro_f1"))).strip().lower()
    if objective not in {"macro_f1", "macro_f1_plus_enrolled_f1", "constrained_enrolled_push"}:
        objective = "macro_f1"
    search_mode = str(raw_cfg.get("search_mode", "single")).strip().lower()
    if search_mode not in {"single", "band"}:
        search_mode = "single"

    default_step = float(raw_cfg.get("step", 0.1))
    if default_step <= 0.0:
        default_step = 0.1
    default_min = float(raw_cfg.get("min", 0.30))
    default_max = float(raw_cfg.get("max", 0.70))
    if default_min > default_max:
        default_min, default_max = default_max, default_min
    default_min = max(0.0, min(1.0, default_min))
    default_max = max(0.0, min(1.0, default_max))

    def _build_grid_for_class(class_name: str) -> list[float]:
        grid_by_class = raw_cfg.get("grid_by_class", {})
        if isinstance(grid_by_class, dict):
            candidate = grid_by_class.get(class_name)
            if isinstance(candidate, (list, tuple)):
                parsed = sorted({float(v) for v in candidate if 0.0 <= float(v) <= 1.0})
                if parsed:
                    return parsed
        points: list[float] = []
        current = default_min
        while current <= default_max + 1e-12:
            points.append(round(current, 6))
            current += default_step
        if not points:
            points = [0.5]
        return sorted({float(v) for v in points})

    class_order = [int(dropout_idx), int(enrolled_idx), int(graduate_idx)]
    class_names = {
        int(dropout_idx): "dropout",
        int(enrolled_idx): "enrolled",
        int(graduate_idx): "graduate",
    }
    default_thresholds = _resolve_two_stage_soft_class_thresholds(
        two_stage_cfg=two_stage_cfg,
        class_metadata=class_metadata,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
    )

    class_grids: dict[int, list[float]] = {}
    for class_idx in class_order:
        class_grids[int(class_idx)] = _build_grid_for_class(class_names[int(class_idx)])

    max_candidates = int(raw_cfg.get("max_candidates", 1500))
    if max_candidates <= 0:
        max_candidates = 1500

    return {
        "enabled": enabled,
        "metric": objective,
        "objective": objective,
        "strategy": str(raw_cfg.get("strategy", "tune" if enabled else "fixed")).strip().lower(),
        "search_mode": search_mode,
        "class_order": class_order,
        "class_grids": class_grids,
        "default_thresholds": default_thresholds,
        "max_candidates": max_candidates,
        "enrolled_push_alpha": float(raw_cfg.get("enrolled_push_alpha", 0.35)),
        "macro_f1_tolerance": float(raw_cfg.get("macro_f1_tolerance", 0.005)),
        "threshold_grid_single": sorted(
            {
                float(v)
                for v in raw_cfg.get("threshold_grid_single", raw_cfg.get("threshold_grid", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]))
            }
        ),
        "threshold_grid_low": sorted({float(v) for v in raw_cfg.get("threshold_grid_low", [0.20, 0.25, 0.30, 0.35, 0.40])}),
        "threshold_grid_high": sorted({float(v) for v in raw_cfg.get("threshold_grid_high", [0.45, 0.50, 0.55, 0.60, 0.65, 0.70])}),
    }


def _resolve_two_stage_stage1_dropout_threshold_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage1_cfg = two_stage_cfg.get("stage1", {}) if isinstance(two_stage_cfg.get("stage1", {}), dict) else {}
    tuning_cfg = (
        two_stage_cfg.get("threshold_tuning", {})
        if isinstance(two_stage_cfg.get("threshold_tuning", {}), dict)
        else {}
    )
    threshold_mode = str(
        stage1_cfg.get(
            "threshold_mode",
            tuning_cfg.get("strategy", ""),
        )
    ).strip().lower()
    if not threshold_mode:
        threshold_mode = "tune" if bool(tuning_cfg.get("enabled", False)) else "fixed"
    if threshold_mode not in {"fixed", "tune", "auto_search"}:
        raise ValueError("two_stage.stage1.threshold_mode must be 'fixed', 'tune', or 'auto_search'.")

    raw_threshold = stage1_cfg.get("dropout_threshold", two_stage_cfg.get("threshold_stage1", 0.5))
    dropout_threshold = float(raw_threshold)
    if dropout_threshold < 0.0 or dropout_threshold > 1.0:
        raise ValueError("two_stage.stage1.dropout_threshold must be within [0.0, 1.0].")

    raw_grid = stage1_cfg.get("threshold_grid", tuning_cfg.get("threshold_grid", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]))
    if not isinstance(raw_grid, list) or len(raw_grid) == 0:
        raise ValueError("two_stage.stage1.threshold_grid must be a non-empty list.")
    threshold_grid = sorted({float(v) for v in raw_grid})
    for value in threshold_grid:
        if value < 0.0 or value > 1.0:
            raise ValueError("two_stage.stage1.threshold_grid values must be within [0.0, 1.0].")

    final_cfg = two_stage_cfg.get("final_decision", {}) if isinstance(two_stage_cfg.get("final_decision", {}), dict) else {}
    middle_band_enabled = bool(final_cfg.get("middle_band_enabled", False))
    search_mode = str(tuning_cfg.get("search_mode", "single")).strip().lower()
    default_low_high_fallback = 0.30 if (middle_band_enabled or search_mode == "band") else dropout_threshold
    default_high_fallback = 0.60 if (middle_band_enabled or search_mode == "band") else dropout_threshold

    low_threshold = float(
        stage1_cfg.get(
            "stage1_dropout_threshold_low",
            stage1_cfg.get(
                "dropout_threshold_low",
                tuning_cfg.get("stage1_dropout_threshold_low", tuning_cfg.get("dropout_threshold_low", default_low_high_fallback)),
            ),
        )
    )
    high_threshold = float(
        stage1_cfg.get(
            "stage1_dropout_threshold_high",
            stage1_cfg.get(
                "dropout_threshold_high",
                tuning_cfg.get("stage1_dropout_threshold_high", tuning_cfg.get("dropout_threshold_high", default_high_fallback)),
            ),
        )
    )
    if low_threshold < 0.0 or low_threshold > 1.0:
        raise ValueError("two_stage.stage1.stage1_dropout_threshold_low must be within [0.0, 1.0].")
    if high_threshold < 0.0 or high_threshold > 1.0:
        raise ValueError("two_stage.stage1.stage1_dropout_threshold_high must be within [0.0, 1.0].")
    if (middle_band_enabled or search_mode == "band") and low_threshold >= high_threshold:
        raise ValueError("two_stage middle-band thresholds require low < high.")

    grid_low_raw = tuning_cfg.get("threshold_grid_low", [0.20, 0.25, 0.30, 0.35, 0.40])
    grid_high_raw = tuning_cfg.get("threshold_grid_high", [0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    if not isinstance(grid_low_raw, list) or not grid_low_raw:
        raise ValueError("two_stage.threshold_tuning.threshold_grid_low must be a non-empty list.")
    if not isinstance(grid_high_raw, list) or not grid_high_raw:
        raise ValueError("two_stage.threshold_tuning.threshold_grid_high must be a non-empty list.")
    threshold_grid_low = sorted({float(v) for v in grid_low_raw})
    threshold_grid_high = sorted({float(v) for v in grid_high_raw})
    for value in [*threshold_grid_low, *threshold_grid_high]:
        if value < 0.0 or value > 1.0:
            raise ValueError("two_stage threshold band grid values must be within [0.0, 1.0].")

    middle_band_behavior = str(final_cfg.get("middle_band_behavior", "force_stage2_soft_fusion")).strip().lower()

    return {
        "mode": threshold_mode,
        "enabled": bool(threshold_mode in {"tune", "auto_search"}),
        "metric": str(tuning_cfg.get("objective", tuning_cfg.get("metric", "macro_f1"))).strip().lower(),
        "selection_split": "validation",
        "dropout_threshold": float(dropout_threshold),
        "threshold_grid": threshold_grid,
        "search_mode": str(
            tuning_cfg.get(
                "search_mode",
                two_stage_cfg.get("auto_balance_search", {}).get("threshold_search_mode", "single")
                if isinstance(two_stage_cfg.get("auto_balance_search", {}), dict)
                else "single",
            )
        ).strip().lower(),
        "threshold_grid_single": sorted(
            {
                float(v)
                for v in tuning_cfg.get(
                    "threshold_grid_single",
                    tuning_cfg.get("threshold_grid", threshold_grid),
                )
            }
        ),
        "stage1_dropout_threshold_low": low_threshold,
        "stage1_dropout_threshold_high": high_threshold,
        "threshold_grid_low": threshold_grid_low,
        "threshold_grid_high": threshold_grid_high,
        "objective": str(tuning_cfg.get("objective", tuning_cfg.get("metric", "macro_f1"))).strip().lower(),
        "enrolled_push_alpha": float(tuning_cfg.get("enrolled_push_alpha", 0.35)),
        "macro_f1_tolerance": float(tuning_cfg.get("macro_f1_tolerance", 0.005)),
        "middle_band_enabled": middle_band_enabled,
        "middle_band_behavior": middle_band_behavior,
    }


def _resolve_two_stage_auto_balance_search_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = two_stage_cfg.get("auto_balance_search", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    enabled = bool(raw_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    search_mode = str(raw_cfg.get("search_mode", "grid")).strip().lower()
    if search_mode not in {"grid"}:
        raise ValueError("two_stage.auto_balance_search.search_mode currently supports only 'grid'.")

    def _float_grid(key: str, default: list[float]) -> list[float]:
        raw_values = raw_cfg.get(key, default)
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"two_stage.auto_balance_search.{key} must be a non-empty list.")
        values = [float(v) for v in raw_values]
        if any(v <= 0.0 for v in values):
            raise ValueError(f"two_stage.auto_balance_search.{key} values must be > 0.")
        return values

    def _threshold_grid(key: str, default: list[float]) -> list[float]:
        values = _float_grid(key, default)
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError(f"two_stage.auto_balance_search.{key} values must be within [0.0, 1.0].")
        return values

    max_configs_per_model = int(raw_cfg.get("max_configs_per_model", 120))
    if max_configs_per_model <= 0:
        raise ValueError("two_stage.auto_balance_search.max_configs_per_model must be >= 1.")

    early_stop_after = raw_cfg.get("early_stop_if_no_improvement_after")
    if early_stop_after is not None:
        early_stop_after = int(early_stop_after)
        if early_stop_after <= 0:
            early_stop_after = None

    return {
        "enabled": enabled,
        "search_mode": search_mode,
        "max_configs_per_model": max_configs_per_model,
        "early_stop_if_no_improvement_after": early_stop_after,
        "random_subsample_configs_if_too_large": bool(raw_cfg.get("random_subsample_configs_if_too_large", True)),
        "threshold_strategy": str(raw_cfg.get("threshold_strategy", "auto_search")).strip().lower(),
        "threshold_search_mode": str(raw_cfg.get("threshold_search_mode", "band")).strip().lower(),
        "stage1_non_dropout_weight_grid": _float_grid(
            "stage1_non_dropout_weight_grid",
            [1.00, 1.05, 1.10, 1.15, 1.20],
        ),
        "stage2_enrolled_weight_grid": _float_grid(
            "stage2_enrolled_weight_grid",
            [1.00, 1.10, 1.20, 1.30, 1.40],
        ),
        "enrolled_push_alpha_grid": _threshold_grid(
            "enrolled_push_alpha_grid",
            [0.00, 0.10, 0.20, 0.30, 0.40],
        ),
        "threshold_grid_low": _threshold_grid(
            "threshold_grid_low",
            [0.20, 0.25, 0.30, 0.35, 0.40],
        ),
        "threshold_grid_high": _threshold_grid(
            "threshold_grid_high",
            [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        ),
    }


def _resolve_two_stage_selection_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = two_stage_cfg.get("selection", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    objective = str(raw_cfg.get("objective", "macro_f1_only")).strip().lower()
    supported_objectives = {
        "macro_f1_only",
        "macro_f1_plus_enrolled_f1",
        "constrained_macro_with_class_floors",
        "constrained_macro_with_soft_penalty",
    }
    if objective not in supported_objectives:
        raise ValueError(
            "two_stage.selection.objective must be one of: "
            "macro_f1_only, macro_f1_plus_enrolled_f1, constrained_macro_with_class_floors, constrained_macro_with_soft_penalty."
        )

    tie_break_order = raw_cfg.get(
        "tie_break_order",
        ["macro_f1", "enrolled_f1", "balanced_accuracy", "macro_recall", "simpler_config"],
    )
    if not isinstance(tie_break_order, list) or not tie_break_order:
        raise ValueError("two_stage.selection.tie_break_order must be a non-empty list.")

    return {
        "objective": objective,
        "dropout_f1_floor": float(raw_cfg.get("dropout_f1_floor", 0.0)),
        "graduate_f1_floor": float(raw_cfg.get("graduate_f1_floor", 0.0)),
        "enrolled_f1_soft_target": float(raw_cfg.get("enrolled_f1_soft_target", 0.0)),
        "penalty_value": float(raw_cfg.get("penalty_value", -999999.0)),
        "tie_break_order": [str(v).strip().lower() for v in tie_break_order],
    }


def _build_two_stage_auto_balance_candidates(
    auto_balance_cfg: dict[str, Any],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not bool(auto_balance_cfg.get("enabled", False)):
        return [], {"full_candidate_count": 0, "selected_candidate_count": 0, "sampling_applied": False}

    candidates: list[dict[str, Any]] = []
    for stage1_weight, stage2_weight, alpha, low_threshold, high_threshold in itertools.product(
        auto_balance_cfg.get("stage1_non_dropout_weight_grid", []),
        auto_balance_cfg.get("stage2_enrolled_weight_grid", []),
        auto_balance_cfg.get("enrolled_push_alpha_grid", []),
        auto_balance_cfg.get("threshold_grid_low", []),
        auto_balance_cfg.get("threshold_grid_high", []),
    ):
        if float(low_threshold) >= float(high_threshold):
            continue
        candidates.append(
            {
                "stage1_non_dropout_weight": float(stage1_weight),
                "stage2_enrolled_weight": float(stage2_weight),
                "enrolled_push_alpha": float(alpha),
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
            }
        )

    full_candidate_count = len(candidates)
    max_configs = int(auto_balance_cfg.get("max_configs_per_model", full_candidate_count or 1))
    if full_candidate_count > max_configs:
        if bool(auto_balance_cfg.get("random_subsample_configs_if_too_large", True)):
            rng = np.random.default_rng(seed)
            selected_idx = sorted(rng.choice(full_candidate_count, size=max_configs, replace=False).tolist())
            candidates = [candidates[idx] for idx in selected_idx]
            sampling_strategy = "deterministic_random_subsample"
        else:
            candidates = candidates[:max_configs]
            sampling_strategy = "deterministic_prefix"
        sampling_applied = True
    else:
        sampling_strategy = "full_grid"
        sampling_applied = False

    return candidates, {
        "full_candidate_count": int(full_candidate_count),
        "selected_candidate_count": int(len(candidates)),
        "sampling_applied": sampling_applied,
        "sampling_strategy": sampling_strategy,
        "max_configs_per_model": max_configs,
    }


def _build_two_stage_stage_weight_configs(
    *,
    stage1_non_dropout_weight: float,
    stage2_enrolled_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage1_non_dropout_weight <= 0.0 or stage2_enrolled_weight <= 0.0:
        raise ValueError("Auto-balance stage weights must be positive.")
    stage1_cfg = {
        "enabled": True,
        "mode": "explicit",
        "strategy": "explicit",
        "values": {"Non-Dropout": float(stage1_non_dropout_weight), "Dropout": 1.0},
        "class_weight_map": {0: float(stage1_non_dropout_weight), 1: 1.0},
        "class_label_to_index": {"Non-Dropout": 0, "Dropout": 1},
    }
    stage2_cfg = {
        "enabled": True,
        "mode": "explicit",
        "strategy": "explicit",
        "values": {"Graduate": 1.0, "Enrolled": float(stage2_enrolled_weight)},
        "class_weight_map": {"Graduate": 1.0, "Enrolled": float(stage2_enrolled_weight)},
        "class_label_to_index": {"Graduate": 0, "Enrolled": 1},
    }
    return stage1_cfg, stage2_cfg


def _score_two_stage_auto_balance_candidate(
    *,
    metrics: dict[str, Any],
    per_class: dict[str, Any],
    selection_cfg: dict[str, Any],
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
    stage2_enrolled_weight: float,
    enrolled_push_alpha: float,
) -> dict[str, Any]:
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    balanced_accuracy = float(metrics.get("balanced_accuracy", 0.0))
    macro_recall = float(metrics.get("macro_recall", 0.0))
    dropout_f1 = float(per_class.get(str(dropout_idx), {}).get("f1", 0.0))
    enrolled_f1 = float(per_class.get(str(enrolled_idx), {}).get("f1", 0.0))
    graduate_f1 = float(per_class.get(str(graduate_idx), {}).get("f1", 0.0))

    objective = str(selection_cfg.get("objective", "macro_f1_only")).strip().lower()
    penalty_value = float(selection_cfg.get("penalty_value", -999999.0))
    dropout_floor = float(selection_cfg.get("dropout_f1_floor", 0.0))
    graduate_floor = float(selection_cfg.get("graduate_f1_floor", 0.0))
    enrolled_soft_target = float(selection_cfg.get("enrolled_f1_soft_target", 0.0))

    dropout_floor_met = dropout_f1 >= dropout_floor
    graduate_floor_met = graduate_f1 >= graduate_floor
    hard_floors_met = bool(dropout_floor_met and graduate_floor_met)
    enrolled_soft_target_met = enrolled_f1 >= enrolled_soft_target

    if objective == "macro_f1_plus_enrolled_f1":
        objective_score = macro_f1 + (float(enrolled_push_alpha) * enrolled_f1)
    elif objective == "constrained_macro_with_class_floors":
        objective_score = macro_f1 if hard_floors_met else penalty_value
    elif objective == "constrained_macro_with_soft_penalty":
        soft_penalty = max(0.0, enrolled_soft_target - enrolled_f1)
        objective_score = (macro_f1 - soft_penalty) if hard_floors_met else penalty_value
    else:
        objective_score = macro_f1

    tie_break_values = {
        "macro_f1": macro_f1,
        "enrolled_f1": enrolled_f1,
        "balanced_accuracy": balanced_accuracy,
        "macro_recall": macro_recall,
        "simpler_config": -abs(float(stage2_enrolled_weight) - 1.0),
    }
    rank_tuple = tuple(float(tie_break_values.get(token, float("-inf"))) for token in selection_cfg.get("tie_break_order", []))

    return {
        "objective_score": float(objective_score),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "macro_recall": macro_recall,
        "dropout_f1": dropout_f1,
        "enrolled_f1": enrolled_f1,
        "graduate_f1": graduate_f1,
        "dropout_floor_met": bool(dropout_floor_met),
        "graduate_floor_met": bool(graduate_floor_met),
        "hard_floors_met": bool(hard_floors_met),
        "enrolled_soft_target_met": bool(enrolled_soft_target_met),
        "rank_tuple": rank_tuple,
    }


def _resolve_two_stage_decision_mode(experiment_mode: str, two_stage_cfg: dict[str, Any]) -> str | None:
    if bool(two_stage_cfg) and not bool(two_stage_cfg.get("enabled", True)):
        return None
    final_cfg = (
        two_stage_cfg.get("final_decision", {})
        if isinstance(two_stage_cfg.get("final_decision", {}), dict)
        else {}
    )
    requested_mode = str(final_cfg.get("mode", "")).strip().lower()
    final_mode_aliases = {
        "hard_routing": "hard_routing",
        "soft_fusion": "soft_fusion",
        "soft_fusion_with_dropout_threshold": "soft_fusion_with_dropout_threshold",
        "soft_fusion_with_middle_band": "soft_fusion_with_middle_band",
        "pure_soft_argmax": "pure_soft_argmax",
    }
    if requested_mode:
        if requested_mode not in final_mode_aliases:
            raise ValueError(
                "two_stage.final_decision.mode must be one of: "
                "hard_routing, soft_fusion, soft_fusion_with_dropout_threshold, soft_fusion_with_middle_band, pure_soft_argmax."
            )
        return final_mode_aliases[requested_mode]

    two_stage_mode_aliases = {
        "two_stage": "hard_routing",
        "hierarchical": "hard_routing",
        "two_stage_uct_3class": "hard_routing",
        "two_stage_soft": "soft_fused",
        "two_stage_uct_3class_soft": "soft_fused",
    }
    return two_stage_mode_aliases.get(experiment_mode)


def _threshold_vector_from_map(labels: list[int], thresholds_map: dict[int, float] | None) -> np.ndarray:
    out = np.zeros(len(labels), dtype=float)
    if not thresholds_map:
        return out
    index = {int(label): i for i, label in enumerate(labels)}
    for class_idx, threshold in thresholds_map.items():
        idx = index.get(int(class_idx))
        if idx is None:
            continue
        value = float(threshold)
        if value < 0.0 or value > 1.0:
            raise ValueError("threshold values must be within [0.0, 1.0].")
        out[idx] = value
    return out


def _predict_two_stage_from_fused_probabilities(
    fused_proba: np.ndarray,
    labels: list[int],
    *,
    decision_mode: str,
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
    dropout_threshold: float,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    class_thresholds: dict[int, float] | None = None,
    stage2_prob_enrolled: np.ndarray | None = None,
    stage2_prob_graduate: np.ndarray | None = None,
    stage2_decision_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    def _finalize_outputs(pred: Any, decision_region: Any, *, branch_name: str) -> tuple[np.ndarray, np.ndarray]:
        pred_arr = np.asarray(pred)
        region_arr = np.asarray(decision_region)
        print(
            "[two_stage][predict_output] "
            f"branch={branch_name} pred_dtype={pred_arr.dtype} pred_ndim={pred_arr.ndim} pred_shape={pred_arr.shape} "
            f"region_dtype={region_arr.dtype} region_ndim={region_arr.ndim} region_shape={region_arr.shape}"
        )
        if pred_arr.ndim != 1:
            raise ValueError(
                f"_predict_two_stage_from_fused_probabilities branch='{branch_name}' returned invalid pred "
                f"with dtype={pred_arr.dtype}, ndim={pred_arr.ndim}, shape={pred_arr.shape}."
            )
        if region_arr.ndim != 1:
            raise ValueError(
                f"_predict_two_stage_from_fused_probabilities branch='{branch_name}' returned invalid decision_region "
                f"with dtype={region_arr.dtype}, ndim={region_arr.ndim}, shape={region_arr.shape}."
            )
        if pred_arr.shape[0] != region_arr.shape[0]:
            raise ValueError(
                f"_predict_two_stage_from_fused_probabilities branch='{branch_name}' length mismatch: "
                f"pred={pred_arr.shape[0]} decision_region={region_arr.shape[0]}."
            )
        try:
            pred_arr = pred_arr.astype(int)
        except Exception as exc:
            raise ValueError(
                f"_predict_two_stage_from_fused_probabilities branch='{branch_name}' could not cast pred to int; "
                f"dtype={pred_arr.dtype}, shape={pred_arr.shape}, error={type(exc).__name__}:{exc}"
            ) from exc
        region_arr = region_arr.astype(str)
        return pred_arr, region_arr

    label_arr = np.asarray([int(v) for v in labels], dtype=int)
    mode = str(decision_mode).strip().lower()
    if mode == "soft_fusion_with_dropout_threshold":
        pred = TwoStageUct3ClassClassifier.predict_with_dropout_threshold_from_fused_probabilities(
            fused_proba=np.asarray(fused_proba, dtype=float),
            classes=label_arr,
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            dropout_threshold=float(dropout_threshold),
            p_enrolled_given_non_dropout=stage2_prob_enrolled,
            p_graduate_given_non_dropout=stage2_prob_graduate,
            stage2_decision_config=stage2_decision_config,
        )
        decision_region = np.where(
            np.asarray(fused_proba, dtype=float)[:, list(label_arr).index(int(dropout_idx))] >= float(dropout_threshold),
            "hard_dropout",
            "safe_non_dropout",
        )
        return _finalize_outputs(pred, decision_region, branch_name="soft_fusion_with_dropout_threshold")
    if mode == "hard_routing":
        fused_proba_arr = np.asarray(fused_proba, dtype=float)
        class_to_idx = {int(label): idx for idx, label in enumerate(label_arr.tolist())}
        p_dropout = fused_proba_arr[:, class_to_idx[int(dropout_idx)]]
        enrolled_preferred = np.asarray(stage2_prob_enrolled, dtype=float) >= np.asarray(stage2_prob_graduate, dtype=float)
        pred = np.where(
            p_dropout >= float(dropout_threshold),
            int(dropout_idx),
            np.where(enrolled_preferred, int(enrolled_idx), int(graduate_idx)),
        ).astype(int)
        pred, _ = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            pred,
            p_enrolled_given_non_dropout=stage2_prob_enrolled,
            p_graduate_given_non_dropout=stage2_prob_graduate,
            p_dropout=p_dropout,
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=stage2_decision_config,
        )
        decision_region = np.where(
            p_dropout >= float(dropout_threshold),
            "hard_dropout",
            "safe_non_dropout",
        )
        return _finalize_outputs(pred, decision_region, branch_name="hard_routing")
    if mode == "soft_fusion_with_middle_band":
        pred, decision_region = TwoStageUct3ClassClassifier.predict_with_middle_band_from_fused_probabilities(
            fused_proba=np.asarray(fused_proba, dtype=float),
            classes=label_arr,
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            low_threshold=float(low_threshold if low_threshold is not None else dropout_threshold),
            high_threshold=float(high_threshold if high_threshold is not None else dropout_threshold),
            p_enrolled_given_non_dropout=stage2_prob_enrolled,
            p_graduate_given_non_dropout=stage2_prob_graduate,
            stage2_decision_config=stage2_decision_config,
        )
        return _finalize_outputs(pred, decision_region, branch_name="soft_fusion_with_middle_band")
    if mode in {"pure_soft_argmax", "soft_fusion"}:
        pred_idx = np.argmax(np.asarray(fused_proba, dtype=float), axis=1)
        pred, _ = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            label_arr[pred_idx],
            p_enrolled_given_non_dropout=stage2_prob_enrolled,
            p_graduate_given_non_dropout=stage2_prob_graduate,
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=stage2_decision_config,
        )
        return _finalize_outputs(
            pred,
            np.full(shape=(np.asarray(fused_proba).shape[0],), fill_value="soft_fusion", dtype=str),
            branch_name=mode,
        )
    thresholds_vec = _threshold_vector_from_map(labels, class_thresholds or {})
    pred = TwoStageUct3ClassClassifier.predict_from_fused_probabilities(
        fused_proba=np.asarray(fused_proba, dtype=float),
        classes=label_arr,
        thresholds=thresholds_vec,
    )
    pred, _ = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
        pred,
        p_enrolled_given_non_dropout=stage2_prob_enrolled,
        p_graduate_given_non_dropout=stage2_prob_graduate,
        dropout_label=int(dropout_idx),
        enrolled_label=int(enrolled_idx),
        graduate_label=int(graduate_idx),
        stage2_decision_config=stage2_decision_config,
    )
    return _finalize_outputs(
        pred,
        np.full(shape=(np.asarray(fused_proba).shape[0],), fill_value="soft_threshold", dtype=str),
        branch_name="soft_threshold",
    )


def _resolve_two_stage_stage2_decision_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    decision_policy_cfg = stage2_cfg.get("decision_policy", {}) if isinstance(stage2_cfg.get("decision_policy", {}), dict) else {}
    raw_cfg = two_stage_cfg.get("stage2_decision", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    use_decision_policy_schema = bool(decision_policy_cfg)
    source_cfg = decision_policy_cfg if use_decision_policy_schema else raw_cfg

    search_cfg = source_cfg.get("search", {}) if isinstance(source_cfg.get("search", {}), dict) else {}
    objective_cfg = source_cfg.get("objective", {}) if isinstance(source_cfg.get("objective", {}), dict) else {}
    acceptance_cfg = source_cfg.get("acceptance", {}) if isinstance(source_cfg.get("acceptance", {}), dict) else {}
    overfit_cfg = source_cfg.get("anti_overfit", {}) if isinstance(source_cfg.get("anti_overfit", {}), dict) else {}
    enabled = bool(source_cfg.get("enabled", False))
    strategy = str(source_cfg.get("strategy", "enrolled_guarded_threshold")).strip().lower()
    mode = str(source_cfg.get("mode", strategy if use_decision_policy_schema else "legacy")).strip().lower()
    if strategy not in {"argmax", "enrolled_guarded_threshold"}:
        raise ValueError("two_stage.stage2.stage2_decision/decision_policy.strategy must be 'argmax' or 'enrolled_guarded_threshold'.")
    if strategy != "enrolled_guarded_threshold":
        enabled = False

    def _bounded_float(value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
        parsed = float(value)
        return min(max(parsed, minimum), maximum)

    def _bounded_signed_float(value: Any, *, minimum: float = -1.0, maximum: float = 1.0) -> float:
        parsed = float(value)
        return min(max(parsed, minimum), maximum)

    def _grid_from_cfg(payload: dict[str, Any], default_min: float, default_max: float, default_step: float) -> list[float]:
        lower = _bounded_float(payload.get("min", default_min))
        upper = _bounded_float(payload.get("max", default_max))
        step = float(payload.get("step", default_step))
        if step <= 0.0:
            raise ValueError("two_stage.stage2_decision.search step values must be > 0.")
        if lower > upper:
            lower, upper = upper, lower
        values: list[float] = []
        current = lower
        while current <= upper + 1e-12:
            values.append(round(current, 6))
            current += step
        return sorted({float(v) for v in values}) or [round(lower, 6)]

    enrolled_search_cfg = (
        search_cfg.get("enrolled_probability_threshold", {})
        if isinstance(search_cfg.get("enrolled_probability_threshold", {}), dict)
        else {}
    )
    margin_search_cfg = (
        search_cfg.get("graduate_margin_guard", {})
        if isinstance(search_cfg.get("graduate_margin_guard", {}), dict)
        else {}
    )
    enrolled_margin_search_cfg = (
        search_cfg.get("enrolled_margin", {})
        if isinstance(search_cfg.get("enrolled_margin", {}), dict)
        else {}
    )
    dropout_guard_search_cfg = (
        search_cfg.get("dropout_probability_guard", {})
        if isinstance(search_cfg.get("dropout_probability_guard", {}), dict)
        else {}
    )
    calibration_method = str(source_cfg.get("calibration_method", "none")).strip().lower()
    if calibration_method not in {"none", "temperature_scaling", "sigmoid", "isotonic"}:
        calibration_method = "none"
    selected_threshold = _bounded_float(source_cfg.get("enrolled_probability_threshold", 0.42))
    selected_margin_guard = _bounded_float(source_cfg.get("graduate_margin_guard", 0.06))
    selected_enrolled_margin = source_cfg.get("enrolled_margin")
    if selected_enrolled_margin is None:
        selected_enrolled_margin = -float(selected_margin_guard)
    selected_dropout_probability_guard = _bounded_float(source_cfg.get("dropout_probability_guard", 1.0))

    return {
        "enabled": enabled,
        "strategy": "enrolled_guarded_threshold" if enabled else "argmax",
        "enrolled_probability_threshold": selected_threshold,
        "graduate_margin_guard": selected_margin_guard,
        "enrolled_margin": _bounded_signed_float(selected_enrolled_margin),
        "dropout_probability_guard": selected_dropout_probability_guard,
        "tune_on_validation": bool(source_cfg.get("tune_on_validation", True)),
        "log_selection": bool(source_cfg.get("log_selection", True)),
        "config_schema": "decision_policy" if use_decision_policy_schema else "legacy_stage2_decision",
        "mode": mode,
        "use_calibrated_proba": bool(source_cfg.get("use_calibrated_proba", False)),
        "calibration_method": calibration_method,
        "search": {
            "enrolled_probability_threshold_grid": _grid_from_cfg(enrolled_search_cfg, 0.30, 0.60, 0.02),
            "graduate_margin_guard_grid": _grid_from_cfg(margin_search_cfg, 0.00, 0.12, 0.02),
            "enrolled_margin_grid": _grid_from_cfg(enrolled_margin_search_cfg, 0.00, 0.12, 0.02),
            "dropout_probability_guard_grid": _grid_from_cfg(dropout_guard_search_cfg, 1.00, 1.00, 1.00),
            "method": str(search_cfg.get("method", "grid")).strip().lower(),
        },
        "objective": {
            "name": str(objective_cfg.get("name", "macro_f1_plus_enrolled_gain_with_graduate_guard")).strip().lower(),
            "mode": str(
                objective_cfg.get("mode", objective_cfg.get("objective_mode", objective_cfg.get("name", "legacy_macro_priority")))
            ).strip().lower(),
            "enrolled_f1_alpha": float(objective_cfg.get("enrolled_f1_alpha", 0.35)),
            "graduate_f1_penalty_beta": float(objective_cfg.get("graduate_f1_penalty_beta", 1.5)),
            "metric": str(objective_cfg.get("metric", "custom")).strip().lower(),
            "alpha_enrolled_f1": float(objective_cfg.get("alpha_enrolled_f1", objective_cfg.get("enrolled_f1_alpha", 0.35))),
            "beta_graduate_drop_penalty": float(
                objective_cfg.get("beta_graduate_drop_penalty", objective_cfg.get("graduate_f1_penalty_beta", 1.5))
            ),
            "gamma_macro_f1": float(objective_cfg.get("gamma_macro_f1", 1.0)),
            "enrolled_f1_floor": float(objective_cfg.get("enrolled_f1_floor", 0.0)),
            "graduate_f1_tolerance_vs_baseline": float(objective_cfg.get("graduate_f1_tolerance_vs_baseline", 0.02)),
        },
        "acceptance": {
            "metric": str(acceptance_cfg.get("metric", "macro_or_enrolled_guarded")).strip().lower(),
            "macro_f1_improvement_epsilon": float(acceptance_cfg.get("macro_f1_improvement_epsilon", 0.001)),
            "min_enrolled_f1_gain": float(acceptance_cfg.get("min_enrolled_f1_gain", 0.02)),
            "graduate_f1_tolerance_vs_baseline": float(
                acceptance_cfg.get(
                    "graduate_f1_tolerance_vs_baseline",
                    objective_cfg.get("graduate_f1_tolerance_vs_baseline", 0.02),
                )
            ),
            "max_final_macro_f1_drop": float(acceptance_cfg.get("max_final_macro_f1_drop", 0.002)),
        },
        "anti_overfit": {
            "strategy": str(overfit_cfg.get("strategy", "stage2_train_inner_split")).strip().lower(),
            "tuning_size": float(overfit_cfg.get("tuning_size", 0.25)),
            "min_tuning_samples": int(overfit_cfg.get("min_tuning_samples", 24)),
            "random_state": int(overfit_cfg.get("random_state", int(two_stage_cfg.get("seed", 42) or 42))),
        },
    }


def _resolve_two_stage_stage2_optuna_tuning_config(two_stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage2_cfg = two_stage_cfg.get("stage2", {}) if isinstance(two_stage_cfg.get("stage2", {}), dict) else {}
    raw_cfg = stage2_cfg.get("optuna_tuning", {}) if isinstance(stage2_cfg.get("optuna_tuning", {}), dict) else {}
    objective_cfg = raw_cfg.get("objective", {}) if isinstance(raw_cfg.get("objective", {}), dict) else {}
    search_space_cfg = raw_cfg.get("search_space", {}) if isinstance(raw_cfg.get("search_space", {}), dict) else {}

    def _normalize_float_space(
        payload: dict[str, Any],
        *,
        default_low: float,
        default_high: float,
    ) -> dict[str, Any]:
        low = float(payload.get("low", default_low))
        high = float(payload.get("high", default_high))
        if high < low:
            low, high = high, low
        return {
            "type": "float",
            "low": float(low),
            "high": float(high),
            "log": bool(payload.get("log", False)),
        }

    enabled = bool(raw_cfg.get("enabled", False))
    method = str(raw_cfg.get("method", "optuna")).strip().lower()
    sampler = str(raw_cfg.get("sampler", "tpe")).strip().lower()
    direction = str(raw_cfg.get("direction", "maximize")).strip().lower()
    if method not in {"optuna", ""}:
        raise ValueError("two_stage.stage2.optuna_tuning.method must be 'optuna'.")
    if sampler not in {"tpe", "random"}:
        raise ValueError("two_stage.stage2.optuna_tuning.sampler must be 'tpe' or 'random'.")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("two_stage.stage2.optuna_tuning.direction must be 'maximize' or 'minimize'.")

    n_trials = int(raw_cfg.get("n_trials", 20))
    if n_trials < 1:
        raise ValueError("two_stage.stage2.optuna_tuning.n_trials must be >= 1 when enabled.")

    return {
        "enabled": enabled and method == "optuna",
        "method": "optuna",
        "n_trials": n_trials,
        "sampler": sampler,
        "direction": direction,
        "retrain_per_trial": bool(raw_cfg.get("retrain_per_trial", True)),
        "objective": {
            "type": str(objective_cfg.get("type", "enrolled_guarded_macro_f1")).strip().lower(),
            "alpha": float(objective_cfg.get("alpha", 0.35)),
            "beta": float(objective_cfg.get("beta", 1.25)),
        },
        "search_space": {
            "enrolled_class_weight_scale": _normalize_float_space(
                search_space_cfg.get("enrolled_class_weight_scale", {})
                if isinstance(search_space_cfg.get("enrolled_class_weight_scale", {}), dict)
                else {},
                default_low=1.0,
                default_high=3.5,
            ),
            "enrolled_probability_threshold": _normalize_float_space(
                search_space_cfg.get("enrolled_probability_threshold", {})
                if isinstance(search_space_cfg.get("enrolled_probability_threshold", {}), dict)
                else {},
                default_low=0.30,
                default_high=0.60,
            ),
            "graduate_margin_guard": _normalize_float_space(
                search_space_cfg.get("graduate_margin_guard", {})
                if isinstance(search_space_cfg.get("graduate_margin_guard", {}), dict)
                else {},
                default_low=0.00,
                default_high=0.12,
            ),
        },
    }


def _resolve_two_stage_stage2_weight_cfg_with_scale(
    stage2_class_weight_cfg: dict[str, Any],
    *,
    enrolled_class_weight_scale: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = dict(stage2_class_weight_cfg) if isinstance(stage2_class_weight_cfg, dict) else {}
    explicit_map = cfg.get("class_weight_map", cfg.get("values", {}))
    explicit_map = explicit_map if isinstance(explicit_map, dict) else {}
    base_enrolled_weight = explicit_map.get("Enrolled", explicit_map.get("enrolled", 1.0))
    base_graduate_weight = explicit_map.get("Graduate", explicit_map.get("graduate", 1.0))
    base_enrolled_weight = float(base_enrolled_weight)
    base_graduate_weight = float(base_graduate_weight)
    selected_enrolled_weight = base_enrolled_weight * float(enrolled_class_weight_scale)
    if selected_enrolled_weight <= 0.0 or base_graduate_weight <= 0.0:
        raise ValueError("Resolved Stage 2 class weights must be positive.")
    cfg.update(
        {
            "enabled": True,
            "mode": "explicit",
            "strategy": "explicit",
            "values": {"Graduate": float(base_graduate_weight), "Enrolled": float(selected_enrolled_weight)},
            "class_weight_map": {"Graduate": float(base_graduate_weight), "Enrolled": float(selected_enrolled_weight)},
        }
    )
    return cfg, {
        "base_enrolled_weight": float(base_enrolled_weight),
        "base_graduate_weight": float(base_graduate_weight),
        "selected_enrolled_weight": float(selected_enrolled_weight),
        "selected_graduate_weight": float(base_graduate_weight),
    }


def _tune_two_stage_stage2_optuna(
    *,
    model_name: str,
    params_stage2: dict[str, Any],
    seed: int,
    X_train_stage2: pd.DataFrame,
    y_train_stage2: pd.Series,
    X_valid_stage2: pd.DataFrame,
    y_valid_stage2_binary: pd.Series,
    y_valid_stage2_original: pd.Series,
    enrolled_idx: int,
    graduate_idx: int,
    stage2_positive_target_label: int,
    stage2_class_weight_cfg: dict[str, Any],
    stage2_decision_cfg: dict[str, Any],
    stage2_optuna_cfg: dict[str, Any],
) -> dict[str, Any]:
    resolved_optuna_cfg = (
        dict(stage2_optuna_cfg)
        if isinstance(stage2_optuna_cfg, dict)
        else {"enabled": False, "method": "optuna", "sampler": "tpe", "direction": "maximize", "n_trials": 0}
    )
    if not bool(resolved_optuna_cfg.get("enabled", False)):
        return {
            "status": "skipped",
            "reason": "stage2_optuna_tuning_disabled",
            "enabled": False,
            "selection_split": "validation",
        }

    if not bool(stage2_decision_cfg.get("enabled", False)):
        return {
            "status": "skipped",
            "reason": "stage2_optuna_requires_guarded_stage2_decision",
            "enabled": True,
            "selection_split": "validation",
        }

    try:
        import optuna
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"optuna_import_failed:{type(exc).__name__}",
            "enabled": True,
            "selection_split": "validation",
        }

    if bool(resolved_optuna_cfg.get("retrain_per_trial", True)) is False:
        return {
            "status": "skipped",
            "reason": "stage2_optuna_requires_retrain_per_trial",
            "enabled": True,
            "selection_split": "validation",
        }

    objective_cfg = resolved_optuna_cfg.get("objective", {}) if isinstance(resolved_optuna_cfg.get("objective", {}), dict) else {}
    search_space = resolved_optuna_cfg.get("search_space", {}) if isinstance(resolved_optuna_cfg.get("search_space", {}), dict) else {}
    label_order_binary = [0, 1]
    y_valid_original_arr = np.asarray(y_valid_stage2_original, dtype=int)
    labels_original = [int(enrolled_idx), int(graduate_idx)]
    candidate_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None

    def _suggest_float(trial: Any, name: str) -> float:
        space = search_space.get(name, {}) if isinstance(search_space.get(name, {}), dict) else {}
        return float(
            trial.suggest_float(
                name,
                float(space.get("low", 0.0)),
                float(space.get("high", 1.0)),
                log=bool(space.get("log", False)),
            )
        )

    def _binary_predictions_to_original(pred: np.ndarray) -> np.ndarray:
        pred_arr = np.asarray(pred, dtype=int)
        if int(stage2_positive_target_label) == int(enrolled_idx):
            return np.where(pred_arr == 1, int(enrolled_idx), int(graduate_idx)).astype(int)
        return np.where(pred_arr == 1, int(graduate_idx), int(enrolled_idx)).astype(int)

    def _probabilities_to_original(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba = np.asarray(probabilities, dtype=float)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("Stage 2 probability output must have shape (n, 2) for Optuna tuning.")
        p_positive = np.clip(proba[:, 1], 0.0, 1.0)
        p_negative = np.clip(1.0 - p_positive, 0.0, 1.0)
        if int(stage2_positive_target_label) == int(enrolled_idx):
            return p_positive, p_negative
        return p_negative, p_positive

    def _score_candidate(
        metrics: dict[str, Any],
        per_class: dict[str, Any],
        baseline_graduate_f1: float,
    ) -> float:
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        enrolled_f1 = float(per_class.get(str(enrolled_idx), {}).get("f1", 0.0))
        graduate_f1 = float(per_class.get(str(graduate_idx), {}).get("f1", 0.0))
        alpha = float(objective_cfg.get("alpha", 0.35))
        beta = float(objective_cfg.get("beta", 1.25))
        graduate_penalty = max(0.0, baseline_graduate_f1 - graduate_f1)
        return macro_f1 + (alpha * enrolled_f1) - (beta * graduate_penalty)

    def _objective(trial: Any) -> float:
        nonlocal best_payload
        enrolled_scale = _suggest_float(trial, "enrolled_class_weight_scale")
        threshold = _suggest_float(trial, "enrolled_probability_threshold")
        margin_guard = _suggest_float(trial, "graduate_margin_guard")
        trial_stage2_weight_cfg, resolved_weight_info = _resolve_two_stage_stage2_weight_cfg_with_scale(
            stage2_class_weight_cfg,
            enrolled_class_weight_scale=float(enrolled_scale),
        )
        trial_result = train_and_evaluate(
            model_name=model_name,
            params=params_stage2,
            X_train=X_train_stage2,
            y_train=y_train_stage2,
            X_valid=X_valid_stage2,
            y_valid=y_valid_stage2_binary,
            X_test=X_valid_stage2,
            y_test=y_valid_stage2_binary,
            eval_config={"seed": seed, "class_weight": trial_stage2_weight_cfg, "label_order": label_order_binary},
        )
        valid_probabilities = trial_result.artifacts.get("y_proba_valid")
        if valid_probabilities is None:
            row = {
                "trial_number": int(trial.number),
                "status": "skipped_no_probabilities",
                "enrolled_class_weight_scale": float(enrolled_scale),
                "enrolled_probability_threshold": float(threshold),
                "graduate_margin_guard": float(margin_guard),
            }
            candidate_rows.append(row)
            trial.set_user_attr("trial_row", row)
            return float("-inf") if resolved_optuna_cfg.get("direction", "maximize") == "maximize" else float("inf")

        baseline_pred_original = _binary_predictions_to_original(np.asarray(trial_result.artifacts.get("y_pred_valid", []), dtype=int))
        p_enrolled, p_graduate = _probabilities_to_original(np.asarray(valid_probabilities, dtype=float))
        baseline_metrics = compute_metrics(pd.Series(y_valid_original_arr), baseline_pred_original)
        baseline_per_class = compute_per_class_metrics(
            pd.Series(y_valid_original_arr),
            baseline_pred_original,
            labels=labels_original,
        )
        baseline_graduate_f1 = float(baseline_per_class.get(str(graduate_idx), {}).get("f1", 0.0))
        selected_cfg = {
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "enrolled_probability_threshold": float(threshold),
            "graduate_margin_guard": float(margin_guard),
            "enrolled_class_weight_scale": float(enrolled_scale),
        }
        tuned_pred, tuned_reasons = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            baseline_pred_original,
            p_enrolled_given_non_dropout=p_enrolled,
            p_graduate_given_non_dropout=p_graduate,
            dropout_label=-1,
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_cfg,
        )
        tuned_metrics = compute_metrics(pd.Series(y_valid_original_arr), tuned_pred)
        tuned_per_class = compute_per_class_metrics(pd.Series(y_valid_original_arr), tuned_pred, labels=labels_original)
        score = float(_score_candidate(tuned_metrics, tuned_per_class, baseline_graduate_f1))
        row = {
            "trial_number": int(trial.number),
            "status": "completed",
            "enrolled_class_weight_scale": float(enrolled_scale),
            "selected_stage2_enrolled_weight": float(resolved_weight_info["selected_enrolled_weight"]),
            "selected_stage2_graduate_weight": float(resolved_weight_info["selected_graduate_weight"]),
            "enrolled_probability_threshold": float(threshold),
            "graduate_margin_guard": float(margin_guard),
            "baseline_macro_f1": float(baseline_metrics.get("macro_f1", 0.0)),
            "baseline_f1_enrolled": float(baseline_per_class.get(str(enrolled_idx), {}).get("f1", 0.0)),
            "baseline_f1_graduate": float(baseline_per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
            "macro_f1": float(tuned_metrics.get("macro_f1", 0.0)),
            "accuracy": float(tuned_metrics.get("accuracy", 0.0)),
            "balanced_accuracy": float(tuned_metrics.get("balanced_accuracy", 0.0)),
            "macro_precision": float(tuned_metrics.get("macro_precision", 0.0)),
            "macro_recall": float(tuned_metrics.get("macro_recall", 0.0)),
            "weighted_f1": float(tuned_metrics.get("weighted_f1", 0.0)),
            "f1_enrolled": float(tuned_per_class.get(str(enrolled_idx), {}).get("f1", 0.0)),
            "f1_graduate": float(tuned_per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
            "objective_score": float(score),
            "enrolled_selected_count": int(np.sum(np.asarray(tuned_pred, dtype=int) == int(enrolled_idx))),
            "graduate_selected_count": int(np.sum(np.asarray(tuned_pred, dtype=int) == int(graduate_idx))),
        }
        candidate_rows.append(row)
        trial.set_user_attr("trial_row", row)
        rank_tuple = (
            float(score),
            float(row["macro_f1"]),
            float(row["f1_enrolled"]),
            float(row["f1_graduate"]),
            -abs(float(threshold) - 0.5),
            -float(margin_guard),
        )
        best_rank_tuple = best_payload.get("rank_tuple") if isinstance(best_payload, dict) else None
        if best_payload is None or rank_tuple > best_rank_tuple:
            best_payload = {
                "rank_tuple": rank_tuple,
                "baseline_validation_metrics": baseline_metrics,
                "baseline_validation_per_class": baseline_per_class,
                "tuned_validation_metrics": tuned_metrics,
                "tuned_validation_per_class": tuned_per_class,
                "validation_objective_score_baseline": float(
                    _score_candidate(baseline_metrics, baseline_per_class, baseline_graduate_f1)
                ),
                "validation_objective_score_selected": float(score),
                "selected_config": selected_cfg,
                "selected_stage2_class_weight_cfg": trial_stage2_weight_cfg,
                "selected_stage2_weights": resolved_weight_info,
                "selected_validation_predictions": np.asarray(tuned_pred, dtype=int).tolist(),
                "selected_validation_decision_reasons": np.asarray(tuned_reasons, dtype=str).tolist(),
            }
        return float(score)

    sampler_name = str(resolved_optuna_cfg.get("sampler", "tpe")).strip().lower()
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=str(resolved_optuna_cfg.get("direction", "maximize")), sampler=sampler)
    study.optimize(_objective, n_trials=int(resolved_optuna_cfg.get("n_trials", 20)))

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if best_payload is None or not completed_trials:
        return {
            "status": "skipped",
            "reason": "stage2_optuna_no_completed_trials",
            "enabled": True,
            "selection_split": "validation",
            "search_results": candidate_rows,
            "optuna": {
                "method": "optuna",
                "sampler": sampler_name,
                "direction": str(resolved_optuna_cfg.get("direction", "maximize")),
                "n_trials_requested": int(resolved_optuna_cfg.get("n_trials", 20)),
                "n_trials_completed": int(len(completed_trials)),
            },
        }

    best_trial = study.best_trial
    return {
        "status": "applied",
        "reason": "stage2_optuna_validation_search_completed",
        "enabled": True,
        "strategy": "enrolled_guarded_threshold",
        "selection_split": "validation",
        "threshold_tuning_requested": True,
        "threshold_tuning_applied": True,
        "threshold_tuning_supported": True,
        "class_weight_tuning_requested": True,
        "class_weight_tuning_applied": True,
        "selected_config": best_payload["selected_config"],
        "selected_stage2_class_weight_cfg": best_payload["selected_stage2_class_weight_cfg"],
        "selected_stage2_weights": best_payload["selected_stage2_weights"],
        "baseline_validation_metrics": best_payload["baseline_validation_metrics"],
        "baseline_validation_per_class": best_payload["baseline_validation_per_class"],
        "tuned_validation_metrics": best_payload["tuned_validation_metrics"],
        "tuned_validation_per_class": best_payload["tuned_validation_per_class"],
        "validation_objective_score_baseline": float(best_payload["validation_objective_score_baseline"]),
        "validation_objective_score_selected": float(best_payload["validation_objective_score_selected"]),
        "selected_validation_predictions": best_payload["selected_validation_predictions"],
        "selected_validation_decision_reasons": best_payload["selected_validation_decision_reasons"],
        "search_results": candidate_rows,
        "search_evaluated_candidates": int(len(candidate_rows)),
        "optuna": {
            "enabled": True,
            "method": "optuna",
            "sampler": sampler_name,
            "direction": str(resolved_optuna_cfg.get("direction", "maximize")),
            "n_trials_requested": int(resolved_optuna_cfg.get("n_trials", 20)),
            "n_trials_completed": int(len(completed_trials)),
            "best_trial_number": int(best_trial.number),
            "best_trial_value": float(best_trial.value),
            "best_trial_params": {
                key: float(value) if isinstance(value, (int, float, np.floating)) else value
                for key, value in best_trial.params.items()
            },
        },
    }


def _fit_stage2_probability_calibrator(
    *,
    p_positive: np.ndarray,
    y_true: pd.Series | np.ndarray,
    method: str,
) -> tuple[Stage2PositiveProbabilityCalibrator | None, dict[str, Any]]:
    resolved_method = str(method).strip().lower()
    p = np.clip(np.asarray(p_positive, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    y_arr = np.asarray(y_true, dtype=int)
    if resolved_method in {"", "none"}:
        return None, {"enabled": False, "applied": False, "method": "none", "reason": "disabled"}
    if p.shape[0] == 0 or y_arr.shape[0] == 0:
        return None, {"enabled": True, "applied": False, "method": resolved_method, "reason": "empty_tuning_split"}
    if p.shape[0] != y_arr.shape[0]:
        return None, {"enabled": True, "applied": False, "method": resolved_method, "reason": "probability_shape_mismatch"}
    if int(pd.Series(y_arr).nunique()) < 2:
        return None, {
            "enabled": True,
            "applied": False,
            "method": resolved_method,
            "reason": "calibration_requires_two_classes",
        }

    try:
        if resolved_method == "temperature_scaling":
            best_temperature = 1.0
            best_loss = float("inf")
            for temperature in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
                calibrator = Stage2PositiveProbabilityCalibrator(
                    method="temperature_scaling",
                    payload={"temperature": float(temperature)},
                )
                calibrated = np.clip(calibrator.transform(p), 1.0e-6, 1.0 - 1.0e-6)
                loss = float(log_loss(y_arr, np.column_stack([1.0 - calibrated, calibrated]), labels=[0, 1]))
                if loss < best_loss:
                    best_loss = loss
                    best_temperature = float(temperature)
            return Stage2PositiveProbabilityCalibrator(
                method="temperature_scaling",
                payload={"temperature": float(best_temperature)},
            ), {
                "enabled": True,
                "applied": True,
                "method": "temperature_scaling",
                "temperature": float(best_temperature),
                "optimization_metric": "log_loss",
                "sample_count": int(len(y_arr)),
            }
        if resolved_method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            model = LogisticRegression(random_state=42, solver="lbfgs")
            model.fit(logits, y_arr)
            return Stage2PositiveProbabilityCalibrator(
                method="sigmoid",
                payload={
                    "coef": float(model.coef_[0][0]),
                    "intercept": float(model.intercept_[0]),
                },
            ), {
                "enabled": True,
                "applied": True,
                "method": "sigmoid",
                "sample_count": int(len(y_arr)),
            }
        if resolved_method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p, y_arr)
            return Stage2PositiveProbabilityCalibrator(
                method="isotonic",
                payload={
                    "x_thresholds": np.asarray(model.X_thresholds_, dtype=float).tolist(),
                    "y_thresholds": np.asarray(model.y_thresholds_, dtype=float).tolist(),
                },
            ), {
                "enabled": True,
                "applied": True,
                "method": "isotonic",
                "sample_count": int(len(y_arr)),
            }
    except Exception as exc:
        return None, {
            "enabled": True,
            "applied": False,
            "method": resolved_method,
            "reason": f"calibration_failed:{type(exc).__name__}:{exc}",
        }
    return None, {
        "enabled": True,
        "applied": False,
        "method": resolved_method,
        "reason": "unsupported_method",
    }


def _apply_stage2_positive_probability_calibrator(
    p_positive: np.ndarray,
    calibrator: Stage2PositiveProbabilityCalibrator | None,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(np.asarray(p_positive, dtype=float), 0.0, 1.0)
    return np.clip(np.asarray(calibrator.transform(p_positive), dtype=float), 0.0, 1.0)


def _compute_stage2_decision_objective_components(
    *,
    metrics: dict[str, Any],
    per_class: dict[str, Any],
    enrolled_idx: int,
    graduate_idx: int,
    objective_cfg: dict[str, Any],
    baseline_graduate_f1: float,
    branch_macro_f1: float | None = None,
) -> dict[str, float]:
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    enrolled_f1 = float(per_class.get(str(enrolled_idx), {}).get("f1", 0.0))
    graduate_f1 = float(per_class.get(str(graduate_idx), {}).get("f1", 0.0))
    gamma = float(objective_cfg.get("gamma_macro_f1", 1.0))
    alpha = float(objective_cfg.get("alpha_enrolled_f1", objective_cfg.get("enrolled_f1_alpha", 0.35)))
    beta = float(
        objective_cfg.get("beta_graduate_drop_penalty", objective_cfg.get("graduate_f1_penalty_beta", 1.5))
    )
    tolerance = float(objective_cfg.get("graduate_f1_tolerance_vs_baseline", 0.02))
    enrolled_floor = float(objective_cfg.get("enrolled_f1_floor", 0.0))
    objective_mode = str(
        objective_cfg.get("mode", objective_cfg.get("objective_mode", objective_cfg.get("name", "legacy_macro_priority")))
    ).strip().lower()
    resolved_branch_macro_f1 = float(branch_macro_f1 if branch_macro_f1 is not None else macro_f1)
    graduate_drop_penalty = max(0.0, baseline_graduate_f1 - graduate_f1 - tolerance)
    enrolled_floor_penalty = max(0.0, enrolled_floor - enrolled_f1)
    if objective_mode == "enrolled_priority":
        score_raw = (
            enrolled_f1 * 1000.0
            + resolved_branch_macro_f1 * 100.0
            + macro_f1 * 10.0
            + graduate_f1
            - (beta * graduate_drop_penalty)
            - enrolled_floor_penalty
        )
    else:
        score_raw = (gamma * macro_f1) + (alpha * enrolled_f1) - (beta * graduate_drop_penalty) - enrolled_floor_penalty
    return {
        "objective_mode": objective_mode,
        "macro_f1": macro_f1,
        "enrolled_f1": enrolled_f1,
        "graduate_f1": graduate_f1,
        "branch_macro_f1": resolved_branch_macro_f1,
        "gamma_macro_f1": gamma,
        "alpha_enrolled_f1": alpha,
        "beta_graduate_drop_penalty": beta,
        "graduate_baseline_f1": float(baseline_graduate_f1),
        "graduate_f1_tolerance": tolerance,
        "graduate_drop_penalty": graduate_drop_penalty,
        "enrolled_f1_floor": enrolled_floor,
        "enrolled_f1_floor_penalty": enrolled_floor_penalty,
        "score_raw": float(score_raw),
        "score_normalized": float(score_raw),
    }


def _stage2_decision_policy_acceptance(
    *,
    baseline_metrics: dict[str, Any],
    baseline_per_class: dict[str, Any],
    tuned_metrics: dict[str, Any],
    tuned_per_class: dict[str, Any],
    enrolled_idx: int,
    graduate_idx: int,
    acceptance_cfg: dict[str, Any],
) -> dict[str, Any]:
    macro_delta = float(tuned_metrics.get("macro_f1", 0.0)) - float(baseline_metrics.get("macro_f1", 0.0))
    enrolled_delta = float(tuned_per_class.get(str(enrolled_idx), {}).get("f1", 0.0)) - float(
        baseline_per_class.get(str(enrolled_idx), {}).get("f1", 0.0)
    )
    graduate_delta = float(tuned_per_class.get(str(graduate_idx), {}).get("f1", 0.0)) - float(
        baseline_per_class.get(str(graduate_idx), {}).get("f1", 0.0)
    )
    epsilon = float(acceptance_cfg.get("macro_f1_improvement_epsilon", 0.001))
    min_enrolled_gain = float(acceptance_cfg.get("min_enrolled_f1_gain", 0.02))
    graduate_tolerance = float(acceptance_cfg.get("graduate_f1_tolerance_vs_baseline", 0.02))
    max_final_macro_f1_drop = float(acceptance_cfg.get("max_final_macro_f1_drop", epsilon))
    metric_mode = str(acceptance_cfg.get("metric", "macro_or_enrolled_guarded")).strip().lower()
    if metric_mode == "enrolled_priority_guarded":
        accepted = bool(
            enrolled_delta >= min_enrolled_gain
            and macro_delta >= (-max_final_macro_f1_drop)
            and graduate_delta >= (-graduate_tolerance)
        )
        if enrolled_delta < min_enrolled_gain:
            reason = "rejected_enrolled_gain_below_epsilon"
        elif macro_delta < (-max_final_macro_f1_drop):
            reason = "rejected_final_macro_guard"
        elif graduate_delta < (-graduate_tolerance):
            reason = "rejected_graduate_guard"
        else:
            reason = "accepted"
    else:
        accepted = bool((macro_delta >= epsilon) or (enrolled_delta >= min_enrolled_gain and graduate_delta >= (-graduate_tolerance)))
        reason = "accepted" if accepted else "rejected_fallback_to_baseline"
    return {
        "accepted": accepted,
        "metric": metric_mode,
        "macro_f1_delta": float(macro_delta),
        "enrolled_f1_delta": float(enrolled_delta),
        "graduate_f1_delta": float(graduate_delta),
        "macro_f1_improvement_epsilon": float(epsilon),
        "min_enrolled_f1_gain": float(min_enrolled_gain),
        "graduate_f1_tolerance_vs_baseline": float(graduate_tolerance),
        "max_final_macro_f1_drop": float(max_final_macro_f1_drop),
        "reason": reason,
    }


def _select_two_stage_stage2_decision_on_full_validation(
    *,
    decision_mode: str,
    y_true_valid: pd.Series,
    fused_proba_valid: np.ndarray,
    labels: list[int],
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
    dropout_threshold: float,
    low_threshold: float | None,
    high_threshold: float | None,
    class_thresholds: dict[int, float] | None,
    stage2_prob_enrolled_valid: np.ndarray,
    stage2_prob_graduate_valid: np.ndarray,
    y_true_valid_stage2: pd.Series,
    stage2_prob_enrolled_valid_stage2: np.ndarray,
    stage2_prob_graduate_valid_stage2: np.ndarray,
    stage2_decision_cfg: dict[str, Any],
) -> dict[str, Any]:
    resolved_cfg = dict(stage2_decision_cfg) if isinstance(stage2_decision_cfg, dict) else {"enabled": False, "strategy": "argmax"}
    baseline_cfg = {"enabled": False, "strategy": "argmax"}
    label_order = [int(v) for v in labels]
    baseline_final_eval = _evaluate_two_stage_policy_on_split(
        y_true=y_true_valid,
        fused_proba=np.asarray(fused_proba_valid, dtype=float),
        labels=label_order,
        decision_mode=decision_mode,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        dropout_threshold=float(dropout_threshold),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        class_thresholds=class_thresholds,
        stage2_prob_enrolled=np.asarray(stage2_prob_enrolled_valid, dtype=float),
        stage2_prob_graduate=np.asarray(stage2_prob_graduate_valid, dtype=float),
        stage2_decision_config=baseline_cfg,
    )
    baseline_final_pred = np.asarray(baseline_final_eval["y_pred"], dtype=int)
    baseline_final_per_class = compute_per_class_metrics(y_true_valid, baseline_final_pred, labels=label_order)
    stage2_subset_base_pred = np.where(
        np.asarray(stage2_prob_enrolled_valid_stage2, dtype=float) >= np.asarray(stage2_prob_graduate_valid_stage2, dtype=float),
        int(enrolled_idx),
        int(graduate_idx),
    ).astype(int)
    baseline_stage2_metrics = compute_metrics(y_true_valid_stage2, stage2_subset_base_pred)
    baseline_stage2_per_class = compute_per_class_metrics(
        y_true_valid_stage2,
        stage2_subset_base_pred,
        labels=[int(enrolled_idx), int(graduate_idx)],
    )
    baseline_components = _compute_stage2_decision_objective_components(
        metrics=baseline_final_eval["metrics"],
        per_class=baseline_final_per_class,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        objective_cfg=resolved_cfg.get("objective", {}),
        baseline_graduate_f1=float(baseline_final_per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
        branch_macro_f1=float(baseline_stage2_metrics.get("macro_f1", 0.0)),
    )
    if not bool(resolved_cfg.get("enabled", False)):
        return {
            "status": "skipped",
            "reason": "stage2_decision_disabled",
            "enabled": False,
            "strategy": "argmax",
            "threshold_tuning_requested": False,
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": True,
            "selection_split": "validation",
            "selected_config": baseline_cfg,
            "baseline_validation_metrics": baseline_stage2_metrics,
            "baseline_validation_per_class": baseline_stage2_per_class,
            "tuned_validation_metrics": baseline_stage2_metrics,
            "tuned_validation_per_class": baseline_stage2_per_class,
            "validation_final_metrics_baseline": dict(baseline_final_eval.get("metrics", {})),
            "validation_final_metrics_selected": dict(baseline_final_eval.get("metrics", {})),
            "validation_objective_score_baseline": float(baseline_components.get("score_normalized", 0.0)),
            "validation_objective_score_selected": float(baseline_components.get("score_normalized", 0.0)),
            "objective_components_baseline": baseline_components,
            "objective_components_selected": baseline_components,
            "search_results": [],
            "search_evaluated_candidates": 0,
            "acceptance": {
                "accepted": False,
                "metric": str(resolved_cfg.get("acceptance", {}).get("metric", "macro_or_enrolled_guarded")),
                "reason": "stage2_decision_disabled",
            },
            "selected_validation_predictions": baseline_final_pred.tolist(),
            "selected_validation_decision_reasons": _build_stage2_fallback_reason_array(len(y_true_valid), reason="argmax"),
        }

    default_cfg = {
        "enabled": True,
        "strategy": "enrolled_guarded_threshold",
        "mode": str(resolved_cfg.get("mode", "legacy")).strip().lower(),
        "enrolled_probability_threshold": float(resolved_cfg.get("enrolled_probability_threshold", 0.42)),
        "graduate_margin_guard": float(resolved_cfg.get("graduate_margin_guard", 0.06)),
        "enrolled_margin": float(resolved_cfg.get("enrolled_margin", -float(resolved_cfg.get("graduate_margin_guard", 0.06)))),
        "dropout_probability_guard": float(resolved_cfg.get("dropout_probability_guard", 1.0)),
        "use_calibrated_proba": bool(resolved_cfg.get("use_calibrated_proba", False)),
        "calibration_method": str(resolved_cfg.get("calibration_method", "none")).strip().lower(),
    }
    candidate_cfgs: list[dict[str, Any]] = []
    if bool(resolved_cfg.get("tune_on_validation", True)):
        for threshold in resolved_cfg.get("search", {}).get("enrolled_probability_threshold_grid", []):
            for margin_guard in resolved_cfg.get("search", {}).get("graduate_margin_guard_grid", []):
                for enrolled_margin in resolved_cfg.get("search", {}).get("enrolled_margin_grid", []):
                    for dropout_probability_guard in resolved_cfg.get("search", {}).get("dropout_probability_guard_grid", []):
                        candidate_cfgs.append(
                            {
                                **default_cfg,
                                "enrolled_probability_threshold": float(threshold),
                                "graduate_margin_guard": float(margin_guard),
                                "enrolled_margin": float(enrolled_margin),
                                "dropout_probability_guard": float(dropout_probability_guard),
                            }
                        )
    else:
        candidate_cfgs = [dict(default_cfg)]
    if not candidate_cfgs:
        candidate_cfgs = [dict(default_cfg)]

    best_cfg = dict(default_cfg)
    best_final_eval = baseline_final_eval
    best_final_per_class = baseline_final_per_class
    best_stage2_metrics = baseline_stage2_metrics
    best_stage2_per_class = baseline_stage2_per_class
    best_reasons = _build_stage2_fallback_reason_array(len(y_true_valid), reason="argmax")
    best_components = baseline_components
    best_rank = (
        float(baseline_components.get("enrolled_f1", 0.0)),
        float(baseline_components.get("branch_macro_f1", 0.0)),
        float(baseline_components.get("macro_f1", 0.0)),
        float(baseline_components.get("graduate_f1", 0.0)),
        -abs(float(default_cfg.get("enrolled_probability_threshold", 0.5)) - 0.5),
        -float(default_cfg.get("graduate_margin_guard", 0.0)),
        -abs(float(default_cfg.get("enrolled_margin", 0.0))),
    )
    candidate_rows: list[dict[str, Any]] = []
    for candidate_cfg in candidate_cfgs:
        final_eval = _evaluate_two_stage_policy_on_split(
            y_true=y_true_valid,
            fused_proba=np.asarray(fused_proba_valid, dtype=float),
            labels=label_order,
            decision_mode=decision_mode,
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(dropout_threshold),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            class_thresholds=class_thresholds,
            stage2_prob_enrolled=np.asarray(stage2_prob_enrolled_valid, dtype=float),
            stage2_prob_graduate=np.asarray(stage2_prob_graduate_valid, dtype=float),
            stage2_decision_config=candidate_cfg,
        )
        final_pred = np.asarray(final_eval["y_pred"], dtype=int)
        final_per_class = compute_per_class_metrics(y_true_valid, final_pred, labels=label_order)
        _, final_reasons = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            final_pred,
            p_enrolled_given_non_dropout=np.asarray(stage2_prob_enrolled_valid, dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage2_prob_graduate_valid, dtype=float),
            p_dropout=None,
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=candidate_cfg,
        )
        stage2_pred, stage2_reasons = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            stage2_subset_base_pred,
            p_enrolled_given_non_dropout=np.asarray(stage2_prob_enrolled_valid_stage2, dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage2_prob_graduate_valid_stage2, dtype=float),
            p_dropout=None,
            dropout_label=-1,
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=candidate_cfg,
        )
        stage2_metrics = compute_metrics(y_true_valid_stage2, stage2_pred)
        stage2_per_class = compute_per_class_metrics(
            y_true_valid_stage2,
            stage2_pred,
            labels=[int(enrolled_idx), int(graduate_idx)],
        )
        components = _compute_stage2_decision_objective_components(
            metrics=final_eval["metrics"],
            per_class=final_per_class,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            objective_cfg=resolved_cfg.get("objective", {}),
            baseline_graduate_f1=float(baseline_final_per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
            branch_macro_f1=float(stage2_metrics.get("macro_f1", 0.0)),
        )
        candidate_rows.append(
            {
                "enrolled_probability_threshold": float(candidate_cfg.get("enrolled_probability_threshold", np.nan)),
                "graduate_margin_guard": float(candidate_cfg.get("graduate_margin_guard", np.nan)),
                "enrolled_margin": float(candidate_cfg.get("enrolled_margin", np.nan)),
                "dropout_probability_guard": float(candidate_cfg.get("dropout_probability_guard", np.nan)),
                "objective_mode": str(components.get("objective_mode", "")),
                "objective_score": float(components.get("score_normalized", 0.0)),
                "final_valid_enrolled_f1": float(components.get("enrolled_f1", 0.0)),
                "stage2_branch_valid_macro_f1": float(stage2_metrics.get("macro_f1", 0.0)),
                "final_valid_macro_f1": float(components.get("macro_f1", 0.0)),
                "final_valid_graduate_f1": float(components.get("graduate_f1", 0.0)),
            }
        )
        rank = (
            float(components.get("enrolled_f1", 0.0)),
            float(stage2_metrics.get("macro_f1", 0.0)),
            float(components.get("macro_f1", 0.0)),
            float(components.get("graduate_f1", 0.0)),
            -abs(float(candidate_cfg["enrolled_probability_threshold"]) - 0.5),
            -float(candidate_cfg["graduate_margin_guard"]),
            -abs(float(candidate_cfg["enrolled_margin"])),
        )
        if rank > best_rank:
            best_cfg = dict(candidate_cfg)
            best_final_eval = final_eval
            best_final_per_class = final_per_class
            best_stage2_metrics = stage2_metrics
            best_stage2_per_class = stage2_per_class
            best_reasons = np.asarray(final_reasons, dtype=str).tolist()
            best_components = components
            best_rank = rank

    acceptance = _stage2_decision_policy_acceptance(
        baseline_metrics=baseline_final_eval["metrics"],
        baseline_per_class=baseline_final_per_class,
        tuned_metrics=best_final_eval["metrics"],
        tuned_per_class=best_final_per_class,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        acceptance_cfg=resolved_cfg.get("acceptance", {}),
    )
    accepted = bool(acceptance.get("accepted", False))
    selected_cfg = best_cfg if accepted else baseline_cfg
    selected_final_eval = best_final_eval if accepted else baseline_final_eval
    selected_stage2_metrics = best_stage2_metrics if accepted else baseline_stage2_metrics
    selected_stage2_per_class = best_stage2_per_class if accepted else baseline_stage2_per_class
    selected_components = best_components if accepted else baseline_components
    selected_reasons = best_reasons if accepted else _build_stage2_fallback_reason_array(len(y_true_valid), reason="argmax")
    return {
        "status": "applied" if accepted else "rejected",
        "reason": "stage2_decision_full_validation_search_completed" if accepted else str(acceptance.get("reason", "rejected_fallback_to_baseline")),
        "enabled": True,
        "strategy": "enrolled_guarded_threshold",
        "threshold_tuning_requested": bool(resolved_cfg.get("tune_on_validation", True)),
        "threshold_tuning_applied": bool(resolved_cfg.get("tune_on_validation", True)),
        "threshold_tuning_supported": True,
        "selection_split": "validation",
        "selected_config": selected_cfg,
        "baseline_validation_metrics": baseline_stage2_metrics,
        "baseline_validation_per_class": baseline_stage2_per_class,
        "tuned_validation_metrics": selected_stage2_metrics,
        "tuned_validation_per_class": selected_stage2_per_class,
        "validation_final_metrics_baseline": dict(baseline_final_eval.get("metrics", {})),
        "validation_final_metrics_selected": dict(selected_final_eval.get("metrics", {})),
        "validation_objective_score_baseline": float(baseline_components.get("score_normalized", 0.0)),
        "validation_objective_score_selected": float(selected_components.get("score_normalized", 0.0)),
        "objective_components_baseline": baseline_components,
        "objective_components_selected": selected_components,
        "acceptance": acceptance,
        "search_results": candidate_rows,
        "search_evaluated_candidates": int(len(candidate_rows)),
        "selected_validation_predictions": np.asarray(selected_final_eval["y_pred"], dtype=int).tolist(),
        "selected_validation_decision_reasons": list(selected_reasons),
    }


def _was_stage2_decision_requested(stage2_decision_cfg: dict[str, Any] | None) -> bool:
    return bool(isinstance(stage2_decision_cfg, dict) and stage2_decision_cfg.get("enabled", False))


def _was_stage2_decision_executed(stage2_decision_result: dict[str, Any] | None) -> bool:
    if not isinstance(stage2_decision_result, dict):
        return False
    status = str(stage2_decision_result.get("status", "")).strip().lower()
    if status in {"applied", "rejected"}:
        return True
    return bool(
        stage2_decision_result.get("search_evaluated_candidates", 0)
        or stage2_decision_result.get("threshold_tuning_applied", False)
        or stage2_decision_result.get("threshold_tuning_requested", False)
    )


def _stage2_decision_rule_string(stage2_decision_cfg: dict[str, Any] | None) -> str:
    cfg = stage2_decision_cfg if isinstance(stage2_decision_cfg, dict) else {}
    strategy = str(cfg.get("strategy", "argmax")).strip().lower() or "argmax"
    if strategy != "enrolled_guarded_threshold" or not bool(cfg.get("enabled", False)):
        return "argmax"
    threshold = cfg.get("enrolled_probability_threshold")
    margin = cfg.get("graduate_margin_guard")
    enrolled_margin = cfg.get("enrolled_margin")
    dropout_guard = cfg.get("dropout_probability_guard")
    return (
        f"{strategy}"
        f"(threshold={threshold},margin={margin},enrolled_margin={enrolled_margin},dropout_guard={dropout_guard})"
    )


def _build_stage2_fallback_reason_array(length: int, reason: str = "argmax_fallback") -> list[str]:
    return np.full(shape=(int(length),), fill_value=str(reason), dtype=object).astype(str).tolist()


def _evaluate_two_stage_policy_on_split(
    *,
    y_true: pd.Series,
    fused_proba: np.ndarray,
    labels: list[int],
    decision_mode: str,
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
    dropout_threshold: float,
    low_threshold: float | None,
    high_threshold: float | None,
    class_thresholds: dict[int, float] | None,
    stage2_prob_enrolled: np.ndarray,
    stage2_prob_graduate: np.ndarray,
    stage2_decision_config: dict[str, Any] | None,
) -> dict[str, Any]:
    pred, decision_regions = _predict_two_stage_from_fused_probabilities(
        fused_proba=np.asarray(fused_proba, dtype=float),
        labels=labels,
        decision_mode=decision_mode,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        dropout_threshold=dropout_threshold,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        class_thresholds=class_thresholds,
        stage2_prob_enrolled=np.asarray(stage2_prob_enrolled, dtype=float),
        stage2_prob_graduate=np.asarray(stage2_prob_graduate, dtype=float),
        stage2_decision_config=stage2_decision_config,
    )
    return {
        "y_pred": np.asarray(pred, dtype=int),
        "decision_regions": np.asarray(decision_regions, dtype=str),
        "metrics": compute_metrics(y_true, np.asarray(pred, dtype=int)),
    }


def _tune_two_stage_stage2_decision_policy_with_inner_split(
    *,
    model_name: str,
    params_stage2: dict[str, Any],
    eval_cfg_stage2: dict[str, Any],
    X_train_stage2: pd.DataFrame,
    y_train_stage2_binary: pd.Series,
    y_train_stage2_original: pd.Series,
    enrolled_idx: int,
    graduate_idx: int,
    stage2_positive_target_label: int,
    stage2_decision_cfg: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], Stage2PositiveProbabilityCalibrator | None]:
    _assert_same_length_arrays(
        context=f"{model_name}:stage2_inner_split_input",
        X_train_stage2=X_train_stage2,
        y_train_stage2_binary=y_train_stage2_binary,
        y_train_stage2_original=y_train_stage2_original,
    )
    anti_overfit_cfg = stage2_decision_cfg.get("anti_overfit", {}) if isinstance(stage2_decision_cfg.get("anti_overfit", {}), dict) else {}
    strategy = str(anti_overfit_cfg.get("strategy", "stage2_train_inner_split")).strip().lower()
    if strategy != "stage2_train_inner_split":
        return {
            "status": "skipped",
            "reason": "unsupported_anti_overfit_strategy",
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "threshold_tuning_requested": True,
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": False,
            "selection_split": "stage2_train_inner_tune",
            "anti_overfit_strategy": strategy,
            "selected_config": {"enabled": False, "strategy": "argmax"},
            "search_results": [],
        }, None

    tuning_size = float(anti_overfit_cfg.get("tuning_size", 0.25))
    tuning_size = min(max(tuning_size, 0.10), 0.50)
    min_tuning_samples = int(anti_overfit_cfg.get("min_tuning_samples", 24))
    random_state = int(anti_overfit_cfg.get("random_state", seed))
    if len(X_train_stage2) < max(min_tuning_samples, 8) or int(pd.Series(y_train_stage2_binary).nunique()) < 2:
        return {
            "status": "skipped",
            "reason": "insufficient_stage2_train_samples_for_inner_tuning",
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "threshold_tuning_requested": True,
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": False,
            "selection_split": "stage2_train_inner_tune",
            "anti_overfit_strategy": strategy,
            "selected_config": {"enabled": False, "strategy": "argmax"},
            "search_results": [],
        }, None

    (
        X_inner_train,
        X_inner_tune,
        y_inner_train_binary,
        y_inner_tune_binary,
        y_inner_train_original,
        y_inner_tune_original,
    ) = train_test_split(
        X_train_stage2,
        y_train_stage2_binary,
        y_train_stage2_original,
        test_size=tuning_size,
        random_state=random_state,
        stratify=y_train_stage2_binary,
    )
    X_inner_train = X_inner_train.reset_index(drop=True)
    X_inner_tune = X_inner_tune.reset_index(drop=True)
    y_inner_train_binary = pd.Series(y_inner_train_binary).reset_index(drop=True)
    y_inner_tune_binary = pd.Series(y_inner_tune_binary).reset_index(drop=True)
    y_inner_tune_original = pd.Series(y_inner_tune_original).reset_index(drop=True)

    inner_result = train_and_evaluate(
        model_name=model_name,
        params=params_stage2,
        X_train=X_inner_train,
        y_train=y_inner_train_binary,
        X_valid=X_inner_tune,
        y_valid=y_inner_tune_binary,
        X_test=X_inner_tune,
        y_test=y_inner_tune_binary,
        eval_config=eval_cfg_stage2,
    )
    inner_model = inner_result.artifacts.get("model")
    if inner_model is None:
        return {
            "status": "skipped",
            "reason": "inner_stage2_model_training_failed",
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "threshold_tuning_requested": True,
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": False,
            "selection_split": "stage2_train_inner_tune",
            "anti_overfit_strategy": strategy,
            "selected_config": {"enabled": False, "strategy": "argmax"},
            "search_results": [],
        }, None

    inner_stage2_proba = np.asarray(inner_model.predict_proba(X_inner_tune), dtype=float)
    inner_stage2_positive = TwoStageUct3ClassClassifier._resolve_probability_column(
        proba=inner_stage2_proba,
        classes=getattr(inner_model, "classes_", None),
        positive_label=1,
    )
    inner_stage2_positive = np.clip(np.asarray(inner_stage2_positive, dtype=float), 0.0, 1.0)
    raw_stage2_positive = inner_stage2_positive
    calibrator: Stage2PositiveProbabilityCalibrator | None = None
    calibration_meta = {
        "enabled": bool(stage2_decision_cfg.get("use_calibrated_proba", False)),
        "applied": False,
        "method": "none",
        "selection_split": "stage2_train_inner_tune",
    }
    if int(stage2_positive_target_label) == int(enrolled_idx):
        p_enrolled_tune = inner_stage2_positive
        p_graduate_tune = 1.0 - inner_stage2_positive
    else:
        p_graduate_tune = inner_stage2_positive
        p_enrolled_tune = 1.0 - inner_stage2_positive
    if bool(stage2_decision_cfg.get("use_calibrated_proba", False)):
        calibrator, calibration_meta = _fit_stage2_probability_calibrator(
            p_positive=raw_stage2_positive,
            y_true=y_inner_tune_binary,
            method=str(stage2_decision_cfg.get("calibration_method", "none")).strip().lower(),
        )
        calibration_meta["selection_split"] = "stage2_train_inner_tune"
        if calibrator is not None:
            calibrated_positive = _apply_stage2_positive_probability_calibrator(raw_stage2_positive, calibrator)
            if int(stage2_positive_target_label) == int(enrolled_idx):
                p_enrolled_tune = calibrated_positive
                p_graduate_tune = 1.0 - calibrated_positive
            else:
                p_graduate_tune = calibrated_positive
                p_enrolled_tune = 1.0 - calibrated_positive

    tuning_result = _tune_two_stage_stage2_decision_thresholds(
        y_true_valid_stage2=y_inner_tune_original,
        p_enrolled_given_non_dropout=p_enrolled_tune,
        p_graduate_given_non_dropout=p_graduate_tune,
        p_dropout=None,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        stage2_decision_cfg=stage2_decision_cfg,
    )
    tuning_result["selection_split"] = "stage2_train_inner_tune"
    tuning_result["anti_overfit_strategy"] = strategy
    tuning_result["inner_split"] = {
        "train_rows": int(len(X_inner_train)),
        "tune_rows": int(len(X_inner_tune)),
        "tuning_size": float(tuning_size),
        "random_state": int(random_state),
    }
    tuning_result["calibration"] = calibration_meta
    _assert_same_length_arrays(
        context=f"{model_name}:stage2_inner_split_selected_validation",
        y_true_valid=y_inner_tune_original,
        selected_validation_predictions=tuning_result.get("selected_validation_predictions", []),
        selected_validation_decision_reasons=tuning_result.get("selected_validation_decision_reasons", []),
    )
    return tuning_result, calibrator


def _tune_two_stage_stage2_decision_thresholds(
    y_true_valid_stage2: pd.Series,
    *,
    p_enrolled_given_non_dropout: np.ndarray | None,
    p_graduate_given_non_dropout: np.ndarray | None,
    p_dropout: np.ndarray | None = None,
    enrolled_idx: int,
    graduate_idx: int,
    stage2_decision_cfg: dict[str, Any],
) -> dict[str, Any]:
    resolved_cfg = dict(stage2_decision_cfg) if isinstance(stage2_decision_cfg, dict) else {"enabled": False, "strategy": "argmax"}
    _assert_same_length_arrays(
        context="stage2_subset_valid:decision_threshold_tuning_inputs",
        y_true_valid_stage2=y_true_valid_stage2,
        p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
        p_graduate_given_non_dropout=p_graduate_given_non_dropout,
        p_dropout=p_dropout,
    )
    baseline_pred = np.where(
        np.asarray(p_enrolled_given_non_dropout if p_enrolled_given_non_dropout is not None else [], dtype=float)
        >= np.asarray(p_graduate_given_non_dropout if p_graduate_given_non_dropout is not None else [], dtype=float),
        int(enrolled_idx),
        int(graduate_idx),
    ).astype(int)
    y_valid_arr = np.asarray(y_true_valid_stage2, dtype=int)
    labels = [int(enrolled_idx), int(graduate_idx)]
    baseline_metrics = (
        compute_metrics(pd.Series(y_valid_arr), baseline_pred)
        if baseline_pred.shape[0] == y_valid_arr.shape[0] and y_valid_arr.size > 0
        else {}
    )
    baseline_per_class = (
        compute_per_class_metrics(pd.Series(y_valid_arr), baseline_pred, labels=labels)
        if baseline_metrics
        else {}
    )
    baseline_graduate_f1 = float(baseline_per_class.get(str(graduate_idx), {}).get("f1", 0.0))

    if not bool(resolved_cfg.get("enabled", False)):
        return {
            "status": "skipped",
            "reason": "stage2_decision_disabled",
            "enabled": False,
            "strategy": "argmax",
            "threshold_tuning_requested": False,
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": True,
            "selection_split": "validation",
            "selected_config": {"enabled": False, "strategy": "argmax"},
            "baseline_validation_metrics": baseline_metrics,
            "baseline_validation_per_class": baseline_per_class,
            "tuned_validation_metrics": baseline_metrics,
            "tuned_validation_per_class": baseline_per_class,
            "search_results": [],
        }

    if p_enrolled_given_non_dropout is None or p_graduate_given_non_dropout is None:
        return {
            "status": "skipped",
            "reason": "stage2_probabilities_unavailable_fallback_to_argmax",
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "threshold_tuning_requested": bool(resolved_cfg.get("tune_on_validation", True)),
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": False,
            "selection_split": "validation",
            "selected_config": {"enabled": False, "strategy": "argmax"},
            "baseline_validation_metrics": baseline_metrics,
            "baseline_validation_per_class": baseline_per_class,
            "tuned_validation_metrics": baseline_metrics,
            "tuned_validation_per_class": baseline_per_class,
            "search_results": [],
        }

    p_enrolled = np.asarray(p_enrolled_given_non_dropout, dtype=float)
    p_graduate = np.asarray(p_graduate_given_non_dropout, dtype=float)
    if p_enrolled.shape[0] != y_valid_arr.shape[0] or p_graduate.shape[0] != y_valid_arr.shape[0]:
        return {
            "status": "skipped",
            "reason": "stage2_probability_shape_mismatch_fallback_to_argmax",
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "threshold_tuning_requested": bool(resolved_cfg.get("tune_on_validation", True)),
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": False,
            "selection_split": "validation",
            "selected_config": {"enabled": False, "strategy": "argmax"},
            "baseline_validation_metrics": baseline_metrics,
            "baseline_validation_per_class": baseline_per_class,
            "tuned_validation_metrics": baseline_metrics,
            "tuned_validation_per_class": baseline_per_class,
            "search_results": [],
        }

    def _score_candidate(metrics: dict[str, Any], per_class: dict[str, Any]) -> tuple[float, dict[str, float]]:
        components = _compute_stage2_decision_objective_components(
            metrics=metrics,
            per_class=per_class,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            objective_cfg=resolved_cfg.get("objective", {}),
            baseline_graduate_f1=baseline_graduate_f1,
        )
        return float(components.get("score_normalized", 0.0)), components

    baseline_score, baseline_components = _score_candidate(baseline_metrics, baseline_per_class)
    default_selected_cfg = {
        "enabled": True,
        "strategy": "enrolled_guarded_threshold",
        "mode": str(resolved_cfg.get("mode", "legacy")).strip().lower(),
        "enrolled_probability_threshold": float(resolved_cfg.get("enrolled_probability_threshold", 0.42)),
        "graduate_margin_guard": float(resolved_cfg.get("graduate_margin_guard", 0.06)),
        "enrolled_margin": float(
            resolved_cfg.get("enrolled_margin", -float(resolved_cfg.get("graduate_margin_guard", 0.06)))
        ),
        "dropout_probability_guard": float(resolved_cfg.get("dropout_probability_guard", 1.0)),
        "use_calibrated_proba": bool(resolved_cfg.get("use_calibrated_proba", False)),
        "calibration_method": str(resolved_cfg.get("calibration_method", "none")).strip().lower(),
    }
    if not bool(resolved_cfg.get("tune_on_validation", True)):
        tuned_pred, tuned_reasons = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            baseline_pred,
            p_enrolled_given_non_dropout=p_enrolled,
            p_graduate_given_non_dropout=p_graduate,
            p_dropout=p_dropout,
            dropout_label=-1,
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=default_selected_cfg,
        )
        tuned_metrics = compute_metrics(pd.Series(y_valid_arr), tuned_pred)
        tuned_per_class = compute_per_class_metrics(pd.Series(y_valid_arr), tuned_pred, labels=labels)
        tuned_score, tuned_components = _score_candidate(tuned_metrics, tuned_per_class)
        acceptance = _stage2_decision_policy_acceptance(
            baseline_metrics=baseline_metrics,
            baseline_per_class=baseline_per_class,
            tuned_metrics=tuned_metrics,
            tuned_per_class=tuned_per_class,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            acceptance_cfg=resolved_cfg.get("acceptance", {}),
        )
        selected_cfg = default_selected_cfg if acceptance["accepted"] else {"enabled": False, "strategy": "argmax"}
        result = {
            "status": "applied" if acceptance["accepted"] else "rejected",
            "reason": "stage2_decision_used_fixed_config" if acceptance["accepted"] else acceptance["reason"],
            "enabled": True,
            "strategy": "enrolled_guarded_threshold",
            "threshold_tuning_requested": False,
            "threshold_tuning_applied": False,
            "threshold_tuning_supported": True,
            "selection_split": "validation",
            "anti_overfit_strategy": "legacy_validation",
            "selected_config": selected_cfg,
            "baseline_validation_metrics": baseline_metrics,
            "baseline_validation_per_class": baseline_per_class,
            "tuned_validation_metrics": tuned_metrics if acceptance["accepted"] else baseline_metrics,
            "tuned_validation_per_class": tuned_per_class if acceptance["accepted"] else baseline_per_class,
            "validation_objective_score_baseline": float(baseline_score),
            "validation_objective_score_selected": float(tuned_score if acceptance["accepted"] else baseline_score),
            "objective_components_baseline": baseline_components,
            "objective_components_selected": tuned_components if acceptance["accepted"] else baseline_components,
            "acceptance": acceptance,
            "search_results": [],
            "selected_validation_predictions": tuned_pred.tolist(),
            "selected_validation_decision_reasons": tuned_reasons.tolist(),
        }
        _assert_same_length_arrays(
            context="stage2_subset_valid:decision_threshold_tuning_selected_fixed",
            y_true_valid_stage2=y_true_valid_stage2,
            selected_validation_predictions=result.get("selected_validation_predictions", []),
            selected_validation_decision_reasons=result.get("selected_validation_decision_reasons", []),
        )
        return result

    candidate_rows: list[dict[str, Any]] = []
    best_cfg = dict(default_selected_cfg)
    best_pred = baseline_pred
    best_reasons = np.full(y_valid_arr.shape[0], "argmax", dtype=str)
    best_metrics = baseline_metrics
    best_per_class = baseline_per_class
    best_score = float(baseline_score)
    best_components = baseline_components
    best_rank = (
        best_score,
        float(best_metrics.get("macro_f1", 0.0)),
        float(best_per_class.get(str(enrolled_idx), {}).get("f1", 0.0)),
        float(best_per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
        -abs(float(best_cfg["enrolled_probability_threshold"]) - 0.5),
        -float(best_cfg["graduate_margin_guard"]),
    )
    for threshold in resolved_cfg.get("search", {}).get("enrolled_probability_threshold_grid", []):
        for margin_guard in resolved_cfg.get("search", {}).get("graduate_margin_guard_grid", []):
            enrolled_margin_grid = resolved_cfg.get("search", {}).get("enrolled_margin_grid", [])
            dropout_guard_grid = resolved_cfg.get("search", {}).get("dropout_probability_guard_grid", [])
            for enrolled_margin in enrolled_margin_grid:
                for dropout_probability_guard in dropout_guard_grid:
                    candidate_cfg = {
                        "enabled": True,
                        "strategy": "enrolled_guarded_threshold",
                        "mode": str(resolved_cfg.get("mode", "legacy")).strip().lower(),
                        "enrolled_probability_threshold": float(threshold),
                        "graduate_margin_guard": float(margin_guard),
                        "enrolled_margin": float(enrolled_margin),
                        "dropout_probability_guard": float(dropout_probability_guard),
                        "use_calibrated_proba": bool(resolved_cfg.get("use_calibrated_proba", False)),
                        "calibration_method": str(resolved_cfg.get("calibration_method", "none")).strip().lower(),
                    }
                    pred, reasons = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
                        baseline_pred,
                        p_enrolled_given_non_dropout=p_enrolled,
                        p_graduate_given_non_dropout=p_graduate,
                        p_dropout=p_dropout,
                        dropout_label=-1,
                        enrolled_label=int(enrolled_idx),
                        graduate_label=int(graduate_idx),
                        stage2_decision_config=candidate_cfg,
                    )
                    metrics = compute_metrics(pd.Series(y_valid_arr), pred)
                    per_class = compute_per_class_metrics(pd.Series(y_valid_arr), pred, labels=labels)
                    score, components = _score_candidate(metrics, per_class)
                    row = {
                        "enrolled_probability_threshold": float(threshold),
                        "graduate_margin_guard": float(margin_guard),
                        "enrolled_margin": float(enrolled_margin),
                        "dropout_probability_guard": float(dropout_probability_guard),
                        "macro_f1": float(metrics.get("macro_f1", 0.0)),
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                        "macro_precision": float(metrics.get("macro_precision", 0.0)),
                        "macro_recall": float(metrics.get("macro_recall", 0.0)),
                        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
                        "f1_enrolled": float(per_class.get(str(enrolled_idx), {}).get("f1", 0.0)),
                        "f1_graduate": float(per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
                        "objective_score": float(score),
                        "objective_components": components,
                        "enrolled_selected_count": int(np.sum(pred == int(enrolled_idx))),
                        "graduate_selected_count": int(np.sum(pred == int(graduate_idx))),
                    }
                    candidate_rows.append(row)
                    rank = (
                        float(score),
                        float(row["macro_f1"]),
                        float(row["f1_enrolled"]),
                        float(row["f1_graduate"]),
                        -abs(float(threshold) - 0.5),
                        -float(margin_guard),
                    )
                    if rank > best_rank:
                        best_cfg = candidate_cfg
                        best_pred = pred
                        best_reasons = reasons
                        best_metrics = metrics
                        best_per_class = per_class
                        best_score = float(score)
                        best_components = components
                        best_rank = rank

    acceptance = _stage2_decision_policy_acceptance(
        baseline_metrics=baseline_metrics,
        baseline_per_class=baseline_per_class,
        tuned_metrics=best_metrics,
        tuned_per_class=best_per_class,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        acceptance_cfg=resolved_cfg.get("acceptance", {}),
    )
    accepted = bool(acceptance.get("accepted", False))
    selected_cfg = best_cfg if accepted else {"enabled": False, "strategy": "argmax"}
    selected_metrics = best_metrics if accepted else baseline_metrics
    selected_per_class = best_per_class if accepted else baseline_per_class
    selected_score = best_score if accepted else baseline_score
    selected_components = best_components if accepted else baseline_components

    result = {
        "status": "applied" if accepted else "rejected",
        "reason": "stage2_decision_validation_grid_search_completed" if accepted else acceptance["reason"],
        "enabled": True,
        "strategy": "enrolled_guarded_threshold",
        "threshold_tuning_requested": True,
        "threshold_tuning_applied": True,
        "threshold_tuning_supported": True,
        "selection_split": "validation",
        "anti_overfit_strategy": "legacy_validation",
        "selected_config": selected_cfg,
        "baseline_validation_metrics": baseline_metrics,
        "baseline_validation_per_class": baseline_per_class,
        "tuned_validation_metrics": selected_metrics,
        "tuned_validation_per_class": selected_per_class,
        "validation_objective_score_baseline": float(baseline_score),
        "validation_objective_score_selected": float(selected_score),
        "objective_components_baseline": baseline_components,
        "objective_components_selected": selected_components,
        "acceptance": acceptance,
        "search_results": candidate_rows,
        "selected_validation_predictions": np.asarray(best_pred if accepted else baseline_pred, dtype=int).tolist(),
        "selected_validation_decision_reasons": (
            np.asarray(best_reasons, dtype=str).tolist()
            if accepted
            else _build_stage2_fallback_reason_array(y_valid_arr.shape[0])
        ),
        "search_evaluated_candidates": int(len(candidate_rows)),
    }
    _assert_same_length_arrays(
        context="stage2_subset_valid:decision_threshold_tuning_selected_grid",
        y_true_valid_stage2=y_true_valid_stage2,
        selected_validation_predictions=result.get("selected_validation_predictions", []),
        selected_validation_decision_reasons=result.get("selected_validation_decision_reasons", []),
    )
    return result


def _tune_two_stage_dropout_threshold(
    y_true_valid: pd.Series,
    y_proba_valid: np.ndarray,
    labels: list[int],
    *,
    dropout_idx: int,
    enrolled_idx: int,
    graduate_idx: int,
    threshold_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
) -> dict[str, Any]:
    objective = str(threshold_cfg.get("objective", threshold_cfg.get("metric", "macro_f1"))).strip().lower()
    search_mode = str(threshold_cfg.get("search_mode", "single")).strip().lower()
    label_order = [int(v) for v in labels]
    proba_valid = np.asarray(y_proba_valid, dtype=float)
    if proba_valid.ndim != 2 or proba_valid.shape[1] != len(label_order):
        raise ValueError("Invalid fused probability shape for two-stage dropout-threshold tuning.")
    y_valid_arr = np.asarray(y_true_valid, dtype=int)
    class_index_to_label = class_metadata.get("class_index_to_label", {})

    default_threshold = float(threshold_cfg.get("dropout_threshold", 0.5))
    default_low_threshold = float(threshold_cfg.get("stage1_dropout_threshold_low", default_threshold))
    default_high_threshold = float(threshold_cfg.get("stage1_dropout_threshold_high", default_threshold))
    baseline_mode = "soft_fusion_with_middle_band" if search_mode == "band" else "soft_fusion_with_dropout_threshold"
    baseline_pred, baseline_regions = _predict_two_stage_from_fused_probabilities(
        fused_proba=proba_valid,
        labels=label_order,
        decision_mode=baseline_mode,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        dropout_threshold=default_threshold,
        low_threshold=default_low_threshold,
        high_threshold=default_high_threshold,
    )
    baseline_metrics = compute_metrics(pd.Series(y_valid_arr), baseline_pred)
    baseline_per_class = compute_per_class_metrics(pd.Series(y_valid_arr), baseline_pred, labels=label_order)
    baseline_enrolled_f1 = float(baseline_per_class.get(str(enrolled_idx), {}).get("f1", 0.0))
    baseline_macro = float(baseline_metrics.get("macro_f1", 0.0))

    def _objective_score(metrics: dict[str, Any], per_class: dict[str, Any]) -> float:
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        enrolled_f1 = float(per_class.get(str(enrolled_idx), {}).get("f1", 0.0))
        if objective == "macro_f1_plus_enrolled_f1":
            alpha = float(threshold_cfg.get("enrolled_push_alpha", 0.35))
            return macro_f1 + (alpha * enrolled_f1)
        if objective == "constrained_enrolled_push":
            tolerance = float(threshold_cfg.get("macro_f1_tolerance", 0.005))
            if macro_f1 < baseline_macro - tolerance:
                return float("-inf")
            return enrolled_f1
        return macro_f1

    candidates: list[dict[str, float]] = []
    if not bool(threshold_cfg.get("enabled", False)):
        candidates = [
            {
                "dropout_threshold": float(default_threshold),
                "low_threshold": float(default_low_threshold),
                "high_threshold": float(default_high_threshold),
            }
        ]
    elif search_mode == "band":
        low_grid = [float(v) for v in threshold_cfg.get("threshold_grid_low", [])]
        high_grid = [float(v) for v in threshold_cfg.get("threshold_grid_high", [])]
        if not low_grid or not high_grid:
            raise ValueError("two_stage band threshold tuning requires non-empty threshold_grid_low and threshold_grid_high.")
        for low_threshold in low_grid:
            for high_threshold in high_grid:
                if float(low_threshold) >= float(high_threshold):
                    continue
                candidates.append(
                    {
                        "dropout_threshold": float(high_threshold),
                        "low_threshold": float(low_threshold),
                        "high_threshold": float(high_threshold),
                    }
                )
        if not candidates:
            raise ValueError("two_stage band threshold tuning produced no valid low/high threshold pairs.")
    else:
        single_grid = [float(v) for v in threshold_cfg.get("threshold_grid_single", threshold_cfg.get("threshold_grid", []))]
        if not single_grid:
            raise ValueError("two_stage single-threshold tuning requires a non-empty threshold grid.")
        candidates = [
            {
                "dropout_threshold": float(threshold),
                "low_threshold": float(threshold),
                "high_threshold": float(threshold),
            }
            for threshold in single_grid
        ]

    rows: list[dict[str, Any]] = []
    best_candidate = {
        "dropout_threshold": float(default_threshold),
        "low_threshold": float(default_low_threshold),
        "high_threshold": float(default_high_threshold),
    }
    best_pred = baseline_pred
    best_regions = baseline_regions
    best_metrics = baseline_metrics
    best_per_class = baseline_per_class
    best_score = _objective_score(baseline_metrics, baseline_per_class)
    best_macro = float(baseline_metrics.get("macro_f1", 0.0))
    best_enrolled_f1 = baseline_enrolled_f1
    best_balanced_accuracy = float(baseline_metrics.get("balanced_accuracy", 0.0))

    for candidate in candidates:
        candidate_mode = "soft_fusion_with_middle_band" if search_mode == "band" else "soft_fusion_with_dropout_threshold"
        pred, decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=proba_valid,
            labels=label_order,
            decision_mode=candidate_mode,
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(candidate["dropout_threshold"]),
            low_threshold=float(candidate["low_threshold"]),
            high_threshold=float(candidate["high_threshold"]),
        )
        metrics = compute_metrics(pd.Series(y_valid_arr), pred)
        per_class = compute_per_class_metrics(pd.Series(y_valid_arr), pred, labels=label_order)
        enrolled_f1 = float(per_class.get(str(enrolled_idx), {}).get("f1", 0.0))
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        balanced_accuracy = float(metrics.get("balanced_accuracy", 0.0))
        enrolled_absorbed = int(np.sum((y_valid_arr == int(enrolled_idx)) & (pred == int(dropout_idx))))
        graduate_absorbed = int(np.sum((y_valid_arr == int(graduate_idx)) & (pred == int(dropout_idx))))
        rows.append(
            {
                "dropout_threshold": float(candidate["dropout_threshold"]),
                "low_threshold": float(candidate["low_threshold"]),
                "high_threshold": float(candidate["high_threshold"]),
                "macro_f1": macro_f1,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "balanced_accuracy": balanced_accuracy,
                "macro_precision": float(metrics.get("macro_precision", 0.0)),
                "macro_recall": float(metrics.get("macro_recall", 0.0)),
                "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
                "objective_score": float(_objective_score(metrics, per_class)),
                "enrolled_absorbed_into_dropout": enrolled_absorbed,
                "graduate_absorbed_into_dropout": graduate_absorbed,
                "f1_dropout": float(per_class.get(str(dropout_idx), {}).get("f1", 0.0)),
                "f1_enrolled": enrolled_f1,
                "f1_graduate": float(per_class.get(str(graduate_idx), {}).get("f1", 0.0)),
                "middle_band_count": int(np.sum(np.asarray(decision_regions) == "middle_band")),
                "hard_dropout_count": int(np.sum(np.asarray(decision_regions) == "hard_dropout")),
                "safe_non_dropout_count": int(np.sum(np.asarray(decision_regions) == "safe_non_dropout")),
            }
        )
        score = _objective_score(metrics, per_class)
        candidate_tuple = (
            float(score),
            macro_f1,
            enrolled_f1,
            balanced_accuracy,
            -float(candidate["high_threshold"]),
            -float(candidate["low_threshold"]),
        )
        best_tuple = (
            float(best_score),
            best_macro,
            best_enrolled_f1,
            best_balanced_accuracy,
            -float(best_candidate["high_threshold"]),
            -float(best_candidate["low_threshold"]),
        )
        if candidate_tuple > best_tuple:
            best_candidate = candidate
            best_pred = pred
            best_regions = decision_regions
            best_metrics = metrics
            best_per_class = per_class
            best_score = score
            best_macro = macro_f1
            best_enrolled_f1 = enrolled_f1
            best_balanced_accuracy = balanced_accuracy

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("objective_score", float("-inf"))),
            float(row.get("macro_f1", float("-inf"))),
            float(row.get("f1_enrolled", float("-inf"))),
            float(row.get("balanced_accuracy", float("-inf"))),
            -float(row.get("high_threshold", 1.0)),
            -float(row.get("low_threshold", 1.0)),
        ),
        reverse=True,
    )

    return {
        "status": "applied" if bool(threshold_cfg.get("enabled", False)) else "skipped",
        "reason": "validation_dropout_threshold_grid_search_completed" if bool(threshold_cfg.get("enabled", False)) else "fixed_threshold",
        "metric": objective,
        "objective": objective,
        "threshold_tuning_requested": bool(threshold_cfg.get("enabled", False)),
        "threshold_tuning_supported": True,
        "threshold_tuning_applied": bool(threshold_cfg.get("enabled", False)),
        "threshold_selection_split": "validation",
        "threshold_applied_to": "test",
        "search_mode": search_mode,
        "selected_thresholds": (
            {
                "dropout": float(best_candidate["dropout_threshold"]),
                "stage1_low_threshold": float(best_candidate["low_threshold"]),
                "stage1_high_threshold": float(best_candidate["high_threshold"]),
            }
            if search_mode == "band"
            else {"dropout": float(best_candidate["dropout_threshold"])}
        ),
        "selected_thresholds_by_index": {str(dropout_idx): float(best_candidate["dropout_threshold"])},
        "selected_dropout_threshold": float(best_candidate["dropout_threshold"]),
        "selected_low_threshold": float(best_candidate["low_threshold"]),
        "selected_high_threshold": float(best_candidate["high_threshold"]),
        "default_dropout_threshold": float(default_threshold),
        "default_low_threshold": float(default_low_threshold),
        "default_high_threshold": float(default_high_threshold),
        "threshold_objective_score": float(best_score),
        "enrolled_push_alpha": float(threshold_cfg.get("enrolled_push_alpha", 0.35)),
        "macro_f1_tolerance": float(threshold_cfg.get("macro_f1_tolerance", 0.005)),
        "validation_baseline_metrics": baseline_metrics,
        "validation_baseline_per_class": baseline_per_class,
        "validation_tuned_metrics": best_metrics,
        "validation_tuned_per_class": best_per_class,
        "validation_objective_score_baseline": float(_objective_score(baseline_metrics, baseline_per_class)),
        "validation_objective_score_at_selected_threshold": float(best_score),
        "search_evaluated_candidates": int(len(rows) if bool(threshold_cfg.get("enabled", False)) else 0),
        "threshold_grid_results": rows,
        "threshold_search_ranking": ranked_rows,
        "class_names": {
            str(dropout_idx): str(class_index_to_label.get(str(dropout_idx), "Dropout")),
            str(enrolled_idx): str(class_index_to_label.get(str(enrolled_idx), "Enrolled")),
            str(graduate_idx): str(class_index_to_label.get(str(graduate_idx), "Graduate")),
        },
        "best_validation_predictions": np.asarray(best_pred, dtype=int).tolist(),
        "best_validation_decision_regions": np.asarray(best_regions, dtype=str).tolist(),
        "middle_band_enabled": bool(threshold_cfg.get("middle_band_enabled", False)),
        "middle_band_behavior": str(threshold_cfg.get("middle_band_behavior", "force_stage2_soft_fusion")),
    }


def _tune_two_stage_fused_thresholds(
    y_true_valid: pd.Series,
    y_proba_valid: np.ndarray,
    labels: list[int],
    threshold_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
) -> dict[str, Any]:
    label_order = [int(v) for v in labels]
    proba_valid = np.asarray(y_proba_valid, dtype=float)
    if proba_valid.ndim != 2 or proba_valid.shape[1] != len(label_order):
        raise ValueError("Invalid fused probability shape for threshold tuning.")
    y_valid_arr = np.asarray(y_true_valid, dtype=int)
    if y_valid_arr.shape[0] != proba_valid.shape[0]:
        raise ValueError("Validation targets and probabilities length mismatch.")

    default_thresholds_map = threshold_cfg.get("default_thresholds", {})
    default_thresholds_vec = _threshold_vector_from_map(label_order, default_thresholds_map)
    baseline_pred = TwoStageUct3ClassClassifier.predict_from_fused_probabilities(
        fused_proba=proba_valid,
        classes=np.asarray(label_order, dtype=int),
        thresholds=default_thresholds_vec,
    )
    baseline_metrics = compute_metrics(pd.Series(y_valid_arr), baseline_pred)
    baseline_per_class = compute_per_class_metrics(
        pd.Series(y_valid_arr),
        baseline_pred,
        labels=label_order,
    )

    if not bool(threshold_cfg.get("enabled", False)):
        return {
            "status": "skipped",
            "reason": "disabled",
            "metric": str(threshold_cfg.get("metric", "macro_f1")),
            "threshold_tuning_requested": False,
            "threshold_tuning_supported": True,
            "threshold_tuning_applied": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "none",
            "default_decision_rule": "argmax_fallback",
            "selected_thresholds": {str(v): float(default_thresholds_vec[i]) for i, v in enumerate(label_order)},
            "selected_thresholds_by_index": {str(v): float(default_thresholds_vec[i]) for i, v in enumerate(label_order)},
            "validation_baseline_metrics": baseline_metrics,
            "validation_baseline_per_class": baseline_per_class,
            "validation_tuned_metrics": baseline_metrics,
            "validation_tuned_per_class": baseline_per_class,
            "search_evaluated_candidates": 0,
        }

    grids_raw = threshold_cfg.get("class_grids", {})
    class_grids = [list(grids_raw.get(int(class_idx), [0.5])) for class_idx in label_order]
    for grid in class_grids:
        if not grid:
            raise ValueError("Threshold tuning grid for a class is empty.")

    best_score = float("-inf")
    best_macro = float("-inf")
    best_thresholds = default_thresholds_vec.copy()
    best_pred = baseline_pred
    evaluated = 0
    max_candidates = int(threshold_cfg.get("max_candidates", 1500))

    for combo in itertools.product(*class_grids):
        if evaluated >= max_candidates:
            break
        thresholds_vec = np.asarray(combo, dtype=float)
        pred = TwoStageUct3ClassClassifier.predict_from_fused_probabilities(
            fused_proba=proba_valid,
            classes=np.asarray(label_order, dtype=int),
            thresholds=thresholds_vec,
        )
        metrics = compute_metrics(pd.Series(y_valid_arr), pred)
        macro_score = float(metrics.get("macro_f1", 0.0))
        score = macro_score
        evaluated += 1

        if (
            score > best_score + 1e-12
            or (abs(score - best_score) <= 1e-12 and macro_score > best_macro + 1e-12)
            or (
                abs(score - best_score) <= 1e-12
                and abs(macro_score - best_macro) <= 1e-12
                and tuple(thresholds_vec.tolist()) < tuple(best_thresholds.tolist())
            )
        ):
            best_score = score
            best_macro = macro_score
            best_thresholds = thresholds_vec
            best_pred = pred

    tuned_metrics = compute_metrics(pd.Series(y_valid_arr), best_pred)
    tuned_per_class = compute_per_class_metrics(
        pd.Series(y_valid_arr),
        best_pred,
        labels=label_order,
    )
    class_index_to_label = class_metadata.get("class_index_to_label", {})
    selected_thresholds_named = {
        str(class_index_to_label.get(str(label_order[i]), label_order[i])): float(best_thresholds[i])
        for i in range(len(label_order))
    }
    return {
        "status": "applied",
        "reason": "validation_grid_search_completed",
        "metric": str(threshold_cfg.get("metric", "macro_f1")),
        "threshold_tuning_requested": True,
        "threshold_tuning_supported": True,
        "threshold_tuning_applied": True,
        "threshold_selection_split": "validation",
        "threshold_applied_to": "test",
        "default_decision_rule": "argmax_fallback",
        "selected_thresholds": selected_thresholds_named,
        "selected_thresholds_by_index": {str(label_order[i]): float(best_thresholds[i]) for i in range(len(label_order))},
        "validation_baseline_metrics": baseline_metrics,
        "validation_baseline_per_class": baseline_per_class,
        "validation_tuned_metrics": tuned_metrics,
        "validation_tuned_per_class": tuned_per_class,
        "search_evaluated_candidates": int(evaluated),
        "search_max_candidates": int(max_candidates),
    }


def _maybe_calibrate_binary_model_prefit(
    model: Any,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    calibration_cfg: dict[str, Any],
    retrained_on_full_train_split: bool,
) -> tuple[Any, dict[str, Any]]:
    stage_name = str(calibration_cfg.get("stage_name", "stage"))
    enabled = bool(calibration_cfg.get("enabled", False))
    if not enabled:
        return model, {"enabled": False, "applied": False, "reason": "disabled"}
    if X_calibration.empty or y_calibration.empty:
        return model, {"enabled": True, "applied": False, "reason": "empty_calibration_split"}
    if int(pd.Series(y_calibration).nunique()) < 2:
        return model, {"enabled": True, "applied": False, "reason": "calibration_requires_two_classes"}

    method = str(calibration_cfg.get("method", "sigmoid")).strip().lower()
    if method not in {"sigmoid", "isotonic"}:
        method = "sigmoid"

    try:
        calibrator = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
    try:
        calibrator.fit(X_calibration, y_calibration)
    except Exception as exc:
        return model, {
            "enabled": True,
            "applied": False,
            "method": method,
            "reason": f"calibration_failed:{type(exc).__name__}:{exc}",
        }

    note = (
        f"{stage_name} calibrated with validation split; model was retrained on train+validation before calibration."
        if retrained_on_full_train_split
        else f"{stage_name} calibrated with held-out validation split."
    )
    return calibrator, {
        "enabled": True,
        "applied": True,
        "method": method,
        "calibration_split": "validation",
        "sample_count": int(len(X_calibration)),
        "note": note,
    }


def _save_two_stage_optuna_artifacts(
    output_dir: Path,
    model_name: str,
    mode_name: str,
    stage1_details: dict[str, Any],
    stage2_details: dict[str, Any],
    stage1_params: dict[str, Any],
    stage2_params: dict[str, Any],
) -> dict[str, str]:
    model_token = _safe_filename_token(model_name)
    trials_path = output_dir / f"optuna_trials_{model_token}.csv"
    best_params_path = output_dir / f"optuna_best_params_{model_token}.json"

    stage1_trials = stage1_details.get("trials_dataframe")
    if not isinstance(stage1_trials, pd.DataFrame):
        stage1_trials = pd.DataFrame(stage1_details.get("trials", []))
    stage1_trials = stage1_trials.copy()
    stage1_trials["stage"] = "stage1_dropout_vs_non_dropout"

    stage2_trials = stage2_details.get("trials_dataframe")
    if not isinstance(stage2_trials, pd.DataFrame):
        stage2_trials = pd.DataFrame(stage2_details.get("trials", []))
    stage2_trials = stage2_trials.copy()
    stage2_trials["stage"] = "stage2_enrolled_vs_graduate"

    merged_trials = pd.concat([stage1_trials, stage2_trials], ignore_index=True, sort=False)
    merged_trials.to_csv(trials_path, index=False)

    payload = {
        "model": model_name,
        "mode": mode_name,
        "stage1": {
            "task": "dropout_vs_non_dropout",
            "best_params": stage1_params,
            "best_value": stage1_details.get("best_value"),
            "objective_source": stage1_details.get("objective_source"),
            "best_validation_metrics": stage1_details.get("best_validation_metrics", {}),
        },
        "stage2": {
            "task": "enrolled_vs_graduate",
            "best_params": stage2_params,
            "best_value": stage2_details.get("best_value"),
            "objective_source": stage2_details.get("objective_source"),
            "best_validation_metrics": stage2_details.get("best_validation_metrics", {}),
        },
    }
    best_params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "trials_csv": str(trials_path),
        "best_params_json": str(best_params_path),
    }


def _run_two_stage_uct_model_auto_balance(
    model_name: str,
    mode_name: str,
    decision_mode: str,
    params_overrides: dict[str, Any],
    seed: int,
    tuning_cfg: dict[str, Any],
    tuning_enabled: bool,
    retrain_on_full_train_split: bool,
    two_stage_cfg: dict[str, Any],
    class_weight_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_stage1: float,
    class_thresholds: dict[int, float],
    threshold_tuning_cfg: dict[str, Any],
    calibration_cfg: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], Any, float | None, dict[str, Any], dict[str, str] | None]:
    dropout_idx, enrolled_idx, graduate_idx = _resolve_uct_three_class_indices(class_metadata)
    auto_balance_cfg = _resolve_two_stage_auto_balance_search_config(two_stage_cfg)
    selection_cfg = _resolve_two_stage_selection_config(two_stage_cfg)
    joint_candidates, sampling_meta = _build_two_stage_auto_balance_candidates(auto_balance_cfg=auto_balance_cfg, seed=seed)
    if not joint_candidates:
        raise ValueError("Auto-balance search is enabled but produced no valid candidate combinations.")

    label_order = [int(v) for v in class_metadata.get("class_indices", [dropout_idx, enrolled_idx, graduate_idx])]
    disabled_threshold_tuning_cfg = dict(threshold_tuning_cfg)
    disabled_threshold_tuning_cfg["enabled"] = False

    grouped_candidates: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for candidate in joint_candidates:
        key = (
            float(candidate["stage1_non_dropout_weight"]),
            float(candidate["stage2_enrolled_weight"]),
        )
        grouped_candidates.setdefault(key, []).append(candidate)

    best_result: dict[str, Any] | None = None
    search_rows: list[dict[str, Any]] = []
    no_improvement_counter = 0
    early_stop_after = auto_balance_cfg.get("early_stop_if_no_improvement_after")

    for (stage1_non_dropout_weight, stage2_enrolled_weight), weight_candidates in grouped_candidates.items():
        candidate_two_stage_cfg = copy.deepcopy(two_stage_cfg)
        candidate_two_stage_cfg.setdefault("auto_balance_search", {})
        candidate_two_stage_cfg["auto_balance_search"]["enabled"] = False
        candidate_two_stage_cfg.setdefault("stage1", {})
        candidate_two_stage_cfg["stage1"]["class_weight_mode"] = "custom"
        candidate_two_stage_cfg["stage1"]["class_weight_positive"] = 1.0
        candidate_two_stage_cfg["stage1"]["class_weight_negative"] = float(stage1_non_dropout_weight)
        candidate_two_stage_cfg.setdefault("stage2", {})
        candidate_two_stage_cfg["stage2"]["class_weight_mode"] = "custom"
        candidate_two_stage_cfg["stage2"]["class_weight_map"] = {
            "enrolled": float(stage2_enrolled_weight),
            "graduate": 1.0,
        }

        try:
            # Reuse the standard two-stage runner per weight setting, then do cheap threshold/objective search from cached probabilities.
            candidate_payload, candidate_model, candidate_tuning_score, candidate_tuning_meta, candidate_tuning_artifacts = _run_two_stage_uct_model(
                model_name=model_name,
                mode_name=mode_name,
                decision_mode=decision_mode,
                params_overrides=params_overrides,
                seed=seed,
                tuning_cfg=tuning_cfg,
                tuning_enabled=tuning_enabled,
                retrain_on_full_train_split=retrain_on_full_train_split,
                two_stage_cfg=candidate_two_stage_cfg,
                class_weight_cfg=class_weight_cfg,
                class_metadata=class_metadata,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                threshold_stage1=threshold_stage1,
                class_thresholds=class_thresholds,
                threshold_tuning_cfg=disabled_threshold_tuning_cfg,
                calibration_cfg=calibration_cfg,
                output_dir=output_dir,
            )
        except Exception as exc:
            for candidate in weight_candidates:
                search_rows.append(
                    {
                        "status": "training_failed",
                        "model": model_name,
                        "stage1_non_dropout_weight": float(candidate["stage1_non_dropout_weight"]),
                        "stage2_enrolled_weight": float(candidate["stage2_enrolled_weight"]),
                        "enrolled_push_alpha": float(candidate["enrolled_push_alpha"]),
                        "low_threshold": float(candidate["low_threshold"]),
                        "high_threshold": float(candidate["high_threshold"]),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
            continue

        fused_valid = np.asarray(candidate_payload.get("artifacts", {}).get("y_proba_valid", []), dtype=float)
        if fused_valid.ndim != 2 or fused_valid.shape[1] != len(label_order):
            raise ValueError("Auto-balance search requires candidate validation fused probabilities.")
        candidate_valid_export = candidate_payload.get("artifacts", {}).get("prediction_export_valid")
        candidate_stage2_decision_cfg = (
            candidate_payload.get("artifacts", {}).get("two_stage", {}).get("stage2_decision", {}).get("selected_config", {})
            if isinstance(candidate_payload.get("artifacts", {}).get("two_stage", {}), dict)
            else {}
        )
        candidate_stage2_prob_enrolled = None
        candidate_stage2_prob_graduate = None
        if isinstance(candidate_valid_export, pd.DataFrame):
            if "stage2_prob_enrolled" in candidate_valid_export.columns:
                candidate_stage2_prob_enrolled = candidate_valid_export["stage2_prob_enrolled"].to_numpy(dtype=float)
            if "stage2_prob_graduate" in candidate_valid_export.columns:
                candidate_stage2_prob_graduate = candidate_valid_export["stage2_prob_graduate"].to_numpy(dtype=float)

        for candidate in weight_candidates:
            y_pred_valid_candidate, valid_regions_candidate = _predict_two_stage_from_fused_probabilities(
                fused_proba=fused_valid,
                labels=label_order,
                decision_mode=decision_mode,
                dropout_idx=dropout_idx,
                enrolled_idx=enrolled_idx,
                graduate_idx=graduate_idx,
                dropout_threshold=float(candidate["high_threshold"]),
                low_threshold=float(candidate["low_threshold"]),
                high_threshold=float(candidate["high_threshold"]),
                stage2_prob_enrolled=candidate_stage2_prob_enrolled,
                stage2_prob_graduate=candidate_stage2_prob_graduate,
                stage2_decision_config=candidate_stage2_decision_cfg,
            )
            valid_metrics_candidate = compute_metrics(y_valid, y_pred_valid_candidate)
            valid_per_class_candidate = compute_per_class_metrics(y_valid, y_pred_valid_candidate, labels=label_order)
            score_info = _score_two_stage_auto_balance_candidate(
                metrics=valid_metrics_candidate,
                per_class=valid_per_class_candidate,
                selection_cfg=selection_cfg,
                dropout_idx=dropout_idx,
                enrolled_idx=enrolled_idx,
                graduate_idx=graduate_idx,
                stage2_enrolled_weight=float(candidate["stage2_enrolled_weight"]),
                enrolled_push_alpha=float(candidate["enrolled_push_alpha"]),
            )
            row = {
                "status": "evaluated",
                "model": model_name,
                "selection_objective": str(selection_cfg.get("objective", "macro_f1_only")),
                "stage1_non_dropout_weight": float(candidate["stage1_non_dropout_weight"]),
                "stage2_enrolled_weight": float(candidate["stage2_enrolled_weight"]),
                "enrolled_push_alpha": float(candidate["enrolled_push_alpha"]),
                "low_threshold": float(candidate["low_threshold"]),
                "high_threshold": float(candidate["high_threshold"]),
                "macro_f1": float(valid_metrics_candidate.get("macro_f1", 0.0)),
                "accuracy": float(valid_metrics_candidate.get("accuracy", 0.0)),
                "balanced_accuracy": float(valid_metrics_candidate.get("balanced_accuracy", 0.0)),
                "macro_precision": float(valid_metrics_candidate.get("macro_precision", 0.0)),
                "macro_recall": float(valid_metrics_candidate.get("macro_recall", 0.0)),
                "weighted_f1": float(valid_metrics_candidate.get("weighted_f1", 0.0)),
                "f1_dropout": float(score_info["dropout_f1"]),
                "f1_enrolled": float(score_info["enrolled_f1"]),
                "f1_graduate": float(score_info["graduate_f1"]),
                "objective_score": float(score_info["objective_score"]),
                "dropout_floor_met": bool(score_info["dropout_floor_met"]),
                "graduate_floor_met": bool(score_info["graduate_floor_met"]),
                "hard_floors_met": bool(score_info["hard_floors_met"]),
                "enrolled_soft_target_met": bool(score_info["enrolled_soft_target_met"]),
                "middle_band_count": int(np.sum(np.asarray(valid_regions_candidate) == "middle_band")),
                "hard_dropout_count": int(np.sum(np.asarray(valid_regions_candidate) == "hard_dropout")),
                "safe_non_dropout_count": int(np.sum(np.asarray(valid_regions_candidate) == "safe_non_dropout")),
            }
            search_rows.append(row)
            rank_tuple = (
                float(score_info["objective_score"]),
                *tuple(float(v) for v in score_info["rank_tuple"]),
                -abs(float(candidate["stage1_non_dropout_weight"]) - 1.0),
                -float(candidate["high_threshold"]),
                -float(candidate["low_threshold"]),
            )
            best_rank_tuple = (
                float(best_result["score_info"]["objective_score"]),
                *tuple(float(v) for v in best_result["score_info"]["rank_tuple"]),
                -abs(float(best_result["candidate"]["stage1_non_dropout_weight"]) - 1.0),
                -float(best_result["candidate"]["high_threshold"]),
                -float(best_result["candidate"]["low_threshold"]),
            ) if best_result is not None else None
            if best_result is None or rank_tuple > best_rank_tuple:
                best_result = {
                    "candidate": dict(candidate),
                    "payload": candidate_payload,
                    "model": candidate_model,
                    "tuning_score": candidate_tuning_score,
                    "tuning_meta": candidate_tuning_meta,
                    "tuning_artifacts": candidate_tuning_artifacts,
                    "score_info": score_info,
                    "y_pred_valid": np.asarray(y_pred_valid_candidate, dtype=int),
                    "valid_regions": np.asarray(valid_regions_candidate, dtype=str),
                }
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
            if early_stop_after is not None and no_improvement_counter >= int(early_stop_after):
                break
        if early_stop_after is not None and no_improvement_counter >= int(early_stop_after):
            break

    if best_result is None:
        raise RuntimeError("Auto-balance search did not produce a valid candidate for this model.")

    selected_candidate = dict(best_result["candidate"])
    selected_two_stage_cfg = copy.deepcopy(two_stage_cfg)
    selected_two_stage_cfg.setdefault("auto_balance_search", {})
    selected_two_stage_cfg["auto_balance_search"]["enabled"] = False
    selected_two_stage_cfg.setdefault("stage1", {})
    selected_two_stage_cfg["stage1"]["class_weight_mode"] = "custom"
    selected_two_stage_cfg["stage1"]["class_weight_positive"] = 1.0
    selected_two_stage_cfg["stage1"]["class_weight_negative"] = float(selected_candidate["stage1_non_dropout_weight"])
    selected_two_stage_cfg.setdefault("stage2", {})
    selected_two_stage_cfg["stage2"]["class_weight_mode"] = "custom"
    selected_two_stage_cfg["stage2"]["class_weight_map"] = {
        "enrolled": float(selected_candidate["stage2_enrolled_weight"]),
        "graduate": 1.0,
    }
    # Materialize the selected weight setting once more so on-disk stage-model artifacts match the chosen configuration.
    payload, combined_model, tuning_score, tuning_meta, tuning_artifacts = _run_two_stage_uct_model(
        model_name=model_name,
        mode_name=mode_name,
        decision_mode=decision_mode,
        params_overrides=params_overrides,
        seed=seed,
        tuning_cfg=tuning_cfg,
        tuning_enabled=tuning_enabled,
        retrain_on_full_train_split=retrain_on_full_train_split,
        two_stage_cfg=selected_two_stage_cfg,
        class_weight_cfg=class_weight_cfg,
        class_metadata=class_metadata,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        threshold_stage1=threshold_stage1,
        class_thresholds=class_thresholds,
        threshold_tuning_cfg=disabled_threshold_tuning_cfg,
        calibration_cfg=calibration_cfg,
        output_dir=output_dir,
    )

    fused_valid = np.asarray(payload.get("artifacts", {}).get("y_proba_valid", []), dtype=float)
    fused_test = np.asarray(payload.get("artifacts", {}).get("y_proba_test", []), dtype=float)
    prediction_export_valid = payload.get("artifacts", {}).get("prediction_export_valid")
    prediction_export_test = payload.get("artifacts", {}).get("prediction_export_test")
    selected_stage2_decision_cfg = (
        payload.get("artifacts", {}).get("two_stage", {}).get("stage2_decision", {}).get("selected_config", {})
        if isinstance(payload.get("artifacts", {}).get("two_stage", {}), dict)
        else {}
    )
    y_pred_valid_final = np.asarray(best_result["y_pred_valid"], dtype=int)
    valid_decision_regions = np.asarray(best_result["valid_regions"], dtype=str)
    y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
        fused_proba=fused_test,
        labels=label_order,
        decision_mode=decision_mode,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        dropout_threshold=float(selected_candidate["high_threshold"]),
        low_threshold=float(selected_candidate["low_threshold"]),
        high_threshold=float(selected_candidate["high_threshold"]),
        stage2_prob_enrolled=(
            prediction_export_test["stage2_prob_enrolled"].to_numpy(dtype=float)
            if isinstance(prediction_export_test, pd.DataFrame) and "stage2_prob_enrolled" in prediction_export_test.columns
            else None
        ),
        stage2_prob_graduate=(
            prediction_export_test["stage2_prob_graduate"].to_numpy(dtype=float)
            if isinstance(prediction_export_test, pd.DataFrame) and "stage2_prob_graduate" in prediction_export_test.columns
            else None
        ),
        stage2_decision_config=selected_stage2_decision_cfg,
    )

    stage_prob_valid = combined_model.predict_stage_probabilities(X_valid)
    stage_prob_test = combined_model.predict_stage_probabilities(X_test)
    combined_model.threshold_stage1 = float(selected_candidate["high_threshold"])
    combined_model.threshold_stage1_low = float(selected_candidate["low_threshold"])
    combined_model.threshold_stage1_high = float(selected_candidate["high_threshold"])
    _, valid_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
        np.asarray(y_pred_valid_final, dtype=int),
        p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
        p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
        dropout_label=int(dropout_idx),
        enrolled_label=int(enrolled_idx),
        graduate_label=int(graduate_idx),
        stage2_decision_config=selected_stage2_decision_cfg,
    )
    _, test_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
        np.asarray(y_pred_test_final, dtype=int),
        p_enrolled_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
        p_graduate_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
        dropout_label=int(dropout_idx),
        enrolled_label=int(enrolled_idx),
        graduate_label=int(graduate_idx),
        stage2_decision_config=selected_stage2_decision_cfg,
    )

    valid_metrics = compute_metrics(y_valid, y_pred_valid_final)
    test_metrics = compute_metrics(y_test, y_pred_test_final)
    per_class_metrics_valid = compute_per_class_metrics(y_valid, y_pred_valid_final, labels=label_order)
    per_class_metrics_test = compute_per_class_metrics(y_test, y_pred_test_final, labels=label_order)
    classification_report_valid = classification_report(
        y_valid,
        y_pred_valid_final,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    classification_report_test = classification_report(
        y_test,
        y_pred_test_final,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, y_pred_test_final, labels=label_order).tolist()

    threshold_tuning_result = {
        "status": "applied",
        "reason": "auto_balance_search_selected_validation_candidate",
        "metric": str(selection_cfg.get("objective", "constrained_macro_with_class_floors")),
        "objective": str(selection_cfg.get("objective", "constrained_macro_with_class_floors")),
        "threshold_tuning_requested": True,
        "threshold_tuning_supported": True,
        "threshold_tuning_applied": True,
        "threshold_selection_split": "validation",
        "threshold_applied_to": "test",
        "search_mode": "band",
        "selected_thresholds": {
            "dropout": float(selected_candidate["high_threshold"]),
            "stage1_low_threshold": float(selected_candidate["low_threshold"]),
            "stage1_high_threshold": float(selected_candidate["high_threshold"]),
        },
        "selected_thresholds_by_index": {str(dropout_idx): float(selected_candidate["high_threshold"])},
        "selected_dropout_threshold": float(selected_candidate["high_threshold"]),
        "selected_low_threshold": float(selected_candidate["low_threshold"]),
        "selected_high_threshold": float(selected_candidate["high_threshold"]),
        "threshold_objective_score": float(best_result["score_info"]["objective_score"]),
        "enrolled_push_alpha": float(selected_candidate["enrolled_push_alpha"]),
        "validation_objective_score_at_selected_threshold": float(best_result["score_info"]["objective_score"]),
        "search_evaluated_candidates": int(len([row for row in search_rows if row.get("status") == "evaluated"])),
        "threshold_grid_results": search_rows,
        "middle_band_enabled": bool(threshold_tuning_cfg.get("middle_band_enabled", False)),
        "middle_band_behavior": str(threshold_tuning_cfg.get("middle_band_behavior", "force_stage2_soft_fusion")),
    }

    selected_balance_config = {
        "stage1_non_dropout_weight": float(selected_candidate["stage1_non_dropout_weight"]),
        "stage2_enrolled_weight": float(selected_candidate["stage2_enrolled_weight"]),
        "enrolled_push_alpha": float(selected_candidate["enrolled_push_alpha"]),
        "selected_low_threshold": float(selected_candidate["low_threshold"]),
        "selected_high_threshold": float(selected_candidate["high_threshold"]),
        "selection_objective": str(selection_cfg.get("objective", "constrained_macro_with_class_floors")),
        "dropout_f1_floor": float(selection_cfg.get("dropout_f1_floor", 0.0)),
        "graduate_f1_floor": float(selection_cfg.get("graduate_f1_floor", 0.0)),
        "enrolled_f1_soft_target": float(selection_cfg.get("enrolled_f1_soft_target", 0.0)),
    }

    payload["metrics"].update({f"valid_{k}": float(v) for k, v in valid_metrics.items()})
    payload["metrics"].update({f"test_{k}": float(v) for k, v in test_metrics.items()})
    payload["metrics"]["selected_dropout_threshold"] = float(selected_candidate["high_threshold"])
    payload["metrics"]["stage1_low_threshold"] = float(selected_candidate["low_threshold"])
    payload["metrics"]["stage1_high_threshold"] = float(selected_candidate["high_threshold"])
    payload["metrics"]["threshold_objective_score"] = float(best_result["score_info"]["objective_score"])
    payload["metrics"]["enrolled_push_alpha"] = float(selected_candidate["enrolled_push_alpha"])
    payload["metrics"]["selected_stage1_non_dropout_weight"] = float(selected_candidate["stage1_non_dropout_weight"])
    payload["metrics"]["selected_stage2_enrolled_weight"] = float(selected_candidate["stage2_enrolled_weight"])
    payload["metrics"]["selected_enrolled_push_alpha"] = float(selected_candidate["enrolled_push_alpha"])
    payload["metrics"]["selected_low_threshold"] = float(selected_candidate["low_threshold"])
    payload["metrics"]["selected_high_threshold"] = float(selected_candidate["high_threshold"])
    payload["metrics"]["selection_objective"] = str(selection_cfg.get("objective", "constrained_macro_with_class_floors"))
    payload["metrics"]["dropout_f1_floor"] = float(selection_cfg.get("dropout_f1_floor", 0.0))
    payload["metrics"]["graduate_f1_floor"] = float(selection_cfg.get("graduate_f1_floor", 0.0))
    payload["metrics"]["enrolled_f1_soft_target"] = float(selection_cfg.get("enrolled_f1_soft_target", 0.0))
    payload["metrics"]["auto_balance_search_enabled"] = 1.0

    payload["class_weight"]["class_weight_requested"] = True
    payload["class_weight"]["class_weight_applied"] = True
    payload["class_weight"]["effective_mechanism"] = "two_stage_auto_balance_search"
    payload["class_weight"]["mode"] = "auto_search"
    payload["class_weight"]["strategy"] = "auto_search"
    payload["class_weight"]["selected_stage1_non_dropout_weight"] = float(selected_candidate["stage1_non_dropout_weight"])
    payload["class_weight"]["selected_stage2_enrolled_weight"] = float(selected_candidate["stage2_enrolled_weight"])

    payload["params"]["threshold_stage1"] = float(selected_candidate["high_threshold"])
    payload["params"]["threshold_stage1_low"] = float(selected_candidate["low_threshold"])
    payload["params"]["threshold_stage1_high"] = float(selected_candidate["high_threshold"])

    payload["artifacts"]["y_pred_valid"] = y_pred_valid_final.tolist()
    payload["artifacts"]["y_pred_test"] = np.asarray(y_pred_test_final, dtype=int).tolist()
    payload["artifacts"]["per_class_metrics_valid"] = per_class_metrics_valid
    payload["artifacts"]["per_class_metrics_test"] = per_class_metrics_test
    payload["artifacts"]["classification_report_valid"] = classification_report_valid
    payload["artifacts"]["classification_report_test"] = classification_report_test
    payload["artifacts"]["confusion_matrix"] = confusion
    payload["artifacts"]["threshold_tuning_results"] = search_rows
    payload["artifacts"]["selected_threshold"] = {
        "dropout_threshold": float(selected_candidate["high_threshold"]),
        "low_threshold": float(selected_candidate["low_threshold"]),
        "high_threshold": float(selected_candidate["high_threshold"]),
        "selected_enrolled_push_alpha": float(selected_candidate["enrolled_push_alpha"]),
        "mode": "auto_search",
        "selection_split": "validation",
        "objective": str(selection_cfg.get("objective", "constrained_macro_with_class_floors")),
    }
    payload["artifacts"]["two_stage"]["threshold_tuning"] = threshold_tuning_result
    payload["artifacts"]["two_stage"]["decision_regions_valid"] = pd.Series(valid_decision_regions).value_counts().to_dict()
    payload["artifacts"]["two_stage"]["decision_regions_test"] = pd.Series(test_decision_regions).value_counts().to_dict()
    payload["artifacts"]["two_stage"]["stage2_decision_reason_counts_valid"] = (
        pd.Series(valid_stage2_decision_reason).value_counts().to_dict()
    )
    payload["artifacts"]["two_stage"]["stage2_decision_reason_counts_test"] = (
        pd.Series(test_stage2_decision_reason).value_counts().to_dict()
    )
    payload["artifacts"]["two_stage"]["threshold_stage1"] = float(selected_candidate["high_threshold"])
    payload["artifacts"]["two_stage"]["threshold_stage1_low"] = float(selected_candidate["low_threshold"])
    payload["artifacts"]["two_stage"]["threshold_stage1_high"] = float(selected_candidate["high_threshold"])
    payload["artifacts"]["two_stage"]["auto_balance_search"] = {
        "enabled": True,
        "search_config": auto_balance_cfg,
        "selection_config": selection_cfg,
        "sampling": sampling_meta,
        "selected_config": selected_balance_config,
    }

    prediction_export_test = pd.DataFrame(
        {
            "selected_threshold": float(selected_candidate["high_threshold"]),
            "selected_low_threshold": float(selected_candidate["low_threshold"]),
            "selected_high_threshold": float(selected_candidate["high_threshold"]),
            "selected_stage1_non_dropout_weight": float(selected_candidate["stage1_non_dropout_weight"]),
            "selected_stage2_enrolled_weight": float(selected_candidate["stage2_enrolled_weight"]),
            "selected_enrolled_push_alpha": float(selected_candidate["enrolled_push_alpha"]),
            "decision_region": np.asarray(test_decision_regions, dtype=str),
            "final_decision_mode": str(decision_mode),
            "stage1_prob_dropout": np.asarray(stage_prob_test.get("stage1_prob_dropout", []), dtype=float),
            "stage1_prob_non_dropout": np.asarray(stage_prob_test.get("stage1_prob_non_dropout", []), dtype=float),
            "stage2_prob_enrolled": np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            "stage2_prob_graduate": np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            "stage2_decision_reason": np.asarray(test_stage2_decision_reason, dtype=str),
        }
    )
    prediction_export_valid = pd.DataFrame(
        {
            "selected_threshold": float(selected_candidate["high_threshold"]),
            "selected_low_threshold": float(selected_candidate["low_threshold"]),
            "selected_high_threshold": float(selected_candidate["high_threshold"]),
            "selected_stage1_non_dropout_weight": float(selected_candidate["stage1_non_dropout_weight"]),
            "selected_stage2_enrolled_weight": float(selected_candidate["stage2_enrolled_weight"]),
            "selected_enrolled_push_alpha": float(selected_candidate["enrolled_push_alpha"]),
            "decision_region": np.asarray(valid_decision_regions, dtype=str),
            "final_decision_mode": str(decision_mode),
            "stage1_prob_dropout": np.asarray(stage_prob_valid.get("stage1_prob_dropout", []), dtype=float),
            "stage1_prob_non_dropout": np.asarray(stage_prob_valid.get("stage1_prob_non_dropout", []), dtype=float),
            "stage2_prob_enrolled": np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            "stage2_prob_graduate": np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            "stage2_decision_reason": np.asarray(valid_stage2_decision_reason, dtype=str),
        }
    )
    payload["artifacts"]["prediction_export_test"] = prediction_export_test
    payload["artifacts"]["prediction_export_valid"] = prediction_export_valid
    payload["artifacts"]["auto_balance_search_results"] = search_rows
    payload["artifacts"]["selected_balance_config"] = selected_balance_config
    payload["threshold_tuning"] = threshold_tuning_result

    auto_balance_results_path = output_dir / f"auto_balance_search_results_{_safe_filename_token(model_name)}.csv"
    selected_balance_config_path = output_dir / f"selected_balance_config_{_safe_filename_token(model_name)}.json"
    pd.DataFrame(search_rows).to_csv(auto_balance_results_path, index=False)
    selected_balance_config_path.write_text(json.dumps(selected_balance_config, indent=2), encoding="utf-8")
    if "artifact_paths" not in payload:
        payload["artifact_paths"] = {}
    payload["artifact_paths"]["auto_balance_search_results_csv"] = str(auto_balance_results_path)
    payload["artifact_paths"]["selected_balance_config_json"] = str(selected_balance_config_path)

    selected_threshold_path = payload["artifact_paths"].get("selected_threshold_json")
    if selected_threshold_path:
        Path(selected_threshold_path).write_text(json.dumps(payload["artifacts"]["selected_threshold"], indent=2), encoding="utf-8")
    threshold_results_path = payload["artifact_paths"].get("threshold_tuning_results_csv")
    if threshold_results_path:
        pd.DataFrame(search_rows).to_csv(Path(threshold_results_path), index=False)
    threshold_metadata_path = payload["artifact_paths"].get("threshold_tuning_metadata")
    if threshold_metadata_path:
        Path(threshold_metadata_path).write_text(
            json.dumps(
                {
                    "model": model_name,
                    "enabled": True,
                    "metric": str(selection_cfg.get("objective", "constrained_macro_with_class_floors")),
                    "search_mode": "band",
                    "selected_thresholds": threshold_tuning_result.get("selected_thresholds", {}),
                    "selected_dropout_threshold": threshold_tuning_result.get("selected_dropout_threshold"),
                    "selected_low_threshold": threshold_tuning_result.get("selected_low_threshold"),
                    "selected_high_threshold": threshold_tuning_result.get("selected_high_threshold"),
                    "validation_score_after": float(valid_metrics.get("macro_f1", 0.0)),
                    "validation_objective_score_after": float(best_result["score_info"]["objective_score"]),
                    "search_evaluated_candidates": int(threshold_tuning_result.get("search_evaluated_candidates", 0)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    two_stage_diagnostics_payload = {
        "model": model_name,
        "decision_mode": decision_mode,
        "selected_dropout_threshold": float(selected_candidate["high_threshold"]),
        "selected_low_threshold": float(selected_candidate["low_threshold"]),
        "selected_high_threshold": float(selected_candidate["high_threshold"]),
        "selected_stage1_non_dropout_weight": float(selected_candidate["stage1_non_dropout_weight"]),
        "selected_stage2_enrolled_weight": float(selected_candidate["stage2_enrolled_weight"]),
        "selected_enrolled_push_alpha": float(selected_candidate["enrolled_push_alpha"]),
        "validation_macro_f1_by_threshold": search_rows,
        "stage2_decision": payload["artifacts"]["two_stage"].get("stage2_decision", {}),
        "validation_decision_regions": pd.Series(valid_decision_regions).value_counts().to_dict(),
        "test_decision_regions": pd.Series(test_decision_regions).value_counts().to_dict(),
        "validation_stage2_decision_reason_counts": pd.Series(valid_stage2_decision_reason).value_counts().to_dict(),
        "test_stage2_decision_reason_counts": pd.Series(test_stage2_decision_reason).value_counts().to_dict(),
    }
    payload["artifacts"]["two_stage_diagnostics"] = two_stage_diagnostics_payload
    two_stage_diagnostics_path = payload["artifact_paths"].get("two_stage_diagnostics_json")
    if two_stage_diagnostics_path:
        Path(two_stage_diagnostics_path).write_text(json.dumps(two_stage_diagnostics_payload, indent=2), encoding="utf-8")

    middle_band_diagnostics_payload = {
        "model": model_name,
        "enabled": bool(threshold_tuning_cfg.get("middle_band_enabled", False)),
        "behavior": str(threshold_tuning_cfg.get("middle_band_behavior", "force_stage2_soft_fusion")),
        "selected_low_threshold": float(selected_candidate["low_threshold"]),
        "selected_high_threshold": float(selected_candidate["high_threshold"]),
        "validation_region_counts": pd.Series(valid_decision_regions).value_counts().to_dict(),
        "test_region_counts": pd.Series(test_decision_regions).value_counts().to_dict(),
    }
    payload["artifacts"]["middle_band_diagnostics"] = middle_band_diagnostics_payload
    middle_band_diagnostics_path = payload["artifact_paths"].get("middle_band_diagnostics_json")
    if middle_band_diagnostics_path:
        Path(middle_band_diagnostics_path).write_text(json.dumps(middle_band_diagnostics_payload, indent=2), encoding="utf-8")

    return payload, combined_model, tuning_score, tuning_meta, tuning_artifacts


def _run_two_stage_uct_model(
    model_name: str,
    mode_name: str,
    decision_mode: str,
    params_overrides: dict[str, Any],
    seed: int,
    tuning_cfg: dict[str, Any],
    tuning_enabled: bool,
    retrain_on_full_train_split: bool,
    two_stage_cfg: dict[str, Any],
    class_weight_cfg: dict[str, Any],
    class_metadata: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_stage1: float,
    class_thresholds: dict[int, float],
    threshold_tuning_cfg: dict[str, Any],
    calibration_cfg: dict[str, dict[str, Any]],
    output_dir: Path,
    X_train_stage2_base: pd.DataFrame | None = None,
    y_train_stage2_base: pd.Series | None = None,
    outlier_cfg: dict[str, Any] | None = None,
    balancing_cfg: dict[str, Any] | None = None,
    stage2_feature_bundle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any, float | None, dict[str, Any], dict[str, str] | None]:
    dropout_idx, enrolled_idx, graduate_idx = _resolve_uct_three_class_indices(class_metadata)
    stage2_positive_target_label = _resolve_two_stage_stage2_positive_target_label(
        two_stage_cfg=two_stage_cfg if isinstance(two_stage_cfg, dict) else {},
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
    )
    feature_bundle = stage2_feature_bundle if isinstance(stage2_feature_bundle, dict) else {}
    stage2_feature_engineering_enabled = bool(feature_bundle.get("enabled", False))
    stage2_requested_groups = list(feature_bundle.get("requested_groups", [])) if stage2_feature_engineering_enabled else []
    stage2_requested_interaction_groups = (
        list(feature_bundle.get("advanced_requested_groups", []))
        if stage2_feature_engineering_enabled
        else []
    )
    stage2_selective_feature_allowlist = (
        list(feature_bundle.get("selective_feature_allowlist", []))
        if stage2_feature_engineering_enabled
        else []
    )
    advanced_stage2_cfg = _resolve_two_stage_stage2_advanced_config(
        two_stage_cfg if isinstance(two_stage_cfg, dict) else {}
    )
    interaction_cfg = advanced_stage2_cfg.get("interaction_features", {})
    prototype_cfg = advanced_stage2_cfg.get("prototype_distance", {})
    finite_sanitation_cfg = _resolve_two_stage_stage2_finite_sanitation_config(
        two_stage_cfg if isinstance(two_stage_cfg, dict) else {}
    )

    X_train_stage2_source = X_train_stage2_base if isinstance(X_train_stage2_base, pd.DataFrame) else X_train
    y_train_stage2_source = y_train_stage2_base if isinstance(y_train_stage2_base, pd.Series) else y_train

    y_train_stage1 = (y_train == dropout_idx).astype(int)
    y_valid_stage1 = (y_valid == dropout_idx).astype(int)
    y_test_stage1 = (y_test == dropout_idx).astype(int)

    train_mask_stage2 = y_train_stage2_source != dropout_idx
    valid_mask_stage2 = y_valid != dropout_idx
    test_mask_stage2 = y_test != dropout_idx

    if int(train_mask_stage2.sum()) == 0:
        raise ValueError("No non-dropout samples available for stage2 training.")
    X_train_stage2_base_filtered = X_train_stage2_source.loc[train_mask_stage2].reset_index(drop=True)
    X_valid_stage2_base_filtered = X_valid.loc[valid_mask_stage2].reset_index(drop=True)
    X_test_stage2_base_filtered = X_test.loc[test_mask_stage2].reset_index(drop=True)
    if int(stage2_positive_target_label) == int(enrolled_idx):
        y_train_stage2 = (y_train_stage2_source.loc[train_mask_stage2] == enrolled_idx).astype(int).reset_index(drop=True)
        y_valid_stage2 = (y_valid.loc[valid_mask_stage2] == enrolled_idx).astype(int).reset_index(drop=True)
        y_test_stage2 = (y_test.loc[test_mask_stage2] == enrolled_idx).astype(int).reset_index(drop=True)
        stage2_class_label_to_index = {"Graduate": 0, "Enrolled": 1}
        stage2_positive_label_name = "Enrolled"
        stage2_negative_label_name = "Graduate"
    else:
        y_train_stage2 = (y_train_stage2_source.loc[train_mask_stage2] == graduate_idx).astype(int).reset_index(drop=True)
        y_valid_stage2 = (y_valid.loc[valid_mask_stage2] == graduate_idx).astype(int).reset_index(drop=True)
        y_test_stage2 = (y_test.loc[test_mask_stage2] == graduate_idx).astype(int).reset_index(drop=True)
        stage2_class_label_to_index = {"Enrolled": 0, "Graduate": 1}
        stage2_positive_label_name = "Graduate"
        stage2_negative_label_name = "Enrolled"

    if int(pd.Series(y_train_stage2).nunique()) < 2:
        raise ValueError("Stage2 training requires both Enrolled and Graduate classes in train split.")

    X_train_stage2_augmented_full = X_train_stage2_source.reset_index(drop=True)
    X_valid_stage2_augmented_full = X_valid.reset_index(drop=True)
    X_test_stage2_augmented_full = X_test.reset_index(drop=True)
    X_train_stage2_inference = X_train_stage2_base_filtered
    X_valid_stage2_inference = X_valid_stage2_base_filtered
    X_test_stage2_inference = X_test_stage2_base_filtered

    stage2_feature_report = (
        dict(feature_bundle.get("report", {}))
        if isinstance(feature_bundle.get("report", {}), dict)
        else {"enabled": False}
    )
    if stage2_feature_engineering_enabled:
        extra_train_full = feature_bundle.get("X_train")
        extra_valid_full = feature_bundle.get("X_valid")
        extra_test_full = feature_bundle.get("X_test")
        if not all(isinstance(item, pd.DataFrame) for item in (extra_train_full, extra_valid_full, extra_test_full)):
            raise ValueError("Stage2 feature sharpening bundle is enabled but missing preprocessed feature tables.")
        X_train_stage2_augmented_full = pd.concat(
            [X_train_stage2_source.reset_index(drop=True), extra_train_full.reset_index(drop=True)],
            axis=1,
        )
        X_valid_stage2_augmented_full = pd.concat(
            [X_valid.reset_index(drop=True), extra_valid_full.reset_index(drop=True)],
            axis=1,
        )
        X_test_stage2_augmented_full = pd.concat(
            [X_test.reset_index(drop=True), extra_test_full.reset_index(drop=True)],
            axis=1,
        )
        X_train_stage2_inference = X_train_stage2_augmented_full.loc[train_mask_stage2].reset_index(drop=True)
        X_valid_stage2_inference = X_valid_stage2_augmented_full.loc[valid_mask_stage2].reset_index(drop=True)
        X_test_stage2_inference = X_test_stage2_augmented_full.loc[test_mask_stage2].reset_index(drop=True)

    prototype_report = (
        dict(stage2_feature_report.get("prototype_distance", {}))
        if isinstance(stage2_feature_report.get("prototype_distance", {}), dict)
        else {"enabled": False}
    )
    if bool(prototype_cfg.get("enabled", False)):
        prototype_splits, prototype_report = build_stage2_prototype_distance_features(
            X_train=X_train_stage2_inference,
            y_train=y_train_stage2,
            X_valid=X_valid_stage2_inference,
            X_test=X_test_stage2_inference,
            feature_cfg=prototype_cfg,
            enrolled_positive_label=1 if int(stage2_positive_target_label) == int(enrolled_idx) else 0,
        )
        train_proto = prototype_splits.get("train", pd.DataFrame(index=X_train_stage2_inference.index))
        valid_proto = prototype_splits.get("valid", pd.DataFrame(index=X_valid_stage2_inference.index))
        test_proto = prototype_splits.get("test", pd.DataFrame(index=X_test_stage2_inference.index))
        if int(train_proto.shape[1]) > 0:
            X_train_stage2_inference = pd.concat([X_train_stage2_inference.reset_index(drop=True), train_proto.reset_index(drop=True)], axis=1)
            X_valid_stage2_inference = pd.concat([X_valid_stage2_inference.reset_index(drop=True), valid_proto.reset_index(drop=True)], axis=1)
            X_test_stage2_inference = pd.concat([X_test_stage2_inference.reset_index(drop=True), test_proto.reset_index(drop=True)], axis=1)

            train_proto_full = pd.DataFrame(np.nan, index=X_train_stage2_augmented_full.index, columns=train_proto.columns)
            valid_proto_full = pd.DataFrame(np.nan, index=X_valid_stage2_augmented_full.index, columns=valid_proto.columns)
            test_proto_full = pd.DataFrame(np.nan, index=X_test_stage2_augmented_full.index, columns=test_proto.columns)
            train_proto_full.loc[np.asarray(train_mask_stage2, dtype=bool), :] = train_proto.to_numpy()
            valid_proto_full.loc[np.asarray(valid_mask_stage2, dtype=bool), :] = valid_proto.to_numpy()
            test_proto_full.loc[np.asarray(test_mask_stage2, dtype=bool), :] = test_proto.to_numpy()
            X_train_stage2_augmented_full = pd.concat([X_train_stage2_augmented_full.reset_index(drop=True), train_proto_full.reset_index(drop=True)], axis=1)
            X_valid_stage2_augmented_full = pd.concat([X_valid_stage2_augmented_full.reset_index(drop=True), valid_proto_full.reset_index(drop=True)], axis=1)
            X_test_stage2_augmented_full = pd.concat([X_test_stage2_augmented_full.reset_index(drop=True), test_proto_full.reset_index(drop=True)], axis=1)

        print(
            "[two_stage][stage2][prototype_distance] "
            f"enabled={bool(prototype_report.get('enabled', False))} "
            f"created_feature_count={int(prototype_report.get('created_feature_count', 0))} "
            f"prototype_source_columns={prototype_report.get('prototype_source_columns', [])}"
        )
        if prototype_report.get("warning"):
            print(f"[v8] prototype augmentation enabled/disabled: warning={prototype_report.get('warning')}")
    stage2_feature_report["prototype_distance"] = prototype_report
    stage2_feature_report["prototype_distance_enabled"] = bool(prototype_report.get("enabled", False))
    stage2_feature_report["prototype_metric_set"] = list(prototype_report.get("metric_set", []))
    stage2_feature_report["selective_feature_allowlist"] = stage2_selective_feature_allowlist

    X_train_stage2 = X_train_stage2_inference
    X_valid_stage2 = X_valid_stage2_inference
    X_test_stage2 = X_test_stage2_inference
    lightgbm_feature_name_artifacts: dict[str, Any] = {
        "applied": False,
        "model": model_name,
        "stage1": {"applied": False, "mapping": {}},
        "stage2": {"applied": False, "mapping": {}},
    }
    stage2_sanitation_report = {"enabled": False}
    if bool(finite_sanitation_cfg.get("enabled", False)):
        sanitized_frames, stage2_sanitation_report = validate_and_sanitize_feature_matrix(
            X_train_stage2,
            X_valid_stage2,
            X_test_stage2,
            model_name=model_name,
            feature_stage="stage2_post_augmentation_pre_fit",
            sanitation_cfg=finite_sanitation_cfg,
            extra_frames={
                "train_full": X_train_stage2_augmented_full,
                "valid_full": X_valid_stage2_augmented_full,
                "test_full": X_test_stage2_augmented_full,
            },
        )
        X_train_stage2 = sanitized_frames["train"]
        X_valid_stage2 = sanitized_frames["valid"]
        X_test_stage2 = sanitized_frames["test"]
        X_train_stage2_augmented_full = sanitized_frames["train_full"]
        X_valid_stage2_augmented_full = sanitized_frames["valid_full"]
        X_test_stage2_augmented_full = sanitized_frames["test_full"]
        print(
            "[v8] prototype augmentation enabled/disabled "
            f"enabled={bool(prototype_report.get('enabled', False))} "
            f"disabled_due_to_failure={bool(prototype_report.get('disabled_due_to_failure', False))}"
        )

    stage1_frames, stage1_lightgbm_meta = _sanitize_lightgbm_feature_frames(
        frames={
            "train": X_train,
            "valid": X_valid,
            "test": X_test,
        },
        model_name=model_name,
        stage_name="stage1",
    )
    X_train = stage1_frames["train"]
    X_valid = stage1_frames["valid"]
    X_test = stage1_frames["test"]

    stage2_frames, stage2_lightgbm_meta = _sanitize_lightgbm_feature_frames(
        frames={
            "train": X_train_stage2,
            "valid": X_valid_stage2,
            "test": X_test_stage2,
            "train_full": X_train_stage2_augmented_full,
            "valid_full": X_valid_stage2_augmented_full,
            "test_full": X_test_stage2_augmented_full,
        },
        model_name=model_name,
        stage_name="stage2",
    )
    X_train_stage2 = stage2_frames["train"]
    X_valid_stage2 = stage2_frames["valid"]
    X_test_stage2 = stage2_frames["test"]
    X_train_stage2_augmented_full = stage2_frames["train_full"]
    X_valid_stage2_augmented_full = stage2_frames["valid_full"]
    X_test_stage2_augmented_full = stage2_frames["test_full"]
    lightgbm_feature_name_artifacts = {
        "applied": bool(stage1_lightgbm_meta.get("applied", False) or stage2_lightgbm_meta.get("applied", False)),
        "model": model_name,
        "stage1": stage1_lightgbm_meta,
        "stage2": stage2_lightgbm_meta,
    }
    print(
        "[two_stage][bundle_lengths] "
        f"model={model_name} "
        f"train={len(X_train_stage2)} valid={len(X_valid_stage2)} test={len(X_test_stage2)} "
        f"train_full={len(X_train_stage2_augmented_full)} valid_full={len(X_valid_stage2_augmented_full)} test_full={len(X_test_stage2_augmented_full)}"
    )

    y_train_stage2_train = y_train_stage2.copy()
    stage2_outlier_meta = {"enabled": False, "method": "disabled"}
    stage2_balancing_meta = {"enabled": False, "method": "disabled"}
    if stage2_feature_engineering_enabled:
        X_train_stage2, y_train_stage2_train, stage2_outlier_meta = apply_outlier_filter(
            X_train_stage2,
            y_train_stage2_train,
            outlier_cfg if isinstance(outlier_cfg, dict) else {"enabled": False},
        )
        X_train_stage2, y_train_stage2_train, stage2_balancing_meta = apply_balancing(
            X_train_stage2,
            y_train_stage2_train,
            balancing_cfg if isinstance(balancing_cfg, dict) else {"enabled": False},
        )
    if bool(finite_sanitation_cfg.get("enabled", False)):
        sanitized_frames_post_balance, stage2_sanitation_post_balance = validate_and_sanitize_feature_matrix(
            X_train_stage2,
            X_valid_stage2,
            X_test_stage2,
            model_name=model_name,
            feature_stage="stage2_post_balance_pre_model",
            sanitation_cfg=finite_sanitation_cfg,
            extra_frames={
                "train_full": X_train_stage2_augmented_full,
                "valid_full": X_valid_stage2_augmented_full,
                "test_full": X_test_stage2_augmented_full,
            },
        )
        X_train_stage2 = sanitized_frames_post_balance["train"]
        X_valid_stage2 = sanitized_frames_post_balance["valid"]
        X_test_stage2 = sanitized_frames_post_balance["test"]
        X_train_stage2_augmented_full = sanitized_frames_post_balance["train_full"]
        X_valid_stage2_augmented_full = sanitized_frames_post_balance["valid_full"]
        X_test_stage2_augmented_full = sanitized_frames_post_balance["test_full"]
        stage2_sanitation_report["post_balance"] = stage2_sanitation_post_balance

    _log_duplicate_feature_check(X_train_stage2, context=f"{model_name}:stage2_train")
    _log_duplicate_feature_check(X_valid_stage2, context=f"{model_name}:stage2_valid")
    _log_duplicate_feature_check(X_test_stage2, context=f"{model_name}:stage2_test")
    _log_duplicate_feature_check(X_train_stage2_augmented_full, context=f"{model_name}:stage2_train_full")
    _log_duplicate_feature_check(X_valid_stage2_augmented_full, context=f"{model_name}:stage2_valid_full")
    _log_duplicate_feature_check(X_test_stage2_augmented_full, context=f"{model_name}:stage2_test_full")

    canonical_stage2_reference = X_train_stage2.copy()
    X_valid_stage2 = align_feature_schema(canonical_stage2_reference, X_valid_stage2, fill_value=0.0)
    print("[v8] schema aligned for split=valid")
    X_test_stage2 = align_feature_schema(canonical_stage2_reference, X_test_stage2, fill_value=0.0)
    print("[v8] schema aligned for split=test")
    X_train_stage2_augmented_full = align_feature_schema(canonical_stage2_reference, X_train_stage2_augmented_full, fill_value=0.0)
    print("[v8] schema aligned for split=train_full")
    X_valid_stage2_augmented_full = align_feature_schema(canonical_stage2_reference, X_valid_stage2_augmented_full, fill_value=0.0)
    print("[v8] schema aligned for split=valid_full")
    X_test_stage2_augmented_full = align_feature_schema(canonical_stage2_reference, X_test_stage2_augmented_full, fill_value=0.0)
    print("[v8] schema aligned for split=test_full")
    print(f"[v8] canonical stage2 feature count: {int(canonical_stage2_reference.shape[1])}")

    validate_feature_schema(canonical_stage2_reference, X_valid_stage2, context="fit(valid)")
    validate_feature_schema(canonical_stage2_reference, X_test_stage2, context="fit(test)")
    validate_feature_schema(canonical_stage2_reference, X_train_stage2_augmented_full, context="predict(train_full)")
    validate_feature_schema(canonical_stage2_reference, X_valid_stage2_augmented_full, context="predict(valid_full)")
    validate_feature_schema(canonical_stage2_reference, X_test_stage2_augmented_full, context="predict(test_full)")
    print(
        "[two_stage][stage2][setup] "
        f"model={model_name} "
        f"feature_sharpening_enabled={bool(stage2_feature_report.get('feature_sharpening', {}).get('enabled', stage2_feature_engineering_enabled))} "
        f"advanced_enrolled_separation_enabled={bool(advanced_stage2_cfg.get('enabled', False))} "
        f"requested_groups={stage2_requested_groups} "
        f"interaction_groups={stage2_requested_interaction_groups} "
        f"selective_interactions={stage2_selective_feature_allowlist} "
        f"prototype_distance_enabled={bool(prototype_report.get('enabled', False))} "
        f"created_features={int(stage2_feature_report.get('created_feature_count', 0)) + int(prototype_report.get('created_feature_count', 0))} "
        f"train_features={int(X_train_stage2.shape[1])}"
    )

    stage2_decision_cfg = _resolve_two_stage_stage2_decision_config(
        two_stage_cfg if isinstance(two_stage_cfg, dict) else {}
    )
    stage2_optuna_cfg = _resolve_two_stage_stage2_optuna_tuning_config(
        two_stage_cfg if isinstance(two_stage_cfg, dict) else {}
    )

    if bool(_resolve_two_stage_auto_balance_search_config(two_stage_cfg if isinstance(two_stage_cfg, dict) else {}).get("enabled", False)):
        return _run_two_stage_uct_model_auto_balance(
            model_name=model_name,
            mode_name=mode_name,
            decision_mode=decision_mode,
            params_overrides=params_overrides,
            seed=seed,
            tuning_cfg=tuning_cfg,
            tuning_enabled=tuning_enabled,
            retrain_on_full_train_split=retrain_on_full_train_split,
            two_stage_cfg=two_stage_cfg,
            class_weight_cfg=class_weight_cfg,
            class_metadata=class_metadata,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            threshold_stage1=threshold_stage1,
            class_thresholds=class_thresholds,
            threshold_tuning_cfg=threshold_tuning_cfg,
            calibration_cfg=calibration_cfg,
            output_dir=output_dir,
        )

    stage1_class_weight_cfg, stage2_class_weight_cfg = _resolve_two_stage_stage_class_weights(
        two_stage_cfg=two_stage_cfg if isinstance(two_stage_cfg, dict) else {},
        class_weight_cfg=class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
    )
    stage2_class_weight_cfg["class_label_to_index"] = stage2_class_label_to_index

    params_stage1: dict[str, Any] = dict(params_overrides)
    params_stage2: dict[str, Any] = dict(params_overrides)
    tuning_score = None
    tuning_meta: dict[str, Any] = {}
    tuning_artifacts: dict[str, str] | None = None

    if tuning_enabled:
        params_stage1, score_stage1, details_stage1 = tune_model_with_optuna(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train_stage1,
            tuning_cfg={
                **tuning_cfg,
                "seed": seed,
                "use_class_weights": _class_weight_requested(class_weight_cfg),
                "class_weight": stage1_class_weight_cfg,
            },
            X_valid=X_valid,
            y_valid=y_valid_stage1,
            fixed_params=params_overrides,
        )
        params_stage2, score_stage2, details_stage2 = tune_model_with_optuna(
            model_name=model_name,
            X_train=X_train_stage2,
            y_train=y_train_stage2_train,
            tuning_cfg={
                **tuning_cfg,
                "seed": seed,
                "use_class_weights": _class_weight_requested(class_weight_cfg),
                "class_weight": stage2_class_weight_cfg,
            },
            X_valid=X_valid_stage2,
            y_valid=y_valid_stage2,
            fixed_params=params_overrides,
        )
        tuning_score = float((float(score_stage1) + float(score_stage2)) / 2.0)
        tuning_meta = {
            "mode": mode_name,
            "stage1": {
                "task": "dropout_vs_non_dropout",
                "score": float(score_stage1),
                "objective_source": details_stage1.get("objective_source"),
                "best_validation_metrics": details_stage1.get("best_validation_metrics", {}),
            },
            "stage2": {
                "task": "enrolled_vs_graduate",
                "score": float(score_stage2),
                "objective_source": details_stage2.get("objective_source"),
                "best_validation_metrics": details_stage2.get("best_validation_metrics", {}),
            },
            "score_aggregation": "mean_stage1_stage2_validation_score",
        }
        tuning_artifacts = _save_two_stage_optuna_artifacts(
            output_dir=output_dir,
            model_name=model_name,
            mode_name=mode_name,
            stage1_details=details_stage1,
            stage2_details=details_stage2,
            stage1_params=params_stage1,
            stage2_params=params_stage2,
        )

    stage2_optuna_tuning_result = _tune_two_stage_stage2_optuna(
        model_name=model_name,
        params_stage2=params_stage2,
        seed=seed,
        X_train_stage2=X_train_stage2,
        y_train_stage2=y_train_stage2_train,
        X_valid_stage2=X_valid_stage2,
        y_valid_stage2_binary=y_valid_stage2,
        y_valid_stage2_original=y_valid.loc[valid_mask_stage2].reset_index(drop=True),
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        stage2_positive_target_label=stage2_positive_target_label,
        stage2_class_weight_cfg=stage2_class_weight_cfg,
        stage2_decision_cfg=stage2_decision_cfg,
        stage2_optuna_cfg=stage2_optuna_cfg,
    )
    if isinstance(stage2_optuna_tuning_result.get("selected_stage2_class_weight_cfg"), dict):
        stage2_class_weight_cfg = dict(stage2_optuna_tuning_result["selected_stage2_class_weight_cfg"])
        stage2_class_weight_cfg["class_label_to_index"] = stage2_class_label_to_index

    eval_cfg_stage1 = {"seed": seed, "class_weight": stage1_class_weight_cfg, "label_order": [0, 1]}
    eval_cfg_stage2 = {"seed": seed, "class_weight": stage2_class_weight_cfg, "label_order": [0, 1]}
    print(
        "[two_stage][stage2][train_start] "
        f"model={model_name} "
        f"train_rows={int(len(X_train_stage2))} "
        f"valid_rows={int(len(X_valid_stage2))} "
        f"feature_count={int(X_train_stage2.shape[1])}"
    )
    validate_feature_schema(canonical_stage2_reference, X_train_stage2, context="fit(train)")

    if retrain_on_full_train_split:
        stage1_prefit = train_and_evaluate(
            model_name=model_name,
            params=params_stage1,
            X_train=X_train,
            y_train=y_train_stage1,
            X_valid=X_valid,
            y_valid=y_valid_stage1,
            X_test=X_test,
            y_test=y_test_stage1,
            eval_config=eval_cfg_stage1,
        )
        X_stage1_full = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
        y_stage1_full = pd.concat([y_train_stage1, y_valid_stage1], axis=0).reset_index(drop=True)
        stage1_result = retrain_on_full_train_and_evaluate_test(
            model_name=model_name,
            params=params_stage1,
            X_train_full=X_stage1_full,
            y_train_full=y_stage1_full,
            X_test=X_test,
            y_test=y_test_stage1,
            eval_config=eval_cfg_stage1,
        )
        for key in ("y_true_valid", "y_pred_valid", "y_proba_valid"):
            stage1_result.artifacts[key] = stage1_prefit.artifacts.get(key, [])

        stage2_prefit = train_and_evaluate(
            model_name=model_name,
            params=params_stage2,
            X_train=X_train_stage2,
            y_train=y_train_stage2_train,
            X_valid=X_valid_stage2,
            y_valid=y_valid_stage2,
            X_test=X_test_stage2,
            y_test=y_test_stage2,
            eval_config=eval_cfg_stage2,
        )
        X_stage2_full = pd.concat([X_train_stage2, X_valid_stage2], axis=0).reset_index(drop=True)
        validate_feature_schema(canonical_stage2_reference, X_stage2_full, context="fit(train_full)")
        y_stage2_full = pd.concat([y_train_stage2_train, y_valid_stage2], axis=0).reset_index(drop=True)
        stage2_result = retrain_on_full_train_and_evaluate_test(
            model_name=model_name,
            params=params_stage2,
            X_train_full=X_stage2_full,
            y_train_full=y_stage2_full,
            X_test=X_test_stage2,
            y_test=y_test_stage2,
            eval_config=eval_cfg_stage2,
        )
        for key in ("y_true_valid", "y_pred_valid", "y_proba_valid"):
            stage2_result.artifacts[key] = stage2_prefit.artifacts.get(key, [])
        _assert_same_length_arrays(
            context=f"{model_name}:stage2_prefit_artifact_merge:stage2_subset_valid",
            y_true_valid=stage2_result.artifacts.get("y_true_valid", []),
            y_pred_valid=stage2_result.artifacts.get("y_pred_valid", []),
            y_proba_valid=stage2_result.artifacts.get("y_proba_valid"),
        )
    else:
        stage1_result = train_and_evaluate(
            model_name=model_name,
            params=params_stage1,
            X_train=X_train,
            y_train=y_train_stage1,
            X_valid=X_valid,
            y_valid=y_valid_stage1,
            X_test=X_test,
            y_test=y_test_stage1,
            eval_config=eval_cfg_stage1,
        )
        stage2_result = train_and_evaluate(
            model_name=model_name,
            params=params_stage2,
            X_train=X_train_stage2,
            y_train=y_train_stage2_train,
            X_valid=X_valid_stage2,
            y_valid=y_valid_stage2,
            X_test=X_test_stage2,
            y_test=y_test_stage2,
            eval_config=eval_cfg_stage2,
        )
    if not X_valid_stage2.empty:
        stage2_valid_artifact_bundle = _validate_two_stage_eval_bundle(
            y_true=stage2_result.artifacts.get("y_true_valid", []),
            y_pred=stage2_result.artifacts.get("y_pred_valid", []),
            y_proba=stage2_result.artifacts.get("y_proba_valid"),
            split_name="valid",
            model_name=f"{model_name}:stage2_prefit",
        )
        if stage2_valid_artifact_bundle["lengths"]["y_true"] == 0:
            raise ValueError(f"[{model_name}] stage2 validation predictions are unavailable for non-empty validation split.")
    _validate_two_stage_eval_bundle(
        y_true=stage2_result.artifacts.get("y_true_test", []),
        y_pred=stage2_result.artifacts.get("y_pred_test", []),
        y_proba=stage2_result.artifacts.get("y_proba_test"),
        split_name="test",
        model_name=f"{model_name}:stage2_prefit",
    )
    print(
        "[two_stage][stage2][train_end] "
        f"model={model_name} "
        f"status=success "
        f"valid_macro_f1={float(stage2_result.metrics.get('valid_macro_f1', np.nan)):.4f} "
        f"test_macro_f1={float(stage2_result.metrics.get('test_macro_f1', np.nan)):.4f}"
    )

    stage1_model = stage1_result.artifacts.get("model")
    stage2_model = stage2_result.artifacts.get("model")
    if stage1_model is None or stage2_model is None:
        raise ValueError("Two-stage training failed to produce stage models.")

    stage1_model, stage1_calibration_meta = _maybe_calibrate_binary_model_prefit(
        model=stage1_model,
        X_calibration=X_valid,
        y_calibration=y_valid_stage1,
        calibration_cfg=calibration_cfg.get("stage1", {}),
        retrained_on_full_train_split=retrain_on_full_train_split,
    )
    stage2_model, stage2_calibration_meta = _maybe_calibrate_binary_model_prefit(
        model=stage2_model,
        X_calibration=X_valid_stage2,
        y_calibration=y_valid_stage2,
        calibration_cfg=calibration_cfg.get("stage2", {}),
        retrained_on_full_train_split=retrain_on_full_train_split,
    )
    stage2_policy_calibrator: Stage2PositiveProbabilityCalibrator | None = None
    if bool(stage2_decision_cfg.get("enabled", False)) and str(stage2_decision_cfg.get("config_schema", "")).strip().lower() == "decision_policy":
        stage2_decision_tuning_result, stage2_policy_calibrator = _tune_two_stage_stage2_decision_policy_with_inner_split(
            model_name=model_name,
            params_stage2=params_stage2,
            eval_cfg_stage2=eval_cfg_stage2,
            X_train_stage2=X_train_stage2,
            y_train_stage2_binary=y_train_stage2_train,
            y_train_stage2_original=pd.Series(
                np.where(
                    np.asarray(y_train_stage2_train, dtype=int) == 1,
                    int(stage2_positive_target_label),
                    int(graduate_idx if int(stage2_positive_target_label) == int(enrolled_idx) else enrolled_idx),
                ),
                name="stage2_original_target",
            ).reset_index(drop=True),
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            stage2_positive_target_label=stage2_positive_target_label,
            stage2_decision_cfg=stage2_decision_cfg,
            seed=seed,
        )
    else:
        stage2_decision_tuning_result = {}
    model_token = _safe_filename_token(model_name)
    stage1_model_path = output_dir / f"two_stage_stage1_model_{model_token}.joblib"
    stage2_model_path = output_dir / f"two_stage_stage2_model_{model_token}.joblib"
    joblib.dump(stage1_model, stage1_model_path)
    joblib.dump(stage2_model, stage2_model_path)

    combined_model = TwoStageUct3ClassClassifier(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        dropout_label=dropout_idx,
        enrolled_label=enrolled_idx,
        graduate_label=graduate_idx,
        decision_mode=decision_mode,
        threshold_stage1=threshold_stage1,
        threshold_stage1_low=float(threshold_tuning_cfg.get("stage1_dropout_threshold_low", threshold_stage1)),
        threshold_stage1_high=float(threshold_tuning_cfg.get("stage1_dropout_threshold_high", threshold_stage1)),
        middle_band_enabled=bool(threshold_tuning_cfg.get("middle_band_enabled", False)),
        middle_band_behavior=str(threshold_tuning_cfg.get("middle_band_behavior", "force_stage2_soft_fusion")),
        stage1_positive_label=1,
        stage2_positive_label=1,
        stage2_positive_target_label=stage2_positive_target_label,
        class_thresholds=class_thresholds,
        stage2_decision_config=stage2_decision_cfg,
        stage2_probability_calibrator=stage2_policy_calibrator,
        stage1_feature_columns=list(X_train.columns),
        stage2_feature_columns=list(X_train_stage2_augmented_full.columns),
    )

    validate_feature_schema(canonical_stage2_reference, X_valid_stage2_augmented_full, context="predict(valid)")
    validate_feature_schema(canonical_stage2_reference, X_test_stage2_augmented_full, context="predict(test)")
    stage_prob_valid = combined_model.predict_stage_probabilities(X_valid_stage2_augmented_full)
    stage_prob_test = combined_model.predict_stage_probabilities(X_test_stage2_augmented_full)
    _assert_same_length_arrays(
        context=f"{model_name}:full_valid_stage_probabilities",
        y_valid=y_valid,
        stage1_prob_dropout=stage_prob_valid.get("stage1_prob_dropout", []),
        stage2_prob_enrolled=stage_prob_valid.get("stage2_prob_enrolled", []),
        stage2_prob_graduate=stage_prob_valid.get("stage2_prob_graduate", []),
    )
    _assert_same_length_arrays(
        context=f"{model_name}:full_test_stage_probabilities",
        y_test=y_test,
        stage1_prob_dropout=stage_prob_test.get("stage1_prob_dropout", []),
        stage2_prob_enrolled=stage_prob_test.get("stage2_prob_enrolled", []),
        stage2_prob_graduate=stage_prob_test.get("stage2_prob_graduate", []),
    )
    stage2_valid_bundle = _validate_two_stage_eval_bundle(
        y_true=y_valid.loc[valid_mask_stage2].reset_index(drop=True),
        y_pred=np.where(
            np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float)[np.asarray(valid_mask_stage2, dtype=bool)]
            >= np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float)[np.asarray(valid_mask_stage2, dtype=bool)],
            int(enrolled_idx),
            int(graduate_idx),
        ),
        y_proba=np.column_stack(
            [
                np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float)[np.asarray(valid_mask_stage2, dtype=bool)],
                np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float)[np.asarray(valid_mask_stage2, dtype=bool)],
            ]
        ),
        split_name="stage2_subset_valid",
        model_name=model_name,
    )
    _validate_two_stage_eval_bundle(
        y_true=y_valid,
        y_pred=combined_model.predict(X_valid_stage2_augmented_full),
        split_name="valid_full",
        model_name=model_name,
    )
    _validate_two_stage_eval_bundle(
        y_true=y_test,
        y_pred=combined_model.predict(X_test_stage2_augmented_full),
        split_name="test_full",
        model_name=model_name,
    )
    if str(stage2_optuna_tuning_result.get("status", "")).strip().lower() == "applied":
        stage2_decision_tuning_result = dict(stage2_optuna_tuning_result)
    elif not stage2_decision_tuning_result:
        stage2_decision_tuning_result = _tune_two_stage_stage2_decision_thresholds(
            y_true_valid_stage2=stage2_valid_bundle["y_true"],
            p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float)[
                np.asarray(valid_mask_stage2, dtype=bool)
            ],
            p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float)[
                np.asarray(valid_mask_stage2, dtype=bool)
            ],
            p_dropout=np.asarray(stage_prob_valid.get("stage1_prob_dropout", []), dtype=float)[np.asarray(valid_mask_stage2, dtype=bool)],
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            stage2_decision_cfg=stage2_decision_cfg,
        )
    selected_stage2_decision_cfg = (
        stage2_decision_tuning_result.get("selected_config", {"enabled": False, "strategy": "argmax"})
        if isinstance(stage2_decision_tuning_result, dict)
        else {"enabled": False, "strategy": "argmax"}
    )
    combined_model.stage2_probability_calibrator = (
        stage2_policy_calibrator
        if stage2_policy_calibrator is not None and bool(selected_stage2_decision_cfg.get("use_calibrated_proba", False))
        else None
    )
    combined_model.stage2_decision_config = dict(selected_stage2_decision_cfg)
    stage_prob_valid = combined_model.predict_stage_probabilities(X_valid_stage2_augmented_full)
    stage_prob_test = combined_model.predict_stage_probabilities(X_test_stage2_augmented_full)
    y_proba_valid_final = combined_model.predict_proba(X_valid_stage2_augmented_full)
    y_proba_test_final = combined_model.predict_proba(X_test_stage2_augmented_full)
    label_order = [int(v) for v in class_metadata.get("class_indices", [dropout_idx, enrolled_idx, graduate_idx])]
    selected_dropout_threshold = float(threshold_stage1)
    selected_low_threshold = float(threshold_tuning_cfg.get("stage1_dropout_threshold_low", threshold_stage1))
    selected_high_threshold = float(threshold_tuning_cfg.get("stage1_dropout_threshold_high", threshold_stage1))
    tuned_thresholds_vec = _threshold_vector_from_map(label_order, class_thresholds)
    if decision_mode == "hard_routing":
        y_pred_valid_final = combined_model.predict(X_valid_stage2_augmented_full)
        y_pred_test_final = combined_model.predict(X_test_stage2_augmented_full)
        print(
            "[two_stage][final_pred_debug] "
            f"branch=hard_routing valid_dtype={np.asarray(y_pred_valid_final).dtype} "
            f"valid_ndim={np.asarray(y_pred_valid_final).ndim} valid_shape={np.asarray(y_pred_valid_final).shape} "
            f"test_dtype={np.asarray(y_pred_test_final).dtype} "
            f"test_ndim={np.asarray(y_pred_test_final).ndim} test_shape={np.asarray(y_pred_test_final).shape}"
        )
        _, valid_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_valid_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        _, test_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_test_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        valid_decision_regions = np.where(
            np.asarray(stage_prob_valid.get("stage1_prob_dropout", []), dtype=float) >= float(selected_dropout_threshold),
            "hard_dropout",
            "safe_non_dropout",
        )
        test_decision_regions = np.where(
            np.asarray(stage_prob_test.get("stage1_prob_dropout", []), dtype=float) >= float(selected_dropout_threshold),
            "hard_dropout",
            "safe_non_dropout",
        )
        threshold_tuning_result = {
            "status": "skipped",
            "reason": "hard_routing_mode_uses_fixed_stage1_threshold",
            "metric": "macro_f1",
            "threshold_tuning_requested": False,
            "threshold_tuning_supported": True,
            "threshold_tuning_applied": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
            "default_decision_rule": "hard_routing",
            "selected_thresholds": {"dropout": float(selected_dropout_threshold)},
            "selected_thresholds_by_index": {str(dropout_idx): float(selected_dropout_threshold)},
            "selected_dropout_threshold": float(selected_dropout_threshold),
            "selected_low_threshold": float(selected_low_threshold),
            "selected_high_threshold": float(selected_high_threshold),
        }
    elif decision_mode in {"soft_fusion_with_dropout_threshold", "soft_fusion_with_middle_band"}:
        # Tune the stage1 decision thresholds against end-to-end 3-class validation behavior.
        threshold_tuning_result = _tune_two_stage_dropout_threshold(
            y_true_valid=y_valid,
            y_proba_valid=np.asarray(y_proba_valid_final, dtype=float),
            labels=label_order,
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            threshold_cfg=threshold_tuning_cfg,
            class_metadata=class_metadata,
        )
        selected_dropout_threshold = float(threshold_tuning_result.get("selected_dropout_threshold", threshold_stage1))
        selected_low_threshold = float(threshold_tuning_result.get("selected_low_threshold", selected_low_threshold))
        selected_high_threshold = float(threshold_tuning_result.get("selected_high_threshold", selected_high_threshold))
        y_pred_valid_final, valid_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_valid_final, dtype=float),
            labels=label_order,
            decision_mode=decision_mode,
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=selected_dropout_threshold,
            low_threshold=selected_low_threshold,
            high_threshold=selected_high_threshold,
            stage2_prob_enrolled=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_test_final, dtype=float),
            labels=label_order,
            decision_mode=decision_mode,
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=selected_dropout_threshold,
            low_threshold=selected_low_threshold,
            high_threshold=selected_high_threshold,
            stage2_prob_enrolled=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        print(
            "[two_stage][final_pred_debug] "
            f"branch={decision_mode} valid_dtype={np.asarray(y_pred_valid_final).dtype} "
            f"valid_ndim={np.asarray(y_pred_valid_final).ndim} valid_shape={np.asarray(y_pred_valid_final).shape} "
            f"test_dtype={np.asarray(y_pred_test_final).dtype} "
            f"test_ndim={np.asarray(y_pred_test_final).ndim} test_shape={np.asarray(y_pred_test_final).shape}"
        )
        _, valid_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_valid_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        _, test_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_test_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
    elif decision_mode == "soft_fusion":
        threshold_tuning_result = {
            "status": "skipped",
            "reason": "soft_fusion_argmax_mode",
            "metric": "macro_f1",
            "threshold_tuning_requested": False,
            "threshold_tuning_supported": True,
            "threshold_tuning_applied": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
            "default_decision_rule": "soft_fusion",
            "selected_thresholds": {},
            "selected_thresholds_by_index": {},
            "selected_dropout_threshold": None,
            "selected_low_threshold": None,
            "selected_high_threshold": None,
        }
        y_pred_valid_final, valid_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_valid_final, dtype=float),
            labels=label_order,
            decision_mode="soft_fusion",
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(threshold_stage1),
            stage2_prob_enrolled=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_test_final, dtype=float),
            labels=label_order,
            decision_mode="soft_fusion",
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(threshold_stage1),
            stage2_prob_enrolled=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        print(
            "[two_stage][final_pred_debug] "
            f"branch=soft_fusion valid_dtype={np.asarray(y_pred_valid_final).dtype} "
            f"valid_ndim={np.asarray(y_pred_valid_final).ndim} valid_shape={np.asarray(y_pred_valid_final).shape} "
            f"test_dtype={np.asarray(y_pred_test_final).dtype} "
            f"test_ndim={np.asarray(y_pred_test_final).ndim} test_shape={np.asarray(y_pred_test_final).shape}"
        )
        _, valid_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_valid_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        _, test_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_test_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
    else:
        threshold_tuning_result = {
            "status": "skipped",
            "reason": "pure_soft_argmax_mode",
            "metric": "macro_f1",
            "threshold_tuning_requested": False,
            "threshold_tuning_supported": True,
            "threshold_tuning_applied": False,
            "threshold_selection_split": "validation",
            "threshold_applied_to": "test",
            "default_decision_rule": "pure_soft_argmax",
            "selected_thresholds": {},
            "selected_thresholds_by_index": {},
            "selected_dropout_threshold": None,
            "selected_low_threshold": None,
            "selected_high_threshold": None,
        }
        y_pred_valid_final, valid_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_valid_final, dtype=float),
            labels=label_order,
            decision_mode="pure_soft_argmax",
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(threshold_stage1),
            stage2_prob_enrolled=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_test_final, dtype=float),
            labels=label_order,
            decision_mode="pure_soft_argmax",
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(threshold_stage1),
            stage2_prob_enrolled=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        print(
            "[two_stage][final_pred_debug] "
            f"branch=pure_soft_argmax valid_dtype={np.asarray(y_pred_valid_final).dtype} "
            f"valid_ndim={np.asarray(y_pred_valid_final).ndim} valid_shape={np.asarray(y_pred_valid_final).shape} "
            f"test_dtype={np.asarray(y_pred_test_final).dtype} "
            f"test_ndim={np.asarray(y_pred_test_final).ndim} test_shape={np.asarray(y_pred_test_final).shape}"
        )
        _, valid_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_valid_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
        _, test_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            np.asarray(y_pred_test_final, dtype=int),
            p_enrolled_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            p_graduate_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            stage2_decision_config=selected_stage2_decision_cfg,
        )
    requested_stage2_decision = _was_stage2_decision_requested(stage2_decision_cfg)
    preserved_stage2_meta = {
        key: stage2_decision_tuning_result.get(key)
        for key in ("selection_split", "anti_overfit_strategy", "inner_split", "calibration", "optuna", "selected_stage2_weights")
        if isinstance(stage2_decision_tuning_result, dict) and key in stage2_decision_tuning_result
    }
    stage2_decision_tuning_result = {
        **_select_two_stage_stage2_decision_on_full_validation(
            decision_mode=decision_mode,
            y_true_valid=y_valid,
            fused_proba_valid=np.asarray(y_proba_valid_final, dtype=float),
            labels=label_order,
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(selected_dropout_threshold),
            low_threshold=float(selected_low_threshold),
            high_threshold=float(selected_high_threshold),
            class_thresholds=class_thresholds,
            stage2_prob_enrolled_valid=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            stage2_prob_graduate_valid=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            y_true_valid_stage2=y_valid.loc[valid_mask_stage2].reset_index(drop=True),
            stage2_prob_enrolled_valid_stage2=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float)[
                np.asarray(valid_mask_stage2, dtype=bool)
            ],
            stage2_prob_graduate_valid_stage2=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float)[
                np.asarray(valid_mask_stage2, dtype=bool)
            ],
            stage2_decision_cfg=stage2_decision_cfg,
        ),
        **preserved_stage2_meta,
    }
    executed_stage2_decision = _was_stage2_decision_executed(stage2_decision_tuning_result)
    stage2_decision_tuning_result["requested"] = bool(requested_stage2_decision)
    stage2_decision_tuning_result["executed"] = bool(executed_stage2_decision)
    selected_stage2_decision_cfg = (
        stage2_decision_tuning_result.get("selected_config", {"enabled": False, "strategy": "argmax"})
        if isinstance(stage2_decision_tuning_result, dict)
        else {"enabled": False, "strategy": "argmax"}
    )
    y_pred_valid_final, valid_decision_regions = _predict_two_stage_from_fused_probabilities(
        fused_proba=np.asarray(y_proba_valid_final, dtype=float),
        labels=label_order,
        decision_mode=decision_mode,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        dropout_threshold=float(selected_dropout_threshold),
        low_threshold=float(selected_low_threshold),
        high_threshold=float(selected_high_threshold),
        stage2_prob_enrolled=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
        stage2_prob_graduate=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
        stage2_decision_config=selected_stage2_decision_cfg,
    )
    y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
        fused_proba=np.asarray(y_proba_test_final, dtype=float),
        labels=label_order,
        decision_mode=decision_mode,
        dropout_idx=dropout_idx,
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
        dropout_threshold=float(selected_dropout_threshold),
        low_threshold=float(selected_low_threshold),
        high_threshold=float(selected_high_threshold),
        stage2_prob_enrolled=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
        stage2_prob_graduate=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
        stage2_decision_config=selected_stage2_decision_cfg,
    )
    _, valid_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
        np.asarray(y_pred_valid_final, dtype=int),
        p_enrolled_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
        p_graduate_given_non_dropout=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
        dropout_label=int(dropout_idx),
        enrolled_label=int(enrolled_idx),
        graduate_label=int(graduate_idx),
        stage2_decision_config=selected_stage2_decision_cfg,
    )
    _, test_stage2_decision_reason = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
        np.asarray(y_pred_test_final, dtype=int),
        p_enrolled_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
        p_graduate_given_non_dropout=np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
        dropout_label=int(dropout_idx),
        enrolled_label=int(enrolled_idx),
        graduate_label=int(graduate_idx),
        stage2_decision_config=selected_stage2_decision_cfg,
    )
    baseline_final_per_class = compute_per_class_metrics(
        y_valid,
        np.asarray(
            _evaluate_two_stage_policy_on_split(
                y_true=y_valid,
                fused_proba=np.asarray(y_proba_valid_final, dtype=float),
                labels=label_order,
                decision_mode=decision_mode,
                dropout_idx=dropout_idx,
                enrolled_idx=enrolled_idx,
                graduate_idx=graduate_idx,
                dropout_threshold=float(selected_dropout_threshold),
                low_threshold=float(selected_low_threshold),
                high_threshold=float(selected_high_threshold),
                class_thresholds=class_thresholds,
                stage2_prob_enrolled=np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
                stage2_prob_graduate=np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
                stage2_decision_config={"enabled": False, "strategy": "argmax"},
            )["y_pred"],
            dtype=int,
        ),
        labels=label_order,
    )
    selected_final_per_class = compute_per_class_metrics(y_valid, np.asarray(y_pred_valid_final, dtype=int), labels=label_order)
    stage2_decision_tuning_result["validation_final_macro_f1_baseline"] = float(
        stage2_decision_tuning_result.get("validation_final_metrics_baseline", {}).get("macro_f1", np.nan)
    )
    stage2_decision_tuning_result["validation_final_macro_f1_selected"] = float(
        stage2_decision_tuning_result.get("validation_final_metrics_selected", {}).get("macro_f1", np.nan)
    )
    stage2_decision_tuning_result["validation_final_macro_f1_delta"] = float(
        float(stage2_decision_tuning_result.get("validation_final_metrics_selected", {}).get("macro_f1", 0.0))
        - float(stage2_decision_tuning_result.get("validation_final_metrics_baseline", {}).get("macro_f1", 0.0))
    )
    stage2_decision_tuning_result["validation_final_enrolled_f1_baseline"] = float(
        baseline_final_per_class.get(str(enrolled_idx), {}).get("f1", np.nan)
    )
    stage2_decision_tuning_result["validation_final_enrolled_f1_selected"] = float(
        selected_final_per_class.get(str(enrolled_idx), {}).get("f1", np.nan)
    )
    stage2_decision_tuning_result["validation_final_graduate_f1_baseline"] = float(
        baseline_final_per_class.get(str(graduate_idx), {}).get("f1", np.nan)
    )
    stage2_decision_tuning_result["validation_final_graduate_f1_selected"] = float(
        selected_final_per_class.get(str(graduate_idx), {}).get("f1", np.nan)
    )
    stage2_acceptance_meta = (
        stage2_decision_tuning_result.get("acceptance", {})
        if isinstance(stage2_decision_tuning_result.get("acceptance", {}), dict)
        else {}
    )
    stage2_acceptance_meta["final_validation_enrolled_f1_before"] = float(
        stage2_decision_tuning_result.get("validation_final_enrolled_f1_baseline", np.nan)
    )
    stage2_acceptance_meta["final_validation_enrolled_f1_after"] = float(
        stage2_decision_tuning_result.get("validation_final_enrolled_f1_selected", np.nan)
    )
    stage2_acceptance_meta["final_validation_macro_f1_before"] = float(
        stage2_decision_tuning_result.get("validation_final_macro_f1_baseline", np.nan)
    )
    stage2_acceptance_meta["final_validation_macro_f1_after"] = float(
        stage2_decision_tuning_result.get("validation_final_macro_f1_selected", np.nan)
    )
    stage2_acceptance_meta["final_validation_graduate_f1_before"] = float(
        stage2_decision_tuning_result.get("validation_final_graduate_f1_baseline", np.nan)
    )
    stage2_acceptance_meta["final_validation_graduate_f1_after"] = float(
        stage2_decision_tuning_result.get("validation_final_graduate_f1_selected", np.nan)
    )
    stage2_decision_tuning_result["acceptance"] = stage2_acceptance_meta
    combined_model.threshold_stage1 = float(selected_dropout_threshold)
    combined_model.threshold_stage1_low = float(selected_low_threshold)
    combined_model.threshold_stage1_high = float(selected_high_threshold)
    combined_model.middle_band_enabled = bool(threshold_tuning_cfg.get("middle_band_enabled", False))
    combined_model.middle_band_behavior = str(threshold_tuning_cfg.get("middle_band_behavior", "force_stage2_soft_fusion"))
    combined_model.stage2_decision_config = dict(selected_stage2_decision_cfg)

    if bool(stage2_decision_cfg.get("enabled", False)) and bool(stage2_decision_cfg.get("log_selection", True)):
        baseline_stage2_metrics = stage2_decision_tuning_result.get("baseline_validation_metrics", {})
        tuned_stage2_metrics = stage2_decision_tuning_result.get("tuned_validation_metrics", {})
        selected_stage2_weights = (
            stage2_decision_tuning_result.get("selected_stage2_weights", {})
            if isinstance(stage2_decision_tuning_result.get("selected_stage2_weights", {}), dict)
            else {}
        )
        optuna_meta = (
            stage2_decision_tuning_result.get("optuna", {})
            if isinstance(stage2_decision_tuning_result.get("optuna", {}), dict)
            else {}
        )
        calibration_meta = (
            stage2_decision_tuning_result.get("calibration", {})
            if isinstance(stage2_decision_tuning_result.get("calibration", {}), dict)
            else {}
        )
        acceptance_meta = (
            stage2_decision_tuning_result.get("acceptance", {})
            if isinstance(stage2_decision_tuning_result.get("acceptance", {}), dict)
            else {}
        )
        objective_meta = (
            stage2_decision_tuning_result.get("objective_components_selected", {})
            if isinstance(stage2_decision_tuning_result.get("objective_components_selected", {}), dict)
            else {}
        )
        print(
            "[two_stage][stage2_decision] "
            f"objective_mode={objective_meta.get('objective_mode', stage2_decision_cfg.get('objective', {}).get('mode', 'legacy_macro_priority'))}"
        )
        print(
            f"[two_stage][{model_name}] stage2_decision="
            f"{stage2_decision_tuning_result.get('status', 'skipped')} "
            f"anti_overfit={stage2_decision_tuning_result.get('anti_overfit_strategy', 'validation')} "
            f"calibration={calibration_meta.get('method', 'none')} "
            f"calibration_applied={bool(calibration_meta.get('applied', False))} "
            f"stage2_optuna_trials={int(optuna_meta.get('n_trials_completed', 0))} "
            f"enrolled_scale={stage2_decision_tuning_result.get('selected_config', {}).get('enrolled_class_weight_scale', 'n/a')} "
            f"enrolled_weight={selected_stage2_weights.get('selected_enrolled_weight', 'n/a')} "
            f"threshold={selected_stage2_decision_cfg.get('enrolled_probability_threshold', 'argmax')} "
            f"margin={selected_stage2_decision_cfg.get('graduate_margin_guard', 'argmax')} "
            f"enrolled_margin={selected_stage2_decision_cfg.get('enrolled_margin', 'n/a')} "
            f"baseline_macro_f1={float(baseline_stage2_metrics.get('macro_f1', 0.0)):.4f} "
            f"tuned_macro_f1={float(tuned_stage2_metrics.get('macro_f1', 0.0)):.4f} "
            f"objective={float(objective_meta.get('score_normalized', np.nan)):.4f} "
            f"accepted={bool(acceptance_meta.get('accepted', False))}"
        )
        print(
            "[two_stage][stage2_decision] "
            f"model={model_name} "
            f"objective_mode={stage2_decision_tuning_result.get('objective_components_selected', {}).get('objective_mode', stage2_decision_cfg.get('objective', {}).get('mode', 'legacy_macro_priority'))} "
            f"requested={bool(stage2_decision_tuning_result.get('requested', requested_stage2_decision))} "
            f"search_started={bool(stage2_decision_tuning_result.get('threshold_tuning_requested', False))} "
            f"executed={bool(stage2_decision_tuning_result.get('executed', executed_stage2_decision))} "
            f"candidates_evaluated={int(stage2_decision_tuning_result.get('search_evaluated_candidates', 0))} "
            f"accepted={bool(acceptance_meta.get('accepted', False) and selected_stage2_decision_cfg.get('enabled', False))} "
            f"reject_reason={stage2_decision_tuning_result.get('reason', '')} "
            f"selected_rule={_stage2_decision_rule_string(selected_stage2_decision_cfg)}"
        )
        print(
            "[two_stage][stage2_decision] "
            f"model={model_name} "
            f"enrolled_f1_before={float(stage2_decision_tuning_result.get('validation_final_enrolled_f1_baseline', np.nan)):.6f} "
            f"enrolled_f1_after={float(stage2_decision_tuning_result.get('validation_final_enrolled_f1_selected', np.nan)):.6f} "
            f"validation_branch_macro_f1_before={float(stage2_decision_tuning_result.get('baseline_validation_metrics', {}).get('macro_f1', np.nan)):.6f} "
            f"validation_branch_macro_f1_after={float(stage2_decision_tuning_result.get('tuned_validation_metrics', {}).get('macro_f1', np.nan)):.6f} "
            f"final_macro_f1_before={float(stage2_decision_tuning_result.get('validation_final_macro_f1_baseline', np.nan)):.6f} "
            f"final_macro_f1_after={float(stage2_decision_tuning_result.get('validation_final_macro_f1_selected', np.nan)):.6f} "
            f"graduate_f1_before={float(stage2_decision_tuning_result.get('validation_final_graduate_f1_baseline', np.nan)):.6f} "
            f"graduate_f1_after={float(stage2_decision_tuning_result.get('validation_final_graduate_f1_selected', np.nan)):.6f} "
            f"trials={int(stage2_decision_tuning_result.get('search_evaluated_candidates', 0))}"
        )

    metrics: dict[str, float] = {}
    _assert_same_length_arrays(
        context=f"{model_name}:full_valid_final_predictions",
        y_valid=y_valid,
        y_pred_valid_final=y_pred_valid_final,
    )
    _assert_same_length_arrays(
        context=f"{model_name}:full_test_final_predictions",
        y_test=y_test,
        y_pred_test_final=y_pred_test_final,
    )
    y_valid_vector = _assert_1d_label_vector(y_valid, name="y_valid", context="two_stage_final_metrics_inputs")
    y_pred_valid_vector = _assert_1d_label_vector(
        y_pred_valid_final,
        name="full_valid prediction",
        context="two_stage_final_metrics_inputs",
    )
    y_test_vector = _assert_1d_label_vector(y_test, name="y_test", context="two_stage_final_metrics_inputs")
    y_pred_test_vector = _assert_1d_label_vector(
        y_pred_test_final,
        name="full_test prediction",
        context="two_stage_final_metrics_inputs",
    )
    try:
        y_pred_valid_final = np.asarray(y_pred_valid_vector).astype(int)
    except Exception as exc:
        raise ValueError(
            "two_stage_final_metrics_inputs: full_valid prediction could not be cast to int; "
            f"dtype={np.asarray(y_pred_valid_vector).dtype}, shape={np.asarray(y_pred_valid_vector).shape}, "
            f"error={type(exc).__name__}:{exc}"
        ) from exc
    try:
        y_pred_test_final = np.asarray(y_pred_test_vector).astype(int)
    except Exception as exc:
        raise ValueError(
            "two_stage_final_metrics_inputs: full_test prediction could not be cast to int; "
            f"dtype={np.asarray(y_pred_test_vector).dtype}, shape={np.asarray(y_pred_test_vector).shape}, "
            f"error={type(exc).__name__}:{exc}"
        ) from exc
    valid_bundle = _validate_two_stage_eval_bundle(
        y_true=y_valid_vector,
        y_pred=y_pred_valid_final,
        y_proba=y_proba_valid_final,
        split_name="full_valid",
        model_name=model_name,
    )
    if not X_valid.empty:
        valid_metrics = compute_metrics(valid_bundle["y_true"], valid_bundle["y_pred"])
        metrics.update({f"valid_{k}": float(v) for k, v in valid_metrics.items()})
    test_bundle = _validate_two_stage_eval_bundle(
        y_true=y_test_vector,
        y_pred=y_pred_test_final,
        y_proba=y_proba_test_final,
        split_name="full_test",
        model_name=model_name,
    )
    test_metrics = compute_metrics(test_bundle["y_true"], test_bundle["y_pred"])
    metrics.update({f"test_{k}": float(v) for k, v in test_metrics.items()})

    per_class_metrics_test = compute_per_class_metrics(test_bundle["y_true"], test_bundle["y_pred"], labels=label_order)
    per_class_metrics_valid = compute_per_class_metrics(valid_bundle["y_true"], valid_bundle["y_pred"], labels=label_order)
    cm = confusion_matrix(test_bundle["y_true"], test_bundle["y_pred"], labels=label_order).tolist()
    classification_report_valid = classification_report(
        valid_bundle["y_true"],
        valid_bundle["y_pred"],
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    classification_report_test = classification_report(
        test_bundle["y_true"],
        test_bundle["y_pred"],
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )

    class_weight_stage1 = dict(stage1_result.artifacts.get("class_weight_info", {}))
    class_weight_stage2 = dict(stage2_result.artifacts.get("class_weight_info", {}))
    class_weight_combined = {
        "class_weight_requested": _class_weight_requested(class_weight_cfg),
        "class_weight_supported": bool(
            class_weight_stage1.get("class_weight_supported", False)
            or class_weight_stage2.get("class_weight_supported", False)
        ),
        "class_weight_applied": bool(
            class_weight_stage1.get("class_weight_applied", False)
            or class_weight_stage2.get("class_weight_applied", False)
        ),
        "effective_mechanism": "two_stage",
        "stages": {
            "stage1_dropout_vs_non_dropout": class_weight_stage1,
            "stage2_enrolled_vs_graduate": class_weight_stage2,
        },
        "class_weight_backend_note": "Two-stage wrapper: stage-specific class-weight handling stored in stages.",
    }

    artifacts = {
        "model": combined_model,
        "params": {
            "mode": mode_name,
            "stage1": params_stage1,
            "stage2": params_stage2,
            "decision_mode": decision_mode,
            "threshold_stage1": float(selected_dropout_threshold),
            "threshold_stage1_low": float(selected_low_threshold) if threshold_tuning_result.get("selected_low_threshold") is not None else None,
            "threshold_stage1_high": float(selected_high_threshold) if threshold_tuning_result.get("selected_high_threshold") is not None else None,
            "final_class_thresholds": {
                str(label_order[i]): float(tuned_thresholds_vec[i]) for i in range(len(label_order))
            },
        },
        "labels": label_order,
        "per_class_metrics_valid": per_class_metrics_valid,
        "per_class_metrics_test": per_class_metrics_test,
        "classification_report_valid": classification_report_valid,
        "classification_report_test": classification_report_test,
        "y_true_valid": y_valid.tolist(),
        "y_pred_valid": np.asarray(y_pred_valid_final, dtype=int).tolist(),
        "y_proba_valid": np.asarray(y_proba_valid_final, dtype=float).tolist(),
        "y_pred_test": np.asarray(y_pred_test_final, dtype=int).tolist(),
        "y_true_test": y_test.tolist(),
        "y_proba_test": np.asarray(y_proba_test_final, dtype=float).tolist(),
        "confusion_matrix": cm,
        "class_weight_info": class_weight_combined,
        "stage1_metrics": {
            "valid": {k: float(v) for k, v in stage1_result.metrics.items() if str(k).startswith("valid_")},
            "test": {k: float(v) for k, v in stage1_result.metrics.items() if str(k).startswith("test_")},
            "classification_report_valid": stage1_result.artifacts.get("classification_report_valid", {}),
            "classification_report_test": stage1_result.artifacts.get("classification_report_test", {}),
        },
        "stage2_metrics": {
            "valid_non_dropout_only": {k: float(v) for k, v in stage2_result.metrics.items() if str(k).startswith("valid_")},
            "test_non_dropout_only": {k: float(v) for k, v in stage2_result.metrics.items() if str(k).startswith("test_")},
            "classification_report_valid": stage2_result.artifacts.get("classification_report_valid", {}),
            "classification_report_test": stage2_result.artifacts.get("classification_report_test", {}),
            "outlier": stage2_outlier_meta,
            "balancing": stage2_balancing_meta,
            "finite_sanitation": stage2_sanitation_report,
            "feature_sharpening": {
                **stage2_feature_report,
                "enabled": stage2_feature_engineering_enabled,
                "requested_groups": stage2_requested_groups,
                "interaction_requested_groups": stage2_requested_interaction_groups,
                "selective_feature_allowlist": stage2_selective_feature_allowlist,
                "feature_count_seen_at_training": int(X_train_stage2.shape[1]),
            },
            "stage2_optuna_tuning": stage2_optuna_tuning_result,
            "stage2_decision_tuning": stage2_decision_tuning_result,
        },
        "selected_threshold": {
            "dropout_threshold": (
                None if threshold_tuning_result.get("selected_dropout_threshold") is None else float(selected_dropout_threshold)
            ),
            "low_threshold": (
                None if threshold_tuning_result.get("selected_low_threshold") is None else float(selected_low_threshold)
            ),
            "high_threshold": (
                None if threshold_tuning_result.get("selected_high_threshold") is None else float(selected_high_threshold)
            ),
            "mode": str(threshold_tuning_cfg.get("mode", "fixed")),
            "selection_split": "validation",
            "objective": str(threshold_tuning_result.get("objective", threshold_tuning_cfg.get("objective", "macro_f1"))),
        },
        "threshold_tuning_results": threshold_tuning_result.get("threshold_grid_results", []),
        "two_stage": {
            "enabled": True,
            "mode": mode_name,
            "decision_mode": decision_mode,
            "threshold_stage1": float(selected_dropout_threshold) if threshold_tuning_result.get("selected_dropout_threshold") is not None else None,
            "threshold_stage1_low": float(selected_low_threshold) if threshold_tuning_result.get("selected_low_threshold") is not None else None,
            "threshold_stage1_high": float(selected_high_threshold) if threshold_tuning_result.get("selected_high_threshold") is not None else None,
            "final_class_thresholds": {
                str(label_order[i]): float(tuned_thresholds_vec[i]) for i in range(len(label_order))
            },
            "stage1_task": "Dropout vs Non-Dropout",
            "stage2_task": "Enrolled vs Graduate",
            "stage1_positive_label": "Dropout",
            "stage2_positive_label": stage2_positive_label_name,
            "stage2_negative_label": stage2_negative_label_name,
            "fusion": {
                "type": "soft_probability_fusion",
                "class_order": [int(dropout_idx), int(enrolled_idx), int(graduate_idx)],
                "formula": {
                    "P(dropout)": "p_dropout",
                    "P(enrolled)": "(1-p_dropout) * p_enrolled_given_non_dropout",
                    "P(graduate)": "(1-p_dropout) * p_graduate_given_non_dropout",
                },
            },
            "calibration": {
                "stage1": stage1_calibration_meta,
                "stage2": stage2_calibration_meta,
            },
            "feature_sharpening": {
                **stage2_feature_report,
                "enabled": stage2_feature_engineering_enabled,
                "requested_groups": stage2_requested_groups,
                "interaction_requested_groups": stage2_requested_interaction_groups,
                "selective_feature_allowlist": stage2_selective_feature_allowlist,
                "feature_count_seen_at_training": int(X_train_stage2.shape[1]),
                "stage1_feature_count": int(X_train.shape[1]),
                "stage2_base_feature_count": int(X_train_stage2_source.shape[1]),
            },
            "finite_sanitation": stage2_sanitation_report,
            "threshold_tuning": threshold_tuning_result,
            "stage2_optuna": stage2_optuna_tuning_result,
            "stage2_decision": stage2_decision_tuning_result,
            "decision_regions_valid": pd.Series(valid_decision_regions).value_counts().to_dict(),
            "decision_regions_test": pd.Series(test_decision_regions).value_counts().to_dict(),
            "stage2_decision_reason_counts_valid": pd.Series(valid_stage2_decision_reason).value_counts().to_dict(),
            "stage2_decision_reason_counts_test": pd.Series(test_stage2_decision_reason).value_counts().to_dict(),
            "stage_model_artifacts": {
                "stage1_model": str(stage1_model_path),
                "stage2_model": str(stage2_model_path),
            },
            "explainability_note": "Best-model explainability uses the wrapped two-stage predictor; visualization fallback may use stage2 model.",
        },
    }
    threshold_payload = dict(threshold_tuning_result)
    if not threshold_payload.get("selected_thresholds"):
        class_index_to_label = class_metadata.get("class_index_to_label", {})
        threshold_payload["selected_thresholds"] = {
            str(class_index_to_label.get(str(label_order[i]), label_order[i])): float(tuned_thresholds_vec[i])
            for i in range(len(label_order))
        }
    payload = {
        "metrics": metrics,
        "artifacts": {k: v for k, v in artifacts.items() if k != "model"},
        "params": artifacts["params"],
        "tuning_score": tuning_score,
        "class_weight": class_weight_combined,
        "threshold_tuning": threshold_payload,
    }
    payload["metrics"]["selected_dropout_threshold"] = (
        float(selected_dropout_threshold) if threshold_tuning_result.get("selected_dropout_threshold") is not None else np.nan
    )
    payload["metrics"]["stage1_low_threshold"] = (
        float(selected_low_threshold) if threshold_tuning_result.get("selected_low_threshold") is not None else np.nan
    )
    payload["metrics"]["stage1_high_threshold"] = (
        float(selected_high_threshold) if threshold_tuning_result.get("selected_high_threshold") is not None else np.nan
    )
    payload["metrics"]["threshold_objective_score"] = float(
        threshold_tuning_result.get("validation_objective_score_at_selected_threshold", np.nan)
    )
    payload["metrics"]["enrolled_push_alpha"] = float(threshold_tuning_result.get("enrolled_push_alpha", np.nan))
    payload["metrics"]["middle_band_enabled"] = (
        1.0 if bool(threshold_tuning_result.get("middle_band_enabled", False)) else 0.0
    )
    payload["metrics"]["stage1_threshold_mode"] = str(threshold_tuning_cfg.get("mode", "fixed"))
    payload["metrics"]["stage2_decision_requested"] = 1.0 if bool(stage2_decision_tuning_result.get("requested", requested_stage2_decision)) else 0.0
    payload["metrics"]["stage2_decision_executed"] = 1.0 if bool(stage2_decision_tuning_result.get("executed", executed_stage2_decision)) else 0.0
    payload["metrics"]["stage2_decision_accepted"] = (
        1.0
        if bool(
            stage2_decision_tuning_result.get("acceptance", {}).get("accepted", False)
            and selected_stage2_decision_cfg.get("enabled", False)
        )
        else 0.0
    )
    payload["metrics"]["stage2_decision_status"] = str(stage2_decision_tuning_result.get("status", "skipped"))
    payload["metrics"]["stage2_decision_objective_mode"] = str(
        stage2_decision_tuning_result.get("objective_components_selected", {}).get(
            "objective_mode",
            stage2_decision_cfg.get("objective", {}).get("mode", "legacy_macro_priority"),
        )
    )
    payload["metrics"]["stage2_decision_reject_reason"] = str(stage2_decision_tuning_result.get("reason", ""))
    payload["metrics"]["stage2_decision_rule"] = _stage2_decision_rule_string(selected_stage2_decision_cfg)
    payload["metrics"]["stage2_decision_enabled"] = (
        1.0 if bool(selected_stage2_decision_cfg.get("enabled", False)) else 0.0
    )
    payload["metrics"]["stage2_decision_threshold"] = float(
        selected_stage2_decision_cfg.get("enrolled_probability_threshold", np.nan)
    )
    payload["metrics"]["stage2_decision_margin"] = float(
        selected_stage2_decision_cfg.get("enrolled_margin", np.nan)
    )
    payload["metrics"]["stage2_enrolled_probability_threshold"] = float(
        selected_stage2_decision_cfg.get("enrolled_probability_threshold", np.nan)
    )
    payload["metrics"]["stage2_graduate_margin_guard"] = float(
        selected_stage2_decision_cfg.get("graduate_margin_guard", np.nan)
    )
    payload["metrics"]["stage2_enrolled_margin"] = float(
        selected_stage2_decision_cfg.get("enrolled_margin", np.nan)
    )
    payload["metrics"]["stage2_dropout_probability_guard"] = float(
        selected_stage2_decision_cfg.get("dropout_probability_guard", np.nan)
    )
    payload["metrics"]["stage2_decision_min_confidence"] = float(
        selected_stage2_decision_cfg.get("min_confidence", np.nan)
    )
    payload["metrics"]["stage2_selected_enrolled_class_weight_scale"] = float(
        selected_stage2_decision_cfg.get("enrolled_class_weight_scale", np.nan)
    )
    payload["metrics"]["stage2_selected_enrolled_weight"] = float(
        stage2_decision_tuning_result.get("selected_stage2_weights", {}).get("selected_enrolled_weight", np.nan)
    )
    payload["metrics"]["stage2_selected_graduate_weight"] = float(
        stage2_decision_tuning_result.get("selected_stage2_weights", {}).get("selected_graduate_weight", np.nan)
    )
    payload["metrics"]["stage2_feature_sharpening_enabled"] = 1.0 if stage2_feature_engineering_enabled else 0.0
    payload["metrics"]["stage2_feature_sharpening_created_count"] = float(
        stage2_feature_report.get("created_feature_count", 0)
    )
    payload["metrics"]["stage2_selective_interaction_count"] = float(
        len(
            stage2_feature_report.get("selective_interactions", {}).get("created_features", [])
            if isinstance(stage2_feature_report.get("selective_interactions", {}), dict)
            else []
        )
    )
    payload["metrics"]["stage2_finite_sanitation_enabled"] = 1.0 if bool(finite_sanitation_cfg.get("enabled", False)) else 0.0
    payload["metrics"]["stage2_prototype_distance_enabled"] = 1.0 if bool(prototype_report.get("enabled", False)) else 0.0
    payload["metrics"]["stage2_prototype_feature_count"] = float(prototype_report.get("created_feature_count", 0))
    payload["metrics"]["stage2_feature_count_seen_at_training"] = float(X_train_stage2.shape[1])
    payload["metrics"]["stage2_optuna_trials"] = float(
        stage2_decision_tuning_result.get("optuna", {}).get("n_trials_completed", 0)
    )
    payload["metrics"]["stage2_decision_trials"] = float(
        stage2_decision_tuning_result.get("search_evaluated_candidates", 0)
    )
    payload["metrics"]["stage2_decision_objective_score"] = float(
        stage2_decision_tuning_result.get("validation_objective_score_selected", np.nan)
    )
    payload["metrics"]["stage2_branch_valid_macro_f1_before"] = float(
        stage2_decision_tuning_result.get("baseline_validation_metrics", {}).get("macro_f1", np.nan)
    )
    payload["metrics"]["stage2_branch_valid_macro_f1_after"] = float(
        stage2_decision_tuning_result.get("tuned_validation_metrics", {}).get("macro_f1", np.nan)
    )
    payload["metrics"]["stage2_final_valid_macro_f1_before"] = float(
        stage2_decision_tuning_result.get("validation_final_macro_f1_baseline", np.nan)
    )
    payload["metrics"]["stage2_final_valid_macro_f1_after"] = float(
        stage2_decision_tuning_result.get("validation_final_macro_f1_selected", np.nan)
    )
    payload["metrics"]["stage2_final_valid_enrolled_f1_before"] = float(
        stage2_decision_tuning_result.get("validation_final_enrolled_f1_baseline", np.nan)
    )
    payload["metrics"]["stage2_final_valid_enrolled_f1_after"] = float(
        stage2_decision_tuning_result.get("validation_final_enrolled_f1_selected", np.nan)
    )
    payload["metrics"]["stage2_final_valid_graduate_f1_before"] = float(
        stage2_decision_tuning_result.get("validation_final_graduate_f1_baseline", np.nan)
    )
    payload["metrics"]["stage2_final_valid_graduate_f1_after"] = float(
        stage2_decision_tuning_result.get("validation_final_graduate_f1_selected", np.nan)
    )
    payload["metrics"]["test_prediction_rate_dropout"] = float(np.mean(np.asarray(y_pred_test_final, dtype=int) == int(dropout_idx)))
    payload["metrics"]["test_prediction_rate_enrolled"] = float(np.mean(np.asarray(y_pred_test_final, dtype=int) == int(enrolled_idx)))
    payload["metrics"]["test_prediction_rate_graduate"] = float(np.mean(np.asarray(y_pred_test_final, dtype=int) == int(graduate_idx)))
    payload["metrics"]["valid_prediction_rate_dropout"] = float(np.mean(np.asarray(y_pred_valid_final, dtype=int) == int(dropout_idx)))
    payload["metrics"]["valid_prediction_rate_enrolled"] = float(np.mean(np.asarray(y_pred_valid_final, dtype=int) == int(enrolled_idx)))
    payload["metrics"]["valid_prediction_rate_graduate"] = float(np.mean(np.asarray(y_pred_valid_final, dtype=int) == int(graduate_idx)))
    if tuning_meta:
        payload["tuning"] = tuning_meta
    two_stage_metadata_path = output_dir / f"two_stage_metadata_{model_token}.json"
    calibration_metadata_path = output_dir / f"calibration_metadata_{model_token}.json"
    threshold_metadata_path = output_dir / f"threshold_tuning_metadata_{model_token}.json"
    lightgbm_feature_name_mapping_path = output_dir / "runtime_artifacts" / "lightgbm_feature_name_map.json"
    two_stage_metadata_payload = {
        "model": model_name,
        "mode": mode_name,
        "decision_mode": decision_mode,
        "class_order": [int(v) for v in label_order],
        "selected_low_threshold": threshold_tuning_result.get("selected_low_threshold"),
        "selected_high_threshold": threshold_tuning_result.get("selected_high_threshold"),
        "stage2_decision": stage2_decision_tuning_result,
        "stage2_optuna_tuning": stage2_optuna_tuning_result,
        "fusion": {
            "P(dropout)": "p_dropout",
            "P(enrolled)": "(1-p_dropout) * p_enrolled_given_non_dropout",
            "P(graduate)": "(1-p_dropout) * p_graduate_given_non_dropout",
        },
        "stage2_positive_label": stage2_positive_label_name,
        "stage2_feature_sharpening": {
            **stage2_feature_report,
            "enabled": stage2_feature_engineering_enabled,
            "requested_groups": stage2_requested_groups,
            "interaction_requested_groups": stage2_requested_interaction_groups,
            "feature_count_seen_at_training": int(X_train_stage2.shape[1]),
        },
    }
    calibration_metadata_payload = {
        "model": model_name,
        "enabled": bool(calibration_cfg.get("stage1", {}).get("enabled", False) or calibration_cfg.get("stage2", {}).get("enabled", False)),
        "method": str(calibration_cfg.get("stage1", {}).get("method", calibration_cfg.get("stage2", {}).get("method", "sigmoid"))),
        "stage1": stage1_calibration_meta,
        "stage2": stage2_calibration_meta,
        "stage2_decision_policy": stage2_decision_tuning_result.get("calibration", {}),
    }
    threshold_metadata_payload = {
        "model": model_name,
        "enabled": bool(threshold_tuning_cfg.get("enabled", False)),
        "metric": str(threshold_tuning_result.get("objective", threshold_tuning_cfg.get("metric", "macro_f1"))),
        "search_mode": str(threshold_tuning_result.get("search_mode", threshold_tuning_cfg.get("search_mode", "single"))),
        "selected_thresholds": threshold_payload.get("selected_thresholds", {}),
        "selected_dropout_threshold": threshold_payload.get("selected_dropout_threshold"),
        "selected_low_threshold": threshold_payload.get("selected_low_threshold"),
        "selected_high_threshold": threshold_payload.get("selected_high_threshold"),
        "validation_score_before": float(threshold_payload.get("validation_baseline_metrics", {}).get("macro_f1", 0.0)),
        "validation_score_after": float(threshold_payload.get("validation_tuned_metrics", {}).get("macro_f1", 0.0)),
        "validation_objective_score_after": float(threshold_payload.get("validation_objective_score_at_selected_threshold", 0.0)),
        "search_evaluated_candidates": int(threshold_payload.get("search_evaluated_candidates", 0)),
        "stage2_decision": stage2_decision_tuning_result,
    }
    two_stage_metadata_path.write_text(json.dumps(two_stage_metadata_payload, indent=2), encoding="utf-8")
    calibration_metadata_path.write_text(json.dumps(calibration_metadata_payload, indent=2), encoding="utf-8")
    threshold_metadata_path.write_text(json.dumps(threshold_metadata_payload, indent=2), encoding="utf-8")
    if bool(lightgbm_feature_name_artifacts.get("applied", False)):
        lightgbm_feature_name_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        lightgbm_feature_name_mapping_path.write_text(
            json.dumps(lightgbm_feature_name_artifacts, indent=2),
            encoding="utf-8",
        )
    threshold_results_path = output_dir / f"threshold_tuning_results_{model_token}.csv"
    selected_threshold_path = output_dir / f"selected_threshold_{model_token}.json"
    two_stage_diagnostics_path = output_dir / f"two_stage_diagnostics_{model_token}.json"
    middle_band_diagnostics_path = output_dir / f"middle_band_diagnostics_{model_token}.json"
    stage1_metrics_path = output_dir / f"stage1_metrics_{model_token}.json"
    stage2_metrics_path = output_dir / f"stage2_metrics_{model_token}.json"
    pd.DataFrame(threshold_tuning_result.get("threshold_grid_results", [])).to_csv(threshold_results_path, index=False)
    selected_threshold_path.write_text(json.dumps(artifacts["selected_threshold"], indent=2), encoding="utf-8")
    two_stage_diagnostics_payload = {
        "model": model_name,
        "decision_mode": decision_mode,
        "stage2_feature_sharpening": {
            **stage2_feature_report,
            "enabled": stage2_feature_engineering_enabled,
            "requested_groups": stage2_requested_groups,
            "interaction_requested_groups": stage2_requested_interaction_groups,
            "feature_count_seen_at_training": int(X_train_stage2.shape[1]),
        },
        "selected_dropout_threshold": threshold_tuning_result.get("selected_dropout_threshold"),
        "selected_low_threshold": threshold_tuning_result.get("selected_low_threshold"),
        "selected_high_threshold": threshold_tuning_result.get("selected_high_threshold"),
        "validation_macro_f1_by_threshold": threshold_tuning_result.get("threshold_grid_results", []),
        "stage2_decision": stage2_decision_tuning_result,
        "validation_enrolled_absorbed_into_dropout_at_selected_threshold": int(
            np.sum((np.asarray(y_valid, dtype=int) == int(enrolled_idx)) & (np.asarray(y_pred_valid_final, dtype=int) == int(dropout_idx)))
        ),
        "validation_decision_regions": pd.Series(valid_decision_regions).value_counts().to_dict(),
        "test_decision_regions": pd.Series(test_decision_regions).value_counts().to_dict(),
        "validation_stage2_decision_reason_counts": pd.Series(valid_stage2_decision_reason).value_counts().to_dict(),
        "test_stage2_decision_reason_counts": pd.Series(test_stage2_decision_reason).value_counts().to_dict(),
        "test_class_distribution": {
            "dropout": int(np.sum(np.asarray(y_pred_test_final, dtype=int) == int(dropout_idx))),
            "enrolled": int(np.sum(np.asarray(y_pred_test_final, dtype=int) == int(enrolled_idx))),
            "graduate": int(np.sum(np.asarray(y_pred_test_final, dtype=int) == int(graduate_idx))),
        },
    }
    middle_band_diagnostics_payload = {
        "model": model_name,
        "enabled": bool(threshold_tuning_result.get("middle_band_enabled", False)),
        "behavior": str(threshold_tuning_result.get("middle_band_behavior", "force_stage2_soft_fusion")),
        "selected_low_threshold": threshold_tuning_result.get("selected_low_threshold"),
        "selected_high_threshold": threshold_tuning_result.get("selected_high_threshold"),
        "validation_region_counts": pd.Series(valid_decision_regions).value_counts().to_dict(),
        "test_region_counts": pd.Series(test_decision_regions).value_counts().to_dict(),
    }
    two_stage_diagnostics_path.write_text(json.dumps(two_stage_diagnostics_payload, indent=2), encoding="utf-8")
    middle_band_diagnostics_path.write_text(json.dumps(middle_band_diagnostics_payload, indent=2), encoding="utf-8")
    stage1_metrics_path.write_text(json.dumps(artifacts["stage1_metrics"], indent=2), encoding="utf-8")
    stage2_metrics_path.write_text(json.dumps(artifacts["stage2_metrics"], indent=2), encoding="utf-8")

    prediction_export_test = pd.DataFrame(
        {
            "selected_threshold": (
                float(selected_dropout_threshold) if threshold_tuning_result.get("selected_dropout_threshold") is not None else np.nan
            ),
            "selected_low_threshold": (
                float(selected_low_threshold) if threshold_tuning_result.get("selected_low_threshold") is not None else np.nan
            ),
            "selected_high_threshold": (
                float(selected_high_threshold) if threshold_tuning_result.get("selected_high_threshold") is not None else np.nan
            ),
            "selected_stage2_enrolled_weight_scale": float(
                selected_stage2_decision_cfg.get("enrolled_class_weight_scale", np.nan)
            ),
            "selected_stage2_enrolled_weight": float(
                stage2_decision_tuning_result.get("selected_stage2_weights", {}).get("selected_enrolled_weight", np.nan)
            ),
            "decision_region": np.asarray(test_decision_regions, dtype=str),
            "final_decision_mode": str(decision_mode),
            "stage1_prob_dropout": np.asarray(stage_prob_test.get("stage1_prob_dropout", []), dtype=float),
            "stage1_prob_non_dropout": np.asarray(stage_prob_test.get("stage1_prob_non_dropout", []), dtype=float),
            "stage2_prob_enrolled": np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            "stage2_prob_graduate": np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
            "stage2_decision_reason": np.asarray(test_stage2_decision_reason, dtype=str),
        }
    )
    prediction_export_valid = pd.DataFrame(
        {
            "selected_threshold": (
                float(selected_dropout_threshold) if threshold_tuning_result.get("selected_dropout_threshold") is not None else np.nan
            ),
            "selected_low_threshold": (
                float(selected_low_threshold) if threshold_tuning_result.get("selected_low_threshold") is not None else np.nan
            ),
            "selected_high_threshold": (
                float(selected_high_threshold) if threshold_tuning_result.get("selected_high_threshold") is not None else np.nan
            ),
            "selected_stage2_enrolled_weight_scale": float(
                selected_stage2_decision_cfg.get("enrolled_class_weight_scale", np.nan)
            ),
            "selected_stage2_enrolled_weight": float(
                stage2_decision_tuning_result.get("selected_stage2_weights", {}).get("selected_enrolled_weight", np.nan)
            ),
            "decision_region": np.asarray(valid_decision_regions, dtype=str),
            "final_decision_mode": str(decision_mode),
            "stage1_prob_dropout": np.asarray(stage_prob_valid.get("stage1_prob_dropout", []), dtype=float),
            "stage1_prob_non_dropout": np.asarray(stage_prob_valid.get("stage1_prob_non_dropout", []), dtype=float),
            "stage2_prob_enrolled": np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            "stage2_prob_graduate": np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
            "stage2_decision_reason": np.asarray(valid_stage2_decision_reason, dtype=str),
        }
    )
    artifacts["prediction_export_test"] = prediction_export_test
    artifacts["prediction_export_valid"] = prediction_export_valid
    artifacts["two_stage_diagnostics"] = two_stage_diagnostics_payload
    artifacts["middle_band_diagnostics"] = middle_band_diagnostics_payload
    payload["artifacts"]["prediction_export_test"] = prediction_export_test
    payload["artifacts"]["prediction_export_valid"] = prediction_export_valid
    payload["artifacts"]["two_stage_diagnostics"] = two_stage_diagnostics_payload
    payload["artifacts"]["middle_band_diagnostics"] = middle_band_diagnostics_payload
    payload["artifacts"]["lightgbm_feature_name_mapping"] = lightgbm_feature_name_artifacts
    payload["artifact_paths"] = {
        "stage1_model": str(stage1_model_path),
        "stage2_model": str(stage2_model_path),
        "two_stage_metadata": str(two_stage_metadata_path),
        "calibration_metadata": str(calibration_metadata_path),
        "threshold_tuning_metadata": str(threshold_metadata_path),
        "threshold_tuning_results_csv": str(threshold_results_path),
        "selected_threshold_json": str(selected_threshold_path),
        "two_stage_diagnostics_json": str(two_stage_diagnostics_path),
        "middle_band_diagnostics_json": str(middle_band_diagnostics_path),
        "stage1_metrics_json": str(stage1_metrics_path),
        "stage2_metrics_json": str(stage2_metrics_path),
    }
    if bool(lightgbm_feature_name_artifacts.get("applied", False)):
        payload["artifact_paths"]["lightgbm_feature_name_mapping_json"] = str(lightgbm_feature_name_mapping_path)
    if tuning_artifacts:
        payload["artifact_paths"].update(tuning_artifacts)
    if stage2_feature_engineering_enabled or bool(prototype_report.get("enabled", False)):
        payload["_runtime_artifact_overrides"] = {
            "X_train": X_train_stage2_augmented_full,
            "X_valid": X_valid_stage2_augmented_full,
            "X_test": X_test_stage2_augmented_full,
            "feature_names": list(X_train_stage2_augmented_full.columns),
        }
    return payload, combined_model, tuning_score, tuning_meta, tuning_artifacts


def _effect_label(enabled_mean: float, disabled_mean: float, metric_name: str) -> str:
    diff = enabled_mean - disabled_mean
    if diff > 0:
        return f"{metric_name}: enabled > disabled ({diff:+.4f})"
    if diff < 0:
        return f"{metric_name}: enabled < disabled ({diff:+.4f})"
    return f"{metric_name}: enabled ~= disabled ({diff:+.4f})"


def _build_ablation_summary_markdown(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "# Ablation Summary\n\n_No ablation rows were generated._\n"

    best_macro_row = summary_df.iloc[0]
    best_enrolled_row = summary_df.sort_values(
        ["test_f1_enrolled", "test_macro_f1"],
        ascending=[False, False],
    ).iloc[0]

    outlier_enabled_mean_macro = float(summary_df.loc[summary_df["outlier_enabled"], "test_macro_f1"].mean())
    outlier_disabled_mean_macro = float(summary_df.loc[~summary_df["outlier_enabled"], "test_macro_f1"].mean())
    outlier_enabled_mean_enrolled = float(summary_df.loc[summary_df["outlier_enabled"], "test_f1_enrolled"].mean())
    outlier_disabled_mean_enrolled = float(summary_df.loc[~summary_df["outlier_enabled"], "test_f1_enrolled"].mean())

    smote_enabled_mean_macro = float(summary_df.loc[summary_df["smote_enabled"], "test_macro_f1"].mean())
    smote_disabled_mean_macro = float(summary_df.loc[~summary_df["smote_enabled"], "test_macro_f1"].mean())
    smote_enabled_mean_enrolled = float(summary_df.loc[summary_df["smote_enabled"], "test_f1_enrolled"].mean())
    smote_disabled_mean_enrolled = float(summary_df.loc[~summary_df["smote_enabled"], "test_f1_enrolled"].mean())

    interaction = (
        summary_df.groupby(["outlier_enabled", "smote_enabled"], as_index=False)[["test_macro_f1", "test_f1_enrolled"]]
        .mean()
        .sort_values(["test_macro_f1", "test_f1_enrolled"], ascending=[False, False])
        .iloc[0]
    )
    interaction_setting = f"outlier={bool(interaction['outlier_enabled'])}, smote={bool(interaction['smote_enabled'])}"

    lines = [
        "# UCT 3-Class Paper-Push Ablation Summary",
        "",
        "- Sort order: `test_macro_f1` desc, then `test_f1_enrolled` desc.",
        "",
        "## Variant x Model Comparison",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            f"- Best setting by `test_macro_f1`: variant `{best_macro_row['ablation_variant']}` "
            f"(model `{best_macro_row['model']}`, macro_f1={best_macro_row['test_macro_f1']:.4f})."
        ),
        (
            f"- Best setting by `test_f1_enrolled`: variant `{best_enrolled_row['ablation_variant']}` "
            f"(model `{best_enrolled_row['model']}`, enrolled_f1={best_enrolled_row['test_f1_enrolled']:.4f})."
        ),
        (
            "- Outlier effect (mean across model rows): "
            + _effect_label(outlier_enabled_mean_macro, outlier_disabled_mean_macro, "macro_f1")
            + "; "
            + _effect_label(outlier_enabled_mean_enrolled, outlier_disabled_mean_enrolled, "enrolled_f1")
            + "."
        ),
        (
            "- SMOTE effect (mean across model rows): "
            + _effect_label(smote_enabled_mean_macro, smote_disabled_mean_macro, "macro_f1")
            + "; "
            + _effect_label(smote_enabled_mean_enrolled, smote_disabled_mean_enrolled, "enrolled_f1")
            + "."
        ),
        (
            f"- Interaction signal (best mean pair): `{interaction_setting}` "
            f"(macro_f1={float(interaction['test_macro_f1']):.4f}, enrolled_f1={float(interaction['test_f1_enrolled']):.4f})."
        ),
    ]
    return "\n".join(lines) + "\n"


def _normalize_enrolled_weight_sweep(raw_values: Any) -> list[float]:
    if not isinstance(raw_values, (list, tuple)):
        return []
    normalized: list[float] = []
    for value in raw_values:
        try:
            token = float(value)
        except (TypeError, ValueError):
            continue
        if token > 0:
            normalized.append(token)
    # Preserve order while removing duplicates.
    deduped: list[float] = []
    seen: set[float] = set()
    for token in normalized:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _weight_token(weight: float) -> str:
    return str(weight).replace(".", "_")


def _replace_path_prefix(value: str, old_prefix: str, new_prefix: str) -> str:
    if value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix):]
    return value


def _build_enrolled_weight_sweep_markdown(
    summary_df: pd.DataFrame,
    selection_cfg: dict[str, Any] | None = None,
) -> str:
    if summary_df.empty:
        return "# Enrolled Weight Sweep Summary\n\n_No sweep rows were generated._\n"

    best_ranked = summary_df.iloc[0]
    best_macro = summary_df.sort_values(
        ["test_macro_f1", "test_f1_enrolled"],
        ascending=[False, False],
    ).iloc[0]
    best_enrolled = summary_df.sort_values(
        ["test_f1_enrolled", "test_macro_f1"],
        ascending=[False, False],
    ).iloc[0]

    mean_by_weight = (
        summary_df.groupby("enrolled_weight", as_index=False)[["test_macro_f1", "test_f1_enrolled"]]
        .mean()
        .sort_values("enrolled_weight")
    )
    macro_peak_idx = int(mean_by_weight["test_macro_f1"].idxmax())
    enrolled_peak_idx = int(mean_by_weight["test_f1_enrolled"].idxmax())
    macro_peak_weight = float(mean_by_weight.iloc[macro_peak_idx]["enrolled_weight"])
    enrolled_peak_weight = float(mean_by_weight.iloc[enrolled_peak_idx]["enrolled_weight"])
    strongest_weight = float(mean_by_weight["enrolled_weight"].max())
    macro_at_strongest = float(mean_by_weight.loc[mean_by_weight["enrolled_weight"] == strongest_weight, "test_macro_f1"].iloc[0])
    macro_at_peak = float(mean_by_weight["test_macro_f1"].max())
    degrades_after_peak = macro_at_strongest < macro_at_peak - 1e-12

    baseline_weight = float(mean_by_weight["enrolled_weight"].min())
    baseline_rows = summary_df[summary_df["enrolled_weight"] == baseline_weight]
    benefit_rows: list[dict[str, Any]] = []
    for model_name in sorted(summary_df["model"].astype(str).unique()):
        model_rows = summary_df[summary_df["model"] == model_name]
        if model_rows.empty:
            continue
        best_model_row = model_rows.sort_values(["test_f1_enrolled", "test_macro_f1"], ascending=[False, False]).iloc[0]
        baseline_model_row = baseline_rows[baseline_rows["model"] == model_name]
        baseline_enrolled = float(baseline_model_row.iloc[0]["test_f1_enrolled"]) if not baseline_model_row.empty else float("nan")
        delta = float(best_model_row["test_f1_enrolled"] - baseline_enrolled) if np.isfinite(baseline_enrolled) else float("nan")
        benefit_rows.append(
            {
                "model": model_name,
                "best_weight": float(best_model_row["enrolled_weight"]),
                "best_enrolled_f1": float(best_model_row["test_f1_enrolled"]),
                "delta_vs_weight_min": delta,
            }
        )
    benefit_df = pd.DataFrame(benefit_rows)
    if not benefit_df.empty:
        top_benefit = benefit_df.sort_values("delta_vs_weight_min", ascending=False).iloc[0]
        top_benefit_line = (
            f"- Model benefiting most in Enrolled F1 vs weight={baseline_weight:.1f}: "
            f"`{top_benefit['model']}` (best_weight={float(top_benefit['best_weight']):.1f}, "
            f"delta={float(top_benefit['delta_vs_weight_min']):+.4f})."
        )
    else:
        top_benefit_line = "- Model benefit comparison unavailable."

    ranking_metrics = (
        selection_cfg.get("ranking_metrics", [])
        if isinstance(selection_cfg, dict)
        else []
    )
    if not isinstance(ranking_metrics, list) or not ranking_metrics:
        ranking_metrics = ["macro_f1", "enrolled_f1"]
    ranking_label = ", ".join([f"`test_{_resolve_metric_column(metric, source='test').removeprefix('test_')}`" for metric in ranking_metrics])

    lines = [
        "# Enrolled Weight Sweep Summary",
        "",
        f"- Sort order: {ranking_label} descending, then `model` ascending.",
        "",
        "## Sweep Comparison",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            f"- Best overall by configured ranking: enrolled_weight={float(best_ranked['enrolled_weight']):.1f}, "
            f"model=`{best_ranked['model']}`."
        ),
        (
            f"- Best overall by Macro F1: enrolled_weight={float(best_macro['enrolled_weight']):.1f}, "
            f"model=`{best_macro['model']}`, macro_f1={float(best_macro['test_macro_f1']):.4f}."
        ),
        (
            f"- Best overall by Enrolled F1: enrolled_weight={float(best_enrolled['enrolled_weight']):.1f}, "
            f"model=`{best_enrolled['model']}`, enrolled_f1={float(best_enrolled['test_f1_enrolled']):.4f}."
        ),
        (
            f"- Mean Macro F1 peak weight: {macro_peak_weight:.1f}; "
            f"mean Enrolled F1 peak weight: {enrolled_peak_weight:.1f}."
        ),
        (
            "- Stronger Enrolled weighting trend: "
            + ("performance appears to peak then degrade at higher weights." if degrades_after_peak else "no clear degradation at the strongest tested weight.")
        ),
        top_benefit_line,
    ]
    return "\n".join(lines) + "\n"


def _run_enrolled_weight_sweep(
    exp_cfg: dict[str, Any],
    compact_summary: bool | None = None,
) -> dict[str, Any]:
    class_weight_cfg = exp_cfg.get("models", {}).get("class_weight", {})
    sweep_weights = _normalize_enrolled_weight_sweep(class_weight_cfg.get("enrolled_weight_sweep"))
    if len(sweep_weights) <= 1:
        raise ValueError("Enrolled weight sweep requires at least 2 positive weights in models.class_weight.enrolled_weight_sweep.")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to run enrolled-weight sweep mode. Install with `pip install pyyaml`.") from exc

    experiment_id = str(exp_cfg.get("experiment", {}).get("id", "exp_weight_sweep"))
    output_dir = resolve_results_dir(exp_cfg, experiment_id=experiment_id)
    ensure_standard_output_layout(output_dir)
    sweep_root = output_dir / "weight_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    selection_cfg = _resolve_model_selection_config(exp_cfg)

    rows: list[dict[str, Any]] = []
    variant_summaries: dict[str, dict[str, Any]] = {}
    required_metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_f1_dropout",
        "test_f1_enrolled",
        "test_f1_graduate",
        "test_macro_precision",
        "test_macro_recall",
        "test_balanced_accuracy",
    ]

    with tempfile.TemporaryDirectory(prefix="weight_sweep_run_") as tmpdir:
        tmp_root = Path(tmpdir)
        for weight in sweep_weights:
            token = _weight_token(weight)
            variant_id = f"ew_{token}"
            variant_output_dir = sweep_root / variant_id

            variant_cfg = copy.deepcopy(exp_cfg)
            variant_cfg.setdefault("experiment", {})
            variant_cfg["experiment"]["mode"] = "benchmark"
            variant_cfg["experiment"]["id"] = f"{experiment_id}_{variant_id}"
            variant_cfg.setdefault("outputs", {})
            variant_cfg["outputs"]["results_dir"] = str(variant_output_dir)
            variant_cfg.setdefault("models", {}).setdefault("class_weight", {})
            variant_cfg["models"]["class_weight"]["enabled"] = True
            variant_cfg["models"]["class_weight"]["strategy"] = "enrolled_boost"
            variant_cfg["models"]["class_weight"]["base_weight"] = 1.0
            variant_cfg["models"]["class_weight"]["enrolled_boost"] = float(weight)
            variant_cfg["models"]["class_weight"]["class_weight_map"] = {
                "Dropout": 1.0,
                "Enrolled": float(weight),
                "Graduate": 1.0,
            }
            variant_cfg["models"]["class_weight"]["enrolled_weight_sweep"] = [float(weight)]
            variant_cfg.setdefault("training", {})
            variant_cfg["training"].pop("class_weight", None)

            variant_cfg_path = tmp_root / f"{variant_id}.yaml"
            variant_cfg_path.write_text(yaml.safe_dump(variant_cfg, sort_keys=False), encoding="utf-8")
            variant_summary = run_experiment(variant_cfg_path, compact_summary=compact_summary)
            variant_summaries[variant_id] = variant_summary

            leaderboard = pd.DataFrame(variant_summary.get("leaderboard", []))
            if leaderboard.empty:
                continue
            for metric in required_metrics:
                if metric not in leaderboard.columns:
                    leaderboard[metric] = np.nan
            for _, lb_row in leaderboard.iterrows():
                model_name = str(lb_row.get("model"))
                model_payload = variant_summary.get("model_results", {}).get(model_name, {})
                threshold_payload = model_payload.get("threshold_tuning", {}) if isinstance(model_payload, dict) else {}
                rows.append(
                    {
                        "enrolled_weight": float(weight),
                        "model": model_name,
                        **{metric: lb_row.get(metric) for metric in required_metrics},
                        "threshold_tuning_applied": bool(threshold_payload.get("threshold_tuning_applied", False)),
                        "selected_thresholds": json.dumps(threshold_payload.get("selected_thresholds", {})),
                        "weight_sweep_variant": variant_id,
                    }
                )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("Enrolled weight sweep produced no leaderboard rows.")
    summary_df, _, _ = _sort_leaderboard_with_tiebreak(
        leaderboard_df=summary_df,
        selection_cfg=selection_cfg,
        source="test",
    )
    if summary_df.empty:
        summary_df = pd.DataFrame(rows)
    summary_df["best_overall_by_macro_f1"] = summary_df["test_macro_f1"] == summary_df["test_macro_f1"].max()
    summary_df["best_overall_by_enrolled_f1"] = summary_df["test_f1_enrolled"] == summary_df["test_f1_enrolled"].max()

    sweep_csv_path = output_dir / "enrolled_weight_sweep_summary.csv"
    sweep_md_path = output_dir / "enrolled_weight_sweep_summary.md"
    summary_df.to_csv(sweep_csv_path, index=False)
    sweep_md_path.write_text(
        _build_enrolled_weight_sweep_markdown(summary_df, selection_cfg=selection_cfg),
        encoding="utf-8",
    )

    best_row = summary_df.iloc[0]
    best_variant_id = str(best_row["weight_sweep_variant"])
    best_variant_summary = copy.deepcopy(variant_summaries[best_variant_id])
    best_variant_dir = sweep_root / best_variant_id

    # Mirror canonical contract files from best sweep variant into parent output directory.
    for folder in ("runtime_artifacts", "model", "figures", "explainability"):
        src = best_variant_dir / folder
        dst = output_dir / folder
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for filename in ("predictions.csv", "artifact_manifest.json"):
        src = best_variant_dir / filename
        dst = output_dir / filename
        if src.exists():
            shutil.copy2(src, dst)

    old_prefix = str(best_variant_dir).replace("\\", "/")
    new_prefix = str(output_dir).replace("\\", "/")
    artifact_paths = best_variant_summary.get("artifact_paths", {})
    if isinstance(artifact_paths, dict):
        for key, value in list(artifact_paths.items()):
            if isinstance(value, str):
                normalized = value.replace("\\", "/")
                artifact_paths[key] = _replace_path_prefix(normalized, old_prefix, new_prefix)

    best_variant_summary["experiment_id"] = experiment_id
    best_variant_summary["output_dir"] = str(output_dir)
    best_variant_summary["artifact_paths"] = artifact_paths
    best_variant_summary["weight_sweep"] = {
        "enabled": True,
        "enrolled_weight_sweep": sweep_weights,
        "variants_root": str(sweep_root),
        "best_variant": best_variant_id,
        "best_enrolled_weight": float(best_row["enrolled_weight"]),
        "best_model": str(best_row["model"]),
        "summary_csv": str(sweep_csv_path),
        "summary_md": str(sweep_md_path),
    }
    best_variant_summary.setdefault("artifact_paths", {})
    best_variant_summary["artifact_paths"]["enrolled_weight_sweep_summary_csv"] = str(sweep_csv_path)
    best_variant_summary["artifact_paths"]["enrolled_weight_sweep_summary_md"] = str(sweep_md_path)

    save_benchmark_summary(best_variant_summary, output_dir, compact=bool(compact_summary))

    metrics_payload = {
        "experiment_id": experiment_id,
        "dataset_name": best_variant_summary.get("dataset_name"),
        "target_formulation": best_variant_summary.get("target_formulation"),
        "primary_metric": best_variant_summary.get("primary_metric"),
        "best_model": best_variant_summary.get("best_model"),
        "class_weight": best_variant_summary.get("class_weight", {}),
        "threshold_tuning": best_variant_summary.get("threshold_tuning", {}),
        "weight_sweep": best_variant_summary.get("weight_sweep", {}),
        "model_mechanism_audit": best_variant_summary.get("model_mechanism_audit", {}),
        "best_model_metrics": (
            best_variant_summary.get("model_results", {}).get(best_variant_summary.get("best_model"), {}).get("metrics", {})
        ),
        "leaderboard": best_variant_summary.get("leaderboard", []),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    runtime_meta_path = output_dir / "runtime_artifacts" / "runtime_metadata.json"
    if runtime_meta_path.exists():
        runtime_meta = json.loads(runtime_meta_path.read_text(encoding="utf-8"))
    else:
        runtime_meta = {}
    runtime_meta["weight_sweep"] = best_variant_summary.get("weight_sweep", {})
    runtime_meta["model_mechanism_audit"] = best_variant_summary.get("model_mechanism_audit", {})
    runtime_meta_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_meta_path.write_text(json.dumps(runtime_meta, indent=2), encoding="utf-8")

    update_artifact_manifest(
        output_dir=output_dir,
        optional_updates={
            "enrolled_weight_sweep_summary_csv": _status_from_path(sweep_csv_path),
            "enrolled_weight_sweep_summary_md": _status_from_path(sweep_md_path),
            "weight_sweep_root": _status_from_path(sweep_root),
        },
        metadata_updates={
            "experiment_id": experiment_id,
            "manifest_scope": "benchmark+weight_sweep",
            "best_weight_sweep_variant": best_variant_id,
            "best_enrolled_weight": float(best_row["enrolled_weight"]),
        },
    )

    return best_variant_summary


def _run_ablation_experiment(
    exp_cfg: dict[str, Any],
    experiment_config_path: Path,
    compact_summary: bool | None = None,
) -> dict[str, Any]:
    ablation_cfg = exp_cfg.get("ablation", {})
    variants = ablation_cfg.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("Ablation mode requires a non-empty 'ablation.variants' list.")

    experiment_id = str(exp_cfg.get("experiment", {}).get("id", "ablation"))
    output_dir = resolve_results_dir(exp_cfg, experiment_id=experiment_id)
    ensure_standard_output_layout(output_dir)

    summary_cfg = ablation_cfg.get("summary", {})
    summary_csv_path = output_dir / str(summary_cfg.get("csv_filename", "ablation_summary.csv"))
    summary_md_path = output_dir / str(summary_cfg.get("md_filename", "ablation_summary.md"))

    rows: list[dict[str, Any]] = []
    variant_outputs: dict[str, str] = {}
    required_metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_f1_dropout",
        "test_f1_enrolled",
        "test_f1_graduate",
        "test_macro_precision",
        "test_macro_recall",
        "test_balanced_accuracy",
    ]

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to run ablation mode. Install with `pip install pyyaml`.") from exc

    with tempfile.TemporaryDirectory(prefix="ablation_run_") as tmpdir:
        tmp_root = Path(tmpdir)
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("Each ablation variant must be a mapping.")
            variant_id = str(variant.get("id", "")).strip()
            if not variant_id:
                raise ValueError("Each ablation variant must define a non-empty 'id'.")

            outlier_enabled = bool(variant.get("outlier_enabled", False))
            smote_enabled = bool(variant.get("smote_enabled", False))

            variant_cfg = copy.deepcopy(exp_cfg)
            variant_cfg.pop("ablation", None)
            variant_cfg.setdefault("experiment", {})
            variant_cfg["experiment"]["mode"] = "benchmark"
            variant_cfg["experiment"]["id"] = f"{experiment_id}_{variant_id}"
            variant_cfg.setdefault("outputs", {})
            variant_output_dir = output_dir / variant_id
            variant_cfg["outputs"]["results_dir"] = str(variant_output_dir)
            variant_cfg.setdefault("preprocessing", {})
            variant_cfg["preprocessing"].setdefault("outlier", {})
            variant_cfg["preprocessing"].setdefault("balancing", {})
            variant_cfg["preprocessing"]["outlier"]["enabled"] = outlier_enabled
            variant_cfg["preprocessing"]["balancing"]["enabled"] = smote_enabled

            variant_cfg_path = tmp_root / f"{variant_id}.yaml"
            variant_cfg_path.write_text(yaml.safe_dump(variant_cfg, sort_keys=False), encoding="utf-8")

            variant_summary = run_experiment(variant_cfg_path, compact_summary=compact_summary)
            variant_outputs[variant_id] = str(variant_output_dir)

            leaderboard = pd.DataFrame(variant_summary.get("leaderboard", []))
            if leaderboard.empty:
                continue
            if "model" not in leaderboard.columns:
                raise ValueError(f"Variant '{variant_id}' leaderboard is missing 'model'.")
            for metric in required_metrics:
                if metric not in leaderboard.columns:
                    leaderboard[metric] = np.nan

            for _, lb_row in leaderboard.iterrows():
                rows.append(
                    {
                        "ablation_variant": variant_id,
                        "outlier_enabled": outlier_enabled,
                        "smote_enabled": smote_enabled,
                        "model": str(lb_row["model"]),
                        **{metric: lb_row[metric] for metric in required_metrics},
                    }
                )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["test_macro_f1", "test_f1_enrolled"],
            ascending=[False, False],
        ).reset_index(drop=True)
        summary_df["best_by_macro_f1"] = summary_df["test_macro_f1"] == summary_df["test_macro_f1"].max()
    else:
        summary_df = pd.DataFrame(
            columns=[
                "ablation_variant",
                "outlier_enabled",
                "smote_enabled",
                "model",
                *required_metrics,
                "best_by_macro_f1",
            ]
        )

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)
    summary_md_path.write_text(_build_ablation_summary_markdown(summary_df), encoding="utf-8")

    return {
        "experiment_id": experiment_id,
        "mode": "ablation",
        "output_dir": str(output_dir),
        "variant_output_dirs": variant_outputs,
        "artifact_paths": {
            "ablation_summary_csv": str(summary_csv_path),
            "ablation_summary_md": str(summary_md_path),
        },
    }

# Main experiment runner
def run_experiment(experiment_config_path: Path, compact_summary: bool | None = None) -> dict[str, Any]:
    exp_cfg = _normalize_experiment_config_schema(load_yaml(experiment_config_path))
    experiment_mode = str(exp_cfg.get("experiment", {}).get("mode", "benchmark")).strip().lower()
    if experiment_mode == "ablation":
        return _run_ablation_experiment(
            exp_cfg=exp_cfg,
            experiment_config_path=experiment_config_path,
            compact_summary=compact_summary,
        )
    if experiment_mode == "error_audit":
        return finalize_error_audit_run(
            result=run_error_audit_mode(exp_cfg=exp_cfg, experiment_config_path=experiment_config_path),
            exp_cfg=exp_cfg,
            experiment_config_path=experiment_config_path,
        )
    if experiment_mode == "threshold_tuning":
        return finalize_threshold_tuning_run(
            result=run_threshold_tuning_mode(exp_cfg=exp_cfg, experiment_config_path=experiment_config_path),
            exp_cfg=exp_cfg,
            experiment_config_path=experiment_config_path,
        )

    class_weight_cfg = exp_cfg.get("models", {}).get("class_weight", {})
    sweep_weights = _normalize_enrolled_weight_sweep(
        class_weight_cfg.get("enrolled_weight_sweep") if isinstance(class_weight_cfg, dict) else []
    )
    if experiment_mode == "benchmark" and len(sweep_weights) > 1:
        return _run_enrolled_weight_sweep(
            exp_cfg=exp_cfg,
            compact_summary=compact_summary,
        )

    if "datasets" in exp_cfg.get("experiment", {}):
        raise NotImplementedError(
            "Shared multi-dataset workflow is experimental and not supported by scripts/run_experiment.py yet. "
            "Use single-dataset configs with experiment.dataset_config."
        )
    dataset_cfg_path = Path(exp_cfg["experiment"]["dataset_config"])
    dataset_cfg = _resolve_experiment_feature_config(
        exp_cfg=exp_cfg,
        dataset_cfg=load_yaml(dataset_cfg_path),
    )
    requested_dataset_token = dataset_cfg_path.stem
    experiment_id = exp_cfg["experiment"]["id"]
    seed = int(exp_cfg["experiment"].get("seed", 42))
    formulation = str(exp_cfg["experiment"].get("target_formulation", "binary"))
    output_cfg = exp_cfg.get("outputs", {})
    output_dir = resolve_results_dir(exp_cfg, experiment_id=experiment_id)
    ensure_standard_output_layout(output_dir)

    target_mapping = _resolve_target_mapping(exp_cfg, dataset_cfg, formulation)
    class_metadata = _resolve_class_metadata(exp_cfg, target_mapping)
    dataset_source_cfg = _resolve_dataset_source_config(dataset_cfg)
    dataset_name = _normalize_dataset_name(str(dataset_cfg.get("dataset", {}).get("name", "")))
    resolved_dataset_token = str(dataset_cfg_path.stem)
    print(f"[dataset] requested={requested_dataset_token}")
    print(f"[dataset] resolved={resolved_dataset_token}")
    print(f"[dataset] config_path={dataset_cfg_path}")
    pre_split_train_feature_source_for_vocab: pd.DataFrame | None = None
    if requested_dataset_token != resolved_dataset_token:
        raise ValueError(
            "Dataset resolution mismatch. "
            f"requested={requested_dataset_token} resolved={resolved_dataset_token} config_path={dataset_cfg_path}"
        )
    if (
        dataset_name == "uct_student"
        and dataset_source_cfg["format"] == "parquet"
        and dataset_source_cfg["split_mode"] == "predefined"
    ):
        if resolved_dataset_token == "uci_student_presplit_parquet":
            raw_dataset_identity = str(dataset_cfg.get("dataset", {}).get("name", "")).strip()
            if raw_dataset_identity != "uci_student_presplit_parquet":
                raise ValueError(
                    "Dataset identity mismatch for uci_student_presplit_parquet: "
                    f"dataset.name={raw_dataset_identity!r} expected='uci_student_presplit_parquet'."
                )
            expected_train_path = "data/processed/uci/uci_12_03_train.parquet"
            expected_test_path = "data/processed/uci/uci_12_03_test.parquet"
            resolved_train_cfg_path = str(dataset_source_cfg.get("train_path", "")).replace("\\", "/")
            resolved_test_cfg_path = str(dataset_source_cfg.get("test_path", "")).replace("\\", "/")
            if resolved_train_cfg_path != expected_train_path or resolved_test_cfg_path != expected_test_path:
                raise ValueError(
                    "Dataset lock violation for uci_student_presplit_parquet: "
                    f"expected train/test paths ({expected_train_path}, {expected_test_path}) but got "
                    f"({resolved_train_cfg_path}, {resolved_test_cfg_path})."
                )
        predefined_splits, dataset_name, id_column, source_target_col, dataset_source_meta = _build_predefined_uci_feature_splits(
            dataset_cfg,
            formulation=formulation,
            target_mapping=target_mapping,
        )
        pre_split_train_feature_source_for_vocab = predefined_splits["train"].reset_index(drop=True).copy()
        train_feature_df, train_missing_meta = _drop_rows_with_missing_values(
            predefined_splits["train"],
            exp_cfg.get("preprocessing", {}),
        )
        test_feature_df, test_missing_meta = _drop_rows_with_missing_values(
            predefined_splits["test"],
            exp_cfg.get("preprocessing", {}),
        )
        if train_feature_df.empty:
            raise ValueError("UCI predefined parquet train split became empty after missing-value filtering.")
        if test_feature_df.empty:
            raise ValueError("UCI predefined parquet test split became empty after missing-value filtering.")
        valid_feature_df_raw = predefined_splits.get("valid")
        internal_valid_source = "predefined_valid" if isinstance(valid_feature_df_raw, pd.DataFrame) else "train_only"
        effective_split_mode = (
            "predefined_train_valid_test"
            if isinstance(valid_feature_df_raw, pd.DataFrame)
            else "train_test_with_internal_valid"
        )
        if isinstance(valid_feature_df_raw, pd.DataFrame):
            valid_feature_df, valid_missing_meta = _drop_rows_with_missing_values(
                valid_feature_df_raw,
                exp_cfg.get("preprocessing", {}),
            )
            if valid_feature_df.empty:
                raise ValueError("UCI predefined parquet valid split became empty after missing-value filtering.")
            splits = {
                "train": train_feature_df.reset_index(drop=True),
                "valid": valid_feature_df.reset_index(drop=True),
                "test": test_feature_df.reset_index(drop=True),
            }
        else:
            validation_size = float(exp_cfg["splits"].get("validation_size", 0.2))
            train_valid_split_cfg = SplitConfig(
                test_size=validation_size,
                validation_size=0.0,
                random_state=seed,
                stratify_column="target",
            )
            train_valid_splits = stratified_train_valid_test_split(train_feature_df, train_valid_split_cfg)
            valid_feature_df = train_valid_splits["test"].reset_index(drop=True)
            train_feature_df = train_valid_splits["train"].reset_index(drop=True)
            valid_missing_meta = {
                "enabled": True,
                "mode": "internal_from_train_only",
                "source_rows_before_split": int(len(predefined_splits["train"])),
                "validation_size": float(validation_size),
                "random_state": int(seed),
                "stratify_column": "target",
            }
            if train_feature_df.empty or valid_feature_df.empty:
                raise ValueError("Internal validation split from predefined train parquet produced an empty train or valid split.")
            splits = {
                "train": train_feature_df,
                "valid": valid_feature_df,
                "test": test_feature_df.reset_index(drop=True),
            }
        missing_value_meta = {
            "enabled": True,
            "mode": "predefined_uci_parquet",
            "train": train_missing_meta,
            "valid": valid_missing_meta,
            "test": test_missing_meta,
            "source": dataset_source_meta,
            "effective_split_mode": effective_split_mode,
            "internal_valid_source": internal_valid_source,
        }
        print(f"[dataset] train_path={dataset_source_cfg.get('train_path')}")
        print(f"[dataset] test_path={dataset_source_cfg.get('test_path')}")
        print(f"[dataset] valid_path={dataset_source_cfg.get('valid_path')}")
        print(f"[dataset] split_mode={effective_split_mode}")
        print(f"[dataset] internal_valid_source={internal_valid_source}")
        print(
            "[dataset][uci] "
            f"source_format={dataset_source_meta.get('source_format')} "
            f"split_mode={dataset_source_meta.get('split_mode')} "
            f"loaded_train_rows={int(len(predefined_splits['train']))} "
            f"loaded_valid_rows={int(len(predefined_splits['valid'])) if isinstance(predefined_splits.get('valid'), pd.DataFrame) else 0} "
            f"loaded_test_rows={int(len(predefined_splits['test']))} "
            f"target_column={source_target_col}"
        )
        print("[dataset][uci] schema_validation_passed=true")
    else:
        feature_df, dataset_name, id_column, source_target_col = _build_feature_table(dataset_cfg)
        if not dataset_name:
            raise ValueError("Resolved dataset name is empty after feature table construction.")
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Resolved dataset name '{dataset_name}' is unsupported. "
                f"Supported datasets: {sorted(SUPPORTED_DATASETS)}."
            )
        feature_df = feature_df.copy()
        feature_df, missing_value_meta = _drop_rows_with_missing_values(feature_df, exp_cfg.get("preprocessing", {}))
        feature_df = _prepare_feature_df_with_target(
            feature_df,
            dataset_name=dataset_name,
            source_target_col=source_target_col,
            formulation=formulation,
            target_mapping=target_mapping,
        )
        split_cfg = SplitConfig(
            test_size=float(exp_cfg["splits"]["test_size"]),
            validation_size=float(exp_cfg["splits"].get("validation_size", 0.2)),
            random_state=seed,
            stratify_column="target",
        )
        splits = stratified_train_valid_test_split(feature_df, split_cfg)

    preprocess_cfg = _prepare_preprocessing_config(
        exp_cfg,
        dataset_cfg,
        id_column=id_column,
        source_target_col=source_target_col,
    )
    if (
        bool(preprocess_cfg.get("onehot", False))
        and bool(preprocess_cfg.get("lock_category_vocabulary_from_pre_split_train", False))
        and isinstance(pre_split_train_feature_source_for_vocab, pd.DataFrame)
        and not pre_split_train_feature_source_for_vocab.empty
    ):
        locked_vocabulary = _build_locked_onehot_vocabulary(
            pre_split_train_feature_source_for_vocab,
            preprocess_cfg,
        )
        preprocess_cfg["onehot_categories"] = locked_vocabulary["categories"]
        preprocess_cfg["onehot_categories_source"] = locked_vocabulary["source"]
        print(
            "[preprocessing][onehot] "
            f"stable_locked_vocabulary_mode=true "
            f"vocabulary_source={locked_vocabulary['source']} "
            f"categorical_columns={locked_vocabulary['categorical_column_count']} "
            f"numerical_columns={locked_vocabulary['numeric_column_count']} "
            f"encoded_categorical_feature_count={locked_vocabulary['encoded_categorical_feature_count']} "
            f"preprocessed_feature_count={locked_vocabulary['preprocessed_feature_count']}"
        )
        print(
            "[preprocessing][onehot] "
            f"per_column_category_counts={json.dumps(locked_vocabulary['column_counts'], sort_keys=True)}"
        )
    artifacts = run_tabular_preprocessing(splits, preprocess_cfg)
    onehot_metadata = _resolve_onehot_metadata_and_validate(
        artifacts=artifacts,
        preprocess_cfg=preprocess_cfg,
        preprocessing_exp_cfg=(exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}),
    )
    if bool(onehot_metadata.get("onehot_enabled", False)):
        artifacts.metadata["locked_category_vocabulary_summary"] = {
            "stable_locked_vocabulary_mode": bool(onehot_metadata.get("stable_locked_vocabulary_mode", False)),
            "vocabulary_source": onehot_metadata.get("vocabulary_source"),
            "categorical_column_count": int(onehot_metadata.get("input_categorical_feature_count", 0)),
            "numerical_column_count": int(onehot_metadata.get("input_numeric_feature_count", 0)),
            "encoded_categorical_feature_count": int(onehot_metadata.get("encoded_categorical_feature_count", 0)),
            "total_final_feature_count": int(onehot_metadata.get("preprocessed_feature_count", 0)),
            "columns": [
                {
                    "column_name": str(column),
                    "category_count": int(count),
                    "category_labels": onehot_metadata.get("per_column_category_labels", {}).get(str(column), []),
                }
                for column, count in onehot_metadata.get("per_column_category_counts", {}).items()
            ],
        }
        print(
            "[preprocessing][onehot] "
            f"stable_locked_vocabulary_mode={str(onehot_metadata.get('stable_locked_vocabulary_mode', False)).lower()} "
            f"vocabulary_source={onehot_metadata.get('vocabulary_source')} "
            f"categorical_column_count={onehot_metadata.get('input_categorical_feature_count', 0)} "
            f"numerical_column_count={onehot_metadata.get('input_numeric_feature_count', 0)} "
            f"encoded_categorical_feature_count={onehot_metadata.get('encoded_categorical_feature_count', 0)} "
            f"preprocessed_feature_count={onehot_metadata.get('preprocessed_feature_count', 0)}"
        )

    outlier_cfg = _resolve_outlier_config(exp_cfg=exp_cfg, seed=seed)
    print(
        "[preprocessing][train_only] "
        f"outlier_enabled={bool(outlier_cfg.get('enabled', False))} "
        f"outlier_method={outlier_cfg.get('method', 'disabled')}"
    )
    outlier_applied_before_preprocessing = bool(outlier_cfg.get("apply_before_preprocessing", False))
    if outlier_applied_before_preprocessing:
        outlier_meta = dict(artifacts.metadata.get("train_only_outlier", {}))
        X_train_filtered = artifacts.X_train
        y_train_filtered = artifacts.y_train
    else:
        X_train_filtered, y_train_filtered, outlier_meta = apply_outlier_filter(
            artifacts.X_train,
            artifacts.y_train,
            outlier_cfg,
        )
    balancing_cfg = _resolve_balancing_config(exp_cfg=exp_cfg, seed=seed)
    print(
        "[preprocessing][train_only] "
        f"smote_enabled={bool(balancing_cfg.get('enabled', False))} "
        f"smote_method={balancing_cfg.get('method', 'disabled')}"
    )
    X_train_bal, y_train_bal, balancing_meta = apply_balancing(X_train_filtered, y_train_filtered, balancing_cfg)
    pre_outlier_source = (
        splits["train"]["target"]
        if outlier_applied_before_preprocessing and "train" in splits and "target" in splits["train"].columns
        else artifacts.y_train
    )
    pre_outlier_class_distribution = {
        str(k): int(v) for k, v in pre_outlier_source.value_counts(dropna=False).sort_index().to_dict().items()
    }
    post_outlier_class_distribution = {
        str(k): int(v) for k, v in y_train_filtered.value_counts(dropna=False).sort_index().to_dict().items()
    }

    model_candidates = _resolve_and_validate_model_candidates(exp_cfg)
    primary_metric = str(exp_cfg.get("metrics", {}).get("primary", "macro_f1"))
    if primary_metric.startswith(("test_", "valid_", "cv_")):
        metric_key = primary_metric
    else:
        metric_key = f"test_{primary_metric}"
    tuning_cfg = exp_cfg.get("models", {}).get("tuning", {})
    if not isinstance(tuning_cfg, dict):
        tuning_cfg = {}
    tuning_backend = str(tuning_cfg.get("backend", tuning_cfg.get("mode", "none"))).lower()
    default_n_trials = int(tuning_cfg.get("n_trials", 0))
    per_model_trial_budgets = _resolve_per_model_trial_budgets(tuning_cfg, model_candidates)
    tuning_enabled = tuning_backend == "optuna" and (default_n_trials > 0 or bool(per_model_trial_budgets))
    retrain_on_full_train_split = bool(exp_cfg.get("models", {}).get("retrain_on_full_train_split", False))
    class_weight_cfg = _resolve_class_weight_config(exp_cfg=exp_cfg, class_metadata=class_metadata)
    threshold_tuning_cfg = _resolve_threshold_tuning_config(exp_cfg)
    two_stage_decision_mode = _resolve_two_stage_decision_mode(
        experiment_mode=experiment_mode,
        two_stage_cfg=exp_cfg.get("two_stage", {}) if isinstance(exp_cfg.get("two_stage", {}), dict) else {},
    )
    two_stage_enabled = two_stage_decision_mode is not None
    if two_stage_enabled and (dataset_name != "uct_student" or formulation != "three_class"):
        raise ValueError(
            "two_stage modes are only supported for dataset=uct_student and target_formulation=three_class."
        )
    two_stage_cfg = exp_cfg.get("two_stage", {}) if isinstance(exp_cfg.get("two_stage", {}), dict) else {}
    stage2_feature_bundle = (
        _prepare_two_stage_stage2_feature_bundle(
            two_stage_cfg=two_stage_cfg,
            splits=splits,
            preprocess_cfg=preprocess_cfg,
        )
        if bool(two_stage_cfg.get("enabled", False))
        else {"enabled": False, "requested_groups": [], "report": {"enabled": False}}
    )
    resolved_stage2_decision_cfg = _resolve_two_stage_stage2_decision_config(two_stage_cfg)
    resolved_stage2_optuna_cfg = _resolve_two_stage_stage2_optuna_tuning_config(two_stage_cfg)
    resolved_two_stage_auto_balance_cfg = _resolve_two_stage_auto_balance_search_config(two_stage_cfg)
    resolved_two_stage_branch = (
        "two_stage_auto_balance"
        if bool(resolved_two_stage_auto_balance_cfg.get("enabled", False))
        else "two_stage_main"
    )
    two_stage_stage1_threshold_cfg = _resolve_two_stage_stage1_dropout_threshold_config(two_stage_cfg)
    threshold_stage1 = float(two_stage_stage1_threshold_cfg.get("dropout_threshold", 0.5))
    two_stage_calibration_cfg = _resolve_two_stage_calibration_config(two_stage_cfg)
    two_stage_class_thresholds: dict[int, float] = {}
    two_stage_threshold_tuning_cfg: dict[str, Any] = {
        "enabled": False,
        "metric": "macro_f1",
        "class_order": [],
        "class_grids": {},
        "default_thresholds": {},
        "max_candidates": 1500,
    }
    if two_stage_enabled:
        dropout_idx_cfg, enrolled_idx_cfg, graduate_idx_cfg = _resolve_uct_three_class_indices(class_metadata)
        two_stage_class_thresholds = _resolve_two_stage_soft_class_thresholds(
            two_stage_cfg=two_stage_cfg,
            class_metadata=class_metadata,
            dropout_idx=dropout_idx_cfg,
            enrolled_idx=enrolled_idx_cfg,
            graduate_idx=graduate_idx_cfg,
        )
        two_stage_threshold_tuning_cfg = _resolve_two_stage_threshold_tuning_config(
            two_stage_cfg=two_stage_cfg,
            class_metadata=class_metadata,
            dropout_idx=dropout_idx_cfg,
            enrolled_idx=enrolled_idx_cfg,
            graduate_idx=graduate_idx_cfg,
        )
        if str(two_stage_decision_mode).strip().lower() in {
            "soft_fusion_with_dropout_threshold",
            "soft_fusion_with_middle_band",
        }:
            two_stage_threshold_tuning_cfg = dict(two_stage_stage1_threshold_cfg)
    if two_stage_enabled:
        threshold_tuning_cfg = {
            "enabled": bool(two_stage_threshold_tuning_cfg.get("enabled", False)),
            "objective": str(two_stage_threshold_tuning_cfg.get("metric", "macro_f1")),
            "focus_class": "all_classes",
            "apply_on_test": True,
            "grid": list(two_stage_threshold_tuning_cfg.get("threshold_grid", [])),
            "reason": "managed_by_two_stage_fused_threshold_tuning",
        }
    training_cfg = exp_cfg.get("training", {}) if isinstance(exp_cfg.get("training", {}), dict) else {}
    training_mode = str(training_cfg.get("mode", "")).strip().lower()
    paper_reproduction_mode = training_mode in {"multiclass_paper_reproduction", "paper_reproduction_cv"}
    if paper_reproduction_mode and two_stage_enabled:
        raise ValueError("Paper reproduction mode requires flat single-stage multiclass; two_stage mode is not allowed.")
    cv_reporting_cfg = _resolve_cv_reporting_config(
        exp_cfg=exp_cfg,
        seed=seed,
        paper_reproduction_mode=paper_reproduction_mode,
    )
    model_selection_cfg = _resolve_model_selection_config(exp_cfg=exp_cfg)
    print(
        "[experiment][resolved] "
        f"dataset={resolved_dataset_token} "
        f"dataset_config={dataset_cfg_path} "
        f"target_formulation={formulation} "
        f"candidate_models={model_candidates}"
    )
    print(
        "[selection][resolved] "
        f"primary_selection_metric={model_selection_cfg.get('primary_selection_metric')} "
        f"ranking_metrics={model_selection_cfg.get('ranking_metrics', [])}"
    )
    print(
        "[cv][resolved] "
        f"enabled={bool(cv_reporting_cfg.get('enabled', False))} "
        f"folds={int(cv_reporting_cfg.get('n_splits', 0))} "
        f"optuna_trials_default={int((exp_cfg.get('models', {}).get('tuning', {}) if isinstance(exp_cfg.get('models', {}).get('tuning', {}), dict) else {}).get('n_trials', 0))}"
    )
    global_balance_guard_cfg = _resolve_global_balance_guard_config(exp_cfg=exp_cfg)
    if paper_reproduction_mode:
        threshold_tuning_cfg = {
            "enabled": False,
            "objective": "macro_f1",
            "focus_class": "enrolled",
            "apply_on_test": False,
            "grid": [],
            "reason": "disabled_for_paper_reproduction_cv",
        }
    decision_rule_cfg = _resolve_decision_rule_config(
        exp_cfg=exp_cfg,
        formulation=formulation,
        two_stage_enabled=two_stage_enabled,
        class_metadata=class_metadata,
    )
    decision_auto_tune_enabled = bool(
        isinstance(decision_rule_cfg.get("multiclass_decision", {}), dict)
        and isinstance(decision_rule_cfg.get("multiclass_decision", {}).get("auto_tune", {}), dict)
        and decision_rule_cfg.get("multiclass_decision", {}).get("auto_tune", {}).get("enabled", False)
    )
    if decision_auto_tune_enabled and bool(threshold_tuning_cfg.get("enabled", False)):
        raise ValueError(
            "Both evaluation.threshold_tuning and inference.multiclass_decision.auto_tune are enabled. "
            "Enable only one tuning mechanism per experiment to keep evaluation semantics explicit."
        )
    print(
        "[decision_policy] "
        f"strategy={decision_rule_cfg.get('decision_rule')} "
        f"params={decision_rule_cfg.get('multiclass_decision', {})} "
        f"override_reason={decision_rule_cfg.get('overridden_reason')}"
    )
    print(
        "[two_stage][resolved] "
        f"experiment_mode={experiment_mode} "
        f"two_stage_enabled={two_stage_enabled} "
        f"branch={resolved_two_stage_branch} "
        f"decision_mode={two_stage_decision_mode} "
        f"models={model_candidates} "
        f"stage2_tuning_enabled={bool(resolved_stage2_optuna_cfg.get('enabled', False))} "
        f"stage2_decision_enabled={bool(resolved_stage2_decision_cfg.get('enabled', False))} "
        f"stage2_decision_strategy={resolved_stage2_decision_cfg.get('strategy', 'argmax')} "
        f"auto_balance_enabled={bool(resolved_two_stage_auto_balance_cfg.get('enabled', False))}"
    )
    if two_stage_enabled:
        print(f"[two_stage][stage1][candidates] models={model_candidates}")
        print(f"[two_stage][stage2][candidates] models={model_candidates}")
    if two_stage_enabled:
        mode_result = run_two_stage_mode(
            experiment_mode=experiment_mode,
            exp_cfg=exp_cfg,
            dataset_cfg=dataset_cfg,
            model_candidates=model_candidates,
            formulation=formulation,
            class_metadata=class_metadata,
            decision_rule_cfg=decision_rule_cfg,
            tuning_cfg=tuning_cfg,
            tuning_backend=tuning_backend,
            default_n_trials=default_n_trials,
            per_model_trial_budgets=per_model_trial_budgets,
            retrain_on_full_train_split=retrain_on_full_train_split,
            class_weight_cfg=class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
            two_stage_decision_mode=str(two_stage_decision_mode),
            resolved_two_stage_branch=resolved_two_stage_branch,
            resolved_stage2_optuna_cfg=resolved_stage2_optuna_cfg,
            resolved_stage2_decision_cfg=resolved_stage2_decision_cfg,
            two_stage_cfg=two_stage_cfg,
            threshold_stage1=threshold_stage1,
            two_stage_class_thresholds=two_stage_class_thresholds,
            two_stage_threshold_tuning_cfg=two_stage_threshold_tuning_cfg,
            two_stage_calibration_cfg=two_stage_calibration_cfg,
            outlier_cfg=outlier_cfg,
            balancing_cfg=balancing_cfg,
            stage2_feature_bundle=stage2_feature_bundle,
            output_dir=output_dir,
            seed=seed,
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            artifacts=artifacts,
            param_overrides_cfg=param_overrides_cfg if isinstance(param_overrides_cfg, dict) else {},
            class_weight_requested_fn=_class_weight_requested,
            add_class_weight_metadata_metrics_fn=_add_class_weight_metadata_metrics,
            run_two_stage_uct_model_fn=_run_two_stage_uct_model,
        )
    else:
        mode_result = run_benchmark_mode(
            experiment_mode=experiment_mode,
            exp_cfg=exp_cfg,
            dataset_cfg=dataset_cfg,
            model_candidates=model_candidates,
            resolved_two_stage_branch=resolved_two_stage_branch,
            formulation=formulation,
            class_metadata=class_metadata,
            decision_rule_cfg=decision_rule_cfg,
            tuning_cfg=tuning_cfg,
            tuning_backend=tuning_backend,
            default_n_trials=default_n_trials,
            per_model_trial_budgets=per_model_trial_budgets,
            retrain_on_full_train_split=retrain_on_full_train_split,
            class_weight_cfg=class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
            threshold_tuning_cfg=threshold_tuning_cfg,
            paper_reproduction_mode=paper_reproduction_mode,
            cv_reporting_cfg=cv_reporting_cfg,
            model_selection_cfg=model_selection_cfg,
            param_overrides_cfg=param_overrides_cfg if isinstance(param_overrides_cfg, dict) else {},
            seed=seed,
            preprocess_cfg=preprocess_cfg,
            outlier_cfg=outlier_cfg,
            balancing_cfg=balancing_cfg,
            splits=splits,
            X_train_bal=X_train_bal,
            y_train_bal=y_train_bal,
            artifacts=artifacts,
            output_dir=output_dir,
            primary_metric=primary_metric,
            class_weight_requested_fn=_class_weight_requested,
            add_class_weight_metadata_metrics_fn=_add_class_weight_metadata_metrics,
            run_multiclass_decision_autotune_fn=_run_multiclass_decision_autotune,
            run_validation_threshold_tuning_fn=_run_validation_threshold_tuning,
        )

    model_results = mode_result["model_results"]
    leaderboard_rows = mode_result["leaderboard_rows"]
    trained_models = mode_result["trained_models"]
    optuna_artifacts = mode_result["optuna_artifacts"]
    model_decision_configs = mode_result["model_decision_configs"]
    runtime_artifact_overrides_by_model = mode_result["runtime_artifact_overrides_by_model"]
    stage2_feature_counts_by_model = mode_result["stage2_feature_counts_by_model"]
    stage2_advanced_reports_by_model = mode_result["stage2_advanced_reports_by_model"]
    successful_models = mode_result["successful_models"]
    failed_models = mode_result["failed_models"]

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    print(
        "[training][summary] "
        f"successful_candidates={len(successful_models)} "
        f"failed_candidates={len(failed_models)} "
        f"successful_models={successful_models}"
    )
    if leaderboard_df.empty:
        error_summary = {
            model_name: payload.get("error")
            for model_name, payload in model_results.items()
            if isinstance(payload, dict) and payload.get("error")
        }
        print(
            "[training][summary] "
            f"no_candidate_models_completed "
            f"requested_models={model_candidates} "
            f"errors={error_summary}"
        )
        if two_stage_enabled:
            print(
                "[two_stage][stage2][summary] "
                "no_stage2_candidate_completed_successfully "
                f"requested_models={model_candidates} "
                f"errors={error_summary}"
            )
        failure_artifacts = _write_benchmark_failure_summary(
            output_dir=output_dir,
            experiment_id=experiment_id,
            requested_models=model_candidates,
            model_results=model_results,
            failed_models=failed_models,
            successful_models=successful_models,
            reason="no_candidate_models_completed",
        )
        raise ValueError(
            "No candidate models completed successfully. "
            f"Requested models={model_candidates}. "
            f"Errors={error_summary}. "
            f"Failure summary artifacts={failure_artifacts}."
        )
    best_by_cv: dict[str, Any] = {"model": None, "ranking_columns": []}
    best_by_test: dict[str, Any] = {"model": None, "ranking_columns": []}
    global_balance_guard_report: dict[str, Any] = {"enabled": False}
    if model_selection_cfg.get("enabled", False):
        leaderboard_df, best_model_by_primary, _ = _sort_leaderboard_with_tiebreak(
            leaderboard_df=leaderboard_df,
            selection_cfg=model_selection_cfg,
            source="test",
        )
        if bool(global_balance_guard_cfg.get("enabled", False)):
            leaderboard_df, global_balance_guard_report = _apply_global_balance_guard(
                leaderboard_df=leaderboard_df,
                guard_cfg=global_balance_guard_cfg,
            )
            best_model_by_primary = str(leaderboard_df.iloc[0]["model"]) if not leaderboard_df.empty else None
        _, best_cv_model, cv_rank_cols = _sort_leaderboard_with_tiebreak(
            leaderboard_df=leaderboard_df,
            selection_cfg=model_selection_cfg,
            source="cv",
        )
        _, best_test_model, test_rank_cols = _sort_leaderboard_with_tiebreak(
            leaderboard_df=leaderboard_df,
            selection_cfg=model_selection_cfg,
            source="test",
        )
        if best_cv_model and cv_rank_cols:
            best_by_cv = {"model": best_cv_model, "ranking_columns": cv_rank_cols}
        if best_test_model and test_rank_cols:
            best_by_test = {"model": best_test_model, "ranking_columns": test_rank_cols}
        best_model = best_model_by_primary
        if not best_model and metric_key in leaderboard_df.columns:
            try:
                metric_sort = pd.to_numeric(leaderboard_df[metric_key], errors="coerce")
                leaderboard_df = leaderboard_df.assign(_metric_sort=metric_sort).sort_values(
                    by=["_metric_sort", "model"],
                    ascending=[False, True],
                    na_position="last",
                ).drop(columns=["_metric_sort"]).reset_index(drop=True)
                if not leaderboard_df.empty:
                    best_model = str(leaderboard_df.iloc[0]["model"])
            except Exception:
                pass
    else:
        if not leaderboard_df.empty and metric_key in leaderboard_df.columns:
            leaderboard_df = leaderboard_df.sort_values(metric_key, ascending=False).reset_index(drop=True)
            best_model = str(leaderboard_df.iloc[0]["model"])
        else:
            best_model = None

    if leaderboard_df.empty:
        print("[selection][summary] leaderboard is empty after training.")
    else:
        print(
            "[selection][summary] "
            f"leaderboard_rows={int(len(leaderboard_df))} "
            f"top_models={leaderboard_df['model'].astype(str).head(5).tolist()}"
        )
    if best_model:
        print(f"[selection][best_model] selected={best_model}")
    else:
        selection_reason = (
            f"metric_key_missing:{metric_key}" if metric_key not in leaderboard_df.columns else "selection_returned_none"
        )
        print(
            "[selection][best_model] "
            f"selected=None reason={selection_reason} "
            f"successful_models={successful_models}"
        )
        if successful_models:
            raise ValueError(
                "Best model could not be selected even though candidate models succeeded. "
                f"successful_models={successful_models}, metric_key={metric_key}, "
                f"leaderboard_columns={leaderboard_df.columns.tolist()}."
            )
    if best_by_cv.get("model") is not None:
        print(
            "[selection][best_by_cv] "
            f"selected={best_by_cv.get('model')} "
            f"ranking_columns={best_by_cv.get('ranking_columns', [])}"
        )

    execution = BenchmarkExecutionResult(
        model_results=model_results,
        leaderboard_df=leaderboard_df,
        trained_models=trained_models,
        optuna_artifacts=optuna_artifacts,
        model_decision_configs=model_decision_configs,
        runtime_artifact_overrides_by_model=runtime_artifact_overrides_by_model,
        stage2_feature_counts_by_model=stage2_feature_counts_by_model,
        stage2_advanced_reports_by_model=stage2_advanced_reports_by_model,
        successful_models=successful_models,
        failed_models=failed_models,
        best_model=best_model,
        best_by_cv=best_by_cv,
        best_by_test=best_by_test,
        global_balance_guard_report=global_balance_guard_report,
    )
    finalization_context = BenchmarkFinalizationContext(
        compact_summary=compact_summary,
        output_cfg=output_cfg,
        output_dir=output_dir,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        dataset_cfg_path=dataset_cfg_path,
        requested_dataset_token=requested_dataset_token,
        resolved_dataset_token=resolved_dataset_token,
        dataset_source_cfg=dataset_source_cfg,
        formulation=formulation,
        class_metadata=class_metadata,
        metric_key=metric_key,
        seed=seed,
        exp_cfg=exp_cfg,
        splits=splits,
        artifacts=artifacts,
        X_train_filtered=X_train_filtered,
        X_train_bal=X_train_bal,
        y_train_bal=y_train_bal,
        missing_value_meta=missing_value_meta,
        outlier_meta=outlier_meta,
        balancing_meta=balancing_meta,
        pre_outlier_class_distribution=pre_outlier_class_distribution,
        post_outlier_class_distribution=post_outlier_class_distribution,
        onehot_metadata=onehot_metadata,
        class_weight_cfg=class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
        threshold_tuning_cfg=threshold_tuning_cfg,
        cv_reporting_cfg=cv_reporting_cfg,
        decision_rule_cfg=decision_rule_cfg,
        two_stage_enabled=two_stage_enabled,
        two_stage_feature_bundle=stage2_feature_bundle,
        model_candidates=model_candidates,
        model_selection_cfg=model_selection_cfg,
        global_balance_guard_cfg=global_balance_guard_cfg,
        runtime_artifact_format=runtime_artifact_format,
        tuning_enabled=tuning_enabled,
        primary_metric=primary_metric,
        benchmark_summary_version=BENCHMARK_SUMMARY_VERSION,
        persist_paper_style_cv_artifacts_fn=_persist_paper_style_cv_artifacts,
        safe_filename_token_fn=_safe_filename_token,
    )
    return finalize_benchmark_run(context=finalization_context, execution=execution)

# CLI main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument(
        "--compact-summary",
        action="store_true",
        help="Write compact benchmark_summary.json by omitting heavy prediction/probability arrays.",
    )
    args = parser.parse_args()
    summary = run_experiment(args.experiment_config, compact_summary=args.compact_summary)
    print(f"Experiment: {summary['experiment_id']}")
    print(f"Best model: {summary.get('best_model')}")
    print(f"Primary metric: {summary.get('primary_metric')}")


if __name__ == "__main__":
    main()
