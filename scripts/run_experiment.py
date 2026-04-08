"""Run config-driven benchmark experiments for UCT Student and OULAD."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.adapters.oulad_adapter import adapt_oulad_schema
from src.data.adapters.uct_student_adapter import adapt_uct_student_schema
from src.data.feature_builders.oulad_paper_features import build_oulad_paper_features
from src.data.feature_builders.uct_student_features import build_uct_student_features
from src.data.loaders.oulad_loader import load_oulad_tables
from src.data.loaders.uct_student_loader import load_uct_student_dataframe
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
from src.models.two_stage_uct import TwoStageUct3ClassClassifier
from src.preprocessing.balancing import apply_balancing
from src.preprocessing.outlier import apply_outlier_filter
from src.preprocessing.tabular_pipeline import run_tabular_preprocessing
from src.reporting.artifact_manifest import update_artifact_manifest
from src.reporting.benchmark_contract import BENCHMARK_SUMMARY_VERSION
from src.reporting.benchmark_summary import save_benchmark_summary
from src.reporting.error_audit import run_uct_3class_error_audit
from src.reporting.generate_all_figures import generate_all_figures
from src.reporting.standard_artifacts import (
    ensure_standard_output_layout,
    resolve_results_dir,
    write_skipped_explainability_report,
)
from src.reporting.threshold_tuning import run_threshold_tuning_experiment

DATASET_NAME_ALIASES = {
    "uct": "uct_student",
    "uct_student": "uct_student",
    "uct-student": "uct_student",
    "oulad": "oulad",
    "open university learning analytics dataset": "oulad",
}
SUPPORTED_DATASETS = {"uct_student", "oulad"}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load experiment configs. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_experiment_config_schema(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept both native and lightweight analysis-only experiment schemas."""
    if "experiment" in exp_cfg:
        return exp_cfg

    experiment_name = exp_cfg.get("experiment_name")
    task_type = str(exp_cfg.get("task_type", "")).strip().lower()
    dataset_name = str(exp_cfg.get("dataset", {}).get("name", "")).strip().lower()
    input_cfg = exp_cfg.get("input", {})
    output_cfg = exp_cfg.get("output", {})
    analysis_cfg = exp_cfg.get("analysis", {})
    evaluation_cfg = exp_cfg.get("evaluation", {})
    preprocessing_cfg = exp_cfg.get("preprocessing", {})

    if not experiment_name:
        return exp_cfg

    mode = "benchmark"
    if task_type == "analysis_only":
        mode = "error_audit"
    elif task_type in {"threshold_tuning", "posthoc_threshold_tuning"}:
        mode = "threshold_tuning"
    target_mode = str(exp_cfg.get("target", {}).get("mode", "")).strip().lower()
    target_formulation = "binary"
    if "3class" in dataset_name or "three_class" in dataset_name or target_mode in {"3class", "three_class"}:
        target_formulation = "three_class"
    elif target_mode in {"4class", "four_class"}:
        target_formulation = "four_class"

    dataset_config = "configs/datasets/uct_student.yaml"
    if dataset_name == "oulad":
        dataset_config = "configs/datasets/oulad.yaml"

    model_aliases = {
        "xgboost_optuna": "xgboost",
        "lightgbm_optuna": "lightgbm",
        "catboost_optuna": "catboost",
    }
    raw_models = exp_cfg.get("models", [])
    candidates: list[str] = []
    if isinstance(raw_models, list):
        for model in raw_models:
            token = str(model).strip().lower()
            candidates.append(model_aliases.get(token, token.removesuffix("_optuna")))
    if not candidates:
        candidates = ["xgboost", "lightgbm", "catboost"]

    optimization_cfg = exp_cfg.get("optimization", {})
    threshold_tuning_cfg = exp_cfg.get("threshold_tuning", {})
    eval_metrics = [str(m) for m in evaluation_cfg.get("metrics", [])]
    objective_metric = str(optimization_cfg.get("objective_metric", "macro_f1"))
    if mode == "threshold_tuning":
        objective_metric = str(threshold_tuning_cfg.get("objective_metric", "macro_f1"))
    scoring_name = "f1_macro" if objective_metric == "macro_f1" else objective_metric
    secondary_metrics = [m for m in eval_metrics if m and m != objective_metric]
    split_cfg = exp_cfg.get("split", {})
    random_state = int(split_cfg.get("random_state", 42))

    output_results_dir = output_cfg.get("results_dir") or output_cfg.get("dir") or f"results/{experiment_name}"

    training_class_weight_cfg = (
        exp_cfg.get("training", {}).get("class_weight", {})
        if isinstance(exp_cfg.get("training", {}).get("class_weight", {}), dict)
        else {}
    )
    class_weight_payload: dict[str, Any] = {"enabled": bool(exp_cfg.get("training", {}).get("use_class_weights", False))}
    if training_class_weight_cfg:
        class_weight_payload = {"enabled": True, **training_class_weight_cfg}

    normalized: dict[str, Any] = {
        "experiment": {
            "id": str(experiment_name),
            "mode": mode,
            "seed": random_state,
            "dataset_config": dataset_config,
            "target_formulation": target_formulation,
        },
        "splits": {
            "test_size": 0.2,
            "validation_size": 0.2,
            "stratify_column": "target",
        },
        "preprocessing": {
            "imputation": preprocessing_cfg.get("imputation", "median_mode"),
            "encoding": preprocessing_cfg.get("encoding", "onehot"),
            "scaling": preprocessing_cfg.get("scaling", "standard"),
            "outlier": preprocessing_cfg.get("outlier", {"enabled": True, "method": "isolation_forest"}),
            "balancing": preprocessing_cfg.get("balancing", {"enabled": True, "method": "smote"}),
        },
        "inputs": {
            "benchmark_summary_path": input_cfg.get("benchmark_summary"),
            "benchmark_results_root": "results",
        },
        "analysis": {
            "top_k_models": analysis_cfg.get("top_k_models"),
            "models": analysis_cfg.get("models"),
            "threshold_grid": analysis_cfg.get("threshold_grid") or threshold_tuning_cfg.get("threshold_grid"),
            "split_source": split_cfg.get("source", "validation"),
        },
        "models": {
            "candidates": candidates,
            "tuning": {
                "backend": str(optimization_cfg.get("engine", "none")),
                "n_trials": int(optimization_cfg.get("n_trials", 0)),
                "cv_folds": 3,
                "scoring": scoring_name,
                "objective_source": "validation",
            },
            "class_weight": class_weight_payload,
            "retrain_on_full_train_split": True,
        },
        "metrics": {
            "primary": objective_metric,
            "secondary": secondary_metrics,
        },
        "evaluation": {
            "metrics": eval_metrics,
            "compare_default_argmax": bool(evaluation_cfg.get("compare_default_argmax", False)),
            "decision_rule": str(evaluation_cfg.get("decision_rule", exp_cfg.get("training", {}).get("decision_rule", "model_predict"))),
            "cross_validation": evaluation_cfg.get("cross_validation", {}),
            "save_confusion_matrix": bool(evaluation_cfg.get("save_confusion_matrix", True)),
            "save_classification_report": bool(evaluation_cfg.get("save_classification_report", True)),
        },
        "inference": exp_cfg.get("inference", {}) if isinstance(exp_cfg.get("inference", {}), dict) else {},
        "artifact_policy": exp_cfg.get("artifact_policy", {}),
        "outputs": {
            "results_dir": output_results_dir,
            "save_artifact_manifest": bool(output_cfg.get("save_artifact_manifest", True)),
        },
    }
    if mode == "threshold_tuning":
        # Preserve defaults in threshold-tuning runner if grid is not explicitly provided.
        normalized["analysis"]["models"] = analysis_cfg.get("models")
        normalized["analysis"]["threshold_grid"] = (
            analysis_cfg.get("threshold_grid")
            or threshold_tuning_cfg.get("threshold_grid")
            or {}
        )
    return normalized


def _save_dataframe(df: pd.DataFrame, preferred_path: Path) -> Path:
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_path.suffix.lower() == ".csv":
        df.to_csv(preferred_path, index=False)
        return preferred_path
    try:
        df.to_parquet(preferred_path, index=False)
        return preferred_path
    except Exception:
        fallback = preferred_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _save_series(series: pd.Series, preferred_path: Path) -> Path:
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    payload = pd.DataFrame({"value": series.reset_index(drop=True)})
    if preferred_path.suffix.lower() == ".csv":
        payload.to_csv(preferred_path, index=False)
        return preferred_path
    try:
        payload.to_parquet(preferred_path, index=False)
        return preferred_path
    except Exception:
        fallback = preferred_path.with_suffix(".csv")
        payload.to_csv(fallback, index=False)
        return fallback


def _persist_runtime_artifacts(
    output_dir: Path,
    best_model_name: str | None,
    trained_models: dict[str, Any],
    preprocessing_artifacts: Any,
    summary: dict[str, Any],
    file_format: str = "parquet",
) -> dict[str, str]:
    runtime_dir = output_dir / "runtime_artifacts"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}

    if best_model_name and best_model_name in trained_models:
        model_path = runtime_dir / "best_model.joblib"
        joblib.dump(trained_models[best_model_name], model_path)
        artifact_paths["best_model"] = str(model_path)

    transformer_path = runtime_dir / "preprocessing_transformer.joblib"
    transformer = preprocessing_artifacts.metadata.get("transformer")
    if transformer is not None:
        joblib.dump(transformer, transformer_path)
        artifact_paths["preprocessing_transformer"] = str(transformer_path)

    ext = ".csv" if str(file_format).strip().lower() == "csv" else ".parquet"
    X_train_path = _save_dataframe(preprocessing_artifacts.X_train, runtime_dir / f"X_train_preprocessed{ext}")
    X_valid_path = _save_dataframe(preprocessing_artifacts.X_valid, runtime_dir / f"X_valid_preprocessed{ext}")
    X_test_path = _save_dataframe(preprocessing_artifacts.X_test, runtime_dir / f"X_test_preprocessed{ext}")
    y_train_path = _save_series(preprocessing_artifacts.y_train, runtime_dir / f"y_train{ext}")
    y_valid_path = _save_series(preprocessing_artifacts.y_valid, runtime_dir / f"y_valid{ext}")
    y_test_path = _save_series(preprocessing_artifacts.y_test, runtime_dir / f"y_test{ext}")

    artifact_paths.update(
        {
            "X_train_preprocessed": str(X_train_path),
            "X_valid_preprocessed": str(X_valid_path),
            "X_test_preprocessed": str(X_test_path),
            "y_train": str(y_train_path),
            "y_valid": str(y_valid_path),
            "y_test": str(y_test_path),
        }
    )

    metadata_path = runtime_dir / "runtime_metadata.json"
    metadata_payload = {
        "experiment_id": summary.get("experiment_id"),
        "dataset_name": summary.get("dataset_name"),
        "target_formulation": summary.get("target_formulation"),
        "class_metadata": summary.get("class_metadata", {}),
        "best_model": best_model_name,
        "feature_names": preprocessing_artifacts.metadata.get("output_feature_names", []),
        "class_weight": summary.get("class_weight", {}),
        "threshold_tuning": summary.get("threshold_tuning", {}),
        "best_model_threshold_tuning": (
            summary.get("model_results", {}).get(best_model_name, {}).get("threshold_tuning", {})
            if best_model_name
            else {}
        ),
        "model_mechanism_audit": summary.get("model_mechanism_audit", {}),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    artifact_paths["runtime_metadata"] = str(metadata_path)
    return artifact_paths


def _persist_required_contract_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    best_model_name: str | None,
    trained_models: dict[str, Any],
    y_test: pd.Series,
    class_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist model/, metrics.json, and predictions.csv required by collaboration contract."""
    paths: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if best_model_name and best_model_name in trained_models and trained_models[best_model_name] is not None:
        model_path = model_dir / "best_model.joblib"
        joblib.dump(trained_models[best_model_name], model_path)
        paths["model_dir"] = str(model_dir)
        paths["best_model_copy"] = str(model_path)
    else:
        paths["model_dir"] = str(model_dir)

    metrics_payload = {
        "experiment_id": summary.get("experiment_id"),
        "dataset_name": summary.get("dataset_name"),
        "target_formulation": summary.get("target_formulation"),
        "primary_metric": summary.get("primary_metric"),
        "best_model": best_model_name,
        "class_weight": summary.get("class_weight", {}),
        "threshold_tuning": summary.get("threshold_tuning", {}),
        "best_model_threshold_tuning": (
            summary.get("model_results", {}).get(best_model_name, {}).get("threshold_tuning", {})
            if best_model_name
            else {}
        ),
        "model_mechanism_audit": summary.get("model_mechanism_audit", {}),
        "best_model_metrics": (
            summary.get("model_results", {}).get(best_model_name, {}).get("metrics", {}) if best_model_name else {}
        ),
        "leaderboard": summary.get("leaderboard", []),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    paths["metrics"] = str(metrics_path)

    if best_model_name:
        best_payload = summary.get("model_results", {}).get(best_model_name, {})
        best_artifacts = best_payload.get("artifacts", {})
        y_pred = best_artifacts.get("y_pred_test")
        y_proba = best_artifacts.get("y_proba_test")
        labels = best_artifacts.get("labels") or []
        pred_df = _build_prediction_export_dataframe(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=labels,
            class_metadata=class_metadata,
            extra_columns=best_artifacts.get("prediction_export_test"),
        )
    else:
        pred_df = pd.DataFrame({"y_true": y_test.reset_index(drop=True)})

    predictions_path = output_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    paths["predictions"] = str(predictions_path)
    return paths


def _status_from_path(path: str | Path, missing_reason: str = "missing_expected_output") -> dict[str, str]:
    resolved = Path(path)
    if resolved.exists():
        return {"status": "generated", "path": str(resolved)}
    return {"status": "failed", "path": str(resolved), "reason": missing_reason}


def _mirror_root_artifacts_to_runtime(
    output_dir: Path,
    runtime_dir: Path,
    model_candidates: list[str],
) -> dict[str, str]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    mirrored: dict[str, str] = {}

    static_files = (
        "artifact_manifest.json",
        "benchmark_summary.json",
        "benchmark_summary.md",
        "leaderboard.csv",
        "summary.csv",
        "metrics.json",
        "predictions.csv",
    )
    for filename in static_files:
        source = output_dir / filename
        if source.exists():
            destination = runtime_dir / filename
            shutil.copy2(source, destination)
            mirrored[filename] = str(destination)

    for model_name in model_candidates:
        cm_file = output_dir / f"confusion_matrix_{model_name}.png"
        if cm_file.exists():
            destination = runtime_dir / cm_file.name
            shutil.copy2(cm_file, destination)
            mirrored[cm_file.name] = str(destination)

        cm_norm_file = output_dir / f"confusion_matrix_{model_name}_normalized.png"
        if cm_norm_file.exists():
            destination = runtime_dir / cm_norm_file.name
            shutil.copy2(cm_norm_file, destination)
            mirrored[cm_norm_file.name] = str(destination)

    return mirrored


def _persist_per_model_run_outputs(
    output_dir: Path,
    run_stamp: str,
    model_results: dict[str, Any],
    trained_models: dict[str, Any],
    class_metadata: dict[str, Any],
) -> dict[str, str]:
    per_model_paths: dict[str, str] = {}
    for model_name, payload in model_results.items():
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        model_dir = output_dir / f"run_{run_stamp}_{_safe_filename_token(model_name)}"
        model_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = model_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload.get("metrics", {}), indent=2), encoding="utf-8")

        report_payload = artifacts.get("classification_report_test")
        if report_payload is None and artifacts.get("y_true_test") is not None and artifacts.get("y_pred_test") is not None:
            report_payload = classification_report(
                artifacts.get("y_true_test"),
                artifacts.get("y_pred_test"),
                labels=artifacts.get("labels"),
                output_dict=True,
                zero_division=0,
            )
        if report_payload is not None:
            report_path = model_dir / "classification_report.json"
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        cm = artifacts.get("confusion_matrix")
        if cm is not None:
            cm_df = pd.DataFrame(cm)
            cm_df.to_csv(model_dir / "confusion_matrix.csv", index=False)

        if artifacts.get("y_true_test") is not None:
            prediction_df = _build_prediction_export_dataframe(
                y_true=pd.Series(artifacts.get("y_true_test")),
                y_pred=artifacts.get("y_pred_test"),
                y_proba=artifacts.get("y_proba_test"),
                labels=artifacts.get("labels"),
                class_metadata=class_metadata,
                extra_columns=artifacts.get("prediction_export_test"),
            )
            prediction_df.to_csv(model_dir / "predictions.csv", index=False)

        for key, filename in (
            ("stage1_metrics", "stage1_metrics.json"),
            ("stage2_metrics", "stage2_metrics.json"),
            ("two_stage_diagnostics", "two_stage_diagnostics.json"),
            ("middle_band_diagnostics", "middle_band_diagnostics.json"),
        ):
            extra_payload = artifacts.get(key)
            if isinstance(extra_payload, dict) and extra_payload:
                (model_dir / filename).write_text(json.dumps(extra_payload, indent=2), encoding="utf-8")

        threshold_rows = artifacts.get("threshold_tuning_results")
        if threshold_rows:
            threshold_df = pd.DataFrame(threshold_rows)
            threshold_df.to_csv(model_dir / "threshold_tuning_results.csv", index=False)
            threshold_df.to_csv(model_dir / "threshold_search_results.csv", index=False)
        selected_threshold = artifacts.get("selected_threshold")
        if isinstance(selected_threshold, dict) and selected_threshold:
            (model_dir / "selected_threshold.json").write_text(json.dumps(selected_threshold, indent=2), encoding="utf-8")

        cv_payload = payload.get("cv_results")
        if isinstance(cv_payload, dict) and cv_payload:
            (model_dir / "cv_results.json").write_text(json.dumps(cv_payload, indent=2), encoding="utf-8")

        optuna_payload = payload.get("optuna_summary")
        if isinstance(optuna_payload, dict) and optuna_payload:
            (model_dir / "optuna_summary.json").write_text(json.dumps(optuna_payload, indent=2), encoding="utf-8")

        trained_model = trained_models.get(model_name)
        if trained_model is not None:
            model_path = model_dir / "model.joblib"
            joblib.dump(trained_model, model_path)

        per_model_paths[f"run_{_safe_filename_token(model_name)}"] = str(model_dir)
    return per_model_paths


def _normalize_dataset_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def _safe_filename_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_").lower()


def _metric_label_token(raw_label: str) -> str:
    value = str(raw_label).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _resolve_decision_rule_config(
    exp_cfg: dict[str, Any],
    formulation: str,
    two_stage_enabled: bool,
    class_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    training_cfg = exp_cfg.get("training", {}) if isinstance(exp_cfg.get("training", {}), dict) else {}
    inference_cfg = exp_cfg.get("inference", {}) if isinstance(exp_cfg.get("inference", {}), dict) else {}
    multiclass_decision_cfg = (
        inference_cfg.get("multiclass_decision", {})
        if isinstance(inference_cfg.get("multiclass_decision", {}), dict)
        else {}
    )

    training_mode = str(training_cfg.get("mode", "")).strip().lower()
    requested_rule_legacy = str(
        evaluation_cfg.get("decision_rule", training_cfg.get("decision_rule", "model_predict"))
    ).strip().lower()
    requested_strategy = str(multiclass_decision_cfg.get("strategy", "")).strip().lower()
    if requested_strategy:
        requested_rule = requested_strategy
    elif training_mode in {"multiclass_argmax", "paper_multiclass_argmax"}:
        requested_rule = "argmax"
    else:
        requested_rule = requested_rule_legacy

    supported_rules = {"model_predict", "argmax", "enrolled_margin", "enrolled_middle_band", "enrolled_push"}
    if requested_rule not in supported_rules:
        raise ValueError(
            "Unsupported decision rule strategy. Use evaluation.decision_rule in "
            "{'model_predict','argmax'} or inference.multiclass_decision.strategy in "
            "{'argmax','enrolled_margin','enrolled_middle_band','enrolled_push'} "
            f"(got '{requested_rule}')."
        )

    label_indices = set()
    if isinstance(class_metadata, dict):
        raw_indices = class_metadata.get("class_indices", [])
        if isinstance(raw_indices, list):
            label_indices = {int(v) for v in raw_indices}

    strategy_params: dict[str, Any] = {"strategy": requested_rule}
    if requested_rule == "enrolled_margin":
        if formulation != "three_class":
            raise ValueError(
                "inference.multiclass_decision.strategy='enrolled_margin' is only supported for "
                "target_formulation='three_class'."
            )
        if label_indices and label_indices != {0, 1, 2}:
            raise ValueError(
                "strategy='enrolled_margin' requires class indices {0,1,2} "
                "(dropout/enrolled/graduate) in class metadata."
            )
        margin_threshold_raw = multiclass_decision_cfg.get("enrolled_margin_threshold", 0.10)
        margin_threshold = float(margin_threshold_raw)
        if margin_threshold < 0.0 or margin_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.enrolled_margin_threshold must be within [0.0, 1.0].")
        strategy_params["enrolled_margin_threshold"] = margin_threshold

    if requested_rule == "enrolled_middle_band":
        if formulation != "three_class":
            raise ValueError(
                "inference.multiclass_decision.strategy='enrolled_middle_band' is only supported for "
                "target_formulation='three_class'."
            )
        if label_indices and label_indices != {0, 1, 2}:
            raise ValueError(
                "strategy='enrolled_middle_band' requires class indices {0,1,2} "
                "(dropout/enrolled/graduate) in class metadata."
            )
        dropout_threshold_raw = multiclass_decision_cfg.get("dropout_threshold", 0.55)
        graduate_threshold_raw = multiclass_decision_cfg.get("graduate_threshold", 0.55)
        dropout_threshold = float(dropout_threshold_raw)
        graduate_threshold = float(graduate_threshold_raw)
        if dropout_threshold < 0.0 or dropout_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.dropout_threshold must be within [0.0, 1.0].")
        if graduate_threshold < 0.0 or graduate_threshold > 1.0:
            raise ValueError("inference.multiclass_decision.graduate_threshold must be within [0.0, 1.0].")
        strategy_params["dropout_threshold"] = dropout_threshold
        strategy_params["graduate_threshold"] = graduate_threshold

    if requested_rule == "enrolled_push":
        if formulation != "three_class":
            raise ValueError(
                "inference.multiclass_decision.strategy='enrolled_push' is only supported for "
                "target_formulation='three_class'."
            )
        if label_indices and label_indices != {0, 1, 2}:
            raise ValueError(
                "strategy='enrolled_push' requires class indices {0,1,2} "
                "(dropout/enrolled/graduate) in class metadata."
            )
        enrolled_class_name = str(multiclass_decision_cfg.get("enrolled_class_name", "Enrolled")).strip() or "Enrolled"
        threshold_block = multiclass_decision_cfg.get("enrolled_probability_threshold", {})
        if not isinstance(threshold_block, dict):
            threshold_block = {"enabled": threshold_block is not None, "value": threshold_block}
        threshold_enabled = bool(threshold_block.get("enabled", False))
        threshold_value_raw = threshold_block.get("value", threshold_block.get("threshold", 0.40))
        threshold_payload: dict[str, Any] = {"enabled": threshold_enabled}
        if threshold_enabled:
            threshold_value = float(threshold_value_raw)
            if threshold_value < 0.0 or threshold_value > 1.0:
                raise ValueError("inference.multiclass_decision.enrolled_probability_threshold must be within [0.0, 1.0].")
            threshold_payload["value"] = threshold_value

        middle_band_raw = multiclass_decision_cfg.get("enrolled_middle_band", {})
        if not isinstance(middle_band_raw, dict):
            middle_band_raw = {}
        middle_band_enabled = bool(middle_band_raw.get("enabled", False))
        middle_band_payload: dict[str, Any] = {"enabled": middle_band_enabled}
        if middle_band_enabled:
            min_prob = float(middle_band_raw.get("min_enrolled_prob", 0.30))
            max_gap = float(middle_band_raw.get("max_top2_gap", 0.05))
            if min_prob < 0.0 or min_prob > 1.0:
                raise ValueError("inference.multiclass_decision.enrolled_middle_band.min_enrolled_prob must be within [0.0, 1.0].")
            if max_gap < 0.0 or max_gap > 1.0:
                raise ValueError("inference.multiclass_decision.enrolled_middle_band.max_top2_gap must be within [0.0, 1.0].")
            middle_band_payload.update(
                {
                    "min_enrolled_prob": min_prob,
                    "max_top2_gap": max_gap,
                }
            )
        strategy_params.update(
            {
                "enrolled_class_name": enrolled_class_name,
                "fallback": str(multiclass_decision_cfg.get("fallback", "argmax")).strip().lower() or "argmax",
                "enrolled_probability_threshold": threshold_payload,
                "enrolled_middle_band": middle_band_payload,
            }
        )

    if requested_rule == "argmax":
        # Always keep strategy payload explicit for downstream auditing and summaries.
        strategy_params = {"strategy": "argmax"}
    if requested_rule == "model_predict":
        strategy_params = {"strategy": "model_predict"}

    auto_tune_cfg_raw = multiclass_decision_cfg.get("auto_tune", {})
    auto_tune_cfg = auto_tune_cfg_raw if isinstance(auto_tune_cfg_raw, dict) else {}
    auto_tune_enabled = bool(auto_tune_cfg.get("enabled", False))
    if auto_tune_enabled:
        if requested_rule not in {"enrolled_margin", "enrolled_middle_band", "enrolled_push"}:
            raise ValueError(
                "inference.multiclass_decision.auto_tune.enabled=true is only supported for "
                "strategies {'enrolled_margin','enrolled_middle_band','enrolled_push'}."
            )
        objective = str(auto_tune_cfg.get("objective", "macro_f1")).strip().lower()
        if objective not in {"macro_f1", "enrolled_f1", "enrolled_recall", "balanced_accuracy"}:
            raise ValueError(
                "inference.multiclass_decision.auto_tune.objective must be one of "
                "{'macro_f1','enrolled_f1','enrolled_recall','balanced_accuracy'}."
            )
        split_name = str(auto_tune_cfg.get("split", "validation")).strip().lower()
        if split_name != "validation":
            raise ValueError("inference.multiclass_decision.auto_tune.split must be 'validation'.")
        search_cfg = auto_tune_cfg.get("search", {}) if isinstance(auto_tune_cfg.get("search", {}), dict) else {}
        search_method = str(search_cfg.get("method", "grid")).strip().lower()
        if search_method != "grid":
            raise ValueError("inference.multiclass_decision.auto_tune.search.method must be 'grid'.")
        if requested_rule == "enrolled_margin":
            grid_vals = search_cfg.get("enrolled_margin_thresholds", [])
            if not isinstance(grid_vals, list) or len(grid_vals) == 0:
                raise ValueError(
                    "inference.multiclass_decision.auto_tune.search.enrolled_margin_thresholds "
                    "must be a non-empty list."
                )
            for raw in grid_vals:
                val = float(raw)
                if val < 0.0 or val > 1.0:
                    raise ValueError("enrolled_margin_threshold search values must be within [0.0, 1.0].")
        if requested_rule == "enrolled_middle_band":
            drop_vals = search_cfg.get("dropout_thresholds", [])
            grad_vals = search_cfg.get("graduate_thresholds", [])
            if not isinstance(drop_vals, list) or len(drop_vals) == 0:
                raise ValueError(
                    "inference.multiclass_decision.auto_tune.search.dropout_thresholds must be a non-empty list."
                )
            if not isinstance(grad_vals, list) or len(grad_vals) == 0:
                raise ValueError(
                    "inference.multiclass_decision.auto_tune.search.graduate_thresholds must be a non-empty list."
                )
            for raw in [*drop_vals, *grad_vals]:
                val = float(raw)
                if val < 0.0 or val > 1.0:
                    raise ValueError("middle-band search threshold values must be within [0.0, 1.0].")
        if requested_rule == "enrolled_push":
            threshold_cfg = strategy_params.get("enrolled_probability_threshold", {})
            middle_band_cfg = strategy_params.get("enrolled_middle_band", {})
            if bool(threshold_cfg.get("enabled", False)):
                threshold_search = search_cfg.get("enrolled_probability_thresholds", [])
                if not isinstance(threshold_search, list) or len(threshold_search) == 0:
                    raise ValueError(
                        "inference.multiclass_decision.auto_tune.search.enrolled_probability_thresholds "
                        "must be a non-empty list when enrolled_probability_threshold is enabled."
                    )
                for raw in threshold_search:
                    val = float(raw)
                    if val < 0.0 or val > 1.0:
                        raise ValueError("enrolled_probability_threshold search values must be within [0.0, 1.0].")
            if bool(middle_band_cfg.get("enabled", False)):
                min_prob_search = search_cfg.get("min_enrolled_probs", [])
                max_gap_search = search_cfg.get("max_top2_gaps", [])
                if not isinstance(min_prob_search, list) or len(min_prob_search) == 0:
                    raise ValueError(
                        "inference.multiclass_decision.auto_tune.search.min_enrolled_probs must be a non-empty list "
                        "when enrolled_middle_band is enabled."
                    )
                if not isinstance(max_gap_search, list) or len(max_gap_search) == 0:
                    raise ValueError(
                        "inference.multiclass_decision.auto_tune.search.max_top2_gaps must be a non-empty list "
                        "when enrolled_middle_band is enabled."
                    )
                for raw in [*min_prob_search, *max_gap_search]:
                    val = float(raw)
                    if val < 0.0 or val > 1.0:
                        raise ValueError("enrolled_push middle-band search values must be within [0.0, 1.0].")
        strategy_params["auto_tune"] = {
            "enabled": True,
            "objective": objective,
            "split": split_name,
            "search": {
                **search_cfg,
                "method": "grid",
            },
        }
    else:
        strategy_params["auto_tune"] = {"enabled": False}

    if two_stage_enabled and requested_rule != "model_predict":
        return {
            "decision_rule": "model_predict",
            "requested_decision_rule": requested_rule,
            "requested_decision_rule_legacy": requested_rule_legacy,
            "multiclass_decision": {"strategy": "model_predict"},
            "overridden_reason": "two_stage_runner_controls_decision_logic",
            "training_mode": training_mode,
            "target_formulation": formulation,
        }
    return {
        "decision_rule": requested_rule,
        "requested_decision_rule": requested_rule,
        "requested_decision_rule_legacy": requested_rule_legacy,
        "multiclass_decision": strategy_params,
        "overridden_reason": None,
        "training_mode": training_mode,
        "target_formulation": formulation,
    }


def _resolve_cv_reporting_config(
    exp_cfg: dict[str, Any],
    seed: int,
    paper_reproduction_mode: bool,
) -> dict[str, Any]:
    evaluation_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    cv_cfg = evaluation_cfg.get("cross_validation", {})
    if not isinstance(cv_cfg, dict):
        cv_cfg = {}

    enabled = bool(cv_cfg.get("enabled", False))
    if paper_reproduction_mode:
        enabled = True

    n_splits = int(cv_cfg.get("n_splits", cv_cfg.get("folds", 5)))
    n_splits = max(2, n_splits)
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", seed))

    return {
        "enabled": enabled,
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
        "optuna_objective_metric": "cv_macro_f1_mean",
    }


def _resolve_model_selection_config(exp_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = exp_cfg.get("selection", {}) if isinstance(exp_cfg.get("selection", {}), dict) else {}
    enabled = bool(raw_cfg)
    primary = str(raw_cfg.get("primary", "macro_f1")).strip()
    secondary = str(raw_cfg.get("secondary", "balanced_accuracy")).strip()
    tertiary = str(raw_cfg.get("tertiary", "accuracy")).strip()
    tie_breakers_raw = raw_cfg.get("tie_breakers", [])
    tie_breakers: list[str] = []
    if isinstance(tie_breakers_raw, list):
        tie_breakers = [str(metric).strip() for metric in tie_breakers_raw if str(metric).strip()]
    ranking_metrics = [metric for metric in [primary or "macro_f1", *tie_breakers] if metric]
    if len(ranking_metrics) == 1:
        for fallback in [secondary or "balanced_accuracy", tertiary or "accuracy"]:
            if fallback and fallback not in ranking_metrics:
                ranking_metrics.append(fallback)
    return {
        "enabled": enabled,
        "primary": primary or "macro_f1",
        "secondary": secondary or "balanced_accuracy",
        "tertiary": tertiary or "accuracy",
        "tie_breakers": tie_breakers,
        "ranking_metrics": ranking_metrics,
    }


def _resolve_experiment_feature_config(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(dataset_cfg)
    experiment_feature_cfg = (
        exp_cfg.get("feature_engineering", {})
        if isinstance(exp_cfg.get("feature_engineering", {}), dict)
        else {}
    )
    if not experiment_feature_cfg:
        return resolved
    dataset_feature_overrides = (
        experiment_feature_cfg.get("dataset_features", {})
        if isinstance(experiment_feature_cfg.get("dataset_features", {}), dict)
        else {}
    )
    if dataset_feature_overrides:
        resolved["features"] = _deep_merge_dicts(
            resolved.get("features", {}) if isinstance(resolved.get("features", {}), dict) else {},
            dataset_feature_overrides,
        )
    return resolved


def _resolve_per_model_trial_budgets(
    tuning_cfg: dict[str, Any],
    model_candidates: list[str],
) -> dict[str, int]:
    if not isinstance(tuning_cfg, dict):
        return {}
    raw_overrides = (
        tuning_cfg.get("per_model_n_trials")
        or tuning_cfg.get("per_model_optuna")
        or tuning_cfg.get("tuning_overrides")
        or {}
    )
    if not isinstance(raw_overrides, dict):
        return {}

    resolved: dict[str, int] = {}
    candidate_set = set(model_candidates)
    for raw_model_name, raw_value in raw_overrides.items():
        model_name = str(raw_model_name).strip().lower()
        if model_name not in candidate_set:
            continue
        try:
            trials = int(raw_value)
        except (TypeError, ValueError):
            continue
        if trials > 0:
            resolved[model_name] = trials
    return resolved


def _resolve_onehot_metadata_and_validate(
    artifacts: Any,
    preprocess_cfg: dict[str, Any],
    preprocessing_exp_cfg: dict[str, Any],
) -> dict[str, Any]:
    metadata = artifacts.metadata if hasattr(artifacts, "metadata") and isinstance(artifacts.metadata, dict) else {}
    transformer = metadata.get("transformer")
    categorical_columns = metadata.get("categorical_columns", [])
    if not isinstance(categorical_columns, list):
        categorical_columns = []
    encoded_categorical_feature_count = 0

    if bool(preprocess_cfg.get("onehot", False)) and transformer is not None:
        try:
            cat_pipeline = transformer.named_transformers_.get("cat")  # type: ignore[attr-defined]
            encoder = cat_pipeline.named_steps.get("encoder") if hasattr(cat_pipeline, "named_steps") else None
            categories = getattr(encoder, "categories_", None)
            if categories is not None:
                encoded_categorical_feature_count = int(sum(len(v) for v in categories))
        except Exception:
            encoded_categorical_feature_count = 0

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
        "input_categorical_feature_count": int(len(categorical_columns)),
        "encoded_categorical_feature_count": int(encoded_categorical_feature_count),
        "preprocessed_feature_count": int(artifacts.X_train.shape[1]),
    }


def _resolve_metric_column(metric_name: str, source: str) -> str:
    token = str(metric_name).strip()
    alias_map = {
        "enrolled_f1": "f1_enrolled",
        "enrolled_recall": "recall_enrolled",
        "enrolled_precision": "precision_enrolled",
    }
    token = alias_map.get(token.lower(), token)
    if token.startswith(("test_", "valid_", "cv_")):
        return token
    if source == "cv":
        return f"cv_{token}_mean"
    return f"test_{token}"


def _sort_leaderboard_with_tiebreak(
    leaderboard_df: pd.DataFrame,
    selection_cfg: dict[str, Any],
    source: str,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    if leaderboard_df.empty:
        return leaderboard_df, None, []
    ranking_columns = [
        _resolve_metric_column(metric_name, source=source)
        for metric_name in selection_cfg.get("ranking_metrics", [])
    ]
    ranking_columns = [c for c in ranking_columns if c in leaderboard_df.columns]
    if not ranking_columns:
        return leaderboard_df, None, []

    ranked = leaderboard_df.copy()
    for col in ranking_columns:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
    ranked = ranked.sort_values(
        by=[*ranking_columns, "model"],
        ascending=[False] * len(ranking_columns) + [True],
        na_position="last",
    ).reset_index(drop=True)
    best_model = str(ranked.iloc[0]["model"]) if not ranked.empty else None
    return ranked, best_model, ranking_columns


def _resolve_model_decision_rule_config(
    exp_cfg: dict[str, Any],
    base_decision_rule_cfg: dict[str, Any],
    model_name: str,
    formulation: str,
    two_stage_enabled: bool,
    class_metadata: dict[str, Any],
) -> dict[str, Any]:
    inference_cfg = exp_cfg.get("inference", {}) if isinstance(exp_cfg.get("inference", {}), dict) else {}
    multiclass_cfg = (
        inference_cfg.get("multiclass_decision", {})
        if isinstance(inference_cfg.get("multiclass_decision", {}), dict)
        else {}
    )
    per_model_cfg = multiclass_cfg.get("per_model", {}) if isinstance(multiclass_cfg.get("per_model", {}), dict) else {}
    raw_override = per_model_cfg.get(model_name, {})
    if not isinstance(raw_override, dict) or not raw_override:
        return copy.deepcopy(base_decision_rule_cfg)

    override_enabled = raw_override.get("enabled")
    if override_enabled is False:
        temp_cfg = copy.deepcopy(exp_cfg)
        temp_cfg.setdefault("evaluation", {})
        temp_cfg["evaluation"]["decision_rule"] = "argmax"
        temp_cfg.setdefault("inference", {})
        temp_cfg["inference"]["multiclass_decision"] = {"strategy": "argmax"}
        return _resolve_decision_rule_config(
            exp_cfg=temp_cfg,
            formulation=formulation,
            two_stage_enabled=two_stage_enabled,
            class_metadata=class_metadata,
        )

    model_override = {k: v for k, v in raw_override.items() if k != "enabled"}
    if not model_override:
        return copy.deepcopy(base_decision_rule_cfg)

    temp_cfg = copy.deepcopy(exp_cfg)
    temp_cfg.setdefault("evaluation", {})
    temp_cfg.setdefault("inference", {})
    base_multiclass_cfg = copy.deepcopy(multiclass_cfg)
    base_multiclass_cfg.pop("per_model", None)
    merged_multiclass_cfg = _deep_merge_dicts(base_multiclass_cfg, model_override)
    temp_cfg["inference"]["multiclass_decision"] = merged_multiclass_cfg
    if "decision_rule" in model_override:
        temp_cfg["evaluation"]["decision_rule"] = str(model_override.get("decision_rule", "argmax"))
    elif "strategy" in merged_multiclass_cfg:
        temp_cfg["evaluation"]["decision_rule"] = str(merged_multiclass_cfg.get("strategy", "argmax"))

    return _resolve_decision_rule_config(
        exp_cfg=temp_cfg,
        formulation=formulation,
        two_stage_enabled=two_stage_enabled,
        class_metadata=class_metadata,
    )


def _build_prediction_export_dataframe(
    y_true: pd.Series,
    y_pred: Any,
    y_proba: Any,
    labels: list[Any] | None,
    class_metadata: dict[str, Any] | None = None,
    extra_columns: dict[str, Any] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.reset_index(drop=True)})
    if y_pred is not None:
        df["y_pred"] = np.asarray(y_pred)

    index_to_label = (
        (class_metadata or {}).get("class_index_to_label", {})
        if isinstance(class_metadata, dict)
        else {}
    )
    if index_to_label:
        df["true_label"] = df["y_true"].map(lambda value: index_to_label.get(str(int(value)), value))
        if "y_pred" in df.columns:
            df["pred_label"] = df["y_pred"].map(lambda value: index_to_label.get(str(int(value)), value))

    if y_proba is None:
        if extra_columns is not None:
            extra_df = extra_columns if isinstance(extra_columns, pd.DataFrame) else pd.DataFrame(extra_columns)
            df = pd.concat([df, extra_df.reset_index(drop=True)], axis=1)
        return df
    probs = np.asarray(y_proba, dtype=float)
    if probs.ndim != 2:
        return df

    ordered_labels = list(labels or [])
    if not ordered_labels:
        ordered_labels = list(range(probs.shape[1]))
    if probs.shape[1] != len(ordered_labels):
        raise ValueError(
            "Probability export failed because label/probability dimensions differ: "
            f"n_labels={len(ordered_labels)}, proba_shape={probs.shape}."
        )

    for idx in range(probs.shape[1]):
        class_idx = ordered_labels[idx] if idx < len(ordered_labels) else idx
        label_name = index_to_label.get(str(class_idx), class_idx)
        token = _metric_label_token(str(label_name))
        df[f"proba_class_{class_idx}"] = probs[:, idx]
        if token:
            df[f"prob_{token}"] = probs[:, idx]
    if extra_columns is not None:
        extra_df = extra_columns if isinstance(extra_columns, pd.DataFrame) else pd.DataFrame(extra_columns)
        df = pd.concat([df, extra_df.reset_index(drop=True)], axis=1)
    return df


def _add_named_per_class_metrics(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        precision_val = float(class_metrics.get("precision", 0.0))
        recall_val = float(class_metrics.get("recall", 0.0))
        f1_val = float(class_metrics.get("f1", 0.0))
        metrics[f"precision_{label_token}"] = precision_val
        metrics[f"recall_{label_token}"] = recall_val
        metrics[f"f1_{label_token}"] = f1_val
        metrics[f"test_precision_{label_token}"] = precision_val
        metrics[f"test_recall_{label_token}"] = recall_val
        metrics[f"test_f1_{label_token}"] = f1_val


def _add_named_per_class_metrics_with_suffix(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
    suffix: str,
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    normalized_suffix = str(suffix).strip()
    if normalized_suffix and not normalized_suffix.startswith("_"):
        normalized_suffix = f"_{normalized_suffix}"
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        precision_val = float(class_metrics.get("precision", 0.0))
        recall_val = float(class_metrics.get("recall", 0.0))
        f1_val = float(class_metrics.get("f1", 0.0))
        metrics[f"precision_{label_token}{normalized_suffix}"] = precision_val
        metrics[f"recall_{label_token}{normalized_suffix}"] = recall_val
        metrics[f"f1_{label_token}{normalized_suffix}"] = f1_val
        metrics[f"test_precision_{label_token}{normalized_suffix}"] = precision_val
        metrics[f"test_recall_{label_token}{normalized_suffix}"] = recall_val
        metrics[f"test_f1_{label_token}{normalized_suffix}"] = f1_val


def _add_named_validation_per_class_metrics(
    metrics: dict[str, Any],
    per_class_metrics: dict[str, dict[str, float]] | None,
    class_index_to_label: dict[str, str],
) -> None:
    if not isinstance(per_class_metrics, dict):
        return
    for class_idx, class_metrics in per_class_metrics.items():
        label_name = class_index_to_label.get(str(class_idx), str(class_idx))
        label_token = _metric_label_token(label_name)
        metrics[f"valid_precision_{label_token}"] = float(class_metrics.get("precision", 0.0))
        metrics[f"valid_recall_{label_token}"] = float(class_metrics.get("recall", 0.0))
        metrics[f"valid_f1_{label_token}"] = float(class_metrics.get("f1", 0.0))


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
        features = build_uct_student_features(adapted, dataset_cfg.get("features", {}))
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


def _prepare_preprocessing_config(
    exp_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    id_column: str,
    source_target_col: str,
) -> dict[str, Any]:
    p_cfg = exp_cfg.get("preprocessing", {})
    ds_p_cfg = dataset_cfg.get("preprocessing", {})
    imputation = str(p_cfg.get("imputation", "median_mode")).lower()
    scaling_raw = str(p_cfg.get("numeric_scaling", p_cfg.get("scaling", "standard"))).strip().lower()
    encoding_raw = str(p_cfg.get("categorical_encoding", p_cfg.get("encoding", "onehot"))).strip().lower()
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
    if global_method not in {"sigmoid", "isotonic"}:
        global_method = "sigmoid"

    def _resolve_stage(stage_name: str) -> dict[str, Any]:
        stage_raw = raw_cfg.get(stage_name, {})
        if not isinstance(stage_raw, dict):
            stage_raw = {}
        enabled = bool(stage_raw.get("enabled", global_enabled))
        method = str(stage_raw.get("method", global_method)).strip().lower()
        if method not in {"sigmoid", "isotonic"}:
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
) -> tuple[np.ndarray, np.ndarray]:
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
        )
        decision_region = np.where(
            np.asarray(fused_proba, dtype=float)[:, list(label_arr).index(int(dropout_idx))] >= float(dropout_threshold),
            "hard_dropout",
            "safe_non_dropout",
        )
        return pred, np.asarray(decision_region, dtype=str)
    if mode == "soft_fusion_with_middle_band":
        pred, decision_region = TwoStageUct3ClassClassifier.predict_with_middle_band_from_fused_probabilities(
            fused_proba=np.asarray(fused_proba, dtype=float),
            classes=label_arr,
            dropout_label=int(dropout_idx),
            enrolled_label=int(enrolled_idx),
            graduate_label=int(graduate_idx),
            low_threshold=float(low_threshold if low_threshold is not None else dropout_threshold),
            high_threshold=float(high_threshold if high_threshold is not None else dropout_threshold),
        )
        return pred, decision_region
    if mode in {"pure_soft_argmax", "soft_fusion"}:
        pred_idx = np.argmax(np.asarray(fused_proba, dtype=float), axis=1)
        return label_arr[pred_idx], np.full(shape=(np.asarray(fused_proba).shape[0],), fill_value="soft_fusion", dtype=str)
    thresholds_vec = _threshold_vector_from_map(labels, class_thresholds or {})
    pred = TwoStageUct3ClassClassifier.predict_from_fused_probabilities(
        fused_proba=np.asarray(fused_proba, dtype=float),
        classes=label_arr,
        thresholds=thresholds_vec,
    )
    return pred, np.full(shape=(np.asarray(fused_proba).shape[0],), fill_value="soft_threshold", dtype=str)


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
    )

    stage_prob_valid = combined_model.predict_stage_probabilities(X_valid)
    stage_prob_test = combined_model.predict_stage_probabilities(X_test)
    combined_model.threshold_stage1 = float(selected_candidate["high_threshold"])
    combined_model.threshold_stage1_low = float(selected_candidate["low_threshold"])
    combined_model.threshold_stage1_high = float(selected_candidate["high_threshold"])

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
        "validation_decision_regions": pd.Series(valid_decision_regions).value_counts().to_dict(),
        "test_decision_regions": pd.Series(test_decision_regions).value_counts().to_dict(),
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
) -> tuple[dict[str, Any], Any, float | None, dict[str, Any], dict[str, str] | None]:
    dropout_idx, enrolled_idx, graduate_idx = _resolve_uct_three_class_indices(class_metadata)
    stage2_positive_target_label = _resolve_two_stage_stage2_positive_target_label(
        two_stage_cfg=two_stage_cfg if isinstance(two_stage_cfg, dict) else {},
        enrolled_idx=enrolled_idx,
        graduate_idx=graduate_idx,
    )

    y_train_stage1 = (y_train == dropout_idx).astype(int)
    y_valid_stage1 = (y_valid == dropout_idx).astype(int)
    y_test_stage1 = (y_test == dropout_idx).astype(int)

    train_mask_stage2 = y_train != dropout_idx
    valid_mask_stage2 = y_valid != dropout_idx
    test_mask_stage2 = y_test != dropout_idx

    if int(train_mask_stage2.sum()) == 0:
        raise ValueError("No non-dropout samples available for stage2 training.")
    X_train_stage2 = X_train.loc[train_mask_stage2].reset_index(drop=True)
    X_valid_stage2 = X_valid.loc[valid_mask_stage2].reset_index(drop=True)
    X_test_stage2 = X_test.loc[test_mask_stage2].reset_index(drop=True)
    if int(stage2_positive_target_label) == int(enrolled_idx):
        y_train_stage2 = (y_train.loc[train_mask_stage2] == enrolled_idx).astype(int).reset_index(drop=True)
        y_valid_stage2 = (y_valid.loc[valid_mask_stage2] == enrolled_idx).astype(int).reset_index(drop=True)
        y_test_stage2 = (y_test.loc[test_mask_stage2] == enrolled_idx).astype(int).reset_index(drop=True)
        stage2_class_label_to_index = {"Graduate": 0, "Enrolled": 1}
        stage2_positive_label_name = "Enrolled"
        stage2_negative_label_name = "Graduate"
    else:
        y_train_stage2 = (y_train.loc[train_mask_stage2] == graduate_idx).astype(int).reset_index(drop=True)
        y_valid_stage2 = (y_valid.loc[valid_mask_stage2] == graduate_idx).astype(int).reset_index(drop=True)
        y_test_stage2 = (y_test.loc[test_mask_stage2] == graduate_idx).astype(int).reset_index(drop=True)
        stage2_class_label_to_index = {"Enrolled": 0, "Graduate": 1}
        stage2_positive_label_name = "Graduate"
        stage2_negative_label_name = "Enrolled"

    if int(pd.Series(y_train_stage2).nunique()) < 2:
        raise ValueError("Stage2 training requires both Enrolled and Graduate classes in train split.")

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
            y_train=y_train_stage2,
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

    eval_cfg_stage1 = {"seed": seed, "class_weight": stage1_class_weight_cfg, "label_order": [0, 1]}
    eval_cfg_stage2 = {"seed": seed, "class_weight": stage2_class_weight_cfg, "label_order": [0, 1]}

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
            y_train=y_train_stage2,
            X_valid=X_valid_stage2,
            y_valid=y_valid_stage2,
            X_test=X_test_stage2,
            y_test=y_test_stage2,
            eval_config=eval_cfg_stage2,
        )
        X_stage2_full = pd.concat([X_train_stage2, X_valid_stage2], axis=0).reset_index(drop=True)
        y_stage2_full = pd.concat([y_train_stage2, y_valid_stage2], axis=0).reset_index(drop=True)
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
            y_train=y_train_stage2,
            X_valid=X_valid_stage2,
            y_valid=y_valid_stage2,
            X_test=X_test_stage2,
            y_test=y_test_stage2,
            eval_config=eval_cfg_stage2,
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
    )

    stage_prob_valid = combined_model.predict_stage_probabilities(X_valid)
    stage_prob_test = combined_model.predict_stage_probabilities(X_test)
    y_proba_valid_final = combined_model.predict_proba(X_valid)
    y_proba_test_final = combined_model.predict_proba(X_test)
    label_order = [int(v) for v in class_metadata.get("class_indices", [dropout_idx, enrolled_idx, graduate_idx])]
    selected_dropout_threshold = float(threshold_stage1)
    selected_low_threshold = float(threshold_tuning_cfg.get("stage1_dropout_threshold_low", threshold_stage1))
    selected_high_threshold = float(threshold_tuning_cfg.get("stage1_dropout_threshold_high", threshold_stage1))
    tuned_thresholds_vec = _threshold_vector_from_map(label_order, class_thresholds)
    if decision_mode == "hard_routing":
        y_pred_valid_final = combined_model.predict(X_valid)
        y_pred_test_final = combined_model.predict(X_test)
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
        )
        y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_test_final, dtype=float),
            labels=label_order,
            decision_mode="soft_fusion",
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(threshold_stage1),
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
        )
        y_pred_test_final, test_decision_regions = _predict_two_stage_from_fused_probabilities(
            fused_proba=np.asarray(y_proba_test_final, dtype=float),
            labels=label_order,
            decision_mode="pure_soft_argmax",
            dropout_idx=dropout_idx,
            enrolled_idx=enrolled_idx,
            graduate_idx=graduate_idx,
            dropout_threshold=float(threshold_stage1),
        )
    combined_model.threshold_stage1 = float(selected_dropout_threshold)
    combined_model.threshold_stage1_low = float(selected_low_threshold)
    combined_model.threshold_stage1_high = float(selected_high_threshold)
    combined_model.middle_band_enabled = bool(threshold_tuning_cfg.get("middle_band_enabled", False))
    combined_model.middle_band_behavior = str(threshold_tuning_cfg.get("middle_band_behavior", "force_stage2_soft_fusion"))

    metrics: dict[str, float] = {}
    if not X_valid.empty:
        valid_metrics = compute_metrics(y_valid, y_pred_valid_final)
        metrics.update({f"valid_{k}": float(v) for k, v in valid_metrics.items()})
    test_metrics = compute_metrics(y_test, y_pred_test_final)
    metrics.update({f"test_{k}": float(v) for k, v in test_metrics.items()})

    per_class_metrics_test = compute_per_class_metrics(y_test, y_pred_test_final, labels=label_order)
    per_class_metrics_valid = compute_per_class_metrics(y_valid, y_pred_valid_final, labels=label_order)
    cm = confusion_matrix(y_test, y_pred_test_final, labels=label_order).tolist()
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
            "threshold_tuning": threshold_tuning_result,
            "decision_regions_valid": pd.Series(valid_decision_regions).value_counts().to_dict(),
            "decision_regions_test": pd.Series(test_decision_regions).value_counts().to_dict(),
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
    two_stage_metadata_payload = {
        "model": model_name,
        "mode": mode_name,
        "decision_mode": decision_mode,
        "class_order": [int(v) for v in label_order],
        "selected_low_threshold": threshold_tuning_result.get("selected_low_threshold"),
        "selected_high_threshold": threshold_tuning_result.get("selected_high_threshold"),
        "fusion": {
            "P(dropout)": "p_dropout",
            "P(enrolled)": "(1-p_dropout) * p_enrolled_given_non_dropout",
            "P(graduate)": "(1-p_dropout) * p_graduate_given_non_dropout",
        },
        "stage2_positive_label": stage2_positive_label_name,
    }
    calibration_metadata_payload = {
        "model": model_name,
        "enabled": bool(calibration_cfg.get("stage1", {}).get("enabled", False) or calibration_cfg.get("stage2", {}).get("enabled", False)),
        "method": str(calibration_cfg.get("stage1", {}).get("method", calibration_cfg.get("stage2", {}).get("method", "sigmoid"))),
        "stage1": stage1_calibration_meta,
        "stage2": stage2_calibration_meta,
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
    }
    two_stage_metadata_path.write_text(json.dumps(two_stage_metadata_payload, indent=2), encoding="utf-8")
    calibration_metadata_path.write_text(json.dumps(calibration_metadata_payload, indent=2), encoding="utf-8")
    threshold_metadata_path.write_text(json.dumps(threshold_metadata_payload, indent=2), encoding="utf-8")
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
        "selected_dropout_threshold": threshold_tuning_result.get("selected_dropout_threshold"),
        "selected_low_threshold": threshold_tuning_result.get("selected_low_threshold"),
        "selected_high_threshold": threshold_tuning_result.get("selected_high_threshold"),
        "validation_macro_f1_by_threshold": threshold_tuning_result.get("threshold_grid_results", []),
        "validation_enrolled_absorbed_into_dropout_at_selected_threshold": int(
            np.sum((np.asarray(y_valid, dtype=int) == int(enrolled_idx)) & (np.asarray(y_pred_valid_final, dtype=int) == int(dropout_idx)))
        ),
        "validation_decision_regions": pd.Series(valid_decision_regions).value_counts().to_dict(),
        "test_decision_regions": pd.Series(test_decision_regions).value_counts().to_dict(),
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
            "decision_region": np.asarray(test_decision_regions, dtype=str),
            "final_decision_mode": str(decision_mode),
            "stage1_prob_dropout": np.asarray(stage_prob_test.get("stage1_prob_dropout", []), dtype=float),
            "stage1_prob_non_dropout": np.asarray(stage_prob_test.get("stage1_prob_non_dropout", []), dtype=float),
            "stage2_prob_enrolled": np.asarray(stage_prob_test.get("stage2_prob_enrolled", []), dtype=float),
            "stage2_prob_graduate": np.asarray(stage_prob_test.get("stage2_prob_graduate", []), dtype=float),
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
            "decision_region": np.asarray(valid_decision_regions, dtype=str),
            "final_decision_mode": str(decision_mode),
            "stage1_prob_dropout": np.asarray(stage_prob_valid.get("stage1_prob_dropout", []), dtype=float),
            "stage1_prob_non_dropout": np.asarray(stage_prob_valid.get("stage1_prob_non_dropout", []), dtype=float),
            "stage2_prob_enrolled": np.asarray(stage_prob_valid.get("stage2_prob_enrolled", []), dtype=float),
            "stage2_prob_graduate": np.asarray(stage_prob_valid.get("stage2_prob_graduate", []), dtype=float),
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
    if tuning_artifacts:
        payload["artifact_paths"].update(tuning_artifacts)
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
        return run_uct_3class_error_audit(exp_cfg=exp_cfg, experiment_config_path=experiment_config_path)
    if experiment_mode == "threshold_tuning":
        return run_threshold_tuning_experiment(exp_cfg=exp_cfg, experiment_config_path=experiment_config_path)

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
    experiment_id = exp_cfg["experiment"]["id"]
    seed = int(exp_cfg["experiment"].get("seed", 42))
    formulation = str(exp_cfg["experiment"].get("target_formulation", "binary"))
    output_cfg = exp_cfg.get("outputs", {})
    output_dir = resolve_results_dir(exp_cfg, experiment_id=experiment_id)
    ensure_standard_output_layout(output_dir)

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
    target_mapping = _resolve_target_mapping(exp_cfg, dataset_cfg, formulation)
    class_metadata = _resolve_class_metadata(exp_cfg, target_mapping)
    mapped_target = _map_target(feature_df, dataset_name, source_target_col, formulation, target_mapping)
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
    if len(mapped_target) != len(feature_df):
        raise ValueError(
            "Mapped target length mismatch: "
            f"len(mapped_target)={len(mapped_target)} vs len(feature_df)={len(feature_df)} "
            f"for dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    feature_df["target"] = mapped_target
    if "target" not in feature_df.columns:
        raise ValueError(
            "Failed to create 'target' column after mapping for "
            f"dataset='{dataset_name}', source_target_col='{source_target_col}'."
        )
    # Some datasets (for example UCT) can map from source column "target" directly.
    # Never drop canonical target after mapping.
    columns_to_drop = [col for col in [source_target_col] if col and col != "target"]
    if columns_to_drop:
        feature_df = feature_df.drop(columns=columns_to_drop, errors="ignore")
    if feature_df["target"].isna().any():
        raise ValueError(
            "Target mapping produced null values for "
            f"dataset='{dataset_name}', source_target_col='{source_target_col}'."
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
    artifacts = run_tabular_preprocessing(splits, preprocess_cfg)
    onehot_metadata = _resolve_onehot_metadata_and_validate(
        artifacts=artifacts,
        preprocess_cfg=preprocess_cfg,
        preprocessing_exp_cfg=(exp_cfg.get("preprocessing", {}) if isinstance(exp_cfg.get("preprocessing", {}), dict) else {}),
    )

    outlier_cfg = _resolve_outlier_config(exp_cfg=exp_cfg, seed=seed)
    X_train_filtered, y_train_filtered, outlier_meta = apply_outlier_filter(
        artifacts.X_train,
        artifacts.y_train,
        outlier_cfg,
    )
    balancing_cfg = _resolve_balancing_config(exp_cfg=exp_cfg, seed=seed)
    X_train_bal, y_train_bal, balancing_meta = apply_balancing(X_train_filtered, y_train_filtered, balancing_cfg)
    pre_outlier_class_distribution = {
        str(k): int(v) for k, v in artifacts.y_train.value_counts(dropna=False).sort_index().to_dict().items()
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
        f"params={decision_rule_cfg.get('multiclass_decision', {})}"
    )
    param_overrides_cfg = exp_cfg.get("models", {}).get("param_overrides", {})
    runtime_artifact_format = str(output_cfg.get("runtime_artifact_format", "parquet")).strip().lower()

    model_results: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    optuna_artifacts: dict[str, dict[str, str]] = {}
    model_decision_configs: dict[str, dict[str, Any]] = {}

    for model_name in model_candidates:
        model_decision_rule_cfg = _resolve_model_decision_rule_config(
            exp_cfg=exp_cfg,
            base_decision_rule_cfg=decision_rule_cfg,
            model_name=model_name,
            formulation=formulation,
            two_stage_enabled=two_stage_enabled,
            class_metadata=class_metadata,
        )
        model_decision_configs[model_name] = copy.deepcopy(model_decision_rule_cfg)
        model_n_trials = int(per_model_trial_budgets.get(model_name, default_n_trials))
        model_tuning_enabled = bool(tuning_backend == "optuna" and model_n_trials > 0)
        model_tuning_cfg = {**tuning_cfg, "n_trials": model_n_trials}
        params: dict[str, Any] = {}
        model_param_overrides = {}
        if isinstance(param_overrides_cfg, dict):
            candidate_override = param_overrides_cfg.get(model_name, {})
            if isinstance(candidate_override, dict):
                model_param_overrides = candidate_override
        tuning_score = None
        tuning_details: dict[str, Any] = {}
        warning_msg: str | None = None
        if two_stage_enabled:
            try:
                payload, trained_model, tuning_score, tuning_details, two_stage_optuna_paths = _run_two_stage_uct_model(
                    model_name=model_name,
                    mode_name=experiment_mode,
                    decision_mode=str(two_stage_decision_mode),
                    params_overrides=model_param_overrides,
                    seed=seed,
                    tuning_cfg=model_tuning_cfg,
                    tuning_enabled=model_tuning_enabled,
                    retrain_on_full_train_split=retrain_on_full_train_split,
                    two_stage_cfg=two_stage_cfg,
                    class_weight_cfg=class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
                    class_metadata=class_metadata,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    threshold_stage1=threshold_stage1,
                    class_thresholds=two_stage_class_thresholds,
                    threshold_tuning_cfg=two_stage_threshold_tuning_cfg,
                    calibration_cfg=two_stage_calibration_cfg,
                    output_dir=output_dir,
                )
                payload["class_weight"] = dict(payload.get("class_weight", {}))
                payload["class_weight"]["class_weight_requested"] = bool(
                    payload["class_weight"].get("class_weight_requested", False)
                    or _class_weight_requested(class_weight_cfg)
                    or bool(payload.get("metrics", {}).get("auto_balance_search_enabled", 0.0))
                )
                payload["class_weight"]["model_name"] = model_name
                _add_class_weight_metadata_metrics(payload["metrics"], payload["class_weight"], class_metadata)
                print(
                    "[class_weight][model] "
                    f"model={model_name} "
                    f"requested={payload['class_weight'].get('class_weight_requested', False)} "
                    f"applied={payload['class_weight'].get('class_weight_applied', False)} "
                    f"method={payload['class_weight'].get('class_weight_application_method', 'none')} "
                    f"mode={payload['class_weight'].get('mode', payload['class_weight'].get('strategy', 'none'))}"
                )
                _add_named_per_class_metrics(
                    payload["metrics"],
                    payload["artifacts"].get("per_class_metrics_test"),
                    class_metadata.get("class_index_to_label", {}),
                )
                if tuning_details:
                    payload["tuning"] = tuning_details
                if two_stage_optuna_paths:
                    optuna_artifacts[model_name] = dict(two_stage_optuna_paths)
                    payload.setdefault("artifact_paths", {}).update(two_stage_optuna_paths)
                model_results[model_name] = payload
                trained_models[model_name] = trained_model
                leaderboard_rows.append({"model": model_name, **payload["metrics"]})
            except Exception as exc:
                model_results[model_name] = {"error": f"Training/evaluation failed: {exc}"}
            continue
        if model_tuning_enabled:
            try:
                objective_source = str(model_tuning_cfg.get("objective_source", "cv")).strip().lower()
                if paper_reproduction_mode:
                    objective_source = "paper_cv"
                params, tuning_score, tuning_details = tune_model_with_optuna(
                    model_name=model_name,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    tuning_cfg={
                        **model_tuning_cfg,
                        "seed": seed,
                        "objective_source": objective_source,
                        "use_class_weights": _class_weight_requested(class_weight_cfg),
                        "class_weight": {
                            **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                            "class_label_to_index": class_metadata.get("class_label_to_index", {}),
                        },
                        "cv_train_df": splits.get("train", pd.DataFrame()).reset_index(drop=True),
                        "cv_preprocess_config": preprocess_cfg,
                        "cv_outlier_config": outlier_cfg,
                        "cv_balancing_config": balancing_cfg,
                        "cv_config": cv_reporting_cfg,
                        "label_order": class_metadata.get("class_indices", []),
                        "decision_rule": model_decision_rule_cfg.get("decision_rule", "model_predict"),
                        "multiclass_decision": model_decision_rule_cfg.get("multiclass_decision", {}),
                        "trial_selection": {
                            "ranking_metrics": model_selection_cfg.get("ranking_metrics", []),
                        },
                    },
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    fixed_params=model_param_overrides,
                )
                model_token = _safe_filename_token(model_name)
                trials_path = output_dir / f"optuna_trials_{model_token}.csv"
                best_params_path = output_dir / f"optuna_best_params_{model_token}.json"
                trials_df = tuning_details.get("trials_dataframe")
                if isinstance(trials_df, pd.DataFrame):
                    trials_df.to_csv(trials_path, index=False)
                else:
                    pd.DataFrame(tuning_details.get("trials", [])).to_csv(trials_path, index=False)
                best_payload = {
                    "model": model_name,
                    "best_params": params,
                    "best_value": tuning_score,
                    "objective_source": tuning_details.get("objective_source"),
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_cv_aggregate_metrics": tuning_details.get("best_cv_aggregate_metrics", {}),
                    "best_per_class_metrics": tuning_details.get("best_per_class_metrics", {}),
                    "best_per_class_f1": tuning_details.get("best_per_class_f1", {}),
                }
                best_params_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
                optuna_artifacts[model_name] = {
                    "trials_csv": str(trials_path),
                    "best_params_json": str(best_params_path),
                }
            except Exception as exc:
                warning_msg = f"Tuning failed; using defaults. Reason: {exc}"
                params = dict(model_param_overrides)
                tuning_score = None
                tuning_details = {}
        elif model_param_overrides:
            params = dict(model_param_overrides)
        try:
            eval_cfg = {
                "seed": seed,
                "class_weight": {
                    **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                    "class_label_to_index": class_metadata.get("class_label_to_index", {}),
                },
                "label_order": class_metadata.get("class_indices", []),
                "decision_rule": model_decision_rule_cfg.get("decision_rule", "model_predict"),
                "multiclass_decision": model_decision_rule_cfg.get("multiclass_decision", {}),
            }
            if retrain_on_full_train_split:
                prefit_result = train_and_evaluate(
                    model_name=model_name,
                    params=params,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    eval_config=eval_cfg,
                )
                X_train_full = pd.concat([X_train_bal, artifacts.X_valid], axis=0).reset_index(drop=True)
                y_train_full = pd.concat([y_train_bal, artifacts.y_valid], axis=0).reset_index(drop=True)
                retrained_result = retrain_on_full_train_and_evaluate_test(
                    model_name=model_name,
                    params=params,
                    X_train_full=X_train_full,
                    y_train_full=y_train_full,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    eval_config=eval_cfg,
                )
                merged_metrics = dict(prefit_result.metrics)
                merged_metrics.update(retrained_result.metrics)
                merged_artifacts = dict(retrained_result.artifacts)
                # Keep validation artifacts from the prefit phase for post-hoc analysis.
                for key in ("y_true_valid", "y_pred_valid", "y_proba_valid"):
                    if key in prefit_result.artifacts:
                        merged_artifacts[key] = prefit_result.artifacts[key]
                result = retrained_result
                result = type(result)(metrics=merged_metrics, artifacts=merged_artifacts)
            else:
                result = train_and_evaluate(
                    model_name=model_name,
                    params=params,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    X_valid=artifacts.X_valid,
                    y_valid=artifacts.y_valid,
                    X_test=artifacts.X_test,
                    y_test=artifacts.y_test,
                    eval_config=eval_cfg,
                )
            payload = {
                "metrics": result.metrics,
                "artifacts": {k: v for k, v in result.artifacts.items() if k != "model"},
                "params": result.artifacts.get("params", params),
                "tuning_score": tuning_score,
            }
            payload["class_weight"] = dict(result.artifacts.get("class_weight_info", {}))
            payload["class_weight"]["class_weight_requested"] = _class_weight_requested(class_weight_cfg)
            payload["class_weight"]["model_name"] = model_name
            _add_class_weight_metadata_metrics(payload["metrics"], payload["class_weight"], class_metadata)
            print(
                "[class_weight][model] "
                f"model={model_name} "
                f"requested={payload['class_weight'].get('class_weight_requested', False)} "
                f"applied={payload['class_weight'].get('class_weight_applied', False)} "
                f"method={payload['class_weight'].get('class_weight_application_method', 'none')} "
                f"mode={payload['class_weight'].get('mode', payload['class_weight'].get('strategy', 'none'))}"
            )

            if cv_reporting_cfg.get("enabled", False):
                cv_eval = run_leakage_safe_stratified_cv(
                    model_name=model_name,
                    params=params,
                    train_df=splits.get("train", pd.DataFrame()).reset_index(drop=True),
                    preprocess_config=preprocess_cfg,
                    outlier_config=outlier_cfg,
                    balancing_config=balancing_cfg,
                    cv_config=cv_reporting_cfg,
                    eval_config={
                        "seed": seed,
                        "class_weight": {
                            **(class_weight_cfg if isinstance(class_weight_cfg, dict) else {}),
                            "class_label_to_index": class_metadata.get("class_label_to_index", {}),
                        },
                        "label_order": class_metadata.get("class_indices", []),
                        "decision_rule": model_decision_rule_cfg.get("decision_rule", "model_predict"),
                        "multiclass_decision": model_decision_rule_cfg.get("multiclass_decision", {}),
                    },
                )
                payload["cv_results"] = cv_eval
                aggregate_metrics = cv_eval.get("aggregate_metrics", {})
                if isinstance(aggregate_metrics, dict):
                    for key, value in aggregate_metrics.items():
                        if str(key) == "cv_num_folds":
                            payload["metrics"][str(key)] = int(value)
                        else:
                            payload["metrics"][str(key)] = float(value)
                    payload["metrics"]["optuna_trials"] = float(int(model_n_trials)) if model_tuning_enabled else 0.0
            if model_tuning_enabled:
                payload["optuna_summary"] = {
                    "model": model_name,
                    "objective_source": tuning_details.get("objective_source"),
                    "n_trials": int(model_n_trials),
                    "best_value": tuning_score,
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_cv_aggregate_metrics": tuning_details.get("best_cv_aggregate_metrics", {}),
                }

            payload["metrics"]["multiclass_decision_auto_tuned"] = 0.0
            payload["metrics"]["multiclass_decision_strategy"] = str(
                model_decision_rule_cfg.get("decision_rule", "model_predict")
            )
            payload["metrics"]["tuning_objective"] = str(
                model_decision_rule_cfg.get("multiclass_decision", {}).get("auto_tune", {}).get("objective", "macro_f1")
            )
            decision_auto_tune_result = _run_multiclass_decision_autotune(
                payload=payload,
                decision_rule_cfg=model_decision_rule_cfg,
                class_metadata=class_metadata,
            )
            payload["decision_policy_tuning"] = decision_auto_tune_result
            payload["decision_rule_config"] = copy.deepcopy(model_decision_rule_cfg)
            payload["metrics"]["multiclass_decision_auto_tuned"] = (
                1.0 if bool(decision_auto_tune_result.get("multiclass_decision_auto_tuned", False)) else 0.0
            )
            print(
                "[decision_policy][model] "
                f"model={model_name} "
                f"strategy={model_decision_rule_cfg.get('decision_rule')} "
                f"auto_tune={decision_auto_tune_result.get('multiclass_decision_auto_tuned', False)} "
                f"grid_size={decision_auto_tune_result.get('search_grid_size', 0)} "
                f"selected={decision_auto_tune_result.get('selected_parameters', {})} "
                f"best_valid={decision_auto_tune_result.get('validation_objective_score_at_selected_threshold')}"
            )

            threshold_result: dict[str, Any] = {"status": "skipped", "reason": "disabled"}
            if bool(threshold_tuning_cfg.get("enabled", False)):
                threshold_result = _run_validation_threshold_tuning(
                    payload=payload,
                    class_metadata=class_metadata,
                    threshold_cfg=threshold_tuning_cfg,
                )
            else:
                threshold_result = {
                    "status": "skipped",
                    "reason": "disabled",
                    "threshold_tuning_requested": False,
                    "threshold_tuning_supported": bool(payload["artifacts"].get("y_proba_valid") is not None),
                    "threshold_selection_split": "validation",
                    "threshold_applied_to": "none",
                    "default_decision_rule": "argmax",
                }
            payload["threshold_tuning"] = threshold_result

            if threshold_result.get("status") == "applied":
                baseline_metrics = {
                    k: float(v)
                    for k, v in payload["metrics"].items()
                    if isinstance(v, (int, float)) and k.startswith("test_")
                }
                for key, value in baseline_metrics.items():
                    payload["metrics"][f"{key}_default"] = float(value)
                baseline_per_class = payload["artifacts"].get("per_class_metrics_test")
                _add_named_per_class_metrics_with_suffix(
                    payload["metrics"],
                    baseline_per_class,
                    class_metadata.get("class_index_to_label", {}),
                    "default",
                )

                payload["artifacts"]["per_class_metrics_test_default"] = baseline_per_class
                payload["artifacts"]["y_pred_test_default"] = payload["artifacts"].get("y_pred_test")
                payload["artifacts"]["confusion_matrix_default"] = payload["artifacts"].get("confusion_matrix")

                tuned_metrics = threshold_result.get("test_tuned_metrics", {})
                payload["metrics"].update({f"test_{k}": float(v) for k, v in tuned_metrics.items()})
                payload["artifacts"]["per_class_metrics_test"] = threshold_result.get("test_tuned_per_class", {})
                if str(threshold_result.get("threshold_applied_to", "test")) == "test":
                    payload["artifacts"]["y_pred_test"] = threshold_result.get("y_pred_test_tuned", [])
                    payload["artifacts"]["confusion_matrix"] = threshold_result.get("confusion_matrix_tuned", [])
                payload["threshold_tuning"]["threshold_tuning_applied"] = bool(
                    str(payload["threshold_tuning"].get("threshold_applied_to", "none")) == "test"
                )
            else:
                payload["threshold_tuning"]["threshold_tuning_applied"] = False

            _add_named_per_class_metrics(
                payload["metrics"],
                payload["artifacts"].get("per_class_metrics_test"),
                class_metadata.get("class_index_to_label", {}),
            )
            _add_named_validation_per_class_metrics(
                payload["metrics"],
                payload["artifacts"].get("per_class_metrics_valid"),
                class_metadata.get("class_index_to_label", {}),
            )
            if tuning_details:
                payload["tuning"] = {
                    "scoring": str(model_tuning_cfg.get("scoring", "f1_macro")),
                    "cv_folds": int(model_tuning_cfg.get("cv_folds", 3)),
                    "objective_source": tuning_details.get("objective_source"),
                    "objective_metric": primary_metric,
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_per_class_metrics": tuning_details.get("best_per_class_metrics", {}),
                    "best_per_class_f1": tuning_details.get("best_per_class_f1", {}),
                    "n_trials": int(model_n_trials),
                }
            if model_name in optuna_artifacts:
                payload.setdefault("artifact_paths", {}).update(optuna_artifacts[model_name])
            if warning_msg:
                payload["warning"] = warning_msg
            model_results[model_name] = payload
            trained_models[model_name] = result.artifacts.get("model")
            leaderboard_rows.append({"model": model_name, **payload["metrics"]})
        except Exception as exc:
            model_results[model_name] = {"error": f"Training/evaluation failed: {exc}"}

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    best_by_cv: dict[str, Any] = {"model": None, "ranking_columns": []}
    best_by_test: dict[str, Any] = {"model": None, "ranking_columns": []}
    if model_selection_cfg.get("enabled", False):
        leaderboard_df, best_model_by_primary, _ = _sort_leaderboard_with_tiebreak(
            leaderboard_df=leaderboard_df,
            selection_cfg=model_selection_cfg,
            source="test",
        )
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

    compact_mode = bool(output_cfg.get("compact_summary", False)) if compact_summary is None else bool(compact_summary)
    summary = {
        "experiment_id": experiment_id,
        "benchmark_summary_version": BENCHMARK_SUMMARY_VERSION,
        "schema_version": BENCHMARK_SUMMARY_VERSION,
        "dataset_name": dataset_name,
        "target_formulation": formulation,
        "class_metadata": class_metadata,
        "primary_metric": metric_key,
        "best_model": best_model,
        "output_dir": str(output_dir),
        "seed": seed,
        "split_config": {
            "test_size": float(exp_cfg["splits"]["test_size"]),
            "validation_size": float(exp_cfg["splits"].get("validation_size", 0.2)),
            "stratify_column": str(exp_cfg["splits"].get("stratify_column", "target")),
        },
        "split_sizes": {
            "train_raw": int(len(splits.get("train", pd.DataFrame()))),
            "valid_raw": int(len(splits.get("valid", pd.DataFrame()))),
            "test_raw": int(len(splits.get("test", pd.DataFrame()))),
            "train_after_preprocessing": int(len(artifacts.X_train)),
            "valid_after_preprocessing": int(len(artifacts.X_valid)),
            "test_after_preprocessing": int(len(artifacts.X_test)),
            "train_after_outlier": int(len(X_train_filtered)),
            "train_after_balancing": int(len(X_train_bal)),
        },
        "model_results": model_results,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "preprocessing": {
            "missing_values": missing_value_meta,
            "outlier": outlier_meta,
            "balancing": balancing_meta,
            "class_distribution_train_before_outlier": pre_outlier_class_distribution,
            "class_distribution_train_after_outlier": post_outlier_class_distribution,
            "feature_count_after_preprocessing": int(artifacts.X_train.shape[1]),
            "preprocessed_feature_count": int(onehot_metadata.get("preprocessed_feature_count", int(artifacts.X_train.shape[1]))),
            "onehot_enabled": bool(onehot_metadata.get("onehot_enabled", False)),
            "require_onehot_encoding": bool(onehot_metadata.get("require_onehot_encoding", False)),
            "input_categorical_feature_count": int(onehot_metadata.get("input_categorical_feature_count", 0)),
            "encoded_categorical_feature_count": int(onehot_metadata.get("encoded_categorical_feature_count", 0)),
        },
        "class_weight": class_weight_cfg if isinstance(class_weight_cfg, dict) else {},
        "threshold_tuning": threshold_tuning_cfg,
        "cross_validation": cv_reporting_cfg,
        "decision_policy": decision_rule_cfg,
        "summary_mode": "compact" if compact_mode else "full",
    }
    if best_by_cv.get("model") is not None:
        summary["best_by_cv"] = best_by_cv
    if best_by_test.get("model") is not None:
        summary["best_by_test"] = best_by_test
    if model_selection_cfg.get("enabled", False):
        summary["selection"] = model_selection_cfg
    model_mechanism_audit: dict[str, Any] = {}
    for model_name, payload in model_results.items():
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue
        class_weight_meta = payload.get("class_weight", {}) if isinstance(payload.get("class_weight", {}), dict) else {}
        threshold_meta = payload.get("threshold_tuning", {}) if isinstance(payload.get("threshold_tuning", {}), dict) else {}
        decision_meta = payload.get("decision_policy_tuning", {}) if isinstance(payload.get("decision_policy_tuning", {}), dict) else {}
        model_decision_cfg = (
            payload.get("decision_rule_config", {})
            if isinstance(payload.get("decision_rule_config", {}), dict)
            else model_decision_configs.get(model_name, {})
        )
        model_mechanism_audit[model_name] = {
            "decision_rule": str(model_decision_cfg.get("decision_rule", "model_predict")),
            "decision_strategy": str(
                model_decision_cfg.get("multiclass_decision", {}).get(
                    "strategy",
                    model_decision_cfg.get("decision_rule", "model_predict"),
                )
            ),
            "multiclass_decision_auto_tuned": bool(decision_meta.get("multiclass_decision_auto_tuned", False)),
            "multiclass_decision_tuning_objective": decision_meta.get("tuning_objective"),
            "multiclass_decision_selected_parameters": decision_meta.get("selected_parameters", {}),
            "validation_objective_score_at_selected_threshold": decision_meta.get(
                "validation_objective_score_at_selected_threshold"
            ),
            "class_weight_requested": bool(class_weight_meta.get("class_weight_requested", False)),
            "class_weight_supported": bool(class_weight_meta.get("class_weight_supported", False)),
            "class_weight_applied": bool(class_weight_meta.get("class_weight_applied", False)),
            "class_weight_effective": str(class_weight_meta.get("effective_mechanism", "none")),
            "class_weight_values": class_weight_meta.get("weight_map", {}),
            "class_weight_backend_note": class_weight_meta.get("class_weight_backend_note"),
            "threshold_tuning_requested": bool(threshold_meta.get("threshold_tuning_requested", False)),
            "threshold_tuning_supported": bool(threshold_meta.get("threshold_tuning_supported", False)),
            "threshold_tuning_applied": bool(threshold_meta.get("threshold_tuning_applied", False)),
            "threshold_selection_split": str(threshold_meta.get("threshold_selection_split", "validation")),
            "threshold_applied_to": str(threshold_meta.get("threshold_applied_to", "none")),
            "selected_thresholds": threshold_meta.get("selected_thresholds", {}),
        }
    summary["model_mechanism_audit"] = model_mechanism_audit
    summary["artifact_paths"] = _persist_runtime_artifacts(
        output_dir=output_dir,
        best_model_name=best_model,
        trained_models=trained_models,
        preprocessing_artifacts=artifacts,
        summary=summary,
        file_format=runtime_artifact_format,
    )

    contract_paths = _persist_required_contract_outputs(
        output_dir=output_dir,
        summary=summary,
        best_model_name=best_model,
        trained_models=trained_models,
        y_test=artifacts.y_test,
        class_metadata=class_metadata,
    )
    summary["artifact_paths"].update(contract_paths)
    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary["artifact_paths"].update(
        _persist_per_model_run_outputs(
            output_dir=output_dir,
            run_stamp=run_stamp,
            model_results=model_results,
            trained_models=trained_models,
            class_metadata=class_metadata,
        )
    )

    figure_status: dict[str, dict[str, Any]] = {}
    if best_model and best_model in trained_models and trained_models[best_model] is not None:
        best_payload = model_results.get(best_model, {})
        figure_result = generate_all_figures(
            model=trained_models[best_model],
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_test=artifacts.X_test,
            y_test=artifacts.y_test,
            y_pred_proba=best_payload.get("artifacts", {}).get("y_proba_test"),
            output_dir=output_dir,
            experiment_name=experiment_id,
            primary_metric=primary_metric,
            random_state=seed,
            include_status=True,
        )
        figure_paths = figure_result.get("artifact_paths", {})
        figure_status = figure_result.get("artifact_status", {})
        summary["artifact_paths"].update(figure_paths)
    else:
        figure_status["learning_curve"] = {
            "status": "skipped",
            "reason": "best_model_missing_or_unavailable",
        }
        figure_status["pr_curve"] = {
            "status": "skipped",
            "reason": "best_model_missing_or_unavailable",
        }
        figure_status["shap_beeswarm"] = {
            "status": "skipped",
            "reason": "best_model_missing_or_unavailable",
        }
        figure_status["shap_waterfall"] = {
            "status": "skipped",
            "reason": "best_model_missing_or_unavailable",
        }

    summary["artifact_paths"]["benchmark_summary"] = str(output_dir / "benchmark_summary.json")
    summary["artifact_paths"]["leaderboard"] = str(output_dir / "leaderboard.csv")
    summary["artifact_paths"]["summary_csv"] = str(output_dir / "summary.csv")
    summary["artifact_paths"]["benchmark_markdown"] = str(output_dir / "benchmark_summary.md")
    summary["artifact_paths"]["artifact_manifest"] = str(output_dir / "artifact_manifest.json")
    for model_name, paths in optuna_artifacts.items():
        model_token = _safe_filename_token(model_name)
        summary["artifact_paths"][f"optuna_trials_{model_token}"] = paths["trials_csv"]
        summary["artifact_paths"][f"optuna_best_params_{model_token}"] = paths["best_params_json"]
    save_benchmark_summary(summary, output_dir, compact=compact_mode)

    mirror_enabled = bool(output_cfg.get("mirror_benchmark_outputs_to_runtime", False))
    if mirror_enabled:
        mirrored = _mirror_root_artifacts_to_runtime(
            output_dir=output_dir,
            runtime_dir=output_dir / "runtime_artifacts",
            model_candidates=model_candidates,
        )
        for name, path in mirrored.items():
            summary["artifact_paths"][f"runtime_{name}"] = path

    explain_json, explain_md = write_skipped_explainability_report(
        output_dir=output_dir,
        reason="explainability_not_run_yet",
    )

    confusion_matrix_paths = sorted(output_dir.glob("confusion_matrix_*.png"))
    normalized_confusion_matrix_paths = sorted(output_dir.glob("confusion_matrix_*_normalized.png"))
    classification_report_paths = sorted(output_dir.glob("classification_report_*.json"))

    mandatory_updates: dict[str, dict[str, Any]] = {
        "benchmark_summary_json": _status_from_path(summary["artifact_paths"]["benchmark_summary"]),
        "benchmark_summary_md": _status_from_path(summary["artifact_paths"]["benchmark_markdown"]),
        "metrics_json": _status_from_path(summary["artifact_paths"]["metrics"]),
        "predictions_csv": _status_from_path(summary["artifact_paths"]["predictions"]),
        "leaderboard_csv": _status_from_path(summary["artifact_paths"]["leaderboard"]),
        "summary_csv": _status_from_path(summary["artifact_paths"]["summary_csv"]),
        "runtime_artifacts_dir": _status_from_path(output_dir / "runtime_artifacts"),
        "model_dir": _status_from_path(output_dir / "model"),
        "learning_curve_png": figure_status.get(
            "learning_curve",
            {"status": "failed", "reason": "figure_generation_error"},
        ),
        "pr_curve_png": figure_status.get(
            "pr_curve",
            {"status": "failed", "reason": "figure_generation_error"},
        ),
    }

    if confusion_matrix_paths:
        mandatory_updates["confusion_matrix_artifacts"] = {
            "status": "generated",
            "paths": [str(p) for p in confusion_matrix_paths],
        }
    else:
        mandatory_updates["confusion_matrix_artifacts"] = {
            "status": "skipped",
            "reason": "not_generated_by_reporting",
        }
    if normalized_confusion_matrix_paths:
        mandatory_updates["normalized_confusion_matrix_artifacts"] = {
            "status": "generated",
            "paths": [str(p) for p in normalized_confusion_matrix_paths],
        }
    else:
        mandatory_updates["normalized_confusion_matrix_artifacts"] = {
            "status": "skipped",
            "reason": "not_generated_by_reporting",
        }
    if classification_report_paths:
        mandatory_updates["classification_report_artifacts"] = {
            "status": "generated",
            "paths": [str(p) for p in classification_report_paths],
        }
    else:
        mandatory_updates["classification_report_artifacts"] = {
            "status": "skipped",
            "reason": "not_generated_by_reporting",
        }
    if optuna_artifacts:
        mandatory_updates["optuna_artifacts"] = {
            "status": "generated",
            "details": optuna_artifacts,
        }
    elif tuning_enabled:
        mandatory_updates["optuna_artifacts"] = {
            "status": "failed",
            "reason": "tuning_enabled_but_no_optuna_artifacts_saved",
        }

    optional_updates: dict[str, dict[str, Any]] = {
        "shap_beeswarm_png": figure_status.get(
            "shap_beeswarm",
            {"status": "skipped", "reason": "not_applicable_for_model_type"},
        ),
        "shap_waterfall_pngs": figure_status.get(
            "shap_waterfall",
            {"status": "skipped", "reason": "not_applicable_for_model_type"},
        ),
        "explainability_dir": {
            "status": "created",
            "path": str(output_dir / "explainability"),
        },
        "explainability_report_json": {
            "status": "created",
            "path": str(explain_json),
            "reason": "explainability_not_run_yet",
        },
        "explainability_report_md": {
            "status": "created",
            "path": str(explain_md),
            "reason": "explainability_not_run_yet",
        },
        "lime_outputs": {
            "status": "skipped",
            "reason": "explainability_not_run_yet",
        },
        "aime_outputs": {
            "status": "skipped",
            "reason": "explainability_not_run_yet",
        },
        "shap_outputs": {
            "status": "skipped",
            "reason": "explainability_not_run_yet",
        },
        "threshold_tuning_inline": {
            "status": "enabled" if bool(threshold_tuning_cfg.get("enabled", False)) else "disabled",
            "objective": str(threshold_tuning_cfg.get("objective", "macro_f1")),
            "focus_class": str(threshold_tuning_cfg.get("focus_class", "Enrolled")),
        },
    }

    update_artifact_manifest(
        output_dir=output_dir,
        mandatory_updates=mandatory_updates,
        optional_updates=optional_updates,
        metadata_updates={
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "target_formulation": formulation,
            "best_model": best_model,
            "manifest_scope": "benchmark",
        },
    )
    if mirror_enabled:
        _mirror_root_artifacts_to_runtime(
            output_dir=output_dir,
            runtime_dir=output_dir / "runtime_artifacts",
            model_candidates=model_candidates,
        )
    return summary


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
