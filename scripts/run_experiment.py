"""Run config-driven benchmark experiments for UCT Student and OULAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

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
from src.models.train_eval import retrain_on_full_train_and_evaluate_test, train_and_evaluate, tune_model_with_optuna
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
            "class_weight": {"enabled": bool(exp_cfg.get("training", {}).get("use_class_weights", False))},
            "retrain_on_full_train_split": True,
        },
        "metrics": {
            "primary": objective_metric,
            "secondary": secondary_metrics,
        },
        "evaluation": {
            "metrics": eval_metrics,
            "compare_default_argmax": bool(evaluation_cfg.get("compare_default_argmax", False)),
            "save_confusion_matrix": bool(evaluation_cfg.get("save_confusion_matrix", True)),
            "save_classification_report": bool(evaluation_cfg.get("save_classification_report", True)),
        },
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
        "best_model_metrics": (
            summary.get("model_results", {}).get(best_model_name, {}).get("metrics", {}) if best_model_name else {}
        ),
        "leaderboard": summary.get("leaderboard", []),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    paths["metrics"] = str(metrics_path)

    pred_df = pd.DataFrame({"y_true": y_test.reset_index(drop=True)})
    if best_model_name:
        best_payload = summary.get("model_results", {}).get(best_model_name, {})
        best_artifacts = best_payload.get("artifacts", {})
        y_pred = best_artifacts.get("y_pred_test")
        y_proba = best_artifacts.get("y_proba_test")
        labels = best_artifacts.get("labels") or []

        if y_pred is not None:
            pred_df["y_pred"] = np.asarray(y_pred)
        if y_proba is not None:
            proba_arr = np.asarray(y_proba)
            if proba_arr.ndim == 2:
                for idx in range(proba_arr.shape[1]):
                    label = labels[idx] if idx < len(labels) else idx
                    pred_df[f"proba_class_{label}"] = proba_arr[:, idx]

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


def _normalize_dataset_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    return DATASET_NAME_ALIASES.get(normalized, normalized)


def _safe_filename_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_").lower()


def _metric_label_token(raw_label: str) -> str:
    value = str(raw_label).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


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


def _drop_rows_with_missing_values(
    df: pd.DataFrame,
    preprocessing_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_cfg = preprocessing_cfg.get("missing_values", {})
    drop_rows = bool(preprocessing_cfg.get("drop_missing_rows", False))
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
        "scaling": str(p_cfg.get("scaling", "standard")).lower() == "standard",
        "onehot": str(p_cfg.get("encoding", "onehot")).lower() == "onehot",
    }


def run_experiment(experiment_config_path: Path, compact_summary: bool | None = None) -> dict[str, Any]:
    exp_cfg = _normalize_experiment_config_schema(load_yaml(experiment_config_path))
    experiment_mode = str(exp_cfg.get("experiment", {}).get("mode", "benchmark")).strip().lower()
    if experiment_mode == "error_audit":
        return run_uct_3class_error_audit(exp_cfg=exp_cfg, experiment_config_path=experiment_config_path)
    if experiment_mode == "threshold_tuning":
        return run_threshold_tuning_experiment(exp_cfg=exp_cfg, experiment_config_path=experiment_config_path)

    if "datasets" in exp_cfg.get("experiment", {}):
        raise NotImplementedError(
            "Shared multi-dataset workflow is experimental and not supported by scripts/run_experiment.py yet. "
            "Use single-dataset configs with experiment.dataset_config."
        )
    dataset_cfg_path = Path(exp_cfg["experiment"]["dataset_config"])
    dataset_cfg = load_yaml(dataset_cfg_path)
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

    outlier_cfg = exp_cfg.get("preprocessing", {}).get("outlier", {"enabled": False})
    outlier_cfg = {**outlier_cfg, "random_state": int(outlier_cfg.get("random_state", seed))}
    X_train_filtered, y_train_filtered, outlier_meta = apply_outlier_filter(
        artifacts.X_train,
        artifacts.y_train,
        outlier_cfg,
    )
    balancing_cfg = exp_cfg.get("preprocessing", {}).get("balancing", {"enabled": False})
    balancing_cfg = {**balancing_cfg, "random_state": int(balancing_cfg.get("random_state", seed))}
    X_train_bal, y_train_bal, balancing_meta = apply_balancing(X_train_filtered, y_train_filtered, balancing_cfg)
    pre_outlier_class_distribution = {
        str(k): int(v) for k, v in artifacts.y_train.value_counts(dropna=False).sort_index().to_dict().items()
    }
    post_outlier_class_distribution = {
        str(k): int(v) for k, v in y_train_filtered.value_counts(dropna=False).sort_index().to_dict().items()
    }

    model_candidates = list(exp_cfg.get("models", {}).get("candidates", []))
    available_models = set(list_available_models())
    primary_metric = str(exp_cfg.get("metrics", {}).get("primary", "macro_f1"))
    metric_key = primary_metric if primary_metric.startswith("test_") else f"test_{primary_metric}"
    tuning_cfg = exp_cfg.get("models", {}).get("tuning", {})
    tuning_enabled = str(tuning_cfg.get("backend", "none")).lower() == "optuna" and int(tuning_cfg.get("n_trials", 0)) > 0
    retrain_on_full_train_split = bool(exp_cfg.get("models", {}).get("retrain_on_full_train_split", False))
    class_weight_cfg = exp_cfg.get("models", {}).get("class_weight", {})
    param_overrides_cfg = exp_cfg.get("models", {}).get("param_overrides", {})
    runtime_artifact_format = str(output_cfg.get("runtime_artifact_format", "parquet")).strip().lower()

    model_results: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    optuna_artifacts: dict[str, dict[str, str]] = {}

    for model_name in model_candidates:
        if model_name not in available_models:
            model_results[model_name] = {"error": "Model not registered."}
            continue
        params: dict[str, Any] = {}
        model_param_overrides = {}
        if isinstance(param_overrides_cfg, dict):
            candidate_override = param_overrides_cfg.get(model_name, {})
            if isinstance(candidate_override, dict):
                model_param_overrides = candidate_override
        tuning_score = None
        tuning_details: dict[str, Any] = {}
        warning_msg: str | None = None
        if tuning_enabled:
            try:
                params, tuning_score, tuning_details = tune_model_with_optuna(
                    model_name=model_name,
                    X_train=X_train_bal,
                    y_train=y_train_bal,
                    tuning_cfg={**tuning_cfg, "seed": seed},
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
                "class_weight": class_weight_cfg,
                "label_order": class_metadata.get("class_indices", []),
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
            _add_named_per_class_metrics(
                payload["metrics"],
                payload["artifacts"].get("per_class_metrics_test"),
                class_metadata.get("class_index_to_label", {}),
            )
            if tuning_details:
                payload["tuning"] = {
                    "scoring": str(tuning_cfg.get("scoring", "f1_macro")),
                    "cv_folds": int(tuning_cfg.get("cv_folds", 3)),
                    "objective_source": tuning_details.get("objective_source"),
                    "objective_metric": primary_metric,
                    "best_validation_metrics": tuning_details.get("best_validation_metrics", {}),
                    "best_per_class_f1": tuning_details.get("best_per_class_f1", {}),
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
        },
        "summary_mode": "compact" if compact_mode else "full",
    }
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
    )
    summary["artifact_paths"].update(contract_paths)

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
