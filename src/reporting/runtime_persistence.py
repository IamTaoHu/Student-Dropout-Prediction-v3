from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.reporting.benchmark_contract import REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS
from src.reporting.prediction_exports import _build_prediction_export_dataframe, _safe_filename_token

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
    runtime_artifact_overrides_by_model: dict[str, dict[str, Any]] | None = None,
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

    override_payload: dict[str, Any] = {}
    if best_model_name and isinstance(runtime_artifact_overrides_by_model, dict):
        override_payload = runtime_artifact_overrides_by_model.get(best_model_name, {}) or {}
    X_train_runtime = override_payload.get("X_train", preprocessing_artifacts.X_train)
    X_valid_runtime = override_payload.get("X_valid", preprocessing_artifacts.X_valid)
    X_test_runtime = override_payload.get("X_test", preprocessing_artifacts.X_test)
    runtime_feature_names = override_payload.get(
        "feature_names",
        preprocessing_artifacts.metadata.get("output_feature_names", []),
    )

    ext = ".csv" if str(file_format).strip().lower() == "csv" else ".parquet"
    X_train_path = _save_dataframe(X_train_runtime, runtime_dir / f"X_train_preprocessed{ext}")
    X_valid_path = _save_dataframe(X_valid_runtime, runtime_dir / f"X_valid_preprocessed{ext}")
    X_test_path = _save_dataframe(X_test_runtime, runtime_dir / f"X_test_preprocessed{ext}")
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
        "feature_names": runtime_feature_names,
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

    locked_vocab_summary = preprocessing_artifacts.metadata.get("locked_category_vocabulary_summary")
    if isinstance(locked_vocab_summary, dict) and locked_vocab_summary:
        vocabulary_summary_path = runtime_dir / "uci_category_vocabulary_summary.json"
        vocabulary_summary_path.write_text(
            json.dumps(locked_vocab_summary, indent=2),
            encoding="utf-8",
        )
        artifact_paths["uci_category_vocabulary_summary"] = str(vocabulary_summary_path)

    train_only_preprocessing_path = runtime_dir / "train_only_preprocessing.json"
    train_only_preprocessing_payload = {
        "experiment_id": summary.get("experiment_id"),
        "dataset_name": summary.get("dataset_name"),
        "preprocessing": summary.get("preprocessing", {}),
        "dataset_rows": summary.get("dataset_rows", {}),
    }
    train_only_preprocessing_path.write_text(
        json.dumps(train_only_preprocessing_payload, indent=2),
        encoding="utf-8",
    )
    artifact_paths["train_only_preprocessing"] = str(train_only_preprocessing_path)
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
        paths["best_model"] = str(model_path)
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


def _ensure_explainability_compatible_artifact_paths(summary: dict[str, Any]) -> None:
    artifact_paths = summary.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError("Cannot write explainability-compatible benchmark_summary: missing artifact_paths object.")

    best_model_path = artifact_paths.get("best_model")
    if not best_model_path:
        fallback_best_model = artifact_paths.get("best_model_copy")
        if fallback_best_model:
            artifact_paths["best_model"] = str(fallback_best_model)
            best_model_path = artifact_paths["best_model"]

    missing: list[str] = []
    missing_files: list[str] = []
    for key in REQUIRED_EXPLAINABILITY_ARTIFACT_KEYS:
        value = artifact_paths.get(key)
        if not value:
            missing.append(key)
            continue
        if not Path(str(value)).exists():
            missing_files.append(f"{key}={value}")

    if missing:
        raise ValueError(
            "Cannot write explainability-compatible benchmark_summary: "
            f"missing required artifact path entries {missing}."
        )
    if missing_files:
        raise ValueError(
            "Cannot write explainability-compatible benchmark_summary: "
            f"required artifacts do not exist on disk: {missing_files}."
        )


def _write_benchmark_failure_summary(
    output_dir: Path,
    *,
    experiment_id: str,
    requested_models: list[str],
    model_results: dict[str, Any],
    failed_models: dict[str, str] | None = None,
    successful_models: list[str] | None = None,
    reason: str = "no_candidate_models_completed",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_payload = failed_models if isinstance(failed_models, dict) else {}
    successful_payload = successful_models if isinstance(successful_models, list) else []
    status = "failed" if not successful_payload else "partial_success"
    failure_payload = {
        "experiment_id": experiment_id,
        "status": status,
        "reason": reason,
        "requested_models": list(requested_models),
        "successful_models": list(successful_payload),
        "successful_candidate_count": int(len(successful_payload)),
        "failed_models": {
            str(model_name): {
                "error": str(error_msg),
                "exception_class": str(str(error_msg).split(":", 1)[0]).strip() if error_msg else "",
                "message": str(str(error_msg).split(":", 1)[1]).strip() if isinstance(error_msg, str) and ":" in error_msg else str(error_msg),
            }
            for model_name, error_msg in failed_payload.items()
        },
        "failed_candidate_count": int(len(failed_payload)),
        "model_results": model_results,
    }
    json_path = output_dir / "benchmark_failure_summary.json"
    md_path = output_dir / "benchmark_failure_summary.md"
    json_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")
    lines = [
        "# Benchmark Failure Summary",
        "",
        f"- Experiment ID: `{experiment_id}`",
        f"- Status: `{status}`",
        f"- Reason: `{reason}`",
        f"- Requested models: `{requested_models}`",
        f"- Successful models: `{successful_payload}`",
        f"- Failed model count: `{len(failed_payload)}`",
        "",
        "## Model Failures",
        "",
    ]
    for model_name in requested_models:
        if model_name not in failed_payload:
            continue
        error_msg = failed_payload.get(model_name, "unknown_failure")
        lines.append(f"- `{model_name}`: `{error_msg}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        "[finalization][failure_summary] "
        f"status={status} reason={reason} "
        f"successful_candidates={len(successful_payload)} "
        f"failed_candidates={len(failed_payload)} "
        f"path={json_path}"
    )
    return {
        "benchmark_failure_summary_json": str(json_path),
        "benchmark_failure_summary_md": str(md_path),
    }


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
