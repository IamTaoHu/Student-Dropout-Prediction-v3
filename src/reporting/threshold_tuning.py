"""Post-hoc threshold tuning for multiclass probability outputs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.reporting.artifact_manifest import update_artifact_manifest
from src.reporting.benchmark_contract import BENCHMARK_SUMMARY_VERSION
from src.reporting.benchmark_summary import save_benchmark_summary
from src.reporting.standard_artifacts import (
    ensure_standard_output_layout,
    infer_source_experiment_name,
    resolve_results_dir,
    write_skipped_explainability_report,
)


def _resolve_source_summary(exp_cfg: dict[str, Any]) -> Path:
    inputs = exp_cfg.get("inputs", {})
    source = inputs.get("benchmark_summary_path") or inputs.get(
        "benchmark_summary", "results/exp_uct_3class_boosting_optuna_v1/benchmark_summary.json"
    )
    path = Path(str(source))
    if path.is_dir():
        path = path / "benchmark_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Threshold tuning source benchmark summary not found: {path}")
    return path


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name)).strip("_").lower()


def _coerce_1d_labels(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _predict_with_thresholds(probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = probabilities / thresholds.reshape(1, -1)
    pred_idx = np.argmax(adjusted, axis=1)
    return labels[pred_idx]


def _plot_confusion(cm: np.ndarray, title: str, out_path: Path, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = f"{cm[i, j]:.2f}" if normalized else str(int(cm[i, j]))
            ax.text(j, i, label, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _normalize_cm(cm: np.ndarray) -> np.ndarray:
    rows = cm.sum(axis=1, keepdims=True)
    return np.divide(cm.astype(float), rows, out=np.zeros_like(cm, dtype=float), where=rows != 0)


def _status_from_path(path: str | Path, missing_reason: str = "missing_expected_output") -> dict[str, str]:
    resolved = Path(path)
    if resolved.exists():
        return {"status": "generated", "path": str(resolved)}
    return {"status": "failed", "path": str(resolved), "reason": missing_reason}


def _search_thresholds(
    y_valid: np.ndarray,
    p_valid: np.ndarray,
    labels: np.ndarray,
    grid_values: list[float],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_acc = -1.0
    best_thresholds = np.ones(p_valid.shape[1], dtype=float)
    best_metrics = _metrics(y_valid, _predict_with_thresholds(p_valid, labels, best_thresholds))

    for thresholds_tuple in itertools.product(grid_values, repeat=p_valid.shape[1]):
        thresholds = np.asarray(thresholds_tuple, dtype=float)
        y_pred = _predict_with_thresholds(p_valid, labels, thresholds)
        metrics = _metrics(y_valid, y_pred)
        report = classification_report(y_valid, y_pred, labels=labels.tolist(), output_dict=True, zero_division=0)
        row = {
            **{f"threshold_{int(labels[i])}": float(thresholds[i]) for i in range(len(labels))},
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "weighted_f1": metrics["weighted_f1"],
        }
        for class_label in labels:
            class_key = str(int(class_label)) if isinstance(class_label, (int, np.integer)) else str(class_label)
            class_metrics = report.get(class_key, {})
            row[f"f1_class_{class_key}"] = float(class_metrics.get("f1-score", 0.0))
            row[f"precision_class_{class_key}"] = float(class_metrics.get("precision", 0.0))
            row[f"recall_class_{class_key}"] = float(class_metrics.get("recall", 0.0))
        rows.append(row)

        if (metrics["macro_f1"] > best_score) or (
            abs(metrics["macro_f1"] - best_score) < 1e-12 and metrics["accuracy"] > best_acc
        ):
            best_score = metrics["macro_f1"]
            best_acc = metrics["accuracy"]
            best_thresholds = thresholds
            best_metrics = metrics

    return pd.DataFrame(rows), best_thresholds, best_metrics


def run_threshold_tuning_experiment(exp_cfg: dict[str, Any], experiment_config_path: Path) -> dict[str, Any]:
    experiment = exp_cfg.get("experiment", {})
    experiment_id = str(experiment.get("id", "exp_uct_3class_threshold_tuning_v1"))
    output_dir = resolve_results_dir(exp_cfg, experiment_id=experiment_id)
    layout = ensure_standard_output_layout(output_dir)
    artifact_policy = exp_cfg.get("artifact_policy", {}) or {}
    allow_inherited = bool(artifact_policy.get("allow_inherited_artifacts", True))
    split_source = str(exp_cfg.get("analysis", {}).get("split_source", "validation")).strip().lower()

    source_summary_path = _resolve_source_summary(exp_cfg)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_artifact_paths = source_summary.get("artifact_paths", {})
    source_models = source_summary.get("model_results", {})
    requested_models = exp_cfg.get("analysis", {}).get("models") or list(source_models.keys())

    grid_cfg = exp_cfg.get("analysis", {}).get("threshold_grid", {})
    start = float(grid_cfg.get("start", 0.6))
    stop = float(grid_cfg.get("stop", 1.4))
    step = float(grid_cfg.get("step", 0.05))
    grid_values = np.round(np.arange(start, stop + 1e-12, step), 6).tolist()

    comparison_rows: list[dict[str, Any]] = []
    model_status: dict[str, dict[str, Any]] = {}
    per_class_gain: dict[str, list[float]] = {}
    benchmark_model_results: dict[str, Any] = {}
    tuned_test_preds: dict[str, list[Any]] = {}
    y_true_test_reference: list[Any] | None = None

    for model_name in requested_models:
        payload = source_models.get(model_name, {})
        artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
        p_valid = artifacts.get("y_proba_valid")
        y_valid = artifacts.get("y_true_valid")
        p_test = artifacts.get("y_proba_test")
        y_test = artifacts.get("y_true_test")
        labels = artifacts.get("labels")

        if split_source != "validation":
            model_status[model_name] = {
                "status": "skipped",
                "reason": f"unsupported_split_source:{split_source}; expected validation",
            }
            continue
        if p_valid is None or y_valid is None:
            model_status[model_name] = {
                "status": "skipped",
                "reason": "missing_validation_probabilities_or_labels",
            }
            continue
        if p_test is None or y_test is None:
            model_status[model_name] = {"status": "skipped", "reason": "missing_test_probabilities_or_labels"}
            continue
        if labels is None:
            model_status[model_name] = {"status": "skipped", "reason": "missing_label_index_order"}
            continue

        try:
            labels_arr = _coerce_1d_labels(labels)
            p_valid_arr = np.asarray(p_valid, dtype=float)
            p_test_arr = np.asarray(p_test, dtype=float)
            y_valid_arr = _coerce_1d_labels(y_valid)
            y_test_arr = _coerce_1d_labels(y_test)
            if p_valid_arr.ndim != 2 or p_test_arr.ndim != 2:
                raise ValueError("probability arrays must be 2D")
            if p_valid_arr.shape[1] != len(labels_arr) or p_test_arr.shape[1] != len(labels_arr):
                raise ValueError("probability columns do not match labels")

            baseline_valid_pred = _predict_with_thresholds(p_valid_arr, labels_arr, np.ones(len(labels_arr)))
            baseline_test_pred = _predict_with_thresholds(p_test_arr, labels_arr, np.ones(len(labels_arr)))

            search_df, best_thresholds, best_valid_metrics = _search_thresholds(
                y_valid=y_valid_arr,
                p_valid=p_valid_arr,
                labels=labels_arr,
                grid_values=grid_values,
            )
            tuned_test_pred = _predict_with_thresholds(p_test_arr, labels_arr, best_thresholds)

            before_metrics = _metrics(y_test_arr, baseline_test_pred)
            after_metrics = _metrics(y_test_arr, tuned_test_pred)
            improved = bool(after_metrics["macro_f1"] > before_metrics["macro_f1"] + 1e-12)

            report_before = classification_report(
                y_test_arr,
                baseline_test_pred,
                labels=labels_arr.tolist(),
                output_dict=True,
                zero_division=0,
            )
            report_after = classification_report(
                y_test_arr,
                tuned_test_pred,
                labels=labels_arr.tolist(),
                output_dict=True,
                zero_division=0,
            )
            for class_label in labels_arr:
                key = str(int(class_label)) if isinstance(class_label, (int, np.integer)) else str(class_label)
                before_f1 = float(report_before.get(key, {}).get("f1-score", 0.0))
                after_f1 = float(report_after.get(key, {}).get("f1-score", 0.0))
                per_class_gain.setdefault(key, []).append(after_f1 - before_f1)

            token = _safe_name(model_name)
            search_path = output_dir / f"threshold_search_results_{token}.csv"
            best_path = output_dir / f"best_thresholds_{token}.json"
            before_report_path = output_dir / f"classification_report_before_{token}.json"
            after_report_path = output_dir / f"classification_report_after_{token}.json"

            search_df.to_csv(search_path, index=False)
            best_payload = {
                "model": model_name,
                "thresholds": {str(labels_arr[i]): float(best_thresholds[i]) for i in range(len(labels_arr))},
                "best_validation_metrics": best_valid_metrics,
                "grid": {"start": start, "stop": stop, "step": step, "num_points": len(grid_values)},
            }
            best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
            before_report_path.write_text(json.dumps(report_before, indent=2), encoding="utf-8")
            after_report_path.write_text(json.dumps(report_after, indent=2), encoding="utf-8")

            cm_before = confusion_matrix(y_test_arr, baseline_test_pred, labels=labels_arr.tolist())
            cm_after = confusion_matrix(y_test_arr, tuned_test_pred, labels=labels_arr.tolist())
            cm_before_path = output_dir / f"confusion_matrix_before_{token}.png"
            cm_after_path = output_dir / f"confusion_matrix_after_{token}.png"
            cm_before_norm_path = output_dir / f"confusion_matrix_before_{token}_normalized.png"
            cm_after_norm_path = output_dir / f"confusion_matrix_after_{token}_normalized.png"
            _plot_confusion(cm_before, f"Confusion Matrix Before - {model_name}", cm_before_path, normalized=False)
            _plot_confusion(cm_after, f"Confusion Matrix After - {model_name}", cm_after_path, normalized=False)
            _plot_confusion(
                _normalize_cm(cm_before),
                f"Normalized Confusion Matrix Before - {model_name}",
                cm_before_norm_path,
                normalized=True,
            )
            _plot_confusion(
                _normalize_cm(cm_after),
                f"Normalized Confusion Matrix After - {model_name}",
                cm_after_norm_path,
                normalized=True,
            )

            comparison_rows.append(
                {
                    "model": model_name,
                    "macro_f1_before": before_metrics["macro_f1"],
                    "macro_f1_after": after_metrics["macro_f1"],
                    "accuracy_before": before_metrics["accuracy"],
                    "accuracy_after": after_metrics["accuracy"],
                    "balanced_accuracy_before": before_metrics["balanced_accuracy"],
                    "balanced_accuracy_after": after_metrics["balanced_accuracy"],
                    "improved": improved,
                }
            )
            benchmark_model_results[model_name] = {
                "metrics": {
                    "test_accuracy": after_metrics["accuracy"],
                    "test_macro_f1": after_metrics["macro_f1"],
                    "test_weighted_f1": after_metrics["weighted_f1"],
                    "test_balanced_accuracy": after_metrics["balanced_accuracy"],
                    "test_macro_f1_before": before_metrics["macro_f1"],
                },
                "artifacts": {
                    "labels": labels_arr.tolist(),
                    "y_true_test": y_test_arr.tolist(),
                    "y_pred_test": tuned_test_pred.tolist(),
                    "y_proba_test": p_test_arr.tolist(),
                    "confusion_matrix": cm_after.tolist(),
                },
            }
            tuned_test_preds[model_name] = tuned_test_pred.tolist()
            y_true_test_reference = y_test_arr.tolist()
            model_status[model_name] = {
                "status": "created",
                "paths": [
                    str(search_path),
                    str(best_path),
                    str(before_report_path),
                    str(after_report_path),
                    str(cm_before_path),
                    str(cm_after_path),
                    str(cm_before_norm_path),
                    str(cm_after_norm_path),
                ],
            }
        except Exception as exc:
            model_status[model_name] = {"status": "failed", "reason": f"threshold_tuning_failed: {exc}"}

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "threshold_comparison.csv"
    if comparison_df.empty:
        comparison_df = pd.DataFrame(
            columns=[
                "model",
                "macro_f1_before",
                "macro_f1_after",
                "accuracy_before",
                "accuracy_after",
                "balanced_accuracy_before",
                "balanced_accuracy_after",
                "improved",
            ]
        )
    comparison_df.to_csv(comparison_path, index=False)
    leaderboard_df = comparison_df.copy()
    if not leaderboard_df.empty:
        leaderboard_df = leaderboard_df.sort_values("macro_f1_after", ascending=False).reset_index(drop=True)

    improved_models = comparison_df.loc[comparison_df["improved"] == True, "model"].tolist() if not comparison_df.empty else []
    class_benefited = None
    if per_class_gain:
        gain_df = pd.DataFrame({"class_name": list(per_class_gain.keys()), "mean_delta_f1": [np.mean(v) for v in per_class_gain.values()]})
        if not gain_df.empty:
            class_benefited = str(gain_df.sort_values("mean_delta_f1", ascending=False).iloc[0]["class_name"])

    worth_keeping = bool(len(improved_models) > 0)
    md_lines = [
        "# Threshold Tuning Before/After Report",
        "",
        f"- Source benchmark: `{source_summary_path}`",
        f"- Models improved: `{', '.join(improved_models) if improved_models else 'none'}`",
        f"- Class benefited most: `{class_benefited if class_benefited is not None else 'N/A'}`",
        f"- Keep threshold tuning in UCT pipeline: `{'yes' if worth_keeping else 'no'}`",
    ]
    report_path = output_dir / "before_after_threshold_report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    best_model = str(leaderboard_df.iloc[0]["model"]) if not leaderboard_df.empty else None
    benchmark_summary_payload = {
        "experiment_id": experiment_id,
        "benchmark_summary_version": BENCHMARK_SUMMARY_VERSION,
        "schema_version": BENCHMARK_SUMMARY_VERSION,
        "dataset_name": source_summary.get("dataset_name", "uct_student"),
        "target_formulation": source_summary.get("target_formulation", "three_class"),
        "primary_metric": "test_macro_f1",
        "best_model": best_model,
        "output_dir": str(output_dir),
        "model_results": benchmark_model_results,
        "leaderboard": [
            {
                "model": row["model"],
                "test_macro_f1": row["macro_f1_after"],
                "test_accuracy": row["accuracy_after"],
                "test_balanced_accuracy": row["balanced_accuracy_after"],
                "test_macro_f1_before": row["macro_f1_before"],
                "improved": bool(row["improved"]),
            }
            for row in leaderboard_df.to_dict(orient="records")
        ],
        "artifact_paths": {
            "benchmark_summary": str(output_dir / "benchmark_summary.json"),
            "leaderboard": str(output_dir / "leaderboard.csv"),
            "benchmark_markdown": str(output_dir / "benchmark_summary.md"),
            "artifact_manifest": str(output_dir / "artifact_manifest.json"),
        },
    }
    for key in (
        "best_model",
        "X_train_preprocessed",
        "X_valid_preprocessed",
        "X_test_preprocessed",
        "y_train",
        "y_valid",
        "y_test",
        "preprocessing_transformer",
    ):
        if allow_inherited and source_artifact_paths.get(key):
            benchmark_summary_payload["artifact_paths"][key] = str(source_artifact_paths[key])
    save_benchmark_summary(benchmark_summary_payload, output_dir, compact=False)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "dataset_name": benchmark_summary_payload["dataset_name"],
                "target_formulation": benchmark_summary_payload["target_formulation"],
                "primary_metric": "test_macro_f1",
                "best_model": best_model,
                "best_model_metrics": benchmark_model_results.get(best_model, {}).get("metrics", {}) if best_model else {},
                "leaderboard": benchmark_summary_payload["leaderboard"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    predictions_path = output_dir / "predictions.csv"
    pred_df = pd.DataFrame({"y_true": y_true_test_reference or []})
    if best_model and best_model in tuned_test_preds:
        pred_df["y_pred"] = tuned_test_preds[best_model]
    pred_df.to_csv(predictions_path, index=False)

    runtime_metadata_path = layout["runtime_artifacts"] / "runtime_metadata.json"
    runtime_metadata_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "mode": "threshold_tuning",
                "source_benchmark_summary": str(source_summary_path),
                "best_model": best_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    explain_json_path, explain_md_path = write_skipped_explainability_report(
        output_dir=output_dir,
        reason="posthoc_threshold_tuning_uses_existing_models",
        details=(
            "Explanations should be computed on the inherited underlying model. "
            "Threshold tuning modifies only the decision layer, not model feature attributions."
        ),
    )

    summary_payload = {
        "experiment_id": experiment_id,
        "mode": "threshold_tuning",
        "source_benchmark_summary": str(source_summary_path),
        "models_requested": requested_models,
        "model_status": model_status,
        "improved_models": improved_models,
        "class_benefited_most": class_benefited,
        "worth_keeping": worth_keeping,
        "artifact_paths": {
            "benchmark_summary": str(output_dir / "benchmark_summary.json"),
            "leaderboard": str(output_dir / "leaderboard.csv"),
            "benchmark_markdown": str(output_dir / "benchmark_summary.md"),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "threshold_comparison_csv": str(comparison_path),
            "before_after_threshold_report_md": str(report_path),
            "artifact_manifest": str(output_dir / "artifact_manifest.json"),
        },
        "config_path": str(experiment_config_path),
    }
    summary_path = output_dir / "threshold_tuning_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    source_experiment = infer_source_experiment_name(source_summary_path)
    confusion_paths = [str(p) for p in sorted(output_dir.glob("confusion_matrix_*.png"))]
    inherited_source_status = "inherited" if allow_inherited else "skipped"
    inherited_source_reason = None if allow_inherited else "allow_inherited_artifacts_disabled_in_config"
    mandatory_updates = {
        "benchmark_summary_json": _status_from_path(output_dir / "benchmark_summary.json"),
        "benchmark_summary_md": _status_from_path(output_dir / "benchmark_summary.md"),
        "leaderboard_csv": _status_from_path(output_dir / "leaderboard.csv"),
        "metrics_json": _status_from_path(metrics_path),
        "predictions_csv": _status_from_path(predictions_path),
        "runtime_artifacts_dir": _status_from_path(layout["runtime_artifacts"]),
        "model_dir": _status_from_path(layout["model"]),
        "figures_dir": _status_from_path(layout["figures"]),
        "explainability_dir": _status_from_path(layout["explainability"]),
        "explainability_report_json": {
            **_status_from_path(explain_json_path),
            "path": str(explain_json_path),
            "reason": "posthoc_threshold_tuning_uses_existing_models",
        },
        "explainability_report_md": {
            **_status_from_path(explain_md_path),
            "path": str(explain_md_path),
            "reason": "posthoc_threshold_tuning_uses_existing_models",
        },
        "source_benchmark_summary_json": {
            "status": inherited_source_status,
            "path": str(source_summary_path),
            "source_experiment": source_experiment,
            "source_path": str(source_summary_path),
            "reason": inherited_source_reason,
        },
        "threshold_tuning_summary_json": _status_from_path(summary_path),
        "threshold_comparison_csv": _status_from_path(comparison_path),
        "before_after_threshold_report_md": _status_from_path(report_path),
        "confusion_matrix_artifacts": (
            {
                "status": "generated",
                "paths": confusion_paths,
            }
            if confusion_paths
            else {
                "status": "skipped",
                "reason": "no_threshold_tuned_models_with_probability_artifacts",
                "path": str(output_dir),
            }
        ),
    }
    inherited_mapping = {
        "best_model": "best_model_reference",
        "X_train_preprocessed": "X_train_preprocessed_reference",
        "X_valid_preprocessed": "X_valid_preprocessed_reference",
        "X_test_preprocessed": "X_test_preprocessed_reference",
        "y_train": "y_train_reference",
        "y_valid": "y_valid_reference",
        "y_test": "y_test_reference",
        "preprocessing_transformer": "preprocessing_transformer_reference",
        "learning_curve": "learning_curve_png",
        "pr_curve": "pr_curve_png",
    }
    for source_key, manifest_key in inherited_mapping.items():
        source_path = source_artifact_paths.get(source_key)
        if not source_path:
            continue
        if allow_inherited:
            mandatory_updates[manifest_key] = {
                "status": "inherited",
                "path": str(source_path),
                "source_experiment": source_experiment,
                "source_path": str(source_path),
            }
        else:
            mandatory_updates[manifest_key] = {
                "status": "skipped",
                "path": str(source_path),
                "reason": "allow_inherited_artifacts_disabled_in_config",
            }
    if "learning_curve_png" not in mandatory_updates:
        mandatory_updates["learning_curve_png"] = {
            "status": "skipped",
            "path": str(layout["figures"] / "learning_curve.png"),
            "reason": "learning_curve_not_available_for_posthoc_threshold_tuning",
        }
    if "pr_curve_png" not in mandatory_updates:
        mandatory_updates["pr_curve_png"] = {
            "status": "skipped",
            "path": str(layout["figures"] / "pr_curve.png"),
            "reason": "pr_curve_not_available_for_posthoc_threshold_tuning",
        }
    generated_models = [m for m, s in model_status.items() if s.get("status") == "generated"]
    optional_updates = {
        "per_model_threshold_tuning_artifacts": {
            "status": "generated" if generated_models else "skipped",
            "details": model_status,
            "reason": None if generated_models else "no_models_with_validation_probabilities",
        }
    }
    update_artifact_manifest(
        output_dir=output_dir,
        mandatory_updates=mandatory_updates,
        optional_updates=optional_updates,
        metadata_updates={
            "experiment_id": experiment_id,
            "manifest_scope": "threshold_tuning",
            "source_benchmark_summary": str(source_summary_path),
        },
    )
    return summary_payload
