"""Error-audit reporting flow for UCT three-class benchmark outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from src.reporting.artifact_manifest import update_artifact_manifest
from src.reporting.benchmark_contract import BENCHMARK_SUMMARY_VERSION
from src.reporting.benchmark_summary import save_benchmark_summary
from src.reporting.standard_artifacts import (
    ensure_standard_output_layout,
    infer_source_experiment_name,
    resolve_results_dir,
    write_skipped_explainability_report,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to run error-audit experiments.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _status_from_path(path: str | Path, missing_reason: str = "missing_expected_output") -> dict[str, str]:
    resolved = Path(path)
    if resolved.exists():
        return {"status": "generated", "path": str(resolved)}
    return {"status": "failed", "path": str(resolved), "reason": missing_reason}


def _safe_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_").lower() or "model"


def _coerce_label(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        return _coerce_label(value[0])
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    return text


def _sort_labels(values: list[Any]) -> list[Any]:
    def _key(item: Any) -> tuple[int, Any]:
        if isinstance(item, (int, np.integer)):
            return (0, int(item))
        return (1, str(item))

    return sorted(values, key=_key)


def _read_series(path: Path) -> pd.Series:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "value" in frame.columns:
            return frame["value"]
        return frame.iloc[:, 0]
    frame = pd.read_parquet(path)
    if "value" in frame.columns:
        return frame["value"]
    return frame.iloc[:, 0]


def _load_y_true(benchmark_dir: Path) -> pd.Series:
    runtime_dir = benchmark_dir / "runtime_artifacts"
    for name in ("y_test.csv", "y_test.parquet"):
        candidate = runtime_dir / name
        if candidate.exists():
            return _read_series(candidate).map(_coerce_label)

    predictions_path = benchmark_dir / "predictions.csv"
    if predictions_path.exists():
        pred_df = pd.read_csv(predictions_path)
        if "y_true" in pred_df.columns:
            return pred_df["y_true"].map(_coerce_label)

    raise FileNotFoundError(
        "Unable to locate y_true values. Expected runtime_artifacts/y_test.csv|parquet or predictions.csv with y_true."
    )


def _build_label_name_map(summary: dict[str, Any], dataset_cfg: dict[str, Any]) -> dict[Any, str]:
    mapping_cfg = dataset_cfg.get("target_mappings", {}).get("three_class", {})
    label_name_map: dict[Any, str] = {}
    for raw_name, mapped_value in mapping_cfg.items():
        label_name_map[_coerce_label(mapped_value)] = str(raw_name)

    # Fallback from first model label list when mapping is unavailable/incomplete.
    for payload in summary.get("model_results", {}).values():
        labels = payload.get("artifacts", {}).get("labels") if isinstance(payload, dict) else None
        if labels:
            for lbl in labels:
                label_name_map.setdefault(_coerce_label(lbl), str(lbl))
            break
    return label_name_map


def _plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[Any],
    label_name_map: dict[Any, str],
    title: str,
    output_path: Path,
    normalized: bool,
) -> None:
    display_labels = [label_name_map.get(lbl, str(lbl)) for lbl in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax)

    if normalized:
        fmt = ".2f"
    else:
        fmt = "d"

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_value = format(value, fmt) if normalized else str(int(value))
            ax.text(j, i, text_value, ha="center", va="center", color="black")

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _resolve_benchmark_summary_path(exp_cfg: dict[str, Any]) -> Path:
    inputs_cfg = exp_cfg.get("inputs", {})
    provided_path = inputs_cfg.get("benchmark_summary_path")
    if provided_path:
        path = Path(str(provided_path))
        if path.is_dir():
            path = path / "benchmark_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Provided benchmark summary path does not exist: {path}")
        return path

    root = Path(str(inputs_cfg.get("benchmark_results_root", "results")))
    candidates = sorted(root.glob("*/benchmark_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset_name = str(payload.get("dataset_name", "")).strip().lower()
        target_formulation = str(payload.get("target_formulation", "")).strip().lower()
        if dataset_name == "uct_student" and target_formulation == "three_class":
            return path

    for path in candidates:
        if "uct" in str(path.parent.name).lower() and "3class" in str(path.parent.name).lower():
            return path

    raise FileNotFoundError(
        "Could not auto-discover a UCT three-class benchmark summary. Set inputs.benchmark_summary_path explicitly."
    )


def _get_top_models(summary: dict[str, Any], benchmark_dir: Path, top_k_models: int | None) -> list[str]:
    leaderboard_path = benchmark_dir / "leaderboard.csv"
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        if "model" in leaderboard.columns and not leaderboard.empty:
            models = leaderboard["model"].dropna().astype(str).tolist()
            return models[:top_k_models] if top_k_models and top_k_models > 0 else models

    models = list(summary.get("model_results", {}).keys())
    return models[:top_k_models] if top_k_models and top_k_models > 0 else models


def run_uct_3class_error_audit(
    exp_cfg: dict[str, Any],
    experiment_config_path: Path,
) -> dict[str, Any]:
    experiment_id = str(exp_cfg.get("experiment", {}).get("id", "exp_uct_3class_error_audit"))
    output_dir = resolve_results_dir(exp_cfg, experiment_id=experiment_id)
    layout = ensure_standard_output_layout(output_dir)
    artifact_policy = exp_cfg.get("artifact_policy", {}) or {}
    allow_inherited = bool(artifact_policy.get("allow_inherited_artifacts", True))

    benchmark_summary_path = _resolve_benchmark_summary_path(exp_cfg)
    benchmark_dir = benchmark_summary_path.parent
    benchmark_summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))

    dataset_cfg_path = Path(exp_cfg.get("experiment", {}).get("dataset_config", "configs/datasets/uct_student.yaml"))
    dataset_cfg = _load_yaml(dataset_cfg_path)
    label_name_map = _build_label_name_map(benchmark_summary, dataset_cfg)

    analysis_cfg = exp_cfg.get("analysis", {})
    top_k_raw = analysis_cfg.get("top_k_models")
    top_k_models = int(top_k_raw) if top_k_raw is not None else None
    top_models = _get_top_models(benchmark_summary, benchmark_dir, top_k_models)

    y_true = _load_y_true(benchmark_dir)
    model_results = benchmark_summary.get("model_results", {})

    comparison_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    aggregate_confusions: list[dict[str, Any]] = []
    per_model_status: dict[str, dict[str, Any]] = {}
    model_macro_f1: dict[str, float] = {}
    model_summary_paths: list[str] = []
    benchmark_model_results: dict[str, Any] = {}
    y_pred_by_model: dict[str, list[Any]] = {}

    for model_name in top_models:
        safe_name = _safe_model_name(model_name)
        payload = model_results.get(model_name, {})
        artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
        y_pred_raw = artifacts.get("y_pred_test")
        if y_pred_raw is None:
            per_model_status[model_name] = {
                "status": "failed",
                "reason": "missing_y_pred_test_in_benchmark_summary",
            }
            continue

        try:
            y_pred = pd.Series(y_pred_raw).map(_coerce_label)
            labels = artifacts.get("labels") or sorted(set(y_true.tolist()) | set(y_pred.tolist()))
            labels = _sort_labels([_coerce_label(lbl) for lbl in labels])

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_path = output_dir / f"confusion_matrix_{safe_name}.png"
            _plot_confusion_matrix(
                matrix=cm.astype(float),
                labels=labels,
                label_name_map=label_name_map,
                title=f"Confusion Matrix - {model_name}",
                output_path=cm_path,
                normalized=False,
            )

            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
            cm_norm_path = output_dir / f"confusion_matrix_{safe_name}_normalized.png"
            _plot_confusion_matrix(
                matrix=cm_norm,
                labels=labels,
                label_name_map=label_name_map,
                title=f"Normalized Confusion Matrix - {model_name}",
                output_path=cm_norm_path,
                normalized=True,
            )

            report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            report_path = output_dir / f"classification_report_{safe_name}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            per_class_rows: list[dict[str, Any]] = []
            for label in labels:
                key = str(label)
                class_metrics = report.get(key, {})
                class_name = label_name_map.get(label, key)
                row = {
                    "class_name": class_name,
                    "precision": float(class_metrics.get("precision", 0.0)),
                    "recall": float(class_metrics.get("recall", 0.0)),
                    "f1": float(class_metrics.get("f1-score", 0.0)),
                    "support": int(class_metrics.get("support", 0)),
                }
                per_class_rows.append(row)
                comparison_rows.append(
                    {
                        "model": model_name,
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                        **row,
                    }
                )

            per_class_path = output_dir / f"per_class_metrics_{safe_name}.csv"
            pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

            confusion_rows: list[dict[str, Any]] = []
            for true_idx, true_label in enumerate(labels):
                total_true = int(cm[true_idx].sum())
                for pred_idx, pred_label in enumerate(labels):
                    if true_idx == pred_idx:
                        continue
                    count = int(cm[true_idx, pred_idx])
                    if count <= 0:
                        continue
                    confusion_row = {
                        "model": model_name,
                        "true_class": label_name_map.get(true_label, str(true_label)),
                        "predicted_class": label_name_map.get(pred_label, str(pred_label)),
                        "count": count,
                        "rate_within_true_class": (float(count) / float(total_true)) if total_true > 0 else 0.0,
                    }
                    confusion_rows.append(confusion_row)
                    aggregate_confusions.append(confusion_row)

            top_confusions_df = pd.DataFrame(confusion_rows).sort_values("count", ascending=False)
            top_confusions_path = output_dir / f"top_confusions_{safe_name}.csv"
            if top_confusions_df.empty:
                top_confusions_df = pd.DataFrame(
                    columns=["model", "true_class", "predicted_class", "count", "rate_within_true_class"]
                )
            top_confusions_df.to_csv(top_confusions_path, index=False)

            model_summary_path = output_dir / f"summary_{safe_name}.md"
            model_summary_lines = [
                f"# Error Audit Summary - {model_name}",
                "",
                f"- Accuracy: `{accuracy_score(y_true, y_pred):.4f}`",
                f"- Macro F1: `{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}`",
                f"- Balanced Accuracy: `{balanced_accuracy_score(y_true, y_pred):.4f}`",
                "",
                "## Top Confusions",
                "",
                top_confusions_df.head(5).to_markdown(index=False) if not top_confusions_df.empty else "_No off-diagonal confusions._",
            ]
            model_summary_path.write_text("\n".join(model_summary_lines), encoding="utf-8")
            model_summary_paths.append(str(model_summary_path))
            accuracy_val = float(accuracy_score(y_true, y_pred))
            macro_f1_val = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            balanced_acc_val = float(balanced_accuracy_score(y_true, y_pred))
            weighted_f1_val = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            model_macro_f1[model_name] = macro_f1_val
            leaderboard_rows.append(
                {
                    "model": model_name,
                    "test_accuracy": accuracy_val,
                    "test_macro_f1": macro_f1_val,
                    "test_weighted_f1": weighted_f1_val,
                    "test_balanced_accuracy": balanced_acc_val,
                }
            )
            y_pred_by_model[model_name] = y_pred.tolist()
            benchmark_model_results[model_name] = {
                "metrics": {
                    "test_accuracy": accuracy_val,
                    "test_macro_f1": macro_f1_val,
                    "test_weighted_f1": weighted_f1_val,
                    "test_balanced_accuracy": balanced_acc_val,
                },
                "artifacts": {
                    "labels": [_coerce_label(lbl) for lbl in labels],
                    "y_true_test": y_true.tolist(),
                    "y_pred_test": y_pred.tolist(),
                    "y_proba_test": None,
                    "confusion_matrix": cm.tolist(),
                },
            }

            per_model_status[model_name] = {
                "status": "generated",
                "paths": [
                    str(cm_path),
                    str(cm_norm_path),
                    str(report_path),
                    str(per_class_path),
                    str(top_confusions_path),
                    str(model_summary_path),
                ],
            }
        except Exception as exc:
            per_model_status[model_name] = {"status": "failed", "reason": f"processing_error: {exc}"}

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "model_comparison_classwise.csv"
    if comparison_df.empty:
        comparison_df = pd.DataFrame(
            columns=[
                "model",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        )
    comparison_df.to_csv(comparison_path, index=False)

    best_model = None
    if model_macro_f1:
        best_model = max(model_macro_f1.items(), key=lambda kv: kv[1])[0]

    weakest_class = None
    if not comparison_df.empty:
        class_avg = comparison_df.groupby("class_name", as_index=False)["f1"].mean()
        if not class_avg.empty:
            weakest_row = class_avg.sort_values("f1", ascending=True).iloc[0]
            weakest_class = str(weakest_row["class_name"])

    frequent_confusion_pair = None
    if aggregate_confusions:
        confusion_df = pd.DataFrame(aggregate_confusions)
        grouped = (
            confusion_df.groupby(["true_class", "predicted_class"], as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
        )
        if not grouped.empty:
            top_row = grouped.iloc[0]
            frequent_confusion_pair = f"{top_row['true_class']} -> {top_row['predicted_class']} (count={int(top_row['count'])})"

    recommendation = "Collect more samples for weak classes and apply class-weighted training focused on top confusion pairs."
    if weakest_class:
        recommendation = (
            f"Prioritize {weakest_class}: tune class weights and add targeted features that separate it from the most common confusion pair."
        )

    error_summary_lines = [
        "# UCT 3-Class Error Analysis Summary",
        "",
        f"- Source benchmark: `{benchmark_summary_path}`",
        f"- Models processed: `{sum(1 for s in per_model_status.values() if s.get('status') == 'generated')}` / `{len(top_models)}`",
        f"- Best model by macro F1: `{best_model}`" if best_model else "- Best model by macro F1: `N/A`",
        f"- Weakest class by average F1 across models: `{weakest_class}`" if weakest_class else "- Weakest class by average F1 across models: `N/A`",
        f"- Most frequent confusion pair: `{frequent_confusion_pair}`"
        if frequent_confusion_pair
        else "- Most frequent confusion pair: `N/A`",
        f"- Recommendation: {recommendation}",
    ]
    error_summary_path = output_dir / "error_analysis_summary.md"
    error_summary_path.write_text("\n".join(error_summary_lines), encoding="utf-8")

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    if not leaderboard_df.empty:
        leaderboard_df = leaderboard_df.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

    benchmark_summary_payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "benchmark_summary_version": BENCHMARK_SUMMARY_VERSION,
        "schema_version": BENCHMARK_SUMMARY_VERSION,
        "dataset_name": benchmark_summary.get("dataset_name", "uct_student"),
        "target_formulation": benchmark_summary.get("target_formulation", "three_class"),
        "primary_metric": "test_macro_f1",
        "best_model": best_model,
        "output_dir": str(output_dir),
        "model_results": benchmark_model_results,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "artifact_paths": {
            "benchmark_summary": str(output_dir / "benchmark_summary.json"),
            "leaderboard": str(output_dir / "leaderboard.csv"),
            "benchmark_markdown": str(output_dir / "benchmark_summary.md"),
            "artifact_manifest": str(output_dir / "artifact_manifest.json"),
        },
    }
    source_artifact_paths = benchmark_summary.get("artifact_paths", {})
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

    metrics_payload = {
        "experiment_id": experiment_id,
        "dataset_name": benchmark_summary_payload["dataset_name"],
        "target_formulation": benchmark_summary_payload["target_formulation"],
        "primary_metric": "test_macro_f1",
        "best_model": best_model,
        "best_model_metrics": benchmark_model_results.get(best_model, {}).get("metrics", {}) if best_model else {},
        "leaderboard": benchmark_summary_payload["leaderboard"],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"y_true": y_true.tolist()})
    if best_model and best_model in y_pred_by_model:
        pred_df["y_pred"] = y_pred_by_model[best_model]
    predictions_path = output_dir / "predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    runtime_metadata_path = layout["runtime_artifacts"] / "runtime_metadata.json"
    runtime_metadata_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "source_benchmark_summary": str(benchmark_summary_path),
                "best_model": best_model,
                "mode": "error_audit",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    explain_json_path, explain_md_path = write_skipped_explainability_report(
        output_dir=output_dir,
        reason="model_objects_unavailable_for_error_audit",
    )

    inherited_best_model = source_artifact_paths.get("best_model")
    if allow_inherited and inherited_best_model:
        benchmark_summary_payload["artifact_paths"]["best_model"] = str(inherited_best_model)

    audit_payload = {
        "experiment_id": experiment_id,
        "mode": "error_audit",
        "primary_metric": "macro_f1",
        "best_model": best_model,
        "source_benchmark_summary": str(benchmark_summary_path),
        "dataset_name": benchmark_summary.get("dataset_name", "uct_student"),
        "target_formulation": benchmark_summary.get("target_formulation", "three_class"),
        "models_requested": top_models,
        "models_status": per_model_status,
        "best_model_by_macro_f1": best_model,
        "weakest_class_avg_f1": weakest_class,
        "most_frequent_confusion_pair": frequent_confusion_pair,
        "artifact_paths": {
            "benchmark_summary": str(output_dir / "benchmark_summary.json"),
            "leaderboard": str(output_dir / "leaderboard.csv"),
            "benchmark_markdown": str(output_dir / "benchmark_summary.md"),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "model_comparison_classwise": str(comparison_path),
            "error_analysis_summary": str(error_summary_path),
            "model_summaries": model_summary_paths,
            "artifact_manifest": str(output_dir / "artifact_manifest.json"),
        },
        "config_path": str(experiment_config_path),
    }
    audit_summary_path = output_dir / "error_audit_summary.json"
    audit_summary_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    generated_models = [name for name, status in per_model_status.items() if status.get("status") == "generated"]
    confusion_paths = [str(p) for p in sorted(output_dir.glob("confusion_matrix_*.png"))]
    inherited_source_status = "inherited" if allow_inherited and benchmark_summary_path.exists() else "skipped"
    inherited_source_reason = (
        None if inherited_source_status == "inherited" else "allow_inherited_artifacts_disabled_in_config"
    )
    mandatory_updates: dict[str, dict[str, Any]] = {
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
            "reason": "model_objects_unavailable_for_error_audit",
        },
        "explainability_report_md": {
            **_status_from_path(explain_md_path),
            "path": str(explain_md_path),
            "reason": "model_objects_unavailable_for_error_audit",
        },
        "learning_curve_png": {
            "status": "skipped",
            "path": str(layout["figures"] / "learning_curve.png"),
            "reason": "analysis_only_experiment_no_training_curve",
        },
        "pr_curve_png": {
            "status": "skipped",
            "path": str(layout["figures"] / "pr_curve.png"),
            "reason": "analysis_only_experiment_no_probability_curve",
        },
        "source_benchmark_summary_json": {
            "status": inherited_source_status,
            "path": str(benchmark_summary_path),
            "source_experiment": infer_source_experiment_name(benchmark_summary_path),
            "source_path": str(benchmark_summary_path),
            "reason": inherited_source_reason,
        },
        "error_audit_summary_json": _status_from_path(audit_summary_path),
        "model_comparison_classwise_csv": _status_from_path(comparison_path),
        "error_analysis_summary_md": _status_from_path(error_summary_path),
        "confusion_matrix_artifacts": {
            "status": "generated" if confusion_paths else "skipped",
            "paths": confusion_paths,
            "reason": None if confusion_paths else "no_models_had_required_prediction_artifacts",
        },
    }
    if allow_inherited and inherited_best_model:
        mandatory_updates["best_model_reference"] = {
            "status": "inherited",
            "path": str(inherited_best_model),
            "source_experiment": infer_source_experiment_name(benchmark_summary_path),
            "source_path": str(inherited_best_model),
        }
    elif inherited_best_model:
        mandatory_updates["best_model_reference"] = {
            "status": "skipped",
            "path": str(inherited_best_model),
            "reason": "allow_inherited_artifacts_disabled_in_config",
        }
    optional_updates: dict[str, dict[str, Any]] = {
        "per_model_error_audit_artifacts": {
            "status": "generated" if generated_models else "failed",
            "details": per_model_status,
            "reason": None if generated_models else "no_models_had_required_prediction_artifacts",
        },
    }

    update_artifact_manifest(
        output_dir=output_dir,
        mandatory_updates=mandatory_updates,
        optional_updates=optional_updates,
        metadata_updates={
            "experiment_id": experiment_id,
            "dataset_name": benchmark_summary.get("dataset_name", "uct_student"),
            "target_formulation": benchmark_summary.get("target_formulation", "three_class"),
            "manifest_scope": "error_audit",
            "source_benchmark_summary": str(benchmark_summary_path),
        },
    )

    return audit_payload
