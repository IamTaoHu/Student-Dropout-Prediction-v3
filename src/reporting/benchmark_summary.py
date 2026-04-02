"""Persistence helpers for benchmark summaries and artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_list()
    return str(value)


def _save_leaderboard_csv(summary: dict[str, Any], output_dir: Path) -> Path:
    leaderboard = pd.DataFrame(summary.get("leaderboard", []))
    if leaderboard.empty and "model_results" in summary:
        rows = []
        for name, payload in summary["model_results"].items():
            metric = payload.get("metrics", {})
            rows.append({"model": name, **metric})
        leaderboard = pd.DataFrame(rows)
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    return leaderboard_path


def _save_markdown_summary(summary: dict[str, Any], leaderboard: pd.DataFrame, output_dir: Path) -> Path:
    md_path = output_dir / "benchmark_summary.md"
    primary_metric = summary.get("primary_metric", "test_macro_f1")
    best_model = summary.get("best_model")
    lines = [
        "# Benchmark Summary",
        "",
        f"- Experiment ID: `{summary.get('experiment_id', 'unknown')}`",
        f"- Primary metric: `{primary_metric}`",
        f"- Best model: `{best_model}`",
        "",
        "## Leaderboard",
        "",
        leaderboard.to_markdown(index=False) if not leaderboard.empty else "_No model results available._",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _save_confusion_matrix_plots(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    paths: list[Path] = []
    for model_name, payload in summary.get("model_results", {}).items():
        cm = payload.get("artifacts", {}).get("confusion_matrix")
        if not cm:
            continue
        cm_df = pd.DataFrame(cm)
        path = output_dir / f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(5, 4))
        if sns is not None:
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        else:
            plt.imshow(cm_df.values, cmap="Blues")
            for i in range(cm_df.shape[0]):
                for j in range(cm_df.shape[1]):
                    plt.text(j, i, str(cm_df.iat[i, j]), ha="center", va="center")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def _save_normalized_confusion_matrix_plots(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for model_name, payload in summary.get("model_results", {}).items():
        cm = payload.get("artifacts", {}).get("confusion_matrix")
        if not cm:
            continue
        cm_arr = np.asarray(cm, dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)
        path = output_dir / f"confusion_matrix_{model_name}_normalized.png"
        plt.figure(figsize=(5, 4))
        plt.imshow(cm_norm, cmap="Blues")
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
        plt.title(f"Normalized Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def _save_classification_reports(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for model_name, payload in summary.get("model_results", {}).items():
        artifacts = payload.get("artifacts", {})
        y_true = artifacts.get("y_true_test")
        y_pred = artifacts.get("y_pred_test")
        labels = artifacts.get("labels")
        if y_true is None or y_pred is None:
            continue
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        path = output_dir / f"classification_report_{model_name}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def _build_persisted_summary(summary: dict[str, Any], compact: bool) -> dict[str, Any]:
    if not compact:
        return summary

    payload = copy.deepcopy(summary)
    model_results = payload.get("model_results")
    if isinstance(model_results, dict):
        for _, model_payload in model_results.items():
            if not isinstance(model_payload, dict):
                continue
            artifacts = model_payload.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            removed_keys: list[str] = []
            for key in ("y_pred_test", "y_proba_test"):
                if key in artifacts:
                    artifacts.pop(key, None)
                    removed_keys.append(key)
            if removed_keys:
                model_payload["compact_omitted_artifacts"] = removed_keys
    payload["summary_mode"] = "compact"
    return payload


def save_benchmark_summary(summary: dict[str, Any], output_dir: Path, compact: bool = False) -> Path:
    """Persist benchmark outputs as JSON/CSV/PNG/Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    persisted_summary = _build_persisted_summary(summary, compact=compact)
    json_path = output_dir / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(persisted_summary, f, indent=2, default=_json_default)

    leaderboard_path = _save_leaderboard_csv(summary, output_dir)
    if leaderboard_path.exists():
        try:
            leaderboard_df = pd.read_csv(leaderboard_path)
        except Exception:
            leaderboard_df = pd.DataFrame()
    else:
        leaderboard_df = pd.DataFrame()
    _save_markdown_summary(summary, leaderboard_df, output_dir)
    _save_confusion_matrix_plots(summary, output_dir)
    _save_normalized_confusion_matrix_plots(summary, output_dir)
    _save_classification_reports(summary, output_dir)
    return json_path
