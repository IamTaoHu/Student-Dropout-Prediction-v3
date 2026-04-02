from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except Exception:
    sns = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.final_result_helpers import (
    load_table,
    normalize_series_labels,
    save_figure,
    save_table,
    safe_mkdir,
    write_markdown,
)


def _pick_predictions_source(
    preferred_sources: list[Path | None],
    warnings: list[str],
) -> Path:
    for source in preferred_sources:
        if source is None:
            continue
        candidate = source / "predictions.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find predictions.csv in provided experiment directories.")


def _load_feature_matrix(exp_dir: Path, expected_len: int) -> pd.DataFrame | None:
    runtime_dir = exp_dir / "runtime_artifacts"
    candidates = [
        runtime_dir / "X_test_preprocessed.csv",
        runtime_dir / "X_test_preprocessed.parquet",
    ]
    for path in candidates:
        frame = load_table(path)
        if frame is not None and len(frame) == expected_len:
            return frame.reset_index(drop=True)
    return None


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.5, message, fontsize=11)
    save_figure(path)


def run_error_audit(
    baseline_exp: Path,
    threshold_exp: Path | None,
    error_audit_exp: Path,
    output_dir: Path,
    examples_limit: int = 40,
) -> dict[str, Any]:
    safe_mkdir(output_dir)
    warnings: list[str] = []

    predictions_path = _pick_predictions_source(
        [threshold_exp, error_audit_exp, baseline_exp],
        warnings=warnings,
    )
    pred_df = pd.read_csv(predictions_path)
    if not {"y_true", "y_pred"}.issubset(pred_df.columns):
        raise ValueError(f"Predictions file missing y_true/y_pred columns: {predictions_path}")

    pred_df = pred_df.reset_index().rename(columns={"index": "instance_index"})
    pred_df["true_label"] = normalize_series_labels(pred_df["y_true"])
    pred_df["predicted_label"] = normalize_series_labels(pred_df["y_pred"])

    fail_errors = pred_df[(pred_df["true_label"] == "FAIL") & (pred_df["predicted_label"] != "FAIL")].copy()
    fail_correct = pred_df[(pred_df["true_label"] == "FAIL") & (pred_df["predicted_label"] == "FAIL")].copy()

    route_df = (
        fail_errors["predicted_label"]
        .value_counts(dropna=False)
        .rename_axis("predicted_label")
        .reset_index(name="count")
    )
    total_fail_errors = max(int(route_df["count"].sum()), 1)
    route_df["pct_of_fail_errors"] = route_df["count"] / total_fail_errors
    save_table(output_dir / "fail_confusion_routes.csv", route_df)

    feature_df = _load_feature_matrix(threshold_exp or error_audit_exp, expected_len=len(pred_df))
    if feature_df is None:
        feature_df = _load_feature_matrix(error_audit_exp, expected_len=len(pred_df))
    if feature_df is None:
        feature_df = _load_feature_matrix(baseline_exp, expected_len=len(pred_df))
    if feature_df is None:
        warnings.append("Could not load aligned X_test_preprocessed.* for feature-level error analysis.")

    example_cols = ["instance_index", "true_label", "predicted_label"]
    proba_cols = [c for c in pred_df.columns if c.startswith("proba_class_")]
    example_cols.extend(proba_cols)
    examples_df = fail_errors[example_cols].copy()

    feature_summary_df = pd.DataFrame(
        columns=["feature", "mean_correct_fail", "mean_wrong_fail", "delta", "abs_delta", "rank"]
    )
    top_features: list[str] = []
    if feature_df is not None and not fail_errors.empty and not fail_correct.empty:
        numeric = feature_df.select_dtypes(include=[np.number]).copy()
        wrong_idx = fail_errors["instance_index"].to_numpy(dtype=int)
        correct_idx = fail_correct["instance_index"].to_numpy(dtype=int)
        wrong_means = numeric.iloc[wrong_idx].mean()
        correct_means = numeric.iloc[correct_idx].mean()
        summary = pd.DataFrame(
            {
                "feature": numeric.columns,
                "mean_correct_fail": correct_means.values,
                "mean_wrong_fail": wrong_means.values,
            }
        )
        summary["delta"] = summary["mean_wrong_fail"] - summary["mean_correct_fail"]
        summary["abs_delta"] = summary["delta"].abs()
        summary = summary.sort_values("abs_delta", ascending=False).reset_index(drop=True)
        summary["rank"] = np.arange(1, len(summary) + 1)
        feature_summary_df = summary
        top_features = summary.head(8)["feature"].astype(str).tolist()
        if top_features:
            snapshot = feature_df.loc[fail_errors.index, top_features].reset_index(drop=True)
            examples_df = pd.concat([examples_df.reset_index(drop=True), snapshot], axis=1)

    save_table(output_dir / "fail_error_examples.csv", examples_df.head(examples_limit))
    save_table(output_dir / "fail_error_feature_summary.csv", feature_summary_df)

    fig_path_1 = output_dir / "fig_fail_confusion_breakdown.png"
    if not route_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(route_df["predicted_label"], route_df["count"])
        ax.set_title("FAIL Misclassification Routes")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig_path_1)
    else:
        _plot_placeholder(fig_path_1, "FAIL Misclassification Routes", "No FAIL misclassifications found.")

    fig_path_2 = output_dir / "fig_fail_misclassification_heatmap.png"
    cm = (
        pred_df.pivot_table(
            index="true_label",
            columns="predicted_label",
            values="instance_index",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=["PASS", "FAIL", "DROPOUT"], columns=["PASS", "FAIL", "DROPOUT"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    else:
        image = ax.imshow(cm.to_numpy(), cmap="Blues")
        fig.colorbar(image, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm.iloc[i, j]), ha="center", va="center")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns.tolist())
        ax.set_yticklabels(cm.index.tolist())
    ax.set_title("Prediction Confusion Heatmap")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    save_figure(fig_path_2)

    fig_path_3 = output_dir / "fig_fail_error_feature_importance.png"
    fig_path_4 = output_dir / "fig_fail_error_top_features.png"
    if not feature_summary_df.empty:
        top = feature_summary_df.head(12).copy()
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"][::-1], top["abs_delta"][::-1])
        ax.set_title("Top Feature Shifts: Wrong FAIL vs Correct FAIL")
        ax.set_xlabel("|mean_wrong_fail - mean_correct_fail|")
        save_figure(fig_path_3)

        melted = top.head(8).melt(
            id_vars="feature",
            value_vars=["mean_correct_fail", "mean_wrong_fail"],
            var_name="group",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.barplot(data=melted, x="value", y="feature", hue="group", ax=ax, orient="h")
        else:
            correct = top.head(8)[["feature", "mean_correct_fail"]].set_index("feature")
            wrong = top.head(8)[["feature", "mean_wrong_fail"]].set_index("feature")
            idx = np.arange(len(correct))
            ax.barh(idx - 0.2, correct["mean_correct_fail"].values, height=0.35, label="mean_correct_fail")
            ax.barh(idx + 0.2, wrong["mean_wrong_fail"].values, height=0.35, label="mean_wrong_fail")
            ax.set_yticks(idx)
            ax.set_yticklabels(correct.index.tolist())
            ax.legend(frameon=False)
        ax.set_title("Top Feature Means by FAIL Outcome Group")
        ax.set_xlabel("Mean standardized value")
        save_figure(fig_path_4)
    else:
        _plot_placeholder(fig_path_3, "Top Feature Shifts", "Feature matrix unavailable for error analysis.")
        _plot_placeholder(fig_path_4, "Top Feature Means", "Feature matrix unavailable for error analysis.")

    dominant_route = "none"
    if not route_df.empty:
        dominant_route = str(route_df.iloc[0]["predicted_label"])

    absorbed_note = "balanced"
    fail_to_pass = int(route_df.loc[route_df["predicted_label"] == "PASS", "count"].sum()) if not route_df.empty else 0
    fail_to_dropout = (
        int(route_df.loc[route_df["predicted_label"] == "DROPOUT", "count"].sum()) if not route_df.empty else 0
    )
    if fail_to_pass > fail_to_dropout:
        absorbed_note = "FAIL is more often absorbed into PASS, indicating boundary weakness."
    elif fail_to_dropout > fail_to_pass:
        absorbed_note = "FAIL is more often absorbed into DROPOUT, indicating overlap with disengagement."

    threshold_note = "Threshold-effect comparison unavailable."
    if threshold_exp is not None and (baseline_exp / "predictions.csv").exists() and (threshold_exp / "predictions.csv").exists():
        base = pd.read_csv(baseline_exp / "predictions.csv")
        tuned = pd.read_csv(threshold_exp / "predictions.csv")
        if {"y_true", "y_pred"}.issubset(base.columns) and {"y_true", "y_pred"}.issubset(tuned.columns):
            base_true = normalize_series_labels(base["y_true"])
            base_pred = normalize_series_labels(base["y_pred"])
            tuned_true = normalize_series_labels(tuned["y_true"])
            tuned_pred = normalize_series_labels(tuned["y_pred"])
            if len(base_true) == len(tuned_true):
                b_fail_to_pass = int(((base_true == "FAIL") & (base_pred == "PASS")).sum())
                t_fail_to_pass = int(((tuned_true == "FAIL") & (tuned_pred == "PASS")).sum())
                b_fail_to_dropout = int(((base_true == "FAIL") & (base_pred == "DROPOUT")).sum())
                t_fail_to_dropout = int(((tuned_true == "FAIL") & (tuned_pred == "DROPOUT")).sum())
                threshold_note = (
                    f"FAIL->PASS changed {b_fail_to_pass}->{t_fail_to_pass}; "
                    f"FAIL->DROPOUT changed {b_fail_to_dropout}->{t_fail_to_dropout} after threshold tuning."
                )

    summary_lines = [
        "# FAIL-Centric Error Audit",
        "",
        f"- Prediction source: `{predictions_path}`",
        f"- FAIL errors: `{len(fail_errors)}` out of `{int((pred_df['true_label'] == 'FAIL').sum())}` FAIL samples.",
        f"- Dominant FAIL confusion route: `{dominant_route}`.",
        f"- FAIL->PASS count: `{fail_to_pass}`; FAIL->DROPOUT count: `{fail_to_dropout}`.",
        f"- Interpretation: {absorbed_note}",
        f"- Threshold note: {threshold_note}",
    ]
    if not feature_summary_df.empty:
        top_list = ", ".join(feature_summary_df.head(5)["feature"].astype(str).tolist())
        summary_lines.append(f"- Features most associated with wrong FAIL predictions: {top_list}.")
    else:
        summary_lines.append("- Feature-level comparison unavailable because aligned test features were missing.")
    if warnings:
        summary_lines.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    write_markdown(output_dir / "error_audit_summary.md", "\n".join(summary_lines))

    return {
        "dominant_route": dominant_route,
        "warnings": warnings,
        "generated_files": [
            "error_audit_summary.md",
            "fail_confusion_routes.csv",
            "fail_error_feature_summary.csv",
            "fail_error_examples.csv",
            "fig_fail_confusion_breakdown.png",
            "fig_fail_misclassification_heatmap.png",
            "fig_fail_error_feature_importance.png",
            "fig_fail_error_top_features.png",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exp", type=Path, required=True)
    parser.add_argument("--error-audit-exp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-exp", type=Path, required=False)
    parser.add_argument("--examples-limit", type=int, default=40)
    args = parser.parse_args()

    result = run_error_audit(
        baseline_exp=args.baseline_exp,
        threshold_exp=args.threshold_exp,
        error_audit_exp=args.error_audit_exp,
        output_dir=args.output_dir,
        examples_limit=args.examples_limit,
    )
    print(f"[error-audit] generated {len(result['generated_files'])} artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
