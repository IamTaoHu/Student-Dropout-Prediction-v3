from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.final_result_helpers import (
    find_report_class_key,
    load_json,
    metrics_from_classification_report,
    save_table,
    safe_mkdir,
    save_figure,
    write_markdown,
    write_markdown_table,
)


def _discover_models(threshold_exp: Path) -> list[str]:
    models: set[str] = set()
    for path in threshold_exp.glob("classification_report_before_*.json"):
        token = path.stem.replace("classification_report_before_", "")
        if (threshold_exp / f"classification_report_after_{token}.json").exists():
            models.add(token)
    return sorted(models)


def _guardrail_conclusion(
    delta_fail_f1: float,
    delta_macro_f1: float,
    delta_balanced: float,
    macro_tol: float,
    balanced_tol: float,
) -> str:
    if np.isnan(delta_fail_f1):
        return "insufficient_data"
    fail_improved = delta_fail_f1 > 0
    guardrail_ok = (delta_macro_f1 >= -macro_tol) and (delta_balanced >= -balanced_tol)
    if fail_improved and guardrail_ok:
        return "guardrail_passed"
    if fail_improved and not guardrail_ok:
        return "fail_improved_but_guardrail_failed"
    return "no_fail_improvement"


def _plot_before_after(
    pivot_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot_df))
    width = 0.35
    ax.bar(x - width / 2, pivot_df[f"{metric}_before"], width=width, label="Before")
    ax.bar(x + width / 2, pivot_df[f"{metric}_after"], width=width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df["model"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: Before vs After Threshold Tuning")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    save_figure(output_path)


def _plot_tradeoff(delta_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.scatter(delta_df["delta_fail_f1"], delta_df["delta_macro_f1"], s=80)
    for _, row in delta_df.iterrows():
        ax.annotate(str(row["model"]), (row["delta_fail_f1"], row["delta_macro_f1"]), fontsize=9)
    ax.set_xlabel("Delta FAIL F1")
    ax.set_ylabel("Delta Macro F1")
    ax.set_title("Threshold Trade-off: FAIL F1 vs Macro F1")
    ax.grid(alpha=0.25)
    save_figure(output_path)


def run_threshold_analysis(
    baseline_exp: Path,
    threshold_exp: Path,
    output_dir: Path,
    macro_tolerance: float = 0.01,
    balanced_tolerance: float = 0.01,
) -> dict[str, Any]:
    _ = baseline_exp
    safe_mkdir(output_dir)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    models = _discover_models(threshold_exp)
    if not models:
        raise FileNotFoundError(
            f"No before/after classification reports found under: {threshold_exp}"
        )

    for model in models:
        before_path = threshold_exp / f"classification_report_before_{model}.json"
        after_path = threshold_exp / f"classification_report_after_{model}.json"
        before_report = load_json(before_path, default={}) or {}
        after_report = load_json(after_path, default={}) or {}
        fail_key_before = find_report_class_key(before_report, "FAIL")
        fail_key_after = find_report_class_key(after_report, "FAIL")
        if fail_key_before is None or fail_key_after is None:
            warnings.append(f"FAIL class not found in reports for model '{model}'.")
        before_metrics = metrics_from_classification_report(before_report, fail_key_before)
        after_metrics = metrics_from_classification_report(after_report, fail_key_after)

        rows.append({"model": model, "condition": "before", **before_metrics})
        rows.append({"model": model, "condition": "after", **after_metrics})

        delta_fail = after_metrics["fail_f1"] - before_metrics["fail_f1"]
        delta_macro = after_metrics["macro_f1"] - before_metrics["macro_f1"]
        delta_bal = after_metrics["balanced_accuracy"] - before_metrics["balanced_accuracy"]
        delta_rows.append(
            {
                "model": model,
                "fail_f1_before": before_metrics["fail_f1"],
                "fail_f1_after": after_metrics["fail_f1"],
                "delta_fail_f1": delta_fail,
                "delta_macro_f1": delta_macro,
                "delta_balanced_accuracy": delta_bal,
                "conclusion": _guardrail_conclusion(
                    delta_fail_f1=delta_fail,
                    delta_macro_f1=delta_macro,
                    delta_balanced=delta_bal,
                    macro_tol=macro_tolerance,
                    balanced_tol=balanced_tolerance,
                ),
            }
        )

    before_after_df = pd.DataFrame(rows)
    delta_df = pd.DataFrame(delta_rows).sort_values("delta_fail_f1", ascending=False)

    save_table(output_dir / "threshold_before_after.csv", before_after_df)
    save_table(output_dir / "fail_class_delta_table.csv", delta_df)

    pivot_df = before_after_df.pivot(index="model", columns="condition")
    pivot_df = pivot_df.sort_values(("fail_f1", "after"), ascending=False).reset_index()
    pivot_df.columns = [
        str(col[0]) if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in pivot_df.columns.to_flat_index()
    ]
    for metric in ["accuracy", "macro_f1", "balanced_accuracy", "fail_precision", "fail_recall", "fail_f1"]:
        pivot_df[f"delta_{metric}"] = pivot_df[f"{metric}_after"] - pivot_df[f"{metric}_before"]
    save_table(output_dir / "per_model_threshold_effect.csv", pivot_df)

    _plot_before_after(pivot_df, "fail_f1", "FAIL F1", output_dir / "fig_fail_f1_before_after.png")
    _plot_before_after(pivot_df, "macro_f1", "Macro F1", output_dir / "fig_macro_f1_before_after.png")
    _plot_before_after(
        pivot_df,
        "balanced_accuracy",
        "Balanced Accuracy",
        output_dir / "fig_balanced_accuracy_before_after.png",
    )
    _plot_tradeoff(delta_df, output_dir / "fig_threshold_tradeoff_fail_vs_macro.png")

    best_row = delta_df.iloc[0]
    improved_precision = pivot_df.loc[pivot_df["model"] == best_row["model"], "delta_fail_precision"].iloc[0]
    improved_recall = pivot_df.loc[pivot_df["model"] == best_row["model"], "delta_fail_recall"].iloc[0]
    precision_or_recall = "recall" if improved_recall >= improved_precision else "precision"
    guardrail_pass_count = int((delta_df["conclusion"] == "guardrail_passed").sum())
    collapsed_cases = delta_df[
        (delta_df["delta_fail_f1"] > 0) & (delta_df["delta_macro_f1"] < -macro_tolerance)
    ]

    summary_lines = [
        "# Threshold Tuning Comparison",
        "",
        f"- Models compared: {', '.join(models)}",
        f"- Strongest FAIL F1 gain: `{best_row['model']}` with delta `{best_row['delta_fail_f1']:.4f}`.",
        f"- For the strongest model, gain is primarily from `{precision_or_recall}`.",
        f"- Guardrail passed in `{guardrail_pass_count}/{len(delta_df)}` models "
        f"(macro tolerance={macro_tolerance:.3f}, balanced tolerance={balanced_tolerance:.3f}).",
    ]
    if not collapsed_cases.empty:
        summary_lines.append(
            "- Cases with FAIL gain but macro collapse: "
            + ", ".join(collapsed_cases["model"].astype(str).tolist())
            + "."
        )
    else:
        summary_lines.append("- No model showed FAIL gain with material macro collapse.")
    if warnings:
        summary_lines.extend(["", "## Missing Inputs / Warnings", *[f"- {w}" for w in warnings]])
    write_markdown(output_dir / "threshold_comparison.md", "\n".join(summary_lines))
    write_markdown_table(output_dir / "threshold_comparison_table.md", delta_df.round(4), "Threshold Delta Table")

    return {
        "models": models,
        "best_fail_model": str(best_row["model"]),
        "warnings": warnings,
        "generated_files": [
            "threshold_comparison.md",
            "threshold_comparison_table.md",
            "threshold_before_after.csv",
            "fail_class_delta_table.csv",
            "per_model_threshold_effect.csv",
            "fig_fail_f1_before_after.png",
            "fig_macro_f1_before_after.png",
            "fig_balanced_accuracy_before_after.png",
            "fig_threshold_tradeoff_fail_vs_macro.png",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exp", type=Path, required=True)
    parser.add_argument("--threshold-exp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--macro-tolerance", type=float, default=0.01)
    parser.add_argument("--balanced-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    result = run_threshold_analysis(
        baseline_exp=args.baseline_exp,
        threshold_exp=args.threshold_exp,
        output_dir=args.output_dir,
        macro_tolerance=args.macro_tolerance,
        balanced_tolerance=args.balanced_tolerance,
    )
    print(f"[threshold] generated {len(result['generated_files'])} artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
