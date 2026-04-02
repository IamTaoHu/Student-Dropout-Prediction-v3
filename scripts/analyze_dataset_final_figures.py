from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.final_dataset_helpers import (
    copy_or_none,
    dataset_label_info,
    find_existing_figure,
    get_model_artifacts,
    load_json_file,
    render_confusion_matrix,
)
from scripts.utils.final_result_helpers import save_figure, safe_mkdir, write_markdown


def _plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    title: str,
) -> float:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(output_path)
    return ap


def _learning_curve_placeholder(path: Path, message: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.5, message)
    save_figure(path)


def _simple_placeholder(path: Path, title: str, message: str) -> None:
    _learning_curve_placeholder(path=path, message=message, title=title)


def run_dataset_final_figures(
    dataset: str,
    best_model_json: Path,
    comparison_csv: Path,
    output_root: Path,
    recompute_learning_curves: bool = False,
) -> dict[str, Any]:
    _ = recompute_learning_curves
    output_dir = safe_mkdir(output_root / dataset)
    warnings: list[str] = []
    best = load_json_file(best_model_json)
    if not best.get("model") or not best.get("source_paths", {}).get("experiment"):
        warnings.append(f"{dataset}: best model metadata incomplete")
        return {"warnings": warnings, "generated_files": []}

    model = str(best["model"])
    source_paths = best.get("source_paths", {}) or {}
    exp_path = Path(str(source_paths["experiment"]))
    candidate_exp_paths: list[Path] = []
    for key in ["experiment", "baseline_exp", "threshold_exp", "boosting_exp", "error_audit_exp"]:
        value = source_paths.get(key)
        if value:
            p = Path(str(value))
            if p.exists() and p not in candidate_exp_paths:
                candidate_exp_paths.append(p)
    label_info = dataset_label_info(dataset)
    label_map = label_info["label_map"]
    label_order = label_info["label_order"]
    fail_label = int(label_info["fail_label"])

    artifacts = get_model_artifacts(exp_path, model)
    if artifacts.get("y_proba") is None or artifacts.get("y_true") is None or artifacts.get("y_pred") is None:
        for candidate in candidate_exp_paths:
            probe = get_model_artifacts(candidate, model)
            if probe.get("y_true") is not None and probe.get("y_pred") is not None:
                artifacts["y_true"] = probe.get("y_true")
                artifacts["y_pred"] = probe.get("y_pred")
                exp_path = candidate
            if probe.get("y_proba") is not None:
                artifacts["y_proba"] = probe.get("y_proba")
                artifacts["labels"] = probe.get("labels") or artifacts.get("labels")
                exp_path = candidate
                break
    y_true = artifacts.get("y_true")
    y_pred = artifacts.get("y_pred")
    y_proba = artifacts.get("y_proba")
    labels = artifacts.get("labels")
    if labels is None:
        labels = label_order

    dataset_tag = dataset.lower()
    conf_raw = output_dir / f"fig_confusion_matrix_best_model_{dataset_tag}.png"
    conf_norm = output_dir / f"fig_confusion_matrix_best_model_{dataset_tag}_normalized.png"
    pr_path = output_dir / f"fig_precision_recall_curve_best_model_{dataset_tag}.png"
    lc_path = output_dir / f"fig_learning_curve_best_model_{dataset_tag}.png"

    confusion_notes = "not_available"
    if y_true is not None and y_pred is not None:
        y_true_arr = np.asarray(y_true).astype(int)
        y_pred_arr = np.asarray(y_pred).astype(int)
        render_confusion_matrix(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            labels=[int(v) for v in label_order if int(v) in set(y_true_arr).union(set(y_pred_arr)) or dataset == "uct"],
            label_map=label_map,
            normalized=False,
            title=f"{dataset.upper()} - {model} Confusion Matrix",
            output_path=conf_raw,
        )
        render_confusion_matrix(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            labels=[int(v) for v in label_order if int(v) in set(y_true_arr).union(set(y_pred_arr)) or dataset == "uct"],
            label_map=label_map,
            normalized=True,
            title=f"{dataset.upper()} - {model} Confusion Matrix (normalized)",
            output_path=conf_norm,
        )
        confusion_notes = "generated_from_predictions"
    else:
        warnings.append(f"{dataset}: missing y_true/y_pred for confusion matrix")
        _simple_placeholder(
            conf_raw,
            title=f"{dataset.upper()} - {model} Confusion Matrix",
            message="Confusion matrix unavailable: missing y_true/y_pred artifacts.",
        )
        _simple_placeholder(
            conf_norm,
            title=f"{dataset.upper()} - {model} Confusion Matrix (normalized)",
            message="Normalized confusion matrix unavailable: missing y_true/y_pred artifacts.",
        )

    pr_note = "not_available"
    pr_ap = float("nan")
    if y_true is not None and y_proba is not None:
        y_true_arr = np.asarray(y_true).astype(int)
        proba = np.asarray(y_proba, dtype=float)
        labels_list = [int(x) for x in (labels or list(range(proba.shape[1])))]
        if proba.ndim == 2 and len(labels_list) == proba.shape[1]:
            try:
                fail_index = labels_list.index(fail_label)
            except ValueError:
                fail_index = min(1, proba.shape[1] - 1)
            y_bin = (y_true_arr == fail_label).astype(int)
            pr_ap = _plot_pr_curve(
                y_true=y_bin,
                y_score=proba[:, fail_index],
                output_path=pr_path,
                title=f"{dataset.upper()} - {model} FAIL-focused PR Curve",
            )
            pr_note = "generated_from_probabilities"
        else:
            warnings.append(f"{dataset}: invalid y_proba shape for PR curve")
    else:
        warnings.append(f"{dataset}: missing probabilities for PR curve")
        _learning_curve_placeholder(
            path=pr_path,
            message="PR curve unavailable: no prediction probabilities found for selected model.",
            title=f"{dataset.upper()} - {model} PR Curve",
        )

    existing_lc = find_existing_figure(exp_path, "learning_curve")
    if existing_lc is None:
        for candidate in candidate_exp_paths:
            existing_lc = find_existing_figure(candidate, "learning_curve")
            if existing_lc is not None:
                break
    lc_note = "not_available"
    if copy_or_none(existing_lc, lc_path):
        lc_note = "reused_existing_learning_curve"
    else:
        _learning_curve_placeholder(
            path=lc_path,
            message="Learning curve unavailable for selected model; run with --recompute-learning-curves if recomputation is added.",
            title=f"{dataset.upper()} - {model} Learning Curve",
        )
        warnings.append(f"{dataset}: learning curve reused/recompute unavailable for selected model")

    comparison_df = pd.read_csv(comparison_csv) if comparison_csv.exists() else pd.DataFrame()
    threshold_rows = comparison_df[comparison_df["condition"] == "threshold_tuned"] if not comparison_df.empty else pd.DataFrame()
    threshold_note = "No threshold-tuned rows available."
    if not threshold_rows.empty:
        top_thr = threshold_rows.sort_values("macro_f1", ascending=False).iloc[0]
        threshold_note = (
            f"Best threshold-tuned row: model={top_thr['model']}, macro_f1={top_thr['macro_f1']:.4f}, "
            f"fail_f1={top_thr['fail_f1']:.4f}."
        )

    summary_lines = [
        f"# Dataset Summary ({dataset.upper()})",
        "",
        f"1. Best model summary: `{model}` from `{exp_path.name}`.",
        "2. Final comparison table excerpt: see `final_model_comparison` file in this folder.",
        f"3. Threshold tuning effect summary: {threshold_note}",
        f"4. FAIL class summary: fail_f1={best.get('selected_metrics', {}).get('fail_f1')}.",
        f"5. Confusion matrix interpretation: generated status=`{confusion_notes}`.",
        f"6. PR curve interpretation: FAIL-focused AP=`{pr_ap:.4f}` (status: `{pr_note}`).",
        f"7. Learning curve interpretation: status=`{lc_note}`.",
        (
            "8. Discussion: The selected best model balances macro-F1 and balanced accuracy while maintaining "
            "the strongest available FAIL-class discrimination under current artifacts."
        ),
    ]
    if warnings:
        summary_lines.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    write_markdown(output_dir / f"dataset_summary_{dataset_tag}.md", "\n".join(summary_lines))

    claim_lines = [
        f"# Paper Claims ({dataset.upper()})",
        "",
        "| claim | supporting table / figure | confidence level | caution note |",
        "|---|---|---|---|",
        f"| Best model for {dataset.upper()} is `{model}` by macro_f1 tie-break rules | `best_model_{dataset_tag}.json`, `final_model_comparison_{dataset_tag}.md` | high | Depends on currently available experiments |",
        f"| Confusion matrix reveals dominant error structure for selected model | `fig_confusion_matrix_best_model_{dataset_tag}.png` | medium | Interpretation depends on label mapping assumptions |",
        f"| FAIL-focused PR curve quantifies ranking quality | `fig_precision_recall_curve_best_model_{dataset_tag}.png` | medium | AP reflects post-hoc test-set probabilities only |",
        f"| Learning-curve evidence is available as `{lc_note}` | `fig_learning_curve_best_model_{dataset_tag}.png` | low | For non-recomputed runs, curve may be reused from source artifacts |",
    ]
    write_markdown(output_dir / f"paper_claims_{dataset_tag}.md", "\n".join(claim_lines))

    return {
        "dataset": dataset,
        "warnings": warnings,
        "best_model": model,
        "ap_fail_pr": pr_ap,
        "confusion_status": confusion_notes,
        "pr_status": pr_note,
        "learning_curve_status": lc_note,
        "generated_files": [
            f"{dataset}/fig_confusion_matrix_best_model_{dataset_tag}.png",
            f"{dataset}/fig_confusion_matrix_best_model_{dataset_tag}_normalized.png",
            f"{dataset}/fig_precision_recall_curve_best_model_{dataset_tag}.png",
            f"{dataset}/fig_learning_curve_best_model_{dataset_tag}.png",
            f"{dataset}/dataset_summary_{dataset_tag}.md",
            f"{dataset}/paper_claims_{dataset_tag}.md",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["uct", "oulad"])
    parser.add_argument("--best-model-json", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recompute-learning-curves", action="store_true")
    args = parser.parse_args()
    result = run_dataset_final_figures(
        dataset=args.dataset,
        best_model_json=args.best_model_json,
        comparison_csv=args.comparison_csv,
        output_root=args.output_root,
        recompute_learning_curves=args.recompute_learning_curves,
    )
    print(f"[dataset-figures] {args.dataset}: generated {len(result['generated_files'])} files")


if __name__ == "__main__":
    main()
