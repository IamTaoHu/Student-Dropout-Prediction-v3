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
    cosine_similarity_to_vector,
    flatten_lime_local,
    flatten_shap_global,
    flatten_shap_local,
    load_json,
    load_table,
    normalize_label,
    normalize_series_labels,
    save_table,
    safe_mkdir,
    save_figure,
    select_best_model,
    write_markdown,
    write_markdown_table,
)


def _choose_selected_model(
    threshold_exp: Path,
    explicit_model: str | None,
) -> tuple[str | None, pd.DataFrame]:
    leaderboard = load_table(threshold_exp / "leaderboard.csv")
    if leaderboard is None:
        return explicit_model, pd.DataFrame()
    leaderboard = leaderboard.copy()
    if explicit_model:
        return explicit_model, leaderboard
    fail_scores: dict[str, float] = {}
    for model in leaderboard["model"].astype(str).tolist():
        report = load_json(threshold_exp / f"classification_report_after_{model}.json", default={}) or {}
        fail_key = None
        for key in report:
            if normalize_label(key) == "FAIL":
                fail_key = key
                break
        if fail_key:
            fail_scores[model] = float(report.get(fail_key, {}).get("f1-score", np.nan))
    return select_best_model(leaderboard, tie_break_fail_f1=fail_scores), leaderboard


def _load_predictions_for_instance_selection(exp_paths: list[Path]) -> tuple[pd.DataFrame, Path]:
    for exp in exp_paths:
        path = exp / "predictions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if {"y_true", "y_pred"}.issubset(frame.columns):
                frame = frame.reset_index().rename(columns={"index": "instance_index"})
                frame["true_label"] = normalize_series_labels(frame["y_true"])
                frame["predicted_label"] = normalize_series_labels(frame["y_pred"])
                return frame, path
    raise FileNotFoundError("Could not locate usable predictions.csv for explainability instance selection.")


def _pick_indices(pred_df: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("correct_fail", (pred_df["true_label"] == "FAIL") & (pred_df["predicted_label"] == "FAIL"), 2),
        ("wrong_fail", (pred_df["true_label"] == "FAIL") & (pred_df["predicted_label"] != "FAIL"), 2),
        ("correct_pass", (pred_df["true_label"] == "PASS") & (pred_df["predicted_label"] == "PASS"), 1),
        ("correct_dropout", (pred_df["true_label"] == "DROPOUT") & (pred_df["predicted_label"] == "DROPOUT"), 1),
    ]
    selected_rows: list[pd.DataFrame] = []
    used: set[int] = set()
    for bucket, mask, n in rules:
        candidate = pred_df.loc[mask & (~pred_df["instance_index"].isin(used))].head(n).copy()
        candidate["bucket"] = bucket
        selected_rows.append(candidate)
        used.update(candidate["instance_index"].astype(int).tolist())
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return selected[["instance_index", "true_label", "predicted_label", "bucket"]]


def _ensure_local_alignment(local_df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if local_df.empty:
        return local_df
    selected_idx = set(selected["instance_index"].astype(int).tolist())
    available_idx = set(local_df["instance_index"].astype(int).unique().tolist())
    overlap = selected_idx.intersection(available_idx)
    if overlap:
        return local_df[local_df["instance_index"].isin(sorted(overlap))].copy()
    return local_df.copy()


def _plot_local_bars(local_df: pd.DataFrame, output_prefix: Path, method_name: str, top_k: int = 10) -> list[str]:
    generated: list[str] = []
    if local_df.empty:
        return generated
    for instance_id in sorted(local_df["instance_index"].astype(int).unique().tolist()):
        sub = local_df[local_df["instance_index"] == instance_id].copy()
        sub["abs_val"] = sub["local_importance"].abs()
        sub = sub.sort_values("abs_val", ascending=False).head(top_k)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#3C8DAD" if v >= 0 else "#D95F5F" for v in sub["local_importance"]]
        ax.barh(sub["feature"][::-1], sub["local_importance"][::-1], color=colors[::-1])
        ax.set_title(f"{method_name.upper()} local explanation - instance {instance_id}")
        ax.set_xlabel("Contribution")
        ax.grid(axis="x", alpha=0.25)
        out_path = output_prefix.parent / f"{output_prefix.name}{instance_id}.png"
        save_figure(out_path)
        generated.append(out_path.name)
    return generated


def run_explainability_comparison(
    baseline_exp: Path,
    threshold_exp: Path,
    boosting_exp: Path,
    output_dir: Path,
    selected_model: str | None = None,
) -> dict[str, Any]:
    safe_mkdir(output_dir)
    shap_dir = safe_mkdir(output_dir / "shap")
    lime_dir = safe_mkdir(output_dir / "lime")
    aime_dir = safe_mkdir(output_dir / "aime")
    warnings: list[str] = []

    chosen_model, leaderboard = _choose_selected_model(threshold_exp, explicit_model=selected_model)
    if chosen_model is None:
        warnings.append("Could not infer selected best model from threshold leaderboard.")

    pred_df, prediction_source = _load_predictions_for_instance_selection([threshold_exp, boosting_exp, baseline_exp])
    selected_instances = _pick_indices(pred_df)
    save_table(output_dir / "selected_instances.csv", selected_instances)

    explain_src = None
    for exp in [threshold_exp, boosting_exp, baseline_exp]:
        if (exp / "explainability").exists():
            explain_src = exp / "explainability"
            break
    if explain_src is None:
        raise FileNotFoundError("No explainability directory found in baseline/threshold/boosting experiments.")

    shap_global_raw = load_table(explain_src / "shap_global_importance.csv")
    shap_local_raw = load_table(explain_src / "shap_local_importance.csv")
    lime_local_raw = load_table(explain_src / "lime_local_importance.csv")
    aime_global_raw = load_table(explain_src / "aime_global_importance.csv")
    aime_local_raw = load_table(explain_src / "aime_local_importance.csv")
    shap_global_raw = shap_global_raw if shap_global_raw is not None else pd.DataFrame()
    shap_local_raw = shap_local_raw if shap_local_raw is not None else pd.DataFrame()
    lime_local_raw = lime_local_raw if lime_local_raw is not None else pd.DataFrame()
    aime_global_raw = aime_global_raw if aime_global_raw is not None else pd.DataFrame()
    aime_local_raw = aime_local_raw if aime_local_raw is not None else pd.DataFrame()

    shap_global = flatten_shap_global(shap_global_raw)
    if shap_global.empty:
        warnings.append("SHAP global importance not available in expected format.")
    save_table(shap_dir / "shap_global_importance.csv", shap_global)

    shap_local = flatten_shap_local(shap_local_raw)
    shap_local = _ensure_local_alignment(shap_local, selected_instances)
    save_table(shap_dir / "shap_local_instances.csv", shap_local)

    lime_local = flatten_lime_local(lime_local_raw)
    lime_local = _ensure_local_alignment(lime_local, selected_instances)
    save_table(lime_dir / "lime_local_instances.csv", lime_local)

    aime_global = aime_global_raw.copy()
    if not aime_global.empty:
        aime_global["class_name"] = "ALL"
        aime_global = aime_global.rename(columns={"importance": "importance", "feature": "feature"})
        aime_global = aime_global[["feature", "class_name", "importance"]]
    save_table(aime_dir / "aime_global_importance.csv", aime_global)

    aime_local = aime_local_raw.copy()
    if not aime_local.empty:
        aime_local["class_name"] = aime_local["predicted_class"].map(normalize_label)
        aime_local = aime_local.rename(
            columns={
                "instance_index": "instance_index",
                "feature": "feature",
                "contribution": "local_importance",
            }
        )
        aime_local = aime_local[["instance_index", "class_name", "feature", "local_importance", "rank"]]
        aime_local = _ensure_local_alignment(aime_local, selected_instances)
    save_table(aime_dir / "aime_local_instances.csv", aime_local)

    runtime_candidates = [threshold_exp, boosting_exp, baseline_exp]
    X_test = None
    y_test = None
    for exp in runtime_candidates:
        x = load_table(exp / "runtime_artifacts" / "X_test_preprocessed.csv")
        y = load_table(exp / "runtime_artifacts" / "y_test.csv")
        if x is not None and y is not None and len(x) == len(pred_df):
            X_test = x.reset_index(drop=True)
            y_test = y.iloc[:, 0].reset_index(drop=True)
            break
    if X_test is None or y_test is None:
        warnings.append("Aligned runtime test features were not found; representative/similarity outputs are limited.")
        X_test = pd.DataFrame()
        y_test = pd.Series(dtype=float)

    if not X_test.empty:
        y_norm = normalize_series_labels(y_test)
        reps: list[dict[str, Any]] = []
        top_features = (
            aime_global_raw.sort_values("importance", ascending=False)["feature"].head(8).astype(str).tolist()
            if not aime_global_raw.empty
            else X_test.columns[:8].tolist()
        )
        for cls in ["PASS", "FAIL", "DROPOUT"]:
            class_idx = y_norm[y_norm == cls].index.to_numpy(dtype=int)
            if len(class_idx) == 0:
                continue
            class_matrix = X_test.iloc[class_idx].to_numpy(dtype=float)
            centroid = class_matrix.mean(axis=0)
            distances = np.linalg.norm(class_matrix - centroid.reshape(1, -1), axis=1)
            local_choice = int(class_idx[int(np.argmin(distances))])
            for feat in top_features:
                reps.append(
                    {
                        "class_name": cls,
                        "representative_instance_id": local_choice,
                        "feature": feat,
                        "feature_value": float(X_test.loc[local_choice, feat]),
                    }
                )
        reps_df = pd.DataFrame(reps)
    else:
        reps_df = pd.DataFrame(columns=["class_name", "representative_instance_id", "feature", "feature_value"])
    save_table(aime_dir / "aime_representative_instances.csv", reps_df)

    similarity_rows: list[dict[str, Any]] = []
    if not X_test.empty and not selected_instances.empty:
        y_norm = normalize_series_labels(y_test)
        class_centroids: dict[str, np.ndarray] = {}
        for cls in ["PASS", "FAIL", "DROPOUT"]:
            cls_idx = y_norm[y_norm == cls].index.to_numpy(dtype=int)
            if len(cls_idx) > 0:
                class_centroids[cls] = X_test.iloc[cls_idx].to_numpy(dtype=float).mean(axis=0)
        selected_idx = selected_instances["instance_index"].astype(int).tolist()
        selected_matrix = X_test.iloc[selected_idx].to_numpy(dtype=float)
        for cls, centroid in class_centroids.items():
            sims = cosine_similarity_to_vector(selected_matrix, centroid)
            for i, instance_id in enumerate(selected_idx):
                similarity_rows.append(
                    {
                        "instance_id": int(instance_id),
                        "true_label": str(selected_instances.loc[selected_instances["instance_index"] == instance_id, "true_label"].iloc[0]),
                        "predicted_label": str(
                            selected_instances.loc[selected_instances["instance_index"] == instance_id, "predicted_label"].iloc[0]
                        ),
                        "class_name": cls,
                        "similarity_to_class_representation": float(sims[i]),
                    }
                )
    similarity_df = pd.DataFrame(similarity_rows)
    save_table(aime_dir / "aime_similarity_distribution.csv", similarity_df)

    if not shap_global.empty:
        top = shap_global.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4C78A8")
        ax.set_title("SHAP Global Importance (mean absolute)")
        ax.set_xlabel("Importance")
        save_figure(shap_dir / "fig_shap_beeswarm.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.5, "SHAP global importance unavailable.")
        save_figure(shap_dir / "fig_shap_beeswarm.png")

    shap_waterfalls = _plot_local_bars(shap_local, shap_dir / "fig_shap_waterfall_instance_", "shap", top_k=10)
    lime_figs = _plot_local_bars(lime_local, lime_dir / "fig_lime_instance_", "lime", top_k=10)

    if not aime_global.empty:
        top = aime_global.sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#72B7B2")
        ax.set_title("AIME Global Importance")
        ax.set_xlabel("Importance")
        save_figure(aime_dir / "fig_aime_global_importance.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.5, "AIME global importance unavailable.")
        save_figure(aime_dir / "fig_aime_global_importance.png")

    aime_local_figs = _plot_local_bars(aime_local, aime_dir / "fig_aime_local_instance_", "aime", top_k=10)

    if not reps_df.empty:
        rep_plot = reps_df.groupby(["class_name", "feature"], as_index=False)["feature_value"].mean()
        rep_plot["abs_val"] = rep_plot["feature_value"].abs()
        rep_plot = rep_plot.sort_values("abs_val", ascending=False).head(18)
        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.barplot(data=rep_plot, x="feature_value", y="feature", hue="class_name", ax=ax, orient="h")
        else:
            classes = sorted(rep_plot["class_name"].unique().tolist())
            features = rep_plot["feature"].unique().tolist()[:10]
            y_ticks = np.arange(len(features))
            width = 0.25
            for i, cls in enumerate(classes):
                sub = rep_plot[rep_plot["class_name"] == cls].set_index("feature").reindex(features).fillna(0.0)
                ax.barh(y_ticks + (i - len(classes) / 2) * width, sub["feature_value"], height=width, label=cls)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(features)
            ax.legend(frameon=False)
        ax.set_title("AIME Representative Estimation Instance Features")
        save_figure(aime_dir / "fig_aime_representative_instances.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.5, "Representative instances unavailable.")
        save_figure(aime_dir / "fig_aime_representative_instances.png")

    if not similarity_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        if sns is not None:
            sns.violinplot(
                data=similarity_df,
                x="class_name",
                y="similarity_to_class_representation",
                hue="true_label",
                cut=0,
                ax=ax,
            )
        else:
            for cls in sorted(similarity_df["class_name"].unique().tolist()):
                sub = similarity_df[similarity_df["class_name"] == cls]
                ax.scatter(
                    [cls] * len(sub),
                    sub["similarity_to_class_representation"],
                    alpha=0.8,
                    label=cls,
                )
            ax.legend(frameon=False)
        ax.set_title("AIME Similarity Distribution (Selected Instances)")
        save_figure(aime_dir / "fig_aime_similarity_distribution.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.5, "Similarity distribution unavailable.")
        save_figure(aime_dir / "fig_aime_similarity_distribution.png")

    comparison_table = pd.DataFrame(
        [
            {
                "method": "SHAP",
                "scope": "Global + local attribution",
                "strengths": "Standard baseline, consistent feature ranking, strong global signal",
                "weaknesses": "Can be dense; zero/near-zero features can clutter local views",
                "best use in our paper": "Primary baseline for global importance + fail-case local waterfall",
            },
            {
                "method": "LIME",
                "scope": "Local surrogate explanation",
                "strengths": "Simple local coefficients and human-readable threshold terms",
                "weaknesses": "Sampling instability and weaker global consistency",
                "best use in our paper": "Secondary local sanity-check on selected FAIL instances",
            },
            {
                "method": "AIME",
                "scope": "Unified global/local + representative similarity",
                "strengths": "Direct bridge from global/local factors to representative-instance similarity",
                "weaknesses": "Here implemented as an approximation to inverse construction",
                "best use in our paper": "Narrative method for class overlap and FAIL boundary interpretation",
            },
        ]
    )
    write_markdown_table(output_dir / "explainability_comparison_table.md", comparison_table, "Explainability Comparison")

    narrative = [
        "# Explainability Comparison (SHAP vs LIME vs AIME)",
        "",
        f"- Selected model: `{chosen_model}`",
        f"- Prediction source for selected instances: `{prediction_source}`",
        f"- Selected instances count: `{len(selected_instances)}`",
        f"- SHAP local rows used: `{len(shap_local)}`; LIME local rows used: `{len(lime_local)}`; AIME local rows used: `{len(aime_local)}`.",
        "- AIME is simpler for storytelling because representative-instance and similarity artifacts connect local effects to class overlap.",
        "- SHAP remains the strongest standardized baseline for global+local attribution.",
        "- LIME is useful for textual local rationale but less stable across instance perturbations.",
        "- For FAIL class storytelling, combine SHAP waterfall for a best FAIL case with AIME similarity distribution to explain overlap.",
    ]
    if warnings:
        narrative.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    write_markdown(output_dir / "explainability_comparison.md", "\n".join(narrative))

    return {
        "selected_model": chosen_model,
        "selected_instances": selected_instances["instance_index"].astype(int).tolist(),
        "warnings": warnings,
        "generated_files": [
            "explainability_comparison.md",
            "explainability_comparison_table.md",
            "selected_instances.csv",
            "shap/shap_global_importance.csv",
            "shap/shap_local_instances.csv",
            "shap/fig_shap_beeswarm.png",
            *[f"shap/{name}" for name in shap_waterfalls],
            "lime/lime_local_instances.csv",
            *[f"lime/{name}" for name in lime_figs],
            "aime/aime_global_importance.csv",
            "aime/aime_local_instances.csv",
            "aime/aime_representative_instances.csv",
            "aime/aime_similarity_distribution.csv",
            "aime/fig_aime_global_importance.png",
            *[f"aime/{name}" for name in aime_local_figs],
            "aime/fig_aime_similarity_distribution.png",
            "aime/fig_aime_representative_instances.png",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exp", type=Path, required=True)
    parser.add_argument("--threshold-exp", type=Path, required=True)
    parser.add_argument("--boosting-exp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-model", type=str, required=False)
    args = parser.parse_args()

    result = run_explainability_comparison(
        baseline_exp=args.baseline_exp,
        threshold_exp=args.threshold_exp,
        boosting_exp=args.boosting_exp,
        output_dir=args.output_dir,
        selected_model=args.selected_model,
    )
    print(f"[explainability] generated {len(result['generated_files'])} artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
