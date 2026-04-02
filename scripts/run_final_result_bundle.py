from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_error_audit import run_error_audit
from scripts.analyze_dataset_final_comparison import run_dataset_final_comparison
from scripts.analyze_dataset_final_figures import run_dataset_final_figures
from scripts.analyze_explainability_comparison import run_explainability_comparison
from scripts.analyze_threshold_tuning import run_threshold_analysis
from scripts.utils.final_dataset_helpers import DatasetPaths, discover_default_dataset_paths
from scripts.utils.final_result_helpers import (
    BundleContext,
    collect_relative_file_list,
    load_table,
    safe_mkdir,
    save_json,
    utc_timestamp,
    write_markdown,
)


def _default_path_if_exists(path: Path) -> Path | None:
    return path if path.exists() else None


def _resolve_inputs(args: argparse.Namespace) -> BundleContext:
    root = Path.cwd()
    baseline = Path(args.baseline_exp) if args.baseline_exp else _default_path_if_exists(root / "results" / "exp_bm_uct_3class")
    threshold = (
        Path(args.threshold_exp)
        if args.threshold_exp
        else _default_path_if_exists(root / "results" / "exp_uct_3class_threshold_tuning_v1")
    )
    error_audit = (
        Path(args.error_audit_exp)
        if args.error_audit_exp
        else _default_path_if_exists(root / "results" / "exp_uct_3class_error_audit")
    )
    boosting = (
        Path(args.boosting_exp)
        if args.boosting_exp
        else _default_path_if_exists(root / "results" / "exp_uct_3class_boosting_optuna_v1")
    )
    output_dir = Path(args.output_dir) if args.output_dir else (root / "results" / "final_result")
    return BundleContext(
        baseline_exp=baseline,
        threshold_exp=threshold,
        error_audit_exp=error_audit,
        boosting_exp=boosting,
        output_dir=output_dir,
    )


def _resolve_dataset_paths(args: argparse.Namespace, dataset: str) -> DatasetPaths:
    defaults = discover_default_dataset_paths(Path.cwd(), dataset)
    if dataset == "uct":
        baseline = Path(args.uct_baseline_exp) if args.uct_baseline_exp else defaults.baseline_exp
        threshold = Path(args.uct_threshold_exp) if args.uct_threshold_exp else defaults.threshold_exp
        error_audit = Path(args.uct_error_audit_exp) if args.uct_error_audit_exp else defaults.error_audit_exp
        boosting = Path(args.uct_boosting_exp) if args.uct_boosting_exp else defaults.boosting_exp
        # Backward compatibility with legacy single-dataset flags.
        baseline = baseline or (Path(args.baseline_exp) if args.baseline_exp else None)
        threshold = threshold or (Path(args.threshold_exp) if args.threshold_exp else None)
        error_audit = error_audit or (Path(args.error_audit_exp) if args.error_audit_exp else None)
        boosting = boosting or (Path(args.boosting_exp) if args.boosting_exp else None)
    else:
        baseline = Path(args.oulad_baseline_exp) if args.oulad_baseline_exp else defaults.baseline_exp
        threshold = Path(args.oulad_threshold_exp) if args.oulad_threshold_exp else defaults.threshold_exp
        error_audit = Path(args.oulad_error_audit_exp) if args.oulad_error_audit_exp else defaults.error_audit_exp
        boosting = Path(args.oulad_boosting_exp) if args.oulad_boosting_exp else defaults.boosting_exp
    return DatasetPaths(
        dataset=dataset,
        baseline_exp=baseline if baseline and baseline.exists() else None,
        threshold_exp=threshold if threshold and threshold.exists() else None,
        error_audit_exp=error_audit if error_audit and error_audit.exists() else None,
        boosting_exp=boosting if boosting and boosting.exists() else None,
    )


def _build_root_readme(ctx: BundleContext) -> str:
    return "\n".join(
        [
            "# final_result",
            "",
            "Unified publication-oriented result bundle with threshold tuning, FAIL-focused error audit, and SHAP/LIME/AIME comparison.",
            "",
            "## Inputs",
            f"- baseline_exp: `{ctx.baseline_exp}`",
            f"- threshold_exp: `{ctx.threshold_exp}`",
            f"- error_audit_exp: `{ctx.error_audit_exp}`",
            f"- boosting_exp: `{ctx.boosting_exp}`",
            "",
            "## Main sections",
            "- `threshold_tuning/`",
            "- `error_audit/`",
            "- `explainability_comparison/`",
            "- `dataset_comparison/`",
            "- `paper_pack/`",
        ]
    )


def _write_paper_pack(root: Path, explainability_result: dict[str, Any] | None) -> list[str]:
    paper_pack = safe_mkdir(root / "paper_pack")
    best_fail_case = "unknown"
    if explainability_result:
        selected = explainability_result.get("selected_instances") or []
        if selected:
            best_fail_case = str(selected[0])

    figures_md = "\n".join(
        [
            "# Paper-Ready Figures",
            "",
            "Priority figures for manuscript-ready narrative:",
            "",
            "1. `threshold_tuning/fig_fail_f1_before_after.png`",
            "2. `threshold_tuning/fig_threshold_tradeoff_fail_vs_macro.png`",
            "3. `error_audit/fig_fail_confusion_breakdown.png`",
            "4. `explainability_comparison/shap/fig_shap_beeswarm.png`",
            f"5. `explainability_comparison/shap/fig_shap_waterfall_instance_{best_fail_case}.png`",
            "6. `explainability_comparison/aime/fig_aime_global_importance.png`",
            f"7. `explainability_comparison/aime/fig_aime_local_instance_{best_fail_case}.png`",
            "8. `explainability_comparison/aime/fig_aime_similarity_distribution.png`",
        ]
    )
    tables_md = "\n".join(
        [
            "# Paper-Ready Tables",
            "",
            "1. `threshold_tuning/threshold_before_after.csv`",
            "2. `threshold_tuning/fail_class_delta_table.csv`",
            "3. `error_audit/fail_confusion_routes.csv`",
            "4. `explainability_comparison/explainability_comparison_table.md`",
            "5. `error_audit/fail_error_feature_summary.csv`",
        ]
    )
    findings_md = "\n".join(
        [
            "# Key Findings For Paper",
            "",
            "- Threshold tuning can improve FAIL class F1 in selected models, but requires macro-F1 and balanced-accuracy guardrails.",
            "- FAIL errors are concentrated in a small set of confusion routes, making targeted boundary interventions feasible.",
            "- SHAP provides the strongest baseline for global and local attribution.",
            "- LIME offers readable local surrogate explanations but can be less stable.",
            "- AIME-style representative-instance similarity views improve interpretation of FAIL/PASS/DROPOUT overlap.",
        ]
    )
    claims_md = "\n".join(
        [
            "# Claims With Evidence",
            "",
            "| claim | supporting file(s) | metric / figure / table evidence | caution level |",
            "|---|---|---|---|",
            "| Threshold tuning improves FAIL for selected models under guardrails | `threshold_tuning/fail_class_delta_table.csv`, `threshold_tuning/fig_fail_f1_before_after.png` | `delta_fail_f1`, guardrail conclusion | medium |",
            "| Dominant FAIL confusion routes are identifiable | `error_audit/fail_confusion_routes.csv`, `error_audit/fig_fail_confusion_breakdown.png` | route count and `%` of FAIL errors | low |",
            "| Fail mistakes correlate with specific feature shifts | `error_audit/fail_error_feature_summary.csv`, `error_audit/fig_fail_error_feature_importance.png` | ranked `abs_delta` across FAIL groups | medium |",
            "| AIME-style similarity plot supports class overlap narrative | `explainability_comparison/aime/aime_similarity_distribution.csv`, `explainability_comparison/aime/fig_aime_similarity_distribution.png` | similarity density across labels | medium |",
        ]
    )
    limitations_md = "\n".join(
        [
            "# Limitations And Cautions",
            "",
            "- Threshold tuning may improve FAIL metrics while harming non-FAIL classes if guardrails are not enforced.",
            "- Row-level error analysis quality depends on the availability of prediction-level and aligned feature artifacts.",
            "- AIME component in this repository is an approximation if exact inverse construction is unavailable.",
            "- SHAP/LIME/AIME are post-hoc explanations and should not be interpreted as causal effects.",
        ]
    )

    write_markdown(paper_pack / "paper_ready_figures.md", figures_md)
    write_markdown(paper_pack / "paper_ready_tables.md", tables_md)
    write_markdown(paper_pack / "key_findings_for_paper.md", findings_md)
    write_markdown(paper_pack / "claims_with_evidence.md", claims_md)
    write_markdown(paper_pack / "limitations_and_cautions.md", limitations_md)
    return [
        "paper_pack/paper_ready_figures.md",
        "paper_pack/paper_ready_tables.md",
        "paper_pack/key_findings_for_paper.md",
        "paper_pack/claims_with_evidence.md",
        "paper_pack/limitations_and_cautions.md",
    ]


def _write_dataset_paper_outputs(root: Path, best_df: pd.DataFrame) -> list[str]:
    paper_pack = safe_mkdir(root / "paper_pack")
    dataset_comparison_md = [
        "# Final Dataset Comparison",
        "",
    ]
    for dataset in ["uct", "oulad"]:
        sub = best_df[best_df["dataset"] == dataset] if not best_df.empty else pd.DataFrame()
        if sub.empty:
            dataset_comparison_md.append(f"- {dataset.upper()}: no best-model entry available.")
            continue
        row = sub.iloc[0]
        dataset_comparison_md.append(
            f"- {dataset.upper()} winner: `{row['best_model']}` from `{row['best_experiment']}` "
            f"(macro_f1={row['macro_f1']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f})."
        )
        dataset_comparison_md.append(
            f"  Threshold usefulness signal: {'yes' if bool(row['threshold_tuned']) else 'no/unknown'}; "
            f"FAIL F1={row['fail_f1']:.4f}."
        )
    dataset_comparison_md.extend(
        [
            "",
            "Both datasets can share the same figure families (confusion matrix, FAIL-focused PR curve, learning curve),",
            "while class-label mapping differences should be documented explicitly (3-class UCT vs binary OULAD).",
        ]
    )

    recommended_tables = "\n".join(
        [
            "# Recommended Main Tables",
            "",
            "Main Table 1:",
            "- `dataset_comparison/uct/final_model_comparison_uct.md`",
            "- `dataset_comparison/oulad/final_model_comparison_oulad.md`",
            "- Optional compact body-table: `dataset_comparison/combined/final_model_comparison_all_datasets.md`",
            "",
            "Main Table 2:",
            "- `dataset_comparison/combined/final_best_models_summary.md`",
        ]
    )
    recommended_figures = "\n".join(
        [
            "# Recommended Main Figures",
            "",
            "For UCT (main paper):",
            "- `dataset_comparison/uct/fig_confusion_matrix_best_model_uct.png`",
            "- `dataset_comparison/uct/fig_precision_recall_curve_best_model_uct.png`",
            "- `dataset_comparison/uct/fig_learning_curve_best_model_uct.png`",
            "",
            "For OULAD (main paper or appendix depending on space):",
            "- `dataset_comparison/oulad/fig_confusion_matrix_best_model_oulad.png`",
            "- `dataset_comparison/oulad/fig_precision_recall_curve_best_model_oulad.png`",
            "- `dataset_comparison/oulad/fig_learning_curve_best_model_oulad.png`",
            "",
            "Appendix recommendation:",
            "- Include normalized confusion matrices for both datasets.",
        ]
    )
    write_markdown(paper_pack / "final_dataset_comparison.md", "\n".join(dataset_comparison_md))
    write_markdown(paper_pack / "recommended_main_tables.md", recommended_tables)
    write_markdown(paper_pack / "recommended_main_figures.md", recommended_figures)
    return [
        "paper_pack/final_dataset_comparison.md",
        "paper_pack/recommended_main_tables.md",
        "paper_pack/recommended_main_figures.md",
    ]


def run_bundle(args: argparse.Namespace) -> dict[str, Any]:
    ctx = _resolve_inputs(args)
    safe_mkdir(ctx.output_dir)
    safe_mkdir(ctx.output_dir / "threshold_tuning")
    safe_mkdir(ctx.output_dir / "error_audit")
    safe_mkdir(ctx.output_dir / "explainability_comparison")
    safe_mkdir(ctx.output_dir / "paper_pack")

    write_markdown(ctx.output_dir / "README.md", _build_root_readme(ctx))

    warnings: list[str] = []
    module_outputs: dict[str, Any] = {}
    dataset_outputs: dict[str, Any] = {}
    selected_model = None
    selected_instances: list[int] = []

    if args.only_dataset_comparison:
        run_threshold = False
        run_error = False
        run_explainability = False
    else:
        run_threshold = not (args.only_error_audit or args.only_explainability)
        run_error = not (args.only_threshold or args.only_explainability)
        run_explainability = not (args.only_threshold or args.only_error_audit)

    if run_threshold:
        if ctx.baseline_exp is None or ctx.threshold_exp is None:
            warnings.append("Threshold analysis skipped: baseline or threshold experiment path missing.")
        else:
            module_outputs["threshold_tuning"] = run_threshold_analysis(
                baseline_exp=ctx.baseline_exp,
                threshold_exp=ctx.threshold_exp,
                output_dir=ctx.output_dir / "threshold_tuning",
                macro_tolerance=args.macro_tolerance,
                balanced_tolerance=args.balanced_tolerance,
            )
            warnings.extend(module_outputs["threshold_tuning"].get("warnings", []))

    if run_error:
        if ctx.baseline_exp is None or ctx.error_audit_exp is None:
            warnings.append("Error audit skipped: baseline or error-audit experiment path missing.")
        else:
            module_outputs["error_audit"] = run_error_audit(
                baseline_exp=ctx.baseline_exp,
                threshold_exp=ctx.threshold_exp,
                error_audit_exp=ctx.error_audit_exp,
                output_dir=ctx.output_dir / "error_audit",
            )
            warnings.extend(module_outputs["error_audit"].get("warnings", []))

    if run_explainability:
        if ctx.baseline_exp is None or ctx.threshold_exp is None or ctx.boosting_exp is None:
            warnings.append("Explainability comparison skipped: baseline/threshold/boosting path missing.")
        else:
            module_outputs["explainability_comparison"] = run_explainability_comparison(
                baseline_exp=ctx.baseline_exp,
                threshold_exp=ctx.threshold_exp,
                boosting_exp=ctx.boosting_exp,
                output_dir=ctx.output_dir / "explainability_comparison",
                selected_model=args.selected_model,
            )
            selected_model = module_outputs["explainability_comparison"].get("selected_model")
            selected_instances = module_outputs["explainability_comparison"].get("selected_instances", [])
            warnings.extend(module_outputs["explainability_comparison"].get("warnings", []))

    if selected_model is None and ctx.threshold_exp is not None:
        leaderboard = load_table(ctx.threshold_exp / "leaderboard.csv")
        if leaderboard is not None and not leaderboard.empty:
            selected_model = str(leaderboard.sort_values("test_macro_f1", ascending=False).iloc[0]["model"])

    run_dataset_comparison = bool(
        args.build_dataset_comparison or args.only_dataset_comparison or args.only_uct or args.only_oulad
    )
    if run_dataset_comparison:
        dataset_root = safe_mkdir(ctx.output_dir / "dataset_comparison")
        combined_root = safe_mkdir(dataset_root / "combined")
        selected_datasets: list[str] = ["uct", "oulad"]
        if args.only_uct:
            selected_datasets = ["uct"]
        if args.only_oulad:
            selected_datasets = ["oulad"]

        comparison_frames: list[pd.DataFrame] = []
        best_rows: list[dict[str, Any]] = []
        for dataset in selected_datasets:
            ds_paths = _resolve_dataset_paths(args, dataset)
            ds_result = run_dataset_final_comparison(
                ds_paths,
                dataset_root,
                backfill_missing_fail_metrics=True,
            )
            dataset_outputs[f"{dataset}_comparison"] = {
                "dataset": ds_result.get("dataset"),
                "best_model": ds_result.get("best_model"),
                "fail_backfill": ds_result.get("fail_backfill", {}),
                "warnings": ds_result.get("warnings", []),
                "generated_files": ds_result.get("generated_files", []),
            }
            warnings.extend(ds_result.get("warnings", []))
            comp_df = ds_result.get("comparison_df", pd.DataFrame())
            if isinstance(comp_df, pd.DataFrame) and not comp_df.empty:
                comparison_frames.append(comp_df)

            best_info = ds_result.get("best_model", {}) or {}
            best_model_path = dataset_root / dataset / f"best_model_{dataset}.json"
            comparison_csv = dataset_root / dataset / f"final_model_comparison_{dataset}.csv"
            fig_result = run_dataset_final_figures(
                dataset=dataset,
                best_model_json=best_model_path,
                comparison_csv=comparison_csv,
                output_root=dataset_root,
                recompute_learning_curves=args.recompute_learning_curves,
            )
            dataset_outputs[f"{dataset}_figures"] = fig_result
            warnings.extend(fig_result.get("warnings", []))
            best_rows.append(
                {
                    "dataset": dataset,
                    "best_model": best_info.get("model"),
                    "best_experiment": best_info.get("experiment_name"),
                    "accuracy": (best_info.get("selected_metrics", {}) or {}).get("accuracy"),
                    "macro_f1": (best_info.get("selected_metrics", {}) or {}).get("macro_f1"),
                    "balanced_accuracy": (best_info.get("selected_metrics", {}) or {}).get("balanced_accuracy"),
                    "fail_f1": (best_info.get("selected_metrics", {}) or {}).get("fail_f1"),
                    "threshold_tuned": (best_info.get("condition") == "threshold_tuned"),
                    "key_takeaway": (
                        f"{dataset.upper()} best={best_info.get('model')} from {best_info.get('experiment_name')}"
                        if best_info.get("model")
                        else "insufficient inputs"
                    ),
                }
            )

        combined_df = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
        if not combined_df.empty:
            combined_df = combined_df.sort_values(["dataset", "macro_f1"], ascending=[True, False])
        combined_csv = combined_root / "final_model_comparison_all_datasets.csv"
        combined_md = combined_root / "final_model_comparison_all_datasets.md"
        combined_df.to_csv(combined_csv, index=False)
        write_markdown(
            combined_md,
            "# Final Model Comparison (All Datasets)\n\n"
            + (combined_df.to_markdown(index=False) if not combined_df.empty else "No comparable rows available.\n"),
        )

        best_df = pd.DataFrame(best_rows)
        best_csv = combined_root / "final_best_models_summary.csv"
        best_md = combined_root / "final_best_models_summary.md"
        best_df.to_csv(best_csv, index=False)
        write_markdown(
            best_md,
            "# Final Best Models Summary\n\n"
            + (best_df.to_markdown(index=False) if not best_df.empty else "No best-model rows available.\n"),
        )
        dataset_outputs["combined_dataset_comparison"] = {
            "generated_files": [
                "dataset_comparison/combined/final_model_comparison_all_datasets.csv",
                "dataset_comparison/combined/final_model_comparison_all_datasets.md",
                "dataset_comparison/combined/final_best_models_summary.csv",
                "dataset_comparison/combined/final_best_models_summary.md",
            ]
        }
        dataset_paper_files = _write_dataset_paper_outputs(ctx.output_dir, best_df)
    else:
        dataset_paper_files = []

    paper_files = _write_paper_pack(ctx.output_dir, module_outputs.get("explainability_comparison"))
    paper_files.extend(dataset_paper_files)

    final_summary_lines = [
        "# Final Result Summary",
        "",
        f"- Generated at: `{utc_timestamp()}`",
        f"- Output root: `{ctx.output_dir}`",
        f"- Selected model: `{selected_model}`",
        f"- Selected instances: `{selected_instances}`",
        "",
        "## Completed Sections",
    ]
    for section in ["threshold_tuning", "error_audit", "explainability_comparison"]:
        status = "completed" if section in module_outputs else "skipped"
        final_summary_lines.append(f"- {section}: {status}")
    if run_dataset_comparison:
        ds_label = "UCT+OULAD"
        if args.only_uct:
            ds_label = "UCT"
        if args.only_oulad:
            ds_label = "OULAD"
        final_summary_lines.append(f"- dataset_comparison: completed ({ds_label})")
        oulad_backfill = (dataset_outputs.get("oulad_comparison", {}) or {}).get("fail_backfill", {})
        if oulad_backfill:
            recovered = oulad_backfill.get("backfilled_models", []) or []
            unavailable = oulad_backfill.get("unavailable_models", []) or []
            src_files = oulad_backfill.get("source_files", []) or []
            final_summary_lines.extend(
                [
                    "",
                    "## OULAD FAIL Metric Backfill",
                    "- recovered FAIL metrics for: " + (", ".join(recovered) if recovered else "(none)"),
                    "- unavailable for: " + (", ".join(unavailable) if unavailable else "(none)"),
                    "- source: " + (", ".join(src_files) if src_files else "summary/report/runtime artifacts"),
                ]
            )
    if warnings:
        final_summary_lines.extend(["", "## Warnings", *[f"- {w}" for w in warnings]])
    write_markdown(ctx.output_dir / "final_summary.md", "\n".join(final_summary_lines))

    all_files = collect_relative_file_list(ctx.output_dir)
    manifest = {
        "generation_timestamp_utc": utc_timestamp(),
        "source_experiment_paths": {
            "baseline_exp": str(ctx.baseline_exp) if ctx.baseline_exp else None,
            "threshold_exp": str(ctx.threshold_exp) if ctx.threshold_exp else None,
            "error_audit_exp": str(ctx.error_audit_exp) if ctx.error_audit_exp else None,
            "boosting_exp": str(ctx.boosting_exp) if ctx.boosting_exp else None,
        },
        "selected_best_model": selected_model,
        "selected_instances": selected_instances,
        "module_outputs": module_outputs,
        "dataset_outputs": dataset_outputs,
        "paper_pack_files": paper_files,
        "all_output_files": all_files,
        "warnings": warnings,
    }
    manifest_path = ctx.output_dir / "artifact_manifest.json"
    save_json(manifest_path, manifest)
    manifest["all_output_files"] = collect_relative_file_list(ctx.output_dir)
    save_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Legacy single-dataset (UCT-first) inputs
    parser.add_argument("--baseline-exp", type=Path, required=False)
    parser.add_argument("--threshold-exp", type=Path, required=False)
    parser.add_argument("--error-audit-exp", type=Path, required=False)
    parser.add_argument("--boosting-exp", type=Path, required=False)
    # Dataset-specific inputs
    parser.add_argument("--uct-baseline-exp", type=Path, required=False)
    parser.add_argument("--uct-threshold-exp", type=Path, required=False)
    parser.add_argument("--uct-error-audit-exp", type=Path, required=False)
    parser.add_argument("--uct-boosting-exp", type=Path, required=False)
    parser.add_argument("--oulad-baseline-exp", type=Path, required=False)
    parser.add_argument("--oulad-threshold-exp", type=Path, required=False)
    parser.add_argument("--oulad-error-audit-exp", type=Path, required=False)
    parser.add_argument("--oulad-boosting-exp", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_result"))
    parser.add_argument("--selected-model", type=str, required=False)
    parser.add_argument("--macro-tolerance", type=float, default=0.01)
    parser.add_argument("--balanced-tolerance", type=float, default=0.01)
    parser.add_argument("--only-threshold", action="store_true")
    parser.add_argument("--only-error-audit", action="store_true")
    parser.add_argument("--only-explainability", action="store_true")
    parser.add_argument("--build-dataset-comparison", action="store_true")
    parser.add_argument("--only-dataset-comparison", action="store_true")
    parser.add_argument("--only-uct", action="store_true")
    parser.add_argument("--only-oulad", action="store_true")
    parser.add_argument("--recompute-learning-curves", action="store_true")
    parser.add_argument("--backfill-missing-fail-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_bundle(args)
    total_files = len(manifest.get("all_output_files", []))
    dataset_outputs = manifest.get("dataset_outputs", {}) or {}
    oulad_backfill = (dataset_outputs.get("oulad_comparison", {}) or {}).get("fail_backfill", {}) or {}
    recovered = oulad_backfill.get("backfilled_models", []) or []
    source_files = oulad_backfill.get("source_files", []) or []
    if recovered:
        print(f"[oulad-backfill] recovered FAIL metrics for models: {', '.join(recovered)}")
    elif oulad_backfill:
        print("[oulad-backfill] recovered FAIL metrics for models: (none)")
    if source_files:
        print(f"[oulad-backfill] source files: {', '.join(source_files)}")
    print(f"[final_result] generated {total_files} files in {args.output_dir}")
    print(f"[final_result] selected model: {manifest.get('selected_best_model')}")
    print(f"[final_result] manifest: {args.output_dir / 'artifact_manifest.json'}")
    if dataset_outputs.get("oulad_comparison"):
        print("[final_result] updated OULAD comparison tables")


if __name__ == "__main__":
    main()
