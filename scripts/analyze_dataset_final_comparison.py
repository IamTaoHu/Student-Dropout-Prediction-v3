from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.final_dataset_helpers import (
    DatasetPaths,
    gather_experiment_model_rows,
    save_markdown_table,
    select_best_model_row,
)
from scripts.utils.final_result_helpers import safe_mkdir, save_json, save_table


def run_dataset_final_comparison(
    dataset_paths: DatasetPaths,
    output_root: Path,
    backfill_missing_fail_metrics: bool = True,
) -> dict[str, Any]:
    dataset = dataset_paths.dataset
    dataset_dir = safe_mkdir(output_root / dataset)
    warnings: list[str] = []

    role_map = {
        "baseline": dataset_paths.baseline_exp,
        "threshold": dataset_paths.threshold_exp,
        "boosting": dataset_paths.boosting_exp,
        "error_audit": dataset_paths.error_audit_exp,
    }
    parts: list[pd.DataFrame] = []
    backfill_models: list[str] = []
    unavailable_models: list[str] = []
    source_files: list[str] = []
    for role, path in role_map.items():
        part_df, part_warnings, part_meta = gather_experiment_model_rows(
            dataset=dataset,
            role=role,
            exp_path=path,
            backfill_missing_fail_metrics=backfill_missing_fail_metrics,
        )
        warnings.extend(part_warnings)
        backfill_models.extend(part_meta.get("backfilled_models", []))
        unavailable_models.extend(part_meta.get("unavailable_models", []))
        source_files.extend(part_meta.get("source_files", []))
        if not part_df.empty:
            parts.append(part_df)

    if parts:
        all_df = pd.concat(parts, ignore_index=True)
    else:
        all_df = pd.DataFrame(
            columns=[
                "dataset",
                "experiment_name",
                "model",
                "condition",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "fail_precision",
                "fail_recall",
                "fail_f1",
                "fail_metrics_source",
                "selected_as_best",
                "notes",
                "source_experiment_path",
            ]
        )
        warnings.append(f"{dataset}: no available experiments to compare")

    best_row = select_best_model_row(all_df)
    best_payload: dict[str, Any] = {
        "dataset": dataset,
        "experiment_name": None,
        "model": None,
        "source_paths": {
            "baseline_exp": str(dataset_paths.baseline_exp) if dataset_paths.baseline_exp else None,
            "threshold_exp": str(dataset_paths.threshold_exp) if dataset_paths.threshold_exp else None,
            "error_audit_exp": str(dataset_paths.error_audit_exp) if dataset_paths.error_audit_exp else None,
            "boosting_exp": str(dataset_paths.boosting_exp) if dataset_paths.boosting_exp else None,
        },
        "selected_metrics": {},
        "why_selected": "No candidate rows available.",
    }
    if best_row is not None:
        mask = (
            (all_df["experiment_name"] == best_row["experiment_name"])
            & (all_df["model"] == best_row["model"])
            & (all_df["condition"] == best_row["condition"])
        )
        all_df.loc[mask, "selected_as_best"] = True
        best_payload = {
            "dataset": dataset,
            "experiment_name": str(best_row["experiment_name"]),
            "model": str(best_row["model"]),
            "condition": str(best_row["condition"]),
            "source_paths": {
                "experiment": str(best_row.get("source_experiment_path", "")),
                "baseline_exp": str(dataset_paths.baseline_exp) if dataset_paths.baseline_exp else None,
                "threshold_exp": str(dataset_paths.threshold_exp) if dataset_paths.threshold_exp else None,
                "error_audit_exp": str(dataset_paths.error_audit_exp) if dataset_paths.error_audit_exp else None,
                "boosting_exp": str(dataset_paths.boosting_exp) if dataset_paths.boosting_exp else None,
            },
            "selected_metrics": {
                "accuracy": float(best_row.get("accuracy", float("nan"))),
                "macro_f1": float(best_row.get("macro_f1", float("nan"))),
                "weighted_f1": float(best_row.get("weighted_f1", float("nan"))),
                "balanced_accuracy": float(best_row.get("balanced_accuracy", float("nan"))),
                "fail_precision": float(best_row.get("fail_precision", float("nan"))),
                "fail_recall": float(best_row.get("fail_recall", float("nan"))),
                "fail_f1": float(best_row.get("fail_f1", float("nan"))),
            },
            "why_selected": (
                "Selected by descending macro_f1, then balanced_accuracy, then fail_f1, then accuracy."
            ),
        }

    all_df = all_df.sort_values(["macro_f1", "balanced_accuracy", "fail_f1", "accuracy"], ascending=False, na_position="last")
    csv_name = f"final_model_comparison_{dataset}.csv"
    md_name = f"final_model_comparison_{dataset}.md"
    best_json_name = f"best_model_{dataset}.json"
    save_table(dataset_dir / csv_name, all_df)
    md_df = all_df.copy()
    md_df.loc[md_df["selected_as_best"] == True, "model"] = md_df["model"].astype(str) + " *BEST*"
    save_markdown_table(dataset_dir / md_name, md_df, title=f"Final Model Comparison ({dataset.upper()})")
    save_json(dataset_dir / best_json_name, best_payload)

    return {
        "dataset": dataset,
        "comparison_df": all_df,
        "best_model": best_payload,
        "warnings": warnings,
        "fail_backfill": {
            "backfilled_models": sorted(set(backfill_models)),
            "unavailable_models": sorted(set(unavailable_models)),
            "source_files": sorted(set(source_files)),
        },
        "generated_files": [
            f"{dataset}/{csv_name}",
            f"{dataset}/{md_name}",
            f"{dataset}/{best_json_name}",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True, choices=["uct", "oulad"])
    parser.add_argument("--baseline-exp", type=Path, required=False)
    parser.add_argument("--threshold-exp", type=Path, required=False)
    parser.add_argument("--error-audit-exp", type=Path, required=False)
    parser.add_argument("--boosting-exp", type=Path, required=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backfill-missing-fail-metrics", action="store_true")
    args = parser.parse_args()

    result = run_dataset_final_comparison(
        dataset_paths=DatasetPaths(
            dataset=args.dataset,
            baseline_exp=args.baseline_exp,
            threshold_exp=args.threshold_exp,
            error_audit_exp=args.error_audit_exp,
            boosting_exp=args.boosting_exp,
        ),
        output_root=args.output_root,
        backfill_missing_fail_metrics=True,
    )
    print(f"[dataset-comparison] {args.dataset}: generated {len(result['generated_files'])} files")


if __name__ == "__main__":
    main()
