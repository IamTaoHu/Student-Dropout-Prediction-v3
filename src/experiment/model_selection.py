from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.reporting.prediction_exports import _resolve_metric_column

def _apply_global_balance_guard(
    leaderboard_df: pd.DataFrame,
    *,
    guard_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": bool(guard_cfg.get("enabled", False)),
        "reference_source": guard_cfg.get("reference_source"),
        "reference_metrics": {},
        "fallback_used": False,
        "candidate_decisions": [],
    }
    if leaderboard_df.empty or not bool(guard_cfg.get("enabled", False)):
        return leaderboard_df, report

    required_cols = {"test_macro_f1", "test_f1_enrolled", "test_f1_graduate"}
    if not required_cols.issubset(set(leaderboard_df.columns)):
        report["fallback_used"] = True
        report["fallback_reason"] = "missing_required_columns"
        print("[v8] fallback to unguarded selection because missing guard columns in leaderboard.")
        return leaderboard_df, report

    ranked_plain = leaderboard_df.copy()
    for col in required_cols:
        ranked_plain[col] = pd.to_numeric(ranked_plain[col], errors="coerce")
    plain_sort_cols = [col for col in ["test_macro_f1", "test_balanced_accuracy", "test_accuracy"] if col in ranked_plain.columns]
    ranked_plain = ranked_plain.sort_values(
        [*plain_sort_cols, "model"],
        ascending=[False] * len(plain_sort_cols) + [True],
        na_position="last",
    ).reset_index(drop=True)
    if ranked_plain.empty:
        return leaderboard_df, report

    reference_row = ranked_plain.iloc[0]
    reference_metrics = {
        "model": str(reference_row["model"]),
        "macro_f1": float(reference_row.get("test_macro_f1", np.nan)),
        "enrolled_f1": float(reference_row.get("test_f1_enrolled", np.nan)),
        "graduate_f1": float(reference_row.get("test_f1_graduate", np.nan)),
        "dropout_f1": float(reference_row.get("test_f1_dropout", np.nan)) if "test_f1_dropout" in ranked_plain.columns else np.nan,
    }
    report["reference_metrics"] = reference_metrics

    guarded = ranked_plain.copy()
    guarded["guard_selection_score"] = guarded["test_macro_f1"].astype(float)
    guarded["guard_pass"] = True
    max_drop = guard_cfg.get("max_graduate_f1_drop")
    min_macro = guard_cfg.get("min_macro_f1")
    min_grad = guard_cfg.get("min_graduate_f1")
    penalty_weight = float(guard_cfg.get("penalty_weight", 0.5))

    for idx, row in guarded.iterrows():
        enrolled_delta = float(row["test_f1_enrolled"] - reference_metrics["enrolled_f1"])
        graduate_delta = float(row["test_f1_graduate"] - reference_metrics["graduate_f1"])
        dropout_delta = (
            float(row["test_f1_dropout"] - reference_metrics["dropout_f1"])
            if "test_f1_dropout" in guarded.columns and np.isfinite(reference_metrics["dropout_f1"])
            else float("nan")
        )
        macro_f1 = float(row["test_macro_f1"])
        graduate_f1 = float(row["test_f1_graduate"])
        violations: list[str] = []
        if min_macro is not None and macro_f1 < float(min_macro):
            violations.append("macro_floor")
        if min_grad is not None and graduate_f1 < float(min_grad):
            violations.append("graduate_floor")
        if max_drop is not None and enrolled_delta > 0.0 and graduate_delta < -float(max_drop):
            violations.append("graduate_drop_exceeded")
        penalty = penalty_weight * max(0.0, -(graduate_delta + float(max_drop or 0.0))) if "graduate_drop_exceeded" in violations else 0.0
        guarded.at[idx, "guard_selection_score"] = macro_f1 - penalty
        guarded.at[idx, "guard_pass"] = len(violations) == 0
        decision = {
            "model": str(row["model"]),
            "enrolled_f1_delta": enrolled_delta,
            "graduate_f1_delta": graduate_delta,
            "dropout_f1_delta": dropout_delta,
            "guard_pass": bool(len(violations) == 0),
            "guarded_selection_score": float(guarded.at[idx, "guard_selection_score"]),
            "violations": violations,
        }
        report["candidate_decisions"].append(decision)
        print(
            "[v8] guard metrics: "
            f"model={decision['model']} enrolled_f1={float(row['test_f1_enrolled']):.4f} "
            f"graduate_f1={graduate_f1:.4f} macro_f1={macro_f1:.4f} "
            f"delta_enrolled={enrolled_delta:+.4f} delta_graduate={graduate_delta:+.4f} "
            f"delta_dropout={dropout_delta:+.4f}"
        )
        print(
            f"[v8] guard decision: model={decision['model']} "
            f"{'pass' if decision['guard_pass'] else 'fail'} score={decision['guarded_selection_score']:.4f} "
            f"violations={violations}"
        )

    passing = guarded[guarded["guard_pass"] == True].copy()
    if passing.empty and bool(guard_cfg.get("fallback_to_plain_macro_f1_if_no_candidate_passes", True)):
        report["fallback_used"] = True
        report["fallback_reason"] = "no_candidate_passed_guard"
        print("[v8] fallback to unguarded selection because no candidate passed the global balance guard.")
        return ranked_plain, report

    ranked_output = passing if not passing.empty else guarded
    guard_sort_cols = [col for col in ["guard_selection_score", "test_macro_f1", "test_balanced_accuracy", "test_accuracy"] if col in ranked_output.columns]
    ranked_output = ranked_output.sort_values(
        [*guard_sort_cols, "model"],
        ascending=[False] * len(guard_sort_cols) + [True],
        na_position="last",
    ).reset_index(drop=True)
    return ranked_output, report


def _sort_leaderboard_with_tiebreak(
    leaderboard_df: pd.DataFrame,
    selection_cfg: dict[str, Any],
    source: str,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    if leaderboard_df.empty:
        return leaderboard_df, None, []
    ranking_columns = [
        _resolve_metric_column(metric_name, source=source)
        for metric_name in selection_cfg.get("ranking_metrics", [])
    ]
    ranking_columns = [c for c in ranking_columns if c in leaderboard_df.columns]
    if not ranking_columns:
        return leaderboard_df, None, []

    ranked = leaderboard_df.copy()
    for col in ranking_columns:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")
    ranked = ranked.sort_values(
        by=[*ranking_columns, "model"],
        ascending=[False] * len(ranking_columns) + [True],
        na_position="last",
    ).reset_index(drop=True)
    best_model = str(ranked.iloc[0]["model"]) if not ranked.empty else None
    return ranked, best_model, ranking_columns
