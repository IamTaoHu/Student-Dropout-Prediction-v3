"""Stage 2-specific UCT feature sharpening helpers for enrolled vs graduate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STAGE2_FEATURE_GROUPS = [
    "grade_drift",
    "approval_ratio",
    "eval_mismatch",
    "continuity",
    "stability",
    "enrolled_middle_state",
    "cross_sem_interactions",
]

ACADEMIC_COLUMN_ALIASES: dict[str, list[str]] = {
    "approved_1st_sem": ["approved_1st_sem", "curricular_units_1st_sem_approved"],
    "approved_2nd_sem": ["approved_2nd_sem", "curricular_units_2nd_sem_approved"],
    "enrolled_1st_sem": ["enrolled_1st_sem", "curricular_units_1st_sem_enrolled"],
    "enrolled_2nd_sem": ["enrolled_2nd_sem", "curricular_units_2nd_sem_enrolled"],
    "evaluations_1st_sem": ["evaluations_1st_sem", "curricular_units_1st_sem_evaluations"],
    "evaluations_2nd_sem": ["evaluations_2nd_sem", "curricular_units_2nd_sem_evaluations"],
    "grade_1st_sem": ["grade_1st_sem", "curricular_units_1st_sem_grade"],
    "grade_2nd_sem": ["grade_2nd_sem", "curricular_units_2nd_sem_grade"],
}

GROUP_FEATURE_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "grade_drift": {
        "stage2_sharp_grade_diff_2nd_minus_1st": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_grade_ratio_2nd_over_1st": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_grade_abs_diff": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_grade_improving_flag": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_grade_declining_flag": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_grade_stable_band_flag": ["grade_1st_sem", "grade_2nd_sem"],
    },
    "approval_ratio": {
        "stage2_sharp_approved_enrolled_ratio_1st": ["approved_1st_sem", "enrolled_1st_sem"],
        "stage2_sharp_approved_enrolled_ratio_2nd": ["approved_2nd_sem", "enrolled_2nd_sem"],
        "stage2_sharp_approval_completion_gap_1st": ["approved_1st_sem", "enrolled_1st_sem"],
        "stage2_sharp_approval_completion_gap_2nd": ["approved_2nd_sem", "enrolled_2nd_sem"],
        "stage2_sharp_ratio_change_2nd_minus_1st": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
    },
    "eval_mismatch": {
        "stage2_sharp_eval_minus_approved_1st": ["evaluations_1st_sem", "approved_1st_sem"],
        "stage2_sharp_eval_minus_approved_2nd": ["evaluations_2nd_sem", "approved_2nd_sem"],
        "stage2_sharp_approved_over_evaluations_1st": ["approved_1st_sem", "evaluations_1st_sem"],
        "stage2_sharp_approved_over_evaluations_2nd": ["approved_2nd_sem", "evaluations_2nd_sem"],
        "stage2_sharp_grade_vs_approval_efficiency_1st": ["grade_1st_sem", "approved_1st_sem", "enrolled_1st_sem"],
        "stage2_sharp_grade_vs_approval_efficiency_2nd": ["grade_2nd_sem", "approved_2nd_sem", "enrolled_2nd_sem"],
    },
    "continuity": {
        "stage2_sharp_enrolled_both_semesters_flag": ["enrolled_1st_sem", "enrolled_2nd_sem"],
        "stage2_sharp_approved_both_semesters_flag": ["approved_1st_sem", "approved_2nd_sem"],
        "stage2_sharp_evaluated_both_semesters_flag": ["evaluations_1st_sem", "evaluations_2nd_sem"],
        "stage2_sharp_activity_continuity_score": [
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
        ],
        "stage2_sharp_progression_continuity_gap": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_sharp_second_sem_present_flag": ["enrolled_2nd_sem"],
        "stage2_sharp_first_to_second_enrollment_retention_flag": ["enrolled_1st_sem", "enrolled_2nd_sem"],
    },
    "stability": {
        "stage2_sharp_total_semester_grade_mean": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_total_semester_grade_std_proxy": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_approval_stability_proxy": ["approved_1st_sem", "approved_2nd_sem"],
        "stage2_sharp_grade_consistency_band": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_sharp_approved_consistency_band": ["approved_1st_sem", "approved_2nd_sem"],
    },
    "enrolled_middle_state": {
        "stage2_sharp_second_sem_enrolled_positive_and_approved_low_flag": ["enrolled_2nd_sem", "approved_2nd_sem"],
        "stage2_sharp_moderate_progress_flag": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_sharp_incomplete_but_active_flag": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_sharp_evaluation_active_but_completion_lag_flag": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
        ],
        "stage2_sharp_persistent_low_completion_flag": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
    },
    "cross_sem_interactions": {
        "stage2_sharp_approved_2nd_minus_enrolled_1st": ["approved_2nd_sem", "enrolled_1st_sem"],
        "stage2_sharp_approved_total_over_enrolled_total": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_sharp_grade_mean_x_approval_ratio": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_sharp_second_sem_weighted_progress": ["approved_2nd_sem", "enrolled_2nd_sem", "grade_2nd_sem"],
        "stage2_sharp_first_sem_to_second_sem_progression_ratio": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
    },
}


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found:
            return found
    return None


def _resolve_semester_sources(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    columns = list(df.columns)
    for canonical, aliases in ACADEMIC_COLUMN_ALIASES.items():
        found = _first_existing(columns, aliases)
        if found:
            resolved[canonical] = found
    return resolved


def _numeric(df: pd.DataFrame, column_name: str) -> pd.Series:
    return pd.to_numeric(df[column_name], errors="coerce").astype(float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = den.abs() > 1.0e-12
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out.replace([np.inf, -np.inf], np.nan)


def _flag(mask: pd.Series) -> pd.Series:
    return mask.fillna(False).astype(float)


def _resolve_requested_groups(feature_cfg: dict[str, Any] | None) -> tuple[bool, list[str]]:
    raw_cfg = feature_cfg if isinstance(feature_cfg, dict) else {}
    enabled = bool(raw_cfg.get("enabled", False))
    raw_groups = raw_cfg.get("groups", [])
    if not enabled:
        return False, []
    if not raw_groups:
        return True, list(DEFAULT_STAGE2_FEATURE_GROUPS)
    requested: list[str] = []
    for item in raw_groups:
        group = str(item).strip().lower()
        if group and group in GROUP_FEATURE_REQUIREMENTS and group not in requested:
            requested.append(group)
    return True, requested or list(DEFAULT_STAGE2_FEATURE_GROUPS)


def _build_stage2_features_for_df(
    df: pd.DataFrame,
    *,
    groups: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    sources = _resolve_semester_sources(df)
    output = pd.DataFrame(index=df.index)
    if target_column in df.columns:
        output[target_column] = df[target_column].copy()

    created_by_group: dict[str, list[str]] = {group: [] for group in groups}
    skipped_by_group: dict[str, list[str]] = {group: [] for group in groups}

    def can_build(group: str, feature_name: str) -> bool:
        required = GROUP_FEATURE_REQUIREMENTS[group][feature_name]
        ready = all(token in sources for token in required)
        if not ready:
            skipped_by_group[group].append(feature_name)
        return ready

    approved_1 = _numeric(df, sources["approved_1st_sem"]) if "approved_1st_sem" in sources else None
    approved_2 = _numeric(df, sources["approved_2nd_sem"]) if "approved_2nd_sem" in sources else None
    enrolled_1 = _numeric(df, sources["enrolled_1st_sem"]) if "enrolled_1st_sem" in sources else None
    enrolled_2 = _numeric(df, sources["enrolled_2nd_sem"]) if "enrolled_2nd_sem" in sources else None
    evaluations_1 = _numeric(df, sources["evaluations_1st_sem"]) if "evaluations_1st_sem" in sources else None
    evaluations_2 = _numeric(df, sources["evaluations_2nd_sem"]) if "evaluations_2nd_sem" in sources else None
    grade_1 = _numeric(df, sources["grade_1st_sem"]) if "grade_1st_sem" in sources else None
    grade_2 = _numeric(df, sources["grade_2nd_sem"]) if "grade_2nd_sem" in sources else None

    ratio_approved_enrolled_1 = _safe_divide(approved_1, enrolled_1) if approved_1 is not None and enrolled_1 is not None else None
    ratio_approved_enrolled_2 = _safe_divide(approved_2, enrolled_2) if approved_2 is not None and enrolled_2 is not None else None
    ratio_approved_eval_1 = _safe_divide(approved_1, evaluations_1) if approved_1 is not None and evaluations_1 is not None else None
    ratio_approved_eval_2 = _safe_divide(approved_2, evaluations_2) if approved_2 is not None and evaluations_2 is not None else None
    approved_total = (approved_1 + approved_2) if approved_1 is not None and approved_2 is not None else None
    enrolled_total = (enrolled_1 + enrolled_2) if enrolled_1 is not None and enrolled_2 is not None else None
    evaluations_total = (evaluations_1 + evaluations_2) if evaluations_1 is not None and evaluations_2 is not None else None
    progress_ratio_total = _safe_divide(approved_total, enrolled_total) if approved_total is not None and enrolled_total is not None else None

    def add(group: str, feature_name: str, values: pd.Series) -> None:
        output[feature_name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        created_by_group[group].append(feature_name)

    if "grade_drift" in groups:
        if can_build("grade_drift", "stage2_sharp_grade_diff_2nd_minus_1st"):
            grade_diff = grade_2 - grade_1
            add("grade_drift", "stage2_sharp_grade_diff_2nd_minus_1st", grade_diff)
            add("grade_drift", "stage2_sharp_grade_ratio_2nd_over_1st", _safe_divide(grade_2, grade_1))
            add("grade_drift", "stage2_sharp_grade_abs_diff", grade_diff.abs())
            add("grade_drift", "stage2_sharp_grade_improving_flag", _flag(grade_diff > 0.5))
            add("grade_drift", "stage2_sharp_grade_declining_flag", _flag(grade_diff < -0.5))
            add("grade_drift", "stage2_sharp_grade_stable_band_flag", _flag(grade_diff.abs() <= 0.5))

    if "approval_ratio" in groups:
        if can_build("approval_ratio", "stage2_sharp_approved_enrolled_ratio_1st"):
            add("approval_ratio", "stage2_sharp_approved_enrolled_ratio_1st", ratio_approved_enrolled_1)
            add("approval_ratio", "stage2_sharp_approval_completion_gap_1st", enrolled_1 - approved_1)
        if can_build("approval_ratio", "stage2_sharp_approved_enrolled_ratio_2nd"):
            add("approval_ratio", "stage2_sharp_approved_enrolled_ratio_2nd", ratio_approved_enrolled_2)
            add("approval_ratio", "stage2_sharp_approval_completion_gap_2nd", enrolled_2 - approved_2)
        if can_build("approval_ratio", "stage2_sharp_ratio_change_2nd_minus_1st"):
            add("approval_ratio", "stage2_sharp_ratio_change_2nd_minus_1st", ratio_approved_enrolled_2 - ratio_approved_enrolled_1)

    if "eval_mismatch" in groups:
        if can_build("eval_mismatch", "stage2_sharp_eval_minus_approved_1st"):
            add("eval_mismatch", "stage2_sharp_eval_minus_approved_1st", evaluations_1 - approved_1)
            add("eval_mismatch", "stage2_sharp_approved_over_evaluations_1st", ratio_approved_eval_1)
        if can_build("eval_mismatch", "stage2_sharp_eval_minus_approved_2nd"):
            add("eval_mismatch", "stage2_sharp_eval_minus_approved_2nd", evaluations_2 - approved_2)
            add("eval_mismatch", "stage2_sharp_approved_over_evaluations_2nd", ratio_approved_eval_2)
        if can_build("eval_mismatch", "stage2_sharp_grade_vs_approval_efficiency_1st"):
            add("eval_mismatch", "stage2_sharp_grade_vs_approval_efficiency_1st", grade_1 * ratio_approved_enrolled_1)
        if can_build("eval_mismatch", "stage2_sharp_grade_vs_approval_efficiency_2nd"):
            add("eval_mismatch", "stage2_sharp_grade_vs_approval_efficiency_2nd", grade_2 * ratio_approved_enrolled_2)

    if "continuity" in groups:
        if can_build("continuity", "stage2_sharp_enrolled_both_semesters_flag"):
            add("continuity", "stage2_sharp_enrolled_both_semesters_flag", _flag((enrolled_1 > 0) & (enrolled_2 > 0)))
            add(
                "continuity",
                "stage2_sharp_first_to_second_enrollment_retention_flag",
                _flag((enrolled_1 > 0) & (enrolled_2 > 0)),
            )
        if can_build("continuity", "stage2_sharp_approved_both_semesters_flag"):
            add("continuity", "stage2_sharp_approved_both_semesters_flag", _flag((approved_1 > 0) & (approved_2 > 0)))
        if can_build("continuity", "stage2_sharp_evaluated_both_semesters_flag"):
            add("continuity", "stage2_sharp_evaluated_both_semesters_flag", _flag((evaluations_1 > 0) & (evaluations_2 > 0)))
        if can_build("continuity", "stage2_sharp_activity_continuity_score"):
            continuity_score = (
                _flag((enrolled_1 > 0) & (enrolled_2 > 0))
                + _flag((approved_1 > 0) & (approved_2 > 0))
                + _flag((evaluations_1 > 0) & (evaluations_2 > 0))
            ) / 3.0
            add("continuity", "stage2_sharp_activity_continuity_score", continuity_score)
        if can_build("continuity", "stage2_sharp_progression_continuity_gap"):
            add("continuity", "stage2_sharp_progression_continuity_gap", (enrolled_2 - approved_2) - (enrolled_1 - approved_1))
        if can_build("continuity", "stage2_sharp_second_sem_present_flag"):
            add("continuity", "stage2_sharp_second_sem_present_flag", _flag(enrolled_2 > 0))

    if "stability" in groups:
        if can_build("stability", "stage2_sharp_total_semester_grade_mean"):
            grade_diff = grade_2 - grade_1
            add("stability", "stage2_sharp_total_semester_grade_mean", (grade_1 + grade_2) / 2.0)
            add("stability", "stage2_sharp_total_semester_grade_std_proxy", grade_diff.abs())
            add("stability", "stage2_sharp_grade_consistency_band", _flag(grade_diff.abs() <= 1.0))
        if can_build("stability", "stage2_sharp_approval_stability_proxy"):
            approval_diff = (approved_2 - approved_1).abs()
            add("stability", "stage2_sharp_approval_stability_proxy", approval_diff)
            add("stability", "stage2_sharp_approved_consistency_band", _flag(approval_diff <= 1.0))

    if "enrolled_middle_state" in groups:
        if can_build("enrolled_middle_state", "stage2_sharp_second_sem_enrolled_positive_and_approved_low_flag"):
            add(
                "enrolled_middle_state",
                "stage2_sharp_second_sem_enrolled_positive_and_approved_low_flag",
                _flag((enrolled_2 > 0) & (ratio_approved_enrolled_2 < 0.50)),
            )
        if can_build("enrolled_middle_state", "stage2_sharp_moderate_progress_flag"):
            add(
                "enrolled_middle_state",
                "stage2_sharp_moderate_progress_flag",
                _flag((progress_ratio_total >= 0.35) & (progress_ratio_total <= 0.80) & (enrolled_total > 0)),
            )
            add(
                "enrolled_middle_state",
                "stage2_sharp_incomplete_but_active_flag",
                _flag((enrolled_total > approved_total) & (enrolled_2 > 0)),
            )
            add(
                "enrolled_middle_state",
                "stage2_sharp_persistent_low_completion_flag",
                _flag((enrolled_1 > 0) & (enrolled_2 > 0) & (ratio_approved_enrolled_1 < 0.50) & (ratio_approved_enrolled_2 < 0.50)),
            )
        if can_build("enrolled_middle_state", "stage2_sharp_evaluation_active_but_completion_lag_flag"):
            add(
                "enrolled_middle_state",
                "stage2_sharp_evaluation_active_but_completion_lag_flag",
                _flag((evaluations_total > approved_total) & (approved_total < evaluations_total)),
            )

    if "cross_sem_interactions" in groups:
        if can_build("cross_sem_interactions", "stage2_sharp_approved_2nd_minus_enrolled_1st"):
            add("cross_sem_interactions", "stage2_sharp_approved_2nd_minus_enrolled_1st", approved_2 - enrolled_1)
        if can_build("cross_sem_interactions", "stage2_sharp_approved_total_over_enrolled_total"):
            add("cross_sem_interactions", "stage2_sharp_approved_total_over_enrolled_total", progress_ratio_total)
        if can_build("cross_sem_interactions", "stage2_sharp_grade_mean_x_approval_ratio"):
            grade_mean = (grade_1 + grade_2) / 2.0
            add("cross_sem_interactions", "stage2_sharp_grade_mean_x_approval_ratio", grade_mean * progress_ratio_total)
        if can_build("cross_sem_interactions", "stage2_sharp_second_sem_weighted_progress"):
            add("cross_sem_interactions", "stage2_sharp_second_sem_weighted_progress", grade_2 * ratio_approved_enrolled_2)
        if can_build("cross_sem_interactions", "stage2_sharp_first_sem_to_second_sem_progression_ratio"):
            first_sem_progress = _safe_divide(approved_1, enrolled_1)
            second_sem_progress = _safe_divide(approved_2, enrolled_2)
            add(
                "cross_sem_interactions",
                "stage2_sharp_first_sem_to_second_sem_progression_ratio",
                _safe_divide(second_sem_progress, first_sem_progress),
            )

    feature_cols = [col for col in output.columns if col != target_column]
    if feature_cols:
        output[feature_cols] = output[feature_cols].replace([np.inf, -np.inf], np.nan)
    return output, created_by_group, skipped_by_group


def build_stage2_feature_sharpening_split_data(
    split_data: dict[str, pd.DataFrame],
    *,
    target_column: str = "target",
    feature_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    enabled, groups = _resolve_requested_groups(feature_cfg)
    report: dict[str, Any] = {
        "enabled": enabled,
        "requested_groups": groups,
        "default_groups": list(DEFAULT_STAGE2_FEATURE_GROUPS),
        "created_features": [],
        "created_feature_count": 0,
        "created_features_by_group": {},
        "skipped_features_by_group": {},
        "source_columns": {},
    }
    if not enabled:
        empty_splits = {
            split_name: pd.DataFrame({target_column: df[target_column].copy()}) if target_column in df.columns else pd.DataFrame(index=df.index)
            for split_name, df in split_data.items()
        }
        return empty_splits, report

    transformed: dict[str, pd.DataFrame] = {}
    created_by_group_final: dict[str, list[str]] = {}
    skipped_by_group_final: dict[str, list[str]] = {}
    for split_name, df in split_data.items():
        feature_df, created_by_group, skipped_by_group = _build_stage2_features_for_df(
            df,
            groups=groups,
            target_column=target_column,
        )
        transformed[split_name] = feature_df
        if split_name == "train":
            created_by_group_final = {group: list(names) for group, names in created_by_group.items()}
            skipped_by_group_final = {group: list(names) for group, names in skipped_by_group.items()}
            report["source_columns"] = _resolve_semester_sources(df)

    created_features = [
        feature_name
        for group in groups
        for feature_name in created_by_group_final.get(group, [])
    ]
    report["created_features"] = created_features
    report["created_feature_count"] = int(len(created_features))
    report["created_features_by_group"] = created_by_group_final
    report["skipped_features_by_group"] = skipped_by_group_final
    return transformed, report
