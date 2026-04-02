"""Paper-style OULAD feature engineering with enrollment-aware joins."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ENROLLMENT_KEYS = ["id_student", "code_module", "code_presentation"]


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _aggregate_weekly_trend(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ENROLLMENT_KEYS + ["weekly_activity_trend"])

    work = df.copy()
    work["week"] = np.floor_divide(work["date"].astype(float), 7).astype("Int64")
    weekly = (
        work.groupby(ENROLLMENT_KEYS + ["week"], dropna=False)[value_col]
        .sum()
        .reset_index()
    )

    def slope(group: pd.DataFrame) -> float:
        values = group[value_col].to_numpy(dtype=float)
        if values.size < 2 or np.std(values) == 0:
            return 0.0
        x = np.arange(values.size, dtype=float)
        return float(np.polyfit(x, values, 1)[0])

    trend = (
        weekly.groupby(ENROLLMENT_KEYS, dropna=False)
        .apply(slope)
        .reset_index(name="weekly_activity_trend")
    )
    return trend


def _build_demographic_base(student_info: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        c
        for c in ENROLLMENT_KEYS
        + [
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "final_result",
        ]
        if c in student_info.columns
    ]
    return student_info[keep_cols].copy()


def _build_registration_features(student_registration: pd.DataFrame) -> pd.DataFrame:
    reg = _to_numeric(student_registration, ["date_registration", "date_unregistration"])
    agg = (
        reg.groupby(ENROLLMENT_KEYS, dropna=False)
        .agg(
            reg_date=("date_registration", "min"),
            unreg_date=("date_unregistration", "min"),
        )
        .reset_index()
    )
    agg["registered_late_flag"] = (agg["reg_date"] > 0).astype(int)
    agg["unregistered_flag"] = agg["unreg_date"].notna().astype(int)
    return agg


def _build_assessment_features(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    cutoff_day: int | None,
) -> pd.DataFrame:
    ass = _to_numeric(assessments, ["date", "weight"])
    sa = _to_numeric(student_assessment, ["score", "date_submitted"])

    merge_cols = [
        c
        for c in [
            "id_assessment",
            "code_module",
            "code_presentation",
            "assessment_type",
            "weight",
            "date",
        ]
        if c in ass.columns
    ]

    merged = sa.merge(
        ass[merge_cols],
        on="id_assessment",
        how="left",
        suffixes=("", "_assessment"),
    )

    # Restore plain enrollment keys if merge produced suffixed columns
    merged = _ensure_enrollment_keys(merged)

    merged["weight"] = merged.get("weight", 0.0)
    merged["weight"] = merged["weight"].fillna(0.0)
    merged["score"] = merged.get("score", 0.0)
    merged["score"] = merged["score"].fillna(0.0)
    merged["weighted_score"] = merged["score"] * merged["weight"] / 100.0

    if cutoff_day is not None and "date" in merged.columns:
        merged = merged.loc[merged["date"].fillna(cutoff_day + 1) <= cutoff_day].copy()

    perf = (
        merged.groupby(ENROLLMENT_KEYS, dropna=False)
        .agg(
            assessment_score_mean=("score", "mean"),
            assessment_score_std=("score", "std"),
            assessment_score_min=("score", "min"),
            assessment_score_max=("score", "max"),
            assessment_count=("id_assessment", "count"),
            weighted_score_sum=("weighted_score", "sum"),
            assessment_date_mean=("date", "mean"),
        )
        .reset_index()
    )

    if "date_submitted" in merged.columns and "date" in merged.columns:
        merged["is_late"] = (merged["date_submitted"] > merged["date"]).astype(int)
        late = (
            merged.groupby(ENROLLMENT_KEYS, dropna=False)["is_late"]
            .mean()
            .reset_index(name="late_submission_rate")
        )
        perf = perf.merge(late, on=ENROLLMENT_KEYS, how="left")

    if "assessment_type" in merged.columns:
        type_scores = (
            merged.pivot_table(
                index=ENROLLMENT_KEYS,
                columns="assessment_type",
                values="score",
                aggfunc="mean",
            )
            .add_prefix("score_mean_")
            .reset_index()
        )
        perf = perf.merge(type_scores, on=ENROLLMENT_KEYS, how="left")

    return perf


def _ensure_enrollment_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure merged OULAD frames expose plain enrollment key columns:
    - id_student
    - code_module
    - code_presentation
    """
    out = df.copy()

    for key in ["code_module", "code_presentation"]:
        if key not in out.columns:
            left_key = f"{key}_x"
            right_key = f"{key}_y"
            alt_left_key = f"{key}_student_vle"
            alt_right_key = f"{key}_vle"
            alt_assessment_key = f"{key}_assessment"

            if left_key in out.columns and right_key in out.columns:
                out[key] = out[left_key].combine_first(out[right_key])
            elif left_key in out.columns:
                out[key] = out[left_key]
            elif right_key in out.columns:
                out[key] = out[right_key]
            elif alt_left_key in out.columns and alt_right_key in out.columns:
                out[key] = out[alt_left_key].combine_first(out[alt_right_key])
            elif alt_left_key in out.columns:
                out[key] = out[alt_left_key]
            elif alt_right_key in out.columns:
                out[key] = out[alt_right_key]
            elif alt_assessment_key in out.columns:
                out[key] = out[alt_assessment_key]

    return out


def _build_vle_features(
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
    cutoff_day: int | None,
) -> pd.DataFrame:
    """
    Build enrollment-aware VLE interaction features.

    Expected studentVle columns:
    - id_student, code_module, code_presentation, id_site, date, sum_click

    Expected vle columns:
    - id_site, code_module, code_presentation, activity_type, week_from, week_to
    """
    sv = _to_numeric(student_vle, ["date", "sum_click"])
    v = _to_numeric(vle, ["week_from", "week_to"])

    if cutoff_day is not None and "date" in sv.columns:
        sv = sv.loc[sv["date"] <= cutoff_day].copy()

    merge_keys = ["id_site"]
    if "code_module" in sv.columns and "code_module" in v.columns:
        merge_keys.append("code_module")
    if "code_presentation" in sv.columns and "code_presentation" in v.columns:
        merge_keys.append("code_presentation")

    vle_keep_cols = [c for c in ["id_site", "code_module", "code_presentation", "activity_type"] if c in v.columns]

    merged = sv.merge(
        v[vle_keep_cols].drop_duplicates(),
        on=merge_keys,
        how="left",
        suffixes=("_student_vle", "_vle"),
    )

    merged = _ensure_enrollment_keys(merged)

    missing_keys = [k for k in ENROLLMENT_KEYS if k not in merged.columns]
    if missing_keys:
        raise ValueError(
            f"Missing enrollment keys after VLE merge: {missing_keys}. "
            f"Available columns: {merged.columns.tolist()}"
        )

    if "sum_click" not in merged.columns:
        raise ValueError(
            f"'sum_click' missing after VLE merge. "
            f"Available columns: {merged.columns.tolist()}"
        )

    agg = (
        merged.groupby(ENROLLMENT_KEYS, dropna=False)
        .agg(
            vle_clicks_total=("sum_click", "sum"),
            vle_clicks_mean=("sum_click", "mean"),
            vle_clicks_std=("sum_click", "std"),
            vle_clicks_max=("sum_click", "max"),
            vle_interaction_count=("sum_click", "count"),
            vle_first_day=("date", "min"),
            vle_last_day=("date", "max"),
            vle_active_days=("date", "nunique"),
        )
        .reset_index()
    )

    if "activity_type" in merged.columns:
        activity = (
            merged.groupby(ENROLLMENT_KEYS, dropna=False)["activity_type"]
            .nunique()
            .reset_index(name="vle_activity_type_nunique")
        )
        agg = agg.merge(activity, on=ENROLLMENT_KEYS, how="left")

    trend = _aggregate_weekly_trend(merged[ENROLLMENT_KEYS + ["date", "sum_click"]].copy(), "sum_click")
    agg = agg.merge(trend, on=ENROLLMENT_KEYS, how="left")

    return agg


def build_oulad_paper_features(adapted: dict[str, Any] | pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Build OULAD features using transparent, enrollment-aware joins."""
    if isinstance(adapted, pd.DataFrame):
        raise TypeError("OULAD feature builder expects adapted dict with 'tables'.")
    if "tables" not in adapted:
        raise KeyError("OULAD feature builder requires adapted['tables'].")

    tables = adapted["tables"]
    cutoff_day = feature_config.get("cutoff_day")

    student_info = tables["studentinfo"].copy()
    student_registration = tables["studentregistration"].copy()
    student_assessment = tables["studentassessment"].copy()
    assessments = tables["assessments"].copy()
    student_vle = tables["studentvle"].copy()
    vle = tables["vle"].copy()

    base = _build_demographic_base(student_info)
    reg_features = _build_registration_features(student_registration)
    assessment_features = _build_assessment_features(
        student_assessment,
        assessments,
        cutoff_day=cutoff_day,
    )
    vle_features = _build_vle_features(student_vle, vle, cutoff_day=cutoff_day)

    final_df = base.merge(reg_features, on=ENROLLMENT_KEYS, how="left")
    final_df = final_df.merge(assessment_features, on=ENROLLMENT_KEYS, how="left")
    final_df = final_df.merge(vle_features, on=ENROLLMENT_KEYS, how="left")

    if {"assessment_count", "vle_interaction_count"}.issubset(final_df.columns):
        denom = final_df["assessment_count"].replace(0, np.nan)
        final_df["interactions_per_assessment"] = (
            final_df["vle_interaction_count"] / denom
        ).fillna(0.0)

    if {"vle_clicks_total", "vle_active_days"}.issubset(final_df.columns):
        denom = final_df["vle_active_days"].replace(0, np.nan)
        final_df["clicks_per_active_day"] = (
            final_df["vle_clicks_total"] / denom
        ).fillna(0.0)

    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = final_df.select_dtypes(exclude=[np.number]).columns.tolist()

    final_df[numeric_cols] = final_df[numeric_cols].fillna(0.0)
    final_df[categorical_cols] = final_df[categorical_cols].fillna("unknown")

    return final_df
