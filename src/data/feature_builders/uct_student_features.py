"""Feature builder for UCT Student tabular benchmark experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ACADEMIC_COLUMN_ALIASES: dict[str, list[str]] = {
    # Canonical normalized names first, followed by known UCI/UCT normalized variants.
    "approved_1st_sem": ["approved_1st_sem", "curricular_units_1st_sem_approved"],
    "approved_2nd_sem": ["approved_2nd_sem", "curricular_units_2nd_sem_approved"],
    "enrolled_1st_sem": ["enrolled_1st_sem", "curricular_units_1st_sem_enrolled"],
    "enrolled_2nd_sem": ["enrolled_2nd_sem", "curricular_units_2nd_sem_enrolled"],
    "evaluations_1st_sem": ["evaluations_1st_sem", "curricular_units_1st_sem_evaluations"],
    "evaluations_2nd_sem": ["evaluations_2nd_sem", "curricular_units_2nd_sem_evaluations"],
    "without_evaluations_1st_sem": [
        "without_evaluations_1st_sem",
        "curricular_units_1st_sem_without_evaluations",
    ],
    "without_evaluations_2nd_sem": [
        "without_evaluations_2nd_sem",
        "curricular_units_2nd_sem_without_evaluations",
    ],
    "grade_1st_sem": ["grade_1st_sem", "curricular_units_1st_sem_grade"],
    "grade_2nd_sem": ["grade_2nd_sem", "curricular_units_2nd_sem_grade"],
}

DEFAULT_ENROLLED_FEATURE_GROUPS: tuple[str, ...] = (
    "efficiency",
    "gap",
    "trend",
    "consistency",
    "near_graduate",
)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        resolved = lower_map.get(candidate.lower())
        if resolved:
            return resolved
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
    return pd.to_numeric(df[column_name], errors="coerce").fillna(0.0).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0).astype(float)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0).astype(float)
    values = np.where(den > 0, num / den, 0.0)
    return pd.Series(values, index=num.index, dtype=float)


def _series_min(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([left, right], axis=1).min(axis=1)


def _series_max(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.concat([left, right], axis=1).max(axis=1)


def _build_enrolled_feature_groups(
    df: pd.DataFrame,
    semester_sources: dict[str, str],
    enrolled_cfg: dict[str, Any],
) -> None:
    enabled = bool(enrolled_cfg.get("enabled", False))
    if not enabled:
        print("[features][uci][enrolled] enabled=false")
        return

    raw_groups = enrolled_cfg.get("groups", [])
    groups: list[str] = []
    if isinstance(raw_groups, list):
        for item in raw_groups:
            group = str(item).strip().lower()
            if group and group not in groups:
                groups.append(group)
    if not groups:
        groups = list(DEFAULT_ENROLLED_FEATURE_GROUPS)

    eps = float(enrolled_cfg.get("eps", 1.0e-8))
    series: dict[str, pd.Series] = {
        token: _numeric(df, column_name) for token, column_name in semester_sources.items()
    }
    created_features: list[str] = []
    skipped_features: list[str] = []
    built_groups: list[str] = []

    def mark_group(group_name: str) -> None:
        if group_name not in built_groups:
            built_groups.append(group_name)

    def missing(tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in series]

    def add(group_name: str, feature_name: str, build_fn: Any, required: list[str]) -> None:
        missing_tokens = missing(required)
        if missing_tokens:
            skipped_features.append(f"{feature_name}:missing={','.join(missing_tokens)}")
            return
        if feature_name in df.columns:
            skipped_features.append(f"{feature_name}:exists")
            return
        output = build_fn()
        df[feature_name] = pd.to_numeric(output, errors="coerce").fillna(0.0).astype(float)
        series[feature_name] = df[feature_name]
        created_features.append(feature_name)
        mark_group(group_name)

    if "efficiency" in groups:
        add(
            "efficiency",
            "sem1_approval_rate",
            lambda: _safe_ratio(series["approved_1st_sem"], series["enrolled_1st_sem"].clip(lower=eps)),
            ["approved_1st_sem", "enrolled_1st_sem"],
        )
        add(
            "efficiency",
            "sem2_approval_rate",
            lambda: _safe_ratio(series["approved_2nd_sem"], series["enrolled_2nd_sem"].clip(lower=eps)),
            ["approved_2nd_sem", "enrolled_2nd_sem"],
        )
        add(
            "efficiency",
            "overall_approval_rate",
            lambda: _safe_ratio(
                series["approved_1st_sem"] + series["approved_2nd_sem"],
                (series["enrolled_1st_sem"] + series["enrolled_2nd_sem"]).clip(lower=eps),
            ),
            ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
        )
        add(
            "efficiency",
            "sem1_grade_efficiency",
            lambda: _safe_ratio(series["grade_1st_sem"], series["approved_1st_sem"].clip(lower=eps)),
            ["grade_1st_sem", "approved_1st_sem"],
        )
        add(
            "efficiency",
            "sem2_grade_efficiency",
            lambda: _safe_ratio(series["grade_2nd_sem"], series["approved_2nd_sem"].clip(lower=eps)),
            ["grade_2nd_sem", "approved_2nd_sem"],
        )
        add(
            "efficiency",
            "overall_grade_efficiency",
            lambda: _safe_ratio(
                series["grade_1st_sem"] + series["grade_2nd_sem"],
                (series["approved_1st_sem"] + series["approved_2nd_sem"]).clip(lower=eps),
            ),
            ["grade_1st_sem", "grade_2nd_sem", "approved_1st_sem", "approved_2nd_sem"],
        )
        add(
            "efficiency",
            "evaluation_to_approval_rate",
            lambda: _safe_ratio(
                series["approved_1st_sem"] + series["approved_2nd_sem"],
                (series["evaluations_1st_sem"] + series["evaluations_2nd_sem"]).clip(lower=eps),
            ),
            ["approved_1st_sem", "approved_2nd_sem", "evaluations_1st_sem", "evaluations_2nd_sem"],
        )

    if "gap" in groups:
        add("gap", "sem1_gap", lambda: series["enrolled_1st_sem"] - series["approved_1st_sem"], ["enrolled_1st_sem", "approved_1st_sem"])
        add("gap", "sem2_gap", lambda: series["enrolled_2nd_sem"] - series["approved_2nd_sem"], ["enrolled_2nd_sem", "approved_2nd_sem"])
        add(
            "gap",
            "persistence_gap",
            lambda: (series["enrolled_1st_sem"] + series["enrolled_2nd_sem"])
            - (series["approved_1st_sem"] + series["approved_2nd_sem"]),
            ["enrolled_1st_sem", "enrolled_2nd_sem", "approved_1st_sem", "approved_2nd_sem"],
        )
        add(
            "gap",
            "evaluation_gap",
            lambda: (series["evaluations_1st_sem"] + series["evaluations_2nd_sem"])
            - (series["approved_1st_sem"] + series["approved_2nd_sem"]),
            ["evaluations_1st_sem", "evaluations_2nd_sem", "approved_1st_sem", "approved_2nd_sem"],
        )
        add(
            "gap",
            "sem1_unfinished_ratio",
            lambda: _safe_ratio(series["sem1_gap"], series["enrolled_1st_sem"].clip(lower=eps)),
            ["sem1_gap", "enrolled_1st_sem"],
        )
        add(
            "gap",
            "sem2_unfinished_ratio",
            lambda: _safe_ratio(series["sem2_gap"], series["enrolled_2nd_sem"].clip(lower=eps)),
            ["sem2_gap", "enrolled_2nd_sem"],
        )
        add(
            "gap",
            "persistence_gap_ratio",
            lambda: _safe_ratio(
                series["persistence_gap"],
                (series["enrolled_1st_sem"] + series["enrolled_2nd_sem"]).clip(lower=eps),
            ),
            ["persistence_gap", "enrolled_1st_sem", "enrolled_2nd_sem"],
        )

    if "trend" in groups:
        add(
            "trend",
            "approval_rate_delta",
            lambda: series["sem2_approval_rate"] - series["sem1_approval_rate"],
            ["sem2_approval_rate", "sem1_approval_rate"],
        )
        add("trend", "grade_delta", lambda: series["grade_2nd_sem"] - series["grade_1st_sem"], ["grade_2nd_sem", "grade_1st_sem"])
        add(
            "trend",
            "grade_efficiency_delta",
            lambda: series["sem2_grade_efficiency"] - series["sem1_grade_efficiency"],
            ["sem2_grade_efficiency", "sem1_grade_efficiency"],
        )
        add("trend", "load_delta", lambda: series["enrolled_2nd_sem"] - series["enrolled_1st_sem"], ["enrolled_2nd_sem", "enrolled_1st_sem"])
        add("trend", "completion_delta", lambda: series["approved_2nd_sem"] - series["approved_1st_sem"], ["approved_2nd_sem", "approved_1st_sem"])
        add("trend", "gap_delta", lambda: series["sem2_gap"] - series["sem1_gap"], ["sem2_gap", "sem1_gap"])

    if "consistency" in groups:
        add(
            "consistency",
            "approval_consistency",
            lambda: 1.0 - (series["sem2_approval_rate"] - series["sem1_approval_rate"]).abs().clip(upper=1.0),
            ["sem2_approval_rate", "sem1_approval_rate"],
        )
        add(
            "consistency",
            "grade_consistency",
            lambda: 1.0
            - _safe_ratio(
                (series["grade_2nd_sem"] - series["grade_1st_sem"]).abs(),
                (series["grade_1st_sem"].abs() + series["grade_2nd_sem"].abs()).clip(lower=eps),
            ).clip(upper=1.0),
            ["grade_2nd_sem", "grade_1st_sem"],
        )
        add(
            "consistency",
            "workload_balance",
            lambda: _safe_ratio(
                _series_min(series["enrolled_1st_sem"], series["enrolled_2nd_sem"]),
                _series_max(series["enrolled_1st_sem"], series["enrolled_2nd_sem"]).clip(lower=eps),
            ),
            ["enrolled_1st_sem", "enrolled_2nd_sem"],
        )
        add(
            "consistency",
            "completion_balance",
            lambda: _safe_ratio(
                _series_min(series["approved_1st_sem"], series["approved_2nd_sem"]),
                _series_max(series["approved_1st_sem"], series["approved_2nd_sem"]).clip(lower=eps),
            ),
            ["approved_1st_sem", "approved_2nd_sem"],
        )
        add(
            "consistency",
            "gap_balance",
            lambda: _safe_ratio(
                _series_min(series["sem1_gap"].abs(), series["sem2_gap"].abs()),
                _series_max(series["sem1_gap"].abs(), series["sem2_gap"].abs()).clip(lower=eps),
            ),
            ["sem1_gap", "sem2_gap"],
        )

    if "near_graduate" in groups:
        add(
            "near_graduate",
            "sem2_completion_strength",
            lambda: series["sem2_approval_rate"] * series["sem2_grade_efficiency"],
            ["sem2_approval_rate", "sem2_grade_efficiency"],
        )
        add(
            "near_graduate",
            "overall_completion_strength",
            lambda: series["overall_approval_rate"] * series["overall_grade_efficiency"],
            ["overall_approval_rate", "overall_grade_efficiency"],
        )
        add(
            "near_graduate",
            "completion_strength_delta",
            lambda: series["sem2_completion_strength"] - (series["sem1_approval_rate"] * series["sem1_grade_efficiency"]),
            ["sem2_completion_strength", "sem1_approval_rate", "sem1_grade_efficiency"],
        )
        add(
            "near_graduate",
            "near_graduate_gap_signal",
            lambda: series["overall_completion_strength"] - series["persistence_gap_ratio"],
            ["overall_completion_strength", "persistence_gap_ratio"],
        )

    skipped_preview = skipped_features[:8]
    print(
        "[features][uci][enrolled] "
        f"enabled=true groups={groups} built_groups={built_groups or ['none']} "
        f"added_columns={len(created_features)} skipped={skipped_preview}"
    )


def _add_if_sources(
    df: pd.DataFrame,
    source_map: dict[str, str],
    output_name: str,
    required: list[str],
    build_fn: Any,
) -> None:
    if not all(token in source_map for token in required):
        return
    output = build_fn()
    df[output_name] = pd.to_numeric(output, errors="coerce").fillna(0.0).astype(float)


def build_uct_student_features(adapted: dict[str, Any] | pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Create a clean UCT feature table while preserving original columns."""
    if isinstance(adapted, dict):
        if "data" not in adapted:
            raise KeyError("UCT feature builder expects adapted schema with 'data' key.")
        df = adapted["data"].copy()
        id_column = adapted.get("id_column")
        target_column = adapted.get("target_column")
    else:
        df = adapted.copy()
        id_column = feature_config.get("id_column")
        target_column = feature_config.get("target_column")

    drop_columns = set(feature_config.get("drop_columns", []))
    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=existing)

    derive_safe_features = bool(feature_config.get("derive_safe_features", True))
    if derive_safe_features:
        if {"studied_credits", "num_of_prev_attempts"}.issubset(df.columns):
            df["credits_per_previous_attempt"] = _safe_ratio(
                df["studied_credits"].astype(float), df["num_of_prev_attempts"].astype(float).replace(0, 1)
            )
        if {"attendance_rate", "engagement_score"}.issubset(df.columns):
            df["attendance_engagement_interaction"] = (
                pd.to_numeric(df["attendance_rate"], errors="coerce")
                * pd.to_numeric(df["engagement_score"], errors="coerce")
            )

        feature_candidates = [c for c in df.columns if c not in {id_column, target_column}]
        numeric_candidates = [c for c in feature_candidates if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_candidates:
            df["missing_numeric_count"] = df[numeric_candidates].isna().sum(axis=1)

        semester_sources = _resolve_semester_sources(df)

        # A. Approval / completion efficiency
        _add_if_sources(
            df,
            semester_sources,
            "approval_rate_1",
            ["approved_1st_sem", "enrolled_1st_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"]),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approval_rate_2",
            ["approved_2nd_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_2nd_sem"]),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "eval_success_rate_1",
            ["approved_1st_sem", "evaluations_1st_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"]),
                _numeric(df, semester_sources["evaluations_1st_sem"]),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "eval_success_rate_2",
            ["approved_2nd_sem", "evaluations_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["evaluations_2nd_sem"]),
            ),
        )

        # B. Pressure / struggle signals
        _add_if_sources(
            df,
            semester_sources,
            "eval_gap_1",
            ["evaluations_1st_sem", "approved_1st_sem"],
            lambda: _numeric(df, semester_sources["evaluations_1st_sem"])
            - _numeric(df, semester_sources["approved_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "eval_gap_2",
            ["evaluations_2nd_sem", "approved_2nd_sem"],
            lambda: _numeric(df, semester_sources["evaluations_2nd_sem"])
            - _numeric(df, semester_sources["approved_2nd_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "no_eval_ratio_1",
            ["without_evaluations_1st_sem", "enrolled_1st_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["without_evaluations_1st_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"]),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "no_eval_ratio_2",
            ["without_evaluations_2nd_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["without_evaluations_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_2nd_sem"]),
            ),
        )

        # C. Trend / persistence
        _add_if_sources(
            df,
            semester_sources,
            "grade_delta_2_minus_1",
            ["grade_1st_sem", "grade_2nd_sem"],
            lambda: _numeric(df, semester_sources["grade_2nd_sem"]) - _numeric(df, semester_sources["grade_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approved_delta_2_minus_1",
            ["approved_1st_sem", "approved_2nd_sem"],
            lambda: _numeric(df, semester_sources["approved_2nd_sem"])
            - _numeric(df, semester_sources["approved_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "enrolled_delta_2_minus_1",
            ["enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _numeric(df, semester_sources["enrolled_2nd_sem"])
            - _numeric(df, semester_sources["enrolled_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approval_rate_delta",
            ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_2nd_sem"]),
            )
            - _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"]),
            ),
        )

        # D. Stability / consistency
        _add_if_sources(
            df,
            semester_sources,
            "grade_mean_12",
            ["grade_1st_sem", "grade_2nd_sem"],
            lambda: (
                _numeric(df, semester_sources["grade_1st_sem"]) + _numeric(df, semester_sources["grade_2nd_sem"])
            )
            / 2.0,
        )
        _add_if_sources(
            df,
            semester_sources,
            "grade_abs_delta",
            ["grade_1st_sem", "grade_2nd_sem"],
            lambda: (
                _numeric(df, semester_sources["grade_2nd_sem"]) - _numeric(df, semester_sources["grade_1st_sem"])
            ).abs(),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approved_consistency",
            ["approved_1st_sem", "approved_2nd_sem"],
            lambda: _safe_ratio(
                pd.concat(
                    [
                        _numeric(df, semester_sources["approved_1st_sem"]),
                        _numeric(df, semester_sources["approved_2nd_sem"]),
                    ],
                    axis=1,
                ).min(axis=1),
                pd.concat(
                    [
                        _numeric(df, semester_sources["approved_1st_sem"]),
                        _numeric(df, semester_sources["approved_2nd_sem"]),
                    ],
                    axis=1,
                ).max(axis=1),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "enrolled_consistency",
            ["enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                pd.concat(
                    [
                        _numeric(df, semester_sources["enrolled_1st_sem"]),
                        _numeric(df, semester_sources["enrolled_2nd_sem"]),
                    ],
                    axis=1,
                ).min(axis=1),
                pd.concat(
                    [
                        _numeric(df, semester_sources["enrolled_1st_sem"]),
                        _numeric(df, semester_sources["enrolled_2nd_sem"]),
                    ],
                    axis=1,
                ).max(axis=1),
            ),
        )

        # E. Commitment ratios
        _add_if_sources(
            df,
            semester_sources,
            "approved_to_enrolled_total",
            ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"])
                + _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"])
                + _numeric(df, semester_sources["enrolled_2nd_sem"]),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approved_to_evaluated_total",
            ["approved_1st_sem", "approved_2nd_sem", "evaluations_1st_sem", "evaluations_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"])
                + _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["evaluations_1st_sem"])
                + _numeric(df, semester_sources["evaluations_2nd_sem"]),
            ),
        )

        enrolled_feature_groups_cfg = (
            feature_config.get("enrolled_feature_groups", {})
            if isinstance(feature_config.get("enrolled_feature_groups", {}), dict)
            else {}
        )
        _build_enrolled_feature_groups(df, semester_sources, enrolled_feature_groups_cfg)

    derive_enrolled_focus_features = bool(feature_config.get("derive_enrolled_focus_features", False))
    if derive_enrolled_focus_features:
        semester_sources = _resolve_semester_sources(df)

        approved_1 = (
            _numeric(df, semester_sources["approved_1st_sem"])
            if "approved_1st_sem" in semester_sources
            else None
        )
        approved_2 = (
            _numeric(df, semester_sources["approved_2nd_sem"])
            if "approved_2nd_sem" in semester_sources
            else None
        )
        enrolled_1 = (
            _numeric(df, semester_sources["enrolled_1st_sem"])
            if "enrolled_1st_sem" in semester_sources
            else None
        )
        enrolled_2 = (
            _numeric(df, semester_sources["enrolled_2nd_sem"])
            if "enrolled_2nd_sem" in semester_sources
            else None
        )
        evaluations_1 = (
            _numeric(df, semester_sources["evaluations_1st_sem"])
            if "evaluations_1st_sem" in semester_sources
            else None
        )
        evaluations_2 = (
            _numeric(df, semester_sources["evaluations_2nd_sem"])
            if "evaluations_2nd_sem" in semester_sources
            else None
        )
        grade_1 = (
            _numeric(df, semester_sources["grade_1st_sem"])
            if "grade_1st_sem" in semester_sources
            else None
        )
        grade_2 = (
            _numeric(df, semester_sources["grade_2nd_sem"])
            if "grade_2nd_sem" in semester_sources
            else None
        )

        _add_if_sources(
            df,
            semester_sources,
            "semester_approved_ratio_1st_sem",
            ["approved_1st_sem", "enrolled_1st_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"]).clip(lower=1.0),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "semester_approved_ratio_2nd_sem",
            ["approved_2nd_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_2nd_sem"]).clip(lower=1.0),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approved_ratio_sem1",
            ["approved_1st_sem", "enrolled_1st_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"]).clip(lower=1.0),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approved_ratio_sem2",
            ["approved_2nd_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_2nd_sem"]).clip(lower=1.0),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "semester_grade_delta",
            ["grade_1st_sem", "grade_2nd_sem"],
            lambda: _numeric(df, semester_sources["grade_2nd_sem"]) - _numeric(df, semester_sources["grade_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "semester_approved_delta",
            ["approved_1st_sem", "approved_2nd_sem"],
            lambda: _numeric(df, semester_sources["approved_2nd_sem"])
            - _numeric(df, semester_sources["approved_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "grade_delta_sem2_minus_sem1",
            ["grade_1st_sem", "grade_2nd_sem"],
            lambda: _numeric(df, semester_sources["grade_2nd_sem"]) - _numeric(df, semester_sources["grade_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "approved_delta_sem2_minus_sem1",
            ["approved_1st_sem", "approved_2nd_sem"],
            lambda: _numeric(df, semester_sources["approved_2nd_sem"])
            - _numeric(df, semester_sources["approved_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "enrolled_delta_sem2_minus_sem1",
            ["enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _numeric(df, semester_sources["enrolled_2nd_sem"])
            - _numeric(df, semester_sources["enrolled_1st_sem"]),
        )
        _add_if_sources(
            df,
            semester_sources,
            "evaluation_to_approved_gap",
            ["approved_1st_sem", "approved_2nd_sem", "evaluations_1st_sem", "evaluations_2nd_sem"],
            lambda: (
                _numeric(df, semester_sources["evaluations_1st_sem"])
                + _numeric(df, semester_sources["evaluations_2nd_sem"])
                - _numeric(df, semester_sources["approved_1st_sem"])
                - _numeric(df, semester_sources["approved_2nd_sem"])
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "evaluation_gap_total",
            ["approved_1st_sem", "approved_2nd_sem", "evaluations_1st_sem", "evaluations_2nd_sem"],
            lambda: (
                _numeric(df, semester_sources["evaluations_1st_sem"])
                + _numeric(df, semester_sources["evaluations_2nd_sem"])
                - _numeric(df, semester_sources["approved_1st_sem"])
                - _numeric(df, semester_sources["approved_2nd_sem"])
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "overall_progress_consistency",
            ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                pd.concat(
                    [
                        _safe_ratio(
                            _numeric(df, semester_sources["approved_1st_sem"]),
                            _numeric(df, semester_sources["enrolled_1st_sem"]).clip(lower=1.0),
                        ),
                        _safe_ratio(
                            _numeric(df, semester_sources["approved_2nd_sem"]),
                            _numeric(df, semester_sources["enrolled_2nd_sem"]).clip(lower=1.0),
                        ),
                    ],
                    axis=1,
                ).min(axis=1),
                pd.concat(
                    [
                        _safe_ratio(
                            _numeric(df, semester_sources["approved_1st_sem"]),
                            _numeric(df, semester_sources["enrolled_1st_sem"]).clip(lower=1.0),
                        ),
                        _safe_ratio(
                            _numeric(df, semester_sources["approved_2nd_sem"]),
                            _numeric(df, semester_sources["enrolled_2nd_sem"]).clip(lower=1.0),
                        ),
                    ],
                    axis=1,
                ).max(axis=1).clip(lower=1.0e-12),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "progress_consistency_ratio",
            ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_1st_sem"])
                + _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"])
                + _numeric(df, semester_sources["enrolled_2nd_sem"]),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "semester_load_retention_ratio",
            ["enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["enrolled_2nd_sem"]),
                _numeric(df, semester_sources["enrolled_1st_sem"]).clip(lower=1.0),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "semester_approval_retention_ratio",
            ["approved_1st_sem", "approved_2nd_sem"],
            lambda: _safe_ratio(
                _numeric(df, semester_sources["approved_2nd_sem"]),
                _numeric(df, semester_sources["approved_1st_sem"]).clip(lower=1.0),
            ),
        )
        _add_if_sources(
            df,
            semester_sources,
            "stability_gap_ratio",
            ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
            lambda: _safe_ratio(
                (
                    _numeric(df, semester_sources["enrolled_1st_sem"])
                    + _numeric(df, semester_sources["enrolled_2nd_sem"])
                    - _numeric(df, semester_sources["approved_1st_sem"])
                    - _numeric(df, semester_sources["approved_2nd_sem"])
                ).clip(lower=0.0),
                (
                    _numeric(df, semester_sources["enrolled_1st_sem"])
                    + _numeric(df, semester_sources["enrolled_2nd_sem"])
                ).clip(lower=1.0),
            ),
        )

        if grade_1 is not None and grade_2 is not None:
            df["sem1_to_sem2_grade_stability"] = (1.0 / (1.0 + (grade_2 - grade_1).abs())).astype(float)

        if approved_1 is not None and approved_2 is not None:
            df["sem1_to_sem2_approval_stability"] = _safe_ratio(
                pd.concat([approved_1, approved_2], axis=1).min(axis=1),
                pd.concat([approved_1, approved_2], axis=1).max(axis=1).clip(lower=1.0),
            )

        if (
            approved_1 is not None
            and approved_2 is not None
            and enrolled_1 is not None
            and enrolled_2 is not None
        ):
            approved_total = approved_1 + approved_2
            enrolled_total = enrolled_1 + enrolled_2
            df["low_progress_but_active"] = (
                ((enrolled_total > 0.0) & (_safe_ratio(approved_total, enrolled_total.clip(lower=1.0)) < 0.5))
                .astype(float)
            )

        if approved_1 is not None and approved_2 is not None:
            approval_delta = approved_2 - approved_1
            df["academic_momentum"] = np.where(
                approval_delta > 0.0,
                1.0,
                np.where(approval_delta < 0.0, -1.0, 0.0),
            ).astype(float)

    # Keep native ordering: id/target first, then feature columns.
    ordered_cols: list[str] = []
    for col in [id_column, target_column]:
        if col and col in df.columns:
            ordered_cols.append(col)
    ordered_cols.extend([c for c in df.columns if c not in ordered_cols])
    return df[ordered_cols]
