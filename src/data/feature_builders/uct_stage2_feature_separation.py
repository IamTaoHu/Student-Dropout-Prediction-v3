"""Targeted Stage 2 feature separation for enrolled vs graduate discrimination."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_builders.uct_stage2_feature_sharpening import (
    _numeric,
    _resolve_semester_sources,
    safe_divide,
)


DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS = [
    "near_graduate_incomplete",
    "effort_outcome_mismatch",
    "temporal_inconsistency",
    "closure_discriminators",
]

FEATURE_GROUP_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "near_graduate_incomplete": {
        "high_grade_low_completion_sem1": ["sem1_grade_efficiency", "sem1_approval_rate"],
        "high_grade_low_completion_sem2": ["sem2_grade_efficiency", "sem2_approval_rate"],
        "strong_sem2_unfinished": ["sem2_grade_efficiency", "completion_balance"],
        "strong_current_term_unresolved_gap": [
            "sem1_grade_efficiency",
            "sem2_grade_efficiency",
            "approval_rate_delta",
        ],
        "near_closure_without_conversion": [
            "sem1_approval_rate",
            "sem2_approval_rate",
            "approved_consistency",
        ],
    },
    "effort_outcome_mismatch": {
        "sem1_effort_outcome_gap": ["load_pressure_sem1", "sem1_approval_rate"],
        "sem2_effort_outcome_gap": ["load_pressure_sem2", "sem2_approval_rate"],
        "enrolled_persistence_without_completion": ["persistence_gap", "completion_balance"],
        "evaluation_success_mismatch": [
            "load_pressure_sem1",
            "load_pressure_sem2",
            "sem1_approval_rate",
            "sem2_approval_rate",
        ],
        "grade_completion_tension": [
            "sem1_grade_efficiency",
            "sem2_grade_efficiency",
            "completion_balance",
        ],
    },
    "temporal_inconsistency": {
        "academic_progress_instability": ["grade_delta", "approval_rate_delta"],
        "improving_but_unfinished": ["grade_delta", "completion_balance"],
        "improving_but_low_conversion": ["grade_delta", "sem2_approval_rate"],
        "stable_but_not_closing": ["enrolled_consistency", "completion_balance"],
        "approval_grade_divergence": ["approval_rate_delta", "normalized_grade_delta"],
    },
    "closure_discriminators": {
        "closure_momentum": ["sem1_approval_rate", "sem2_approval_rate"],
        "closure_strength": ["sem2_approval_rate", "sem2_grade_efficiency"],
        "completion_readiness_gap": ["sem2_grade_efficiency", "completion_balance"],
        "graduate_like_resolution_score": [
            "sem2_approval_rate",
            "sem2_grade_efficiency",
            "completion_balance",
            "approved_consistency",
            "persistence_gap",
            "approval_rate_delta",
        ],
        "enrolled_like_unresolved_score": [
            "high_grade_low_completion_sem2",
            "sem2_effort_outcome_gap",
            "stable_but_not_closing",
            "improving_but_unfinished",
        ],
    },
}


def _safe_mean(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        raise ValueError("_safe_mean requires at least one series.")
    frame = pd.concat([pd.to_numeric(s, errors="coerce").astype(float) for s in series_list], axis=1)
    return frame.mean(axis=1).astype(float)


def _safe_clip(values: pd.Series, *, lower: float | None = None, upper: float | None = None) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").astype(float)
    if lower is not None or upper is not None:
        out = out.clip(lower=lower, upper=upper)
    return out


def _sanitize_numeric(values: pd.Series, default: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(float(default))
        .astype(float)
    )


def _resolve_requested_groups(feature_cfg: dict[str, Any] | None) -> tuple[bool, list[str], bool, bool]:
    raw_cfg = feature_cfg if isinstance(feature_cfg, dict) else {}
    enabled = bool(raw_cfg.get("enabled", False))
    strict_mode = bool(raw_cfg.get("strict_mode", False))
    create_composite_scores = bool(raw_cfg.get("create_composite_scores", True))
    if not enabled:
        return False, [], strict_mode, create_composite_scores
    groups: list[str] = []
    raw_groups = raw_cfg.get("feature_groups", raw_cfg.get("groups", []))
    if isinstance(raw_groups, list):
        for item in raw_groups:
            group = str(item).strip().lower()
            if group and group in FEATURE_GROUP_REQUIREMENTS and group not in groups:
                groups.append(group)
    return True, groups or list(DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS), strict_mode, create_composite_scores


def _resolve_existing_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return _sanitize_numeric(df[found])
    return None


def _build_signal_library(df: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, str], list[str]]:
    signals: dict[str, pd.Series] = {}
    sources = _resolve_semester_sources(df)
    source_notes: list[str] = []

    def use_existing(signal_name: str, candidates: list[str]) -> bool:
        series = _resolve_existing_series(df, candidates)
        if series is None:
            return False
        signals[signal_name] = series
        source_notes.append(f"{signal_name}:existing")
        return True

    def add(signal_name: str, values: pd.Series, note: str) -> None:
        signals[signal_name] = _sanitize_numeric(values)
        source_notes.append(f"{signal_name}:{note}")

    approved_1 = _numeric(df, sources["approved_1st_sem"]) if "approved_1st_sem" in sources else None
    approved_2 = _numeric(df, sources["approved_2nd_sem"]) if "approved_2nd_sem" in sources else None
    enrolled_1 = _numeric(df, sources["enrolled_1st_sem"]) if "enrolled_1st_sem" in sources else None
    enrolled_2 = _numeric(df, sources["enrolled_2nd_sem"]) if "enrolled_2nd_sem" in sources else None
    evaluations_1 = _numeric(df, sources["evaluations_1st_sem"]) if "evaluations_1st_sem" in sources else None
    evaluations_2 = _numeric(df, sources["evaluations_2nd_sem"]) if "evaluations_2nd_sem" in sources else None
    grade_1 = _numeric(df, sources["grade_1st_sem"]) if "grade_1st_sem" in sources else None
    grade_2 = _numeric(df, sources["grade_2nd_sem"]) if "grade_2nd_sem" in sources else None

    if not use_existing("sem1_approval_rate", ["sem1_approval_rate", "approval_rate_1", "approved_ratio_sem1", "semester_approved_ratio_1st_sem"]) and approved_1 is not None and enrolled_1 is not None:
        add("sem1_approval_rate", safe_divide(approved_1, enrolled_1, default=0.0), "derived")
    if not use_existing("sem2_approval_rate", ["sem2_approval_rate", "approval_rate_2", "approved_ratio_sem2", "semester_approved_ratio_2nd_sem"]) and approved_2 is not None and enrolled_2 is not None:
        add("sem2_approval_rate", safe_divide(approved_2, enrolled_2, default=0.0), "derived")

    if not use_existing("sem1_grade_efficiency", ["sem1_grade_efficiency"]) and grade_1 is not None and approved_1 is not None:
        add("sem1_grade_efficiency", safe_divide(grade_1, approved_1, default=0.0), "derived")
    if not use_existing("sem2_grade_efficiency", ["sem2_grade_efficiency"]) and grade_2 is not None and approved_2 is not None:
        add("sem2_grade_efficiency", safe_divide(grade_2, approved_2, default=0.0), "derived")

    if not use_existing("approval_rate_delta", ["approval_rate_delta"]) and "sem1_approval_rate" in signals and "sem2_approval_rate" in signals:
        add("approval_rate_delta", signals["sem2_approval_rate"] - signals["sem1_approval_rate"], "derived")
    if not use_existing("grade_delta", ["grade_delta", "grade_delta_2_minus_1", "semester_grade_delta", "grade_delta_sem2_minus_sem1"]) and grade_1 is not None and grade_2 is not None:
        add("grade_delta", grade_2 - grade_1, "derived")

    if not use_existing("persistence_gap", ["persistence_gap"]) and approved_1 is not None and approved_2 is not None and enrolled_1 is not None and enrolled_2 is not None:
        add("persistence_gap", (enrolled_1 + enrolled_2) - (approved_1 + approved_2), "derived")

    if not use_existing("completion_balance", ["completion_balance"]) and approved_1 is not None and approved_2 is not None:
        add(
            "completion_balance",
            safe_divide(pd.concat([approved_1, approved_2], axis=1).min(axis=1), pd.concat([approved_1, approved_2], axis=1).max(axis=1), default=0.0),
            "derived",
        )

    if not use_existing("approved_consistency", ["approved_consistency"]) and approved_1 is not None and approved_2 is not None:
        add(
            "approved_consistency",
            safe_divide(pd.concat([approved_1, approved_2], axis=1).min(axis=1), pd.concat([approved_1, approved_2], axis=1).max(axis=1), default=0.0),
            "derived",
        )
    if not use_existing("enrolled_consistency", ["enrolled_consistency"]) and enrolled_1 is not None and enrolled_2 is not None:
        add(
            "enrolled_consistency",
            safe_divide(pd.concat([enrolled_1, enrolled_2], axis=1).min(axis=1), pd.concat([enrolled_1, enrolled_2], axis=1).max(axis=1), default=0.0),
            "derived",
        )

    if not use_existing("load_pressure_sem1", ["load_pressure_sem1", "sem1_gap"]) and enrolled_1 is not None and approved_1 is not None:
        add("load_pressure_sem1", (enrolled_1 - approved_1).clip(lower=0.0), "derived")
    if not use_existing("load_pressure_sem2", ["load_pressure_sem2", "sem2_gap"]) and enrolled_2 is not None and approved_2 is not None:
        add("load_pressure_sem2", (enrolled_2 - approved_2).clip(lower=0.0), "derived")

    if "grade_delta" in signals and not use_existing("normalized_grade_delta", ["normalized_grade_delta"]):
        denom = None
        if grade_1 is not None and grade_2 is not None:
            denom = (grade_1.abs() + grade_2.abs()).clip(lower=1.0)
        elif "sem1_grade_efficiency" in signals and "sem2_grade_efficiency" in signals:
            denom = (signals["sem1_grade_efficiency"].abs() + signals["sem2_grade_efficiency"].abs()).clip(lower=1.0)
        if denom is not None:
            add("normalized_grade_delta", safe_divide(signals["grade_delta"], denom, default=0.0), "derived")

    if not use_existing("evaluation_success_rate", ["evaluation_success_rate", "evaluation_to_approval_rate"]) and approved_1 is not None and approved_2 is not None and evaluations_1 is not None and evaluations_2 is not None:
        add(
            "evaluation_success_rate",
            safe_divide(approved_1 + approved_2, evaluations_1 + evaluations_2, default=0.0),
            "derived",
        )

    return signals, sources, source_notes


def _build_stage2_feature_separation_for_df(
    df: pd.DataFrame,
    *,
    groups: list[str],
    strict_mode: bool,
    create_composite_scores: bool,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = pd.DataFrame(index=df.index)
    if target_column in df.columns:
        output[target_column] = df[target_column].copy()

    signals, source_columns, source_notes = _build_signal_library(df)
    created_features: list[str] = []
    missing_base_columns: list[str] = []
    skipped_features: list[str] = []
    skipped_existing_features: list[str] = []
    created_by_group: dict[str, list[str]] = {group: [] for group in groups}
    skipped_by_group: dict[str, list[str]] = {group: [] for group in groups}

    def require(feature_name: str, group_name: str, required: list[str]) -> bool:
        missing = [token for token in required if token not in signals]
        if not missing:
            return True
        skipped_by_group[group_name].append(feature_name)
        skipped_features.append(f"{feature_name}:missing={','.join(missing)}")
        for token in missing:
            if token not in missing_base_columns:
                missing_base_columns.append(token)
        if strict_mode:
            raise ValueError(
                f"[two_stage][stage2][feature_separation] strict_mode=true missing base signals for {feature_name}: {missing}"
            )
        return False

    def add(group_name: str, feature_name: str, values: pd.Series) -> None:
        if feature_name in df.columns or feature_name in output.columns:
            existing_series = _resolve_existing_series(df, [feature_name])
            if existing_series is not None:
                signals[feature_name] = existing_series
            skipped_existing_features.append(feature_name)
            skipped_by_group[group_name].append(f"{feature_name}:exists")
            return
        sanitized = _sanitize_numeric(values)
        output[feature_name] = sanitized
        signals[feature_name] = sanitized
        created_features.append(feature_name)
        created_by_group[group_name].append(feature_name)

    for group_name in groups:
        if group_name == "near_graduate_incomplete":
            if require("high_grade_low_completion_sem1", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["high_grade_low_completion_sem1"]):
                add(group_name, "high_grade_low_completion_sem1", signals["sem1_grade_efficiency"] * (1.0 - signals["sem1_approval_rate"]).clip(lower=0.0))
            if require("high_grade_low_completion_sem2", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["high_grade_low_completion_sem2"]):
                add(group_name, "high_grade_low_completion_sem2", signals["sem2_grade_efficiency"] * (1.0 - signals["sem2_approval_rate"]).clip(lower=0.0))
            if require("strong_sem2_unfinished", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["strong_sem2_unfinished"]):
                add(group_name, "strong_sem2_unfinished", signals["sem2_grade_efficiency"] * (1.0 - signals["completion_balance"]).clip(lower=0.0))
            if require("strong_current_term_unresolved_gap", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["strong_current_term_unresolved_gap"]):
                add(
                    group_name,
                    "strong_current_term_unresolved_gap",
                    pd.concat([signals["sem1_grade_efficiency"], signals["sem2_grade_efficiency"]], axis=1).max(axis=1) * signals["approval_rate_delta"].abs(),
                )
            if require("near_closure_without_conversion", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["near_closure_without_conversion"]):
                add(
                    group_name,
                    "near_closure_without_conversion",
                    _safe_mean(signals["sem1_approval_rate"], signals["sem2_approval_rate"]) * (1.0 - signals["approved_consistency"]).clip(lower=0.0),
                )

        if group_name == "effort_outcome_mismatch":
            if require("sem1_effort_outcome_gap", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["sem1_effort_outcome_gap"]):
                add(group_name, "sem1_effort_outcome_gap", signals["load_pressure_sem1"] * (1.0 - signals["sem1_approval_rate"]).clip(lower=0.0))
            if require("sem2_effort_outcome_gap", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["sem2_effort_outcome_gap"]):
                add(group_name, "sem2_effort_outcome_gap", signals["load_pressure_sem2"] * (1.0 - signals["sem2_approval_rate"]).clip(lower=0.0))
            if require("enrolled_persistence_without_completion", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["enrolled_persistence_without_completion"]):
                add(group_name, "enrolled_persistence_without_completion", signals["persistence_gap"] * (1.0 - signals["completion_balance"]).clip(lower=0.0))
            if require("evaluation_success_mismatch", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["evaluation_success_mismatch"]):
                add(
                    group_name,
                    "evaluation_success_mismatch",
                    _safe_mean(signals["load_pressure_sem1"], signals["load_pressure_sem2"]) * (1.0 - _safe_mean(signals["sem1_approval_rate"], signals["sem2_approval_rate"])).clip(lower=0.0),
                )
            if require("grade_completion_tension", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["grade_completion_tension"]):
                add(group_name, "grade_completion_tension", _safe_mean(signals["sem1_grade_efficiency"], signals["sem2_grade_efficiency"]) - signals["completion_balance"])

        if group_name == "temporal_inconsistency":
            if require("academic_progress_instability", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["academic_progress_instability"]):
                add(group_name, "academic_progress_instability", signals["grade_delta"].abs() + signals["approval_rate_delta"].abs())
            if require("improving_but_unfinished", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["improving_but_unfinished"]):
                add(group_name, "improving_but_unfinished", signals["grade_delta"].clip(lower=0.0) * (1.0 - signals["completion_balance"]).clip(lower=0.0))
            if require("improving_but_low_conversion", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["improving_but_low_conversion"]):
                add(group_name, "improving_but_low_conversion", signals["grade_delta"].clip(lower=0.0) * (1.0 - signals["sem2_approval_rate"]).clip(lower=0.0))
            if require("stable_but_not_closing", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["stable_but_not_closing"]):
                add(group_name, "stable_but_not_closing", signals["enrolled_consistency"] * (1.0 - signals["completion_balance"]).clip(lower=0.0))
            if require("approval_grade_divergence", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["approval_grade_divergence"]):
                add(group_name, "approval_grade_divergence", (signals["approval_rate_delta"] - signals["normalized_grade_delta"]).abs())

        if group_name == "closure_discriminators":
            if require("closure_momentum", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["closure_momentum"]):
                add(group_name, "closure_momentum", signals["sem2_approval_rate"] - signals["sem1_approval_rate"])
            if require("closure_strength", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["closure_strength"]):
                add(group_name, "closure_strength", signals["sem2_approval_rate"] * signals["sem2_grade_efficiency"])
            if require("completion_readiness_gap", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["completion_readiness_gap"]):
                add(group_name, "completion_readiness_gap", signals["sem2_grade_efficiency"] - signals["completion_balance"])
            if create_composite_scores and require("graduate_like_resolution_score", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["graduate_like_resolution_score"]):
                add(
                    group_name,
                    "graduate_like_resolution_score",
                    _safe_mean(
                        signals["sem2_approval_rate"],
                        signals["sem2_grade_efficiency"],
                        signals["completion_balance"],
                        signals["approved_consistency"],
                    ) - _safe_mean(signals["persistence_gap"], signals["approval_rate_delta"].abs()),
                )
            if create_composite_scores and require("enrolled_like_unresolved_score", group_name, FEATURE_GROUP_REQUIREMENTS[group_name]["enrolled_like_unresolved_score"]):
                add(
                    group_name,
                    "enrolled_like_unresolved_score",
                    _safe_mean(
                        output["high_grade_low_completion_sem2"],
                        output["sem2_effort_outcome_gap"],
                        output["stable_but_not_closing"],
                        output["improving_but_unfinished"],
                    ),
                )

    feature_cols = [col for col in output.columns if col != target_column]
    if feature_cols:
        output[feature_cols] = output[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    report = {
        "enabled": True,
        "strict_mode": bool(strict_mode),
        "create_composite_scores": bool(create_composite_scores),
        "created_features": created_features,
        "created_feature_count": int(len(created_features)),
        "created_features_by_group": created_by_group,
        "skipped_features_by_group": skipped_by_group,
        "missing_base_columns": sorted(missing_base_columns),
        "skipped_features": skipped_features,
        "skipped_existing_features": skipped_existing_features,
        "source_columns": source_columns,
        "signal_sources": source_notes,
    }
    return output, report


def build_advanced_enrolled_feature_separation_split_data(
    split_data: dict[str, pd.DataFrame],
    *,
    target_column: str = "target",
    feature_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    enabled, groups, strict_mode, create_composite_scores = _resolve_requested_groups(feature_cfg)
    report: dict[str, Any] = {
        "enabled": enabled,
        "strict_mode": strict_mode,
        "feature_groups": groups,
        "default_feature_groups": list(DEFAULT_ADVANCED_ENROLLED_FEATURE_SEPARATION_GROUPS),
        "create_composite_scores": create_composite_scores,
        "apply_only_to_stage2": bool((feature_cfg or {}).get("apply_only_to_stage2", True)) if isinstance(feature_cfg, dict) else True,
        "created_features": [],
        "created_feature_count": 0,
        "created_features_by_group": {},
        "skipped_features_by_group": {},
        "missing_base_columns": [],
        "skipped_features": [],
        "skipped_existing_features": [],
        "source_columns": {},
        "signal_sources": [],
    }
    if not enabled:
        empty = {
            split_name: pd.DataFrame({target_column: df[target_column].copy()}) if target_column in df.columns else pd.DataFrame(index=df.index)
            for split_name, df in split_data.items()
        }
        return empty, report

    transformed: dict[str, pd.DataFrame] = {}
    for split_name, df in split_data.items():
        feature_df, split_report = _build_stage2_feature_separation_for_df(
            df,
            groups=groups,
            strict_mode=strict_mode,
            create_composite_scores=create_composite_scores,
            target_column=target_column,
        )
        transformed[split_name] = feature_df
        if split_name == "train":
            report.update(split_report)
            report["feature_groups"] = groups
            report["apply_only_to_stage2"] = bool((feature_cfg or {}).get("apply_only_to_stage2", True)) if isinstance(feature_cfg, dict) else True
    return transformed, report
