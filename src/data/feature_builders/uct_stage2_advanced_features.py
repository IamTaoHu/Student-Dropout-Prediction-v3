"""Advanced Stage 2 feature helpers for enrolled vs graduate separation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.feature_builders.uct_stage2_feature_sharpening import (
    _flag,
    _numeric,
    _resolve_semester_sources,
    _safe_divide,
    safe_divide,
)


DEFAULT_INTERACTION_GROUPS = [
    "multiplicative",
    "difference_conditioned",
    "ratio_cross",
    "flag_conditioned",
    "limited_polynomial",
]

DEFAULT_PROTOTYPE_METRIC_SET = [
    "l1",
    "l2",
    "margin",
    "inverse_similarity",
]

DEFAULT_SELECTIVE_INTERACTION_ALLOWLIST = [
    "sem1_approval_rate",
    "sem2_approval_rate",
    "sem1_grade_efficiency",
    "sem2_grade_efficiency",
    "approval_rate_delta",
    "grade_delta",
    "persistence_gap",
    "load_pressure_sem1",
    "load_pressure_sem2",
    "completion_balance",
]

SELECTIVE_INTERACTION_REQUIREMENTS: dict[str, list[str]] = {
    "sem1_approval_rate": ["approved_1st_sem", "enrolled_1st_sem"],
    "sem2_approval_rate": ["approved_2nd_sem", "enrolled_2nd_sem"],
    "sem1_grade_efficiency": ["grade_1st_sem", "approved_1st_sem", "enrolled_1st_sem"],
    "sem2_grade_efficiency": ["grade_2nd_sem", "approved_2nd_sem", "enrolled_2nd_sem"],
    "approval_rate_delta": ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
    "grade_delta": ["grade_1st_sem", "grade_2nd_sem"],
    "persistence_gap": ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
    "load_pressure_sem1": ["approved_1st_sem", "enrolled_1st_sem"],
    "load_pressure_sem2": ["approved_2nd_sem", "enrolled_2nd_sem"],
    "completion_balance": ["approved_1st_sem", "approved_2nd_sem", "enrolled_1st_sem", "enrolled_2nd_sem"],
}

INTERACTION_FEATURE_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "multiplicative": {
        "stage2_interaction_grade_mean_x_approval_ratio_total": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_grade_diff_x_completion_gap_total": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_continuity_x_approval_ratio_2nd": [
            "approved_2nd_sem",
            "enrolled_2nd_sem",
            "enrolled_1st_sem",
            "approved_1st_sem",
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
        ],
        "stage2_interaction_eval_mismatch_x_grade_decline_flag": [
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
            "grade_1st_sem",
            "grade_2nd_sem",
        ],
        "stage2_interaction_activity_continuity_x_incomplete_active_flag": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
        ],
    },
    "difference_conditioned": {
        "stage2_interaction_grade_diff_weighted_by_approved_total": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
        ],
        "stage2_interaction_ratio_change_weighted_by_enrolled_total": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_approval_gap_2nd_weighted_by_second_sem_presence": [
            "approved_2nd_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_drift_x_second_sem_active_flag": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "enrolled_2nd_sem",
        ],
    },
    "ratio_cross": {
        "stage2_interaction_approved_ratio_1st_over_2nd": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_approval_efficiency_2nd_over_1st": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
        ],
        "stage2_interaction_evaluation_pressure_over_completion_ratio": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "evaluations_1st_sem",
            "evaluations_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_grade_ratio_x_approved_ratio_change": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
    },
    "flag_conditioned": {
        "stage2_interaction_second_sem_grade_if_active": [
            "grade_2nd_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_completion_gap_if_persistent": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_grade_decline_if_enrolled_both_semesters": [
            "grade_1st_sem",
            "grade_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_approval_ratio_if_incomplete_but_active": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
    },
    "limited_polynomial": {
        "stage2_interaction_grade_diff_sq": ["grade_1st_sem", "grade_2nd_sem"],
        "stage2_interaction_completion_gap_sq": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_approval_ratio_delta_abs": [
            "approved_1st_sem",
            "approved_2nd_sem",
            "enrolled_1st_sem",
            "enrolled_2nd_sem",
        ],
        "stage2_interaction_abs_grade_drift": ["grade_1st_sem", "grade_2nd_sem"],
    },
}


def _resolve_requested_groups(
    enabled: bool,
    raw_groups: Any,
    default_groups: list[str],
    valid_groups: dict[str, Any],
) -> list[str]:
    if not enabled:
        return []
    requested: list[str] = []
    if isinstance(raw_groups, list):
        for item in raw_groups:
            group = str(item).strip().lower()
            if group and group in valid_groups and group not in requested:
                requested.append(group)
    return requested or list(default_groups)


def _resolve_interaction_config(feature_cfg: dict[str, Any] | None) -> tuple[bool, list[str]]:
    raw_cfg = feature_cfg if isinstance(feature_cfg, dict) else {}
    enabled = bool(raw_cfg.get("enabled", False))
    return enabled, _resolve_requested_groups(
        enabled,
        raw_cfg.get("groups", []),
        DEFAULT_INTERACTION_GROUPS,
        INTERACTION_FEATURE_REQUIREMENTS,
    )


def _resolve_selective_interaction_config(feature_cfg: dict[str, Any] | None) -> tuple[bool, list[str]]:
    raw_cfg = feature_cfg if isinstance(feature_cfg, dict) else {}
    enabled = bool(raw_cfg.get("enabled", False))
    if not enabled:
        return False, []
    allowlist: list[str] = []
    raw_allowlist = raw_cfg.get("feature_allowlist", [])
    if isinstance(raw_allowlist, list):
        for item in raw_allowlist:
            feature_name = str(item).strip().lower()
            if feature_name and feature_name in SELECTIVE_INTERACTION_REQUIREMENTS and feature_name not in allowlist:
                allowlist.append(feature_name)
    return True, allowlist or list(DEFAULT_SELECTIVE_INTERACTION_ALLOWLIST)


def _resolve_metric_set(feature_cfg: dict[str, Any] | None) -> tuple[bool, list[str]]:
    raw_cfg = feature_cfg if isinstance(feature_cfg, dict) else {}
    enabled = bool(raw_cfg.get("enabled", False))
    if not enabled:
        return False, []
    requested: list[str] = []
    raw_metrics = raw_cfg.get("metric_set", [])
    if isinstance(raw_metrics, list):
        for item in raw_metrics:
            metric = str(item).strip().lower()
            if metric and metric not in requested:
                requested.append(metric)
    metric_set = requested or list(DEFAULT_PROTOTYPE_METRIC_SET)
    normalized: list[str] = []
    for metric in metric_set:
        if metric in {"l1", "l2", "margin", "inverse_similarity", "normalized"} and metric not in normalized:
            normalized.append(metric)
    return True, normalized or list(DEFAULT_PROTOTYPE_METRIC_SET)


def _resolve_robust_prototype_config(feature_cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw_cfg = feature_cfg if isinstance(feature_cfg, dict) else {}
    enabled = bool(raw_cfg.get("enabled", False))
    return {
        "enabled": enabled,
        "add_distance_features": bool(raw_cfg.get("add_distance_features", True)),
        "add_ratio_features": bool(raw_cfg.get("add_ratio_features", True)),
        "add_margin_feature": bool(raw_cfg.get("add_margin_feature", True)),
        "add_cosine_similarity": bool(raw_cfg.get("add_cosine_similarity", False)),
        "eps": float(raw_cfg.get("eps", 1.0e-8)),
        "on_failure": str(raw_cfg.get("on_failure", "disable_and_continue")).strip().lower(),
    }


def _sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _build_stage2_signal_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    sources = _resolve_semester_sources(df)
    signal = pd.DataFrame(index=df.index)
    for canonical, actual in sources.items():
        signal[canonical] = _numeric(df, actual)

    approved_1 = signal["approved_1st_sem"] if "approved_1st_sem" in signal else None
    approved_2 = signal["approved_2nd_sem"] if "approved_2nd_sem" in signal else None
    enrolled_1 = signal["enrolled_1st_sem"] if "enrolled_1st_sem" in signal else None
    enrolled_2 = signal["enrolled_2nd_sem"] if "enrolled_2nd_sem" in signal else None
    evaluations_1 = signal["evaluations_1st_sem"] if "evaluations_1st_sem" in signal else None
    evaluations_2 = signal["evaluations_2nd_sem"] if "evaluations_2nd_sem" in signal else None
    grade_1 = signal["grade_1st_sem"] if "grade_1st_sem" in signal else None
    grade_2 = signal["grade_2nd_sem"] if "grade_2nd_sem" in signal else None

    if grade_1 is not None and grade_2 is not None:
        signal["grade_diff"] = grade_2 - grade_1
        signal["grade_ratio"] = _safe_divide(grade_2, grade_1)
        signal["grade_mean"] = (grade_1 + grade_2) / 2.0
        signal["grade_decline_flag"] = _flag(signal["grade_diff"] < -0.5)
    if approved_1 is not None and enrolled_1 is not None:
        signal["approved_ratio_1st"] = _safe_divide(approved_1, enrolled_1)
    if approved_2 is not None and enrolled_2 is not None:
        signal["approved_ratio_2nd"] = _safe_divide(approved_2, enrolled_2)
        signal["second_sem_present_flag"] = _flag(enrolled_2 > 0)
    if approved_1 is not None and approved_2 is not None:
        signal["approved_total"] = approved_1 + approved_2
    if enrolled_1 is not None and enrolled_2 is not None:
        signal["enrolled_total"] = enrolled_1 + enrolled_2
        signal["enrolled_both_semesters_flag"] = _flag((enrolled_1 > 0) & (enrolled_2 > 0))
    if evaluations_1 is not None and evaluations_2 is not None:
        signal["evaluations_total"] = evaluations_1 + evaluations_2
    if "approved_total" in signal and "enrolled_total" in signal:
        signal["approval_ratio_total"] = _safe_divide(signal["approved_total"], signal["enrolled_total"])
        signal["completion_gap_total"] = signal["enrolled_total"] - signal["approved_total"]
    if "approved_ratio_1st" in signal and "approved_ratio_2nd" in signal:
        signal["ratio_change"] = signal["approved_ratio_2nd"] - signal["approved_ratio_1st"]
    if approved_1 is not None and evaluations_1 is not None:
        signal["approval_efficiency_1st"] = _safe_divide(approved_1, evaluations_1)
    if approved_2 is not None and evaluations_2 is not None:
        signal["approval_efficiency_2nd"] = _safe_divide(approved_2, evaluations_2)
    if evaluations_1 is not None and approved_1 is not None and evaluations_2 is not None and approved_2 is not None:
        signal["eval_mismatch_total"] = (evaluations_1 - approved_1) + (evaluations_2 - approved_2)
    if approved_2 is not None and enrolled_2 is not None:
        signal["completion_gap_2nd"] = enrolled_2 - approved_2
    if approved_1 is not None and enrolled_1 is not None and approved_2 is not None and enrolled_2 is not None:
        signal["persistent_low_completion_flag"] = _flag(
            (_safe_divide(approved_1, enrolled_1) < 0.50)
            & (_safe_divide(approved_2, enrolled_2) < 0.50)
            & (enrolled_1 > 0)
            & (enrolled_2 > 0)
        )
        signal["incomplete_but_active_flag"] = _flag(((enrolled_1 + enrolled_2) > (approved_1 + approved_2)) & (enrolled_2 > 0))
    if approved_1 is not None and approved_2 is not None and enrolled_1 is not None and enrolled_2 is not None and evaluations_1 is not None and evaluations_2 is not None:
        signal["activity_continuity_score"] = (
            _flag((enrolled_1 > 0) & (enrolled_2 > 0))
            + _flag((approved_1 > 0) & (approved_2 > 0))
            + _flag((evaluations_1 > 0) & (evaluations_2 > 0))
        ) / 3.0

    signal = signal.replace([np.inf, -np.inf], np.nan)
    return signal, sources


def _build_selective_interactions_for_df(
    df: pd.DataFrame,
    *,
    allowlist: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], dict[str, str]]:
    signal, sources = _build_stage2_signal_frame(df)
    output = pd.DataFrame(index=df.index)
    if target_column in df.columns:
        output[target_column] = df[target_column].copy()

    created_features: list[str] = []
    skipped_features: list[str] = []
    skipped_existing_features: list[str] = []

    def missing_sources(feature_name: str) -> bool:
        required = SELECTIVE_INTERACTION_REQUIREMENTS.get(feature_name, [])
        ready = all(token in sources for token in required)
        if not ready:
            skipped_features.append(feature_name)
        return not ready

    def add(feature_name: str, values: pd.Series) -> None:
        output[feature_name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        created_features.append(feature_name)

    sem1_approval_rate = signal.get("approved_ratio_1st")
    sem2_approval_rate = signal.get("approved_ratio_2nd")
    grade_1 = signal.get("grade_1st_sem")
    grade_2 = signal.get("grade_2nd_sem")
    approved_1 = signal.get("approved_1st_sem")
    approved_2 = signal.get("approved_2nd_sem")
    enrolled_1 = signal.get("enrolled_1st_sem")
    enrolled_2 = signal.get("enrolled_2nd_sem")
    approved_total = signal.get("approved_total")
    enrolled_total = signal.get("enrolled_total")

    for feature_name in allowlist:
        if feature_name in output.columns or feature_name in df.columns:
            skipped_existing_features.append(feature_name)
            continue
        if feature_name not in SELECTIVE_INTERACTION_REQUIREMENTS:
            skipped_features.append(feature_name)
            continue
        if missing_sources(feature_name):
            continue
        if feature_name == "sem1_approval_rate":
            add(feature_name, safe_divide(approved_1, enrolled_1, default=0.0))
        elif feature_name == "sem2_approval_rate":
            add(feature_name, safe_divide(approved_2, enrolled_2, default=0.0))
        elif feature_name == "sem1_grade_efficiency":
            add(feature_name, grade_1 * safe_divide(approved_1, enrolled_1, default=0.0))
        elif feature_name == "sem2_grade_efficiency":
            add(feature_name, grade_2 * safe_divide(approved_2, enrolled_2, default=0.0))
        elif feature_name == "approval_rate_delta":
            add(feature_name, sem2_approval_rate.fillna(0.0) - sem1_approval_rate.fillna(0.0))
        elif feature_name == "grade_delta":
            add(feature_name, grade_2 - grade_1)
        elif feature_name == "persistence_gap":
            add(feature_name, (sem2_approval_rate.fillna(0.0) - sem1_approval_rate.fillna(0.0)).abs())
        elif feature_name == "load_pressure_sem1":
            add(feature_name, enrolled_1 - approved_1)
        elif feature_name == "load_pressure_sem2":
            add(feature_name, enrolled_2 - approved_2)
        elif feature_name == "completion_balance":
            add(feature_name, safe_divide(approved_total, enrolled_total, default=0.0))

    feature_cols = [col for col in output.columns if col != target_column]
    if feature_cols:
        output[feature_cols] = output[feature_cols].replace([np.inf, -np.inf], np.nan)
    return output, created_features, skipped_features, skipped_existing_features, sources


def build_stage2_selective_interaction_split_data(
    split_data: dict[str, pd.DataFrame],
    *,
    target_column: str = "target",
    feature_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    enabled, allowlist = _resolve_selective_interaction_config(feature_cfg)
    report: dict[str, Any] = {
        "enabled": enabled,
        "feature_allowlist": allowlist,
        "default_feature_allowlist": list(DEFAULT_SELECTIVE_INTERACTION_ALLOWLIST),
        "created_features": [],
        "created_feature_count": 0,
        "skipped_features": [],
        "skipped_existing_features": [],
        "source_columns": {},
    }
    if not enabled:
        empty_splits = {
            split_name: pd.DataFrame({target_column: df[target_column].copy()}) if target_column in df.columns else pd.DataFrame(index=df.index)
            for split_name, df in split_data.items()
        }
        return empty_splits, report

    transformed: dict[str, pd.DataFrame] = {}
    created_features: list[str] = []
    skipped_features: list[str] = []
    skipped_existing_features: list[str] = []
    for split_name, df in split_data.items():
        feature_df, split_created, split_skipped, split_existing, sources = _build_selective_interactions_for_df(
            df,
            allowlist=allowlist,
            target_column=target_column,
        )
        transformed[split_name] = feature_df
        if split_name == "train":
            created_features = list(split_created)
            skipped_features = list(split_skipped)
            skipped_existing_features = list(split_existing)
            report["source_columns"] = dict(sources)

    report["created_features"] = created_features
    report["created_feature_count"] = int(len(created_features))
    report["skipped_features"] = skipped_features
    report["skipped_existing_features"] = skipped_existing_features
    return transformed, report


def _build_interaction_features_for_df(
    df: pd.DataFrame,
    *,
    groups: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    signal, sources = _build_stage2_signal_frame(df)
    output = pd.DataFrame(index=df.index)
    if target_column in df.columns:
        output[target_column] = df[target_column].copy()

    created_by_group: dict[str, list[str]] = {group: [] for group in groups}
    skipped_by_group: dict[str, list[str]] = {group: [] for group in groups}

    def can_build(group: str, feature_name: str) -> bool:
        required = INTERACTION_FEATURE_REQUIREMENTS[group][feature_name]
        ready = all(token in sources for token in required)
        if not ready:
            skipped_by_group[group].append(feature_name)
        return ready

    def add(group: str, feature_name: str, values: pd.Series) -> None:
        output[feature_name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        created_by_group[group].append(feature_name)

    if "multiplicative" in groups:
        if can_build("multiplicative", "stage2_interaction_grade_mean_x_approval_ratio_total"):
            add("multiplicative", "stage2_interaction_grade_mean_x_approval_ratio_total", signal["grade_mean"] * signal["approval_ratio_total"])
        if can_build("multiplicative", "stage2_interaction_grade_diff_x_completion_gap_total"):
            add("multiplicative", "stage2_interaction_grade_diff_x_completion_gap_total", signal["grade_diff"] * signal["completion_gap_total"])
        if can_build("multiplicative", "stage2_interaction_continuity_x_approval_ratio_2nd"):
            add("multiplicative", "stage2_interaction_continuity_x_approval_ratio_2nd", signal["activity_continuity_score"] * signal["approved_ratio_2nd"])
        if can_build("multiplicative", "stage2_interaction_eval_mismatch_x_grade_decline_flag"):
            add("multiplicative", "stage2_interaction_eval_mismatch_x_grade_decline_flag", signal["eval_mismatch_total"] * signal["grade_decline_flag"])
        if can_build("multiplicative", "stage2_interaction_activity_continuity_x_incomplete_active_flag"):
            add("multiplicative", "stage2_interaction_activity_continuity_x_incomplete_active_flag", signal["activity_continuity_score"] * signal["incomplete_but_active_flag"])

    if "difference_conditioned" in groups:
        if can_build("difference_conditioned", "stage2_interaction_grade_diff_weighted_by_approved_total"):
            add("difference_conditioned", "stage2_interaction_grade_diff_weighted_by_approved_total", signal["grade_diff"] * signal["approved_total"])
        if can_build("difference_conditioned", "stage2_interaction_ratio_change_weighted_by_enrolled_total"):
            add("difference_conditioned", "stage2_interaction_ratio_change_weighted_by_enrolled_total", signal["ratio_change"] * signal["enrolled_total"])
        if can_build("difference_conditioned", "stage2_interaction_approval_gap_2nd_weighted_by_second_sem_presence"):
            add("difference_conditioned", "stage2_interaction_approval_gap_2nd_weighted_by_second_sem_presence", signal["completion_gap_2nd"] * signal["second_sem_present_flag"])
        if can_build("difference_conditioned", "stage2_interaction_drift_x_second_sem_active_flag"):
            add("difference_conditioned", "stage2_interaction_drift_x_second_sem_active_flag", signal["grade_diff"] * signal["second_sem_present_flag"])

    if "ratio_cross" in groups:
        if can_build("ratio_cross", "stage2_interaction_approved_ratio_1st_over_2nd"):
            add("ratio_cross", "stage2_interaction_approved_ratio_1st_over_2nd", _safe_divide(signal["approved_ratio_1st"], signal["approved_ratio_2nd"]))
        if can_build("ratio_cross", "stage2_interaction_approval_efficiency_2nd_over_1st"):
            add("ratio_cross", "stage2_interaction_approval_efficiency_2nd_over_1st", _safe_divide(signal["approval_efficiency_2nd"], signal["approval_efficiency_1st"]))
        if can_build("ratio_cross", "stage2_interaction_evaluation_pressure_over_completion_ratio"):
            add("ratio_cross", "stage2_interaction_evaluation_pressure_over_completion_ratio", _safe_divide(signal["evaluations_total"] - signal["approved_total"], signal["completion_gap_total"]))
        if can_build("ratio_cross", "stage2_interaction_grade_ratio_x_approved_ratio_change"):
            add("ratio_cross", "stage2_interaction_grade_ratio_x_approved_ratio_change", signal["grade_ratio"] * signal["ratio_change"])

    if "flag_conditioned" in groups:
        if can_build("flag_conditioned", "stage2_interaction_second_sem_grade_if_active"):
            add("flag_conditioned", "stage2_interaction_second_sem_grade_if_active", signal["grade_2nd_sem"] * signal["second_sem_present_flag"])
        if can_build("flag_conditioned", "stage2_interaction_completion_gap_if_persistent"):
            add("flag_conditioned", "stage2_interaction_completion_gap_if_persistent", signal["completion_gap_total"] * signal["persistent_low_completion_flag"])
        if can_build("flag_conditioned", "stage2_interaction_grade_decline_if_enrolled_both_semesters"):
            add("flag_conditioned", "stage2_interaction_grade_decline_if_enrolled_both_semesters", signal["grade_diff"] * signal["enrolled_both_semesters_flag"] * signal["grade_decline_flag"])
        if can_build("flag_conditioned", "stage2_interaction_approval_ratio_if_incomplete_but_active"):
            add("flag_conditioned", "stage2_interaction_approval_ratio_if_incomplete_but_active", signal["approval_ratio_total"] * signal["incomplete_but_active_flag"])

    if "limited_polynomial" in groups:
        if can_build("limited_polynomial", "stage2_interaction_grade_diff_sq"):
            add("limited_polynomial", "stage2_interaction_grade_diff_sq", signal["grade_diff"] ** 2)
        if can_build("limited_polynomial", "stage2_interaction_completion_gap_sq"):
            add("limited_polynomial", "stage2_interaction_completion_gap_sq", signal["completion_gap_total"] ** 2)
        if can_build("limited_polynomial", "stage2_interaction_approval_ratio_delta_abs"):
            add("limited_polynomial", "stage2_interaction_approval_ratio_delta_abs", signal["ratio_change"].abs())
        if can_build("limited_polynomial", "stage2_interaction_abs_grade_drift"):
            add("limited_polynomial", "stage2_interaction_abs_grade_drift", signal["grade_diff"].abs())

    feature_cols = [col for col in output.columns if col != target_column]
    if feature_cols:
        output[feature_cols] = output[feature_cols].replace([np.inf, -np.inf], np.nan)
    return output, created_by_group, skipped_by_group, sources


def build_stage2_interaction_split_data(
    split_data: dict[str, pd.DataFrame],
    *,
    target_column: str = "target",
    feature_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    enabled, groups = _resolve_interaction_config(feature_cfg)
    report: dict[str, Any] = {
        "enabled": enabled,
        "requested_groups": groups,
        "default_groups": list(DEFAULT_INTERACTION_GROUPS),
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
        feature_df, created_by_group, skipped_by_group, sources = _build_interaction_features_for_df(
            df,
            groups=groups,
            target_column=target_column,
        )
        transformed[split_name] = feature_df
        if split_name == "train":
            created_by_group_final = {group: list(names) for group, names in created_by_group.items()}
            skipped_by_group_final = {group: list(names) for group, names in skipped_by_group.items()}
            report["source_columns"] = sources

    created_features = [feature_name for group in groups for feature_name in created_by_group_final.get(group, [])]
    report["created_features"] = created_features
    report["created_feature_count"] = int(len(created_features))
    report["created_features_by_group"] = created_by_group_final
    report["skipped_features_by_group"] = skipped_by_group_final
    return transformed, report


def _build_legacy_stage2_prototype_distance_features(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cfg: dict[str, Any] | None = None,
    enrolled_positive_label: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    enabled, metric_set = _resolve_metric_set(feature_cfg)
    report: dict[str, Any] = {
        "enabled": enabled,
        "metric_set": metric_set,
        "prototype_source_columns": [],
        "prototype_feature_columns": [],
        "prototype_definition": "classwise_median",
        "fitted_on_train_only": True,
    }
    if not enabled:
        return {
            "train": pd.DataFrame(index=X_train.index),
            "valid": pd.DataFrame(index=X_valid.index),
            "test": pd.DataFrame(index=X_test.index),
        }, report

    positive_label = int(enrolled_positive_label)
    train_numeric = X_train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    preferred_columns = [
        col
        for col in train_numeric.columns
        if (
            col in {
                "approved_1st_sem",
                "approved_2nd_sem",
                "enrolled_1st_sem",
                "enrolled_2nd_sem",
                "evaluations_1st_sem",
                "evaluations_2nd_sem",
                "grade_1st_sem",
                "grade_2nd_sem",
            }
            or col.startswith("stage2_sharp_")
            or col.startswith("stage2_interaction_")
        )
    ]
    source_columns = [col for col in preferred_columns if col in X_valid.columns and col in X_test.columns]
    if not source_columns:
        report["skip_reason"] = "no_shared_numeric_source_columns"
        return {
            "train": pd.DataFrame(index=X_train.index),
            "valid": pd.DataFrame(index=X_valid.index),
            "test": pd.DataFrame(index=X_test.index),
        }, report

    y_arr = pd.Series(y_train).reset_index(drop=True).astype(int)
    X_train_source = train_numeric.loc[:, source_columns].reset_index(drop=True)
    enrolled_mask = y_arr == positive_label
    graduate_mask = y_arr != positive_label
    if int(enrolled_mask.sum()) == 0 or int(graduate_mask.sum()) == 0:
        report["skip_reason"] = "missing_stage2_class_for_prototypes"
        report["prototype_source_columns"] = list(source_columns)
        return {
            "train": pd.DataFrame(index=X_train.index),
            "valid": pd.DataFrame(index=X_valid.index),
            "test": pd.DataFrame(index=X_test.index),
        }, report

    fill_values = X_train_source.median(axis=0, numeric_only=True).replace([np.inf, -np.inf], np.nan)
    fill_values = fill_values.fillna(0.0)
    train_filled = X_train_source.fillna(fill_values)
    enrolled_proto = train_filled.loc[enrolled_mask].median(axis=0, numeric_only=True)
    graduate_proto = train_filled.loc[graduate_mask].median(axis=0, numeric_only=True)

    def transform_frame(df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.loc[:, source_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        filled = numeric.fillna(fill_values)
        enrolled_delta = filled.subtract(enrolled_proto, axis=1)
        graduate_delta = filled.subtract(graduate_proto, axis=1)
        dist_enrolled_l1 = enrolled_delta.abs().sum(axis=1)
        dist_graduate_l1 = graduate_delta.abs().sum(axis=1)
        dist_enrolled_l2 = np.sqrt((enrolled_delta ** 2).sum(axis=1))
        dist_graduate_l2 = np.sqrt((graduate_delta ** 2).sum(axis=1))
        out = pd.DataFrame(index=df.index)
        if "l1" in metric_set:
            out["stage2_proto_dist_to_enrolled_proto_l1"] = dist_enrolled_l1.astype(float)
            out["stage2_proto_dist_to_graduate_proto_l1"] = dist_graduate_l1.astype(float)
        if "l2" in metric_set:
            out["stage2_proto_dist_to_enrolled_proto_l2"] = dist_enrolled_l2.astype(float)
            out["stage2_proto_dist_to_graduate_proto_l2"] = dist_graduate_l2.astype(float)

        enrolled_anchor = dist_enrolled_l2 if "l2" in metric_set else dist_enrolled_l1
        graduate_anchor = dist_graduate_l2 if "l2" in metric_set else dist_graduate_l1
        if "margin" in metric_set:
            out["stage2_proto_dist_margin_graduate_minus_enrolled"] = (graduate_anchor - enrolled_anchor).astype(float)
            out["stage2_proto_dist_margin_abs"] = (graduate_anchor - enrolled_anchor).abs().astype(float)
            out["stage2_proto_ratio_enrolled_over_graduate"] = _safe_divide(enrolled_anchor, graduate_anchor).astype(float)
            out["stage2_proto_nearest_class_flag"] = (enrolled_anchor <= graduate_anchor).astype(float)
        if "inverse_similarity" in metric_set:
            out["stage2_proto_enrolled_similarity_inverse"] = (1.0 / (1.0 + enrolled_anchor)).astype(float)
            out["stage2_proto_graduate_similarity_inverse"] = (1.0 / (1.0 + graduate_anchor)).astype(float)
        if "normalized" in metric_set:
            denom = enrolled_anchor + graduate_anchor
            out["stage2_proto_normalized_dist_to_enrolled"] = _safe_divide(enrolled_anchor, denom).astype(float)
            out["stage2_proto_normalized_dist_to_graduate"] = _safe_divide(graduate_anchor, denom).astype(float)
        return out.replace([np.inf, -np.inf], np.nan)

    transformed = {
        "train": transform_frame(X_train),
        "valid": transform_frame(X_valid),
        "test": transform_frame(X_test),
    }
    report["prototype_source_columns"] = list(source_columns)
    report["prototype_feature_columns"] = list(transformed["train"].columns)
    report["created_feature_count"] = int(len(report["prototype_feature_columns"]))
    return transformed, report


def _build_robust_stage2_prototype_distance_features(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cfg: dict[str, Any] | None = None,
    enrolled_positive_label: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cfg = _resolve_robust_prototype_config(feature_cfg)
    report: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "metric_set": ["robust_distance"],
        "prototype_source_columns": [],
        "prototype_feature_columns": [],
        "prototype_definition": "classwise_mean_standardized",
        "fitted_on_train_only": True,
        "standardized_feature_space": True,
        "failure_mode": cfg.get("on_failure", "disable_and_continue"),
    }
    empty = {
        "train": pd.DataFrame(index=X_train.index),
        "valid": pd.DataFrame(index=X_valid.index),
        "test": pd.DataFrame(index=X_test.index),
    }
    if not bool(cfg.get("enabled", False)):
        return empty, report

    try:
        source_columns = [
            col
            for col in X_train.columns
            if col in X_valid.columns and col in X_test.columns and pd.api.types.is_numeric_dtype(X_train[col])
        ]
        if not source_columns:
            report["skip_reason"] = "no_shared_numeric_source_columns"
            return empty, report

        y_arr = pd.Series(y_train).reset_index(drop=True).astype(int)
        positive_label = int(enrolled_positive_label)
        enrolled_mask = y_arr == positive_label
        graduate_mask = y_arr != positive_label
        if int(enrolled_mask.sum()) == 0 or int(graduate_mask.sum()) == 0:
            report["skip_reason"] = "missing_stage2_class_for_prototypes"
            report["prototype_source_columns"] = list(source_columns)
            return empty, report

        train_numeric = _sanitize_feature_frame(X_train.loc[:, source_columns]).reset_index(drop=True)
        valid_numeric = _sanitize_feature_frame(X_valid.loc[:, source_columns]).reset_index(drop=True)
        test_numeric = _sanitize_feature_frame(X_test.loc[:, source_columns]).reset_index(drop=True)
        fill_values = train_numeric.median(axis=0, numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_filled = train_numeric.fillna(fill_values)
        valid_filled = valid_numeric.fillna(fill_values)
        test_filled = test_numeric.fillna(fill_values)

        scaler = StandardScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(train_filled), columns=source_columns, index=X_train.index)
        valid_scaled = pd.DataFrame(scaler.transform(valid_filled), columns=source_columns, index=X_valid.index)
        test_scaled = pd.DataFrame(scaler.transform(test_filled), columns=source_columns, index=X_test.index)

        enrolled_proto = train_scaled.reset_index(drop=True).loc[enrolled_mask].mean(axis=0)
        graduate_proto = train_scaled.reset_index(drop=True).loc[graduate_mask].mean(axis=0)
        eps = max(float(cfg.get("eps", 1.0e-8)), 1.0e-12)

        def transform_frame(df_scaled: pd.DataFrame) -> pd.DataFrame:
            enrolled_delta = df_scaled.subtract(enrolled_proto, axis=1)
            graduate_delta = df_scaled.subtract(graduate_proto, axis=1)
            dist_enrolled = np.sqrt((enrolled_delta ** 2).sum(axis=1)).astype(float)
            dist_graduate = np.sqrt((graduate_delta ** 2).sum(axis=1)).astype(float)
            out = pd.DataFrame(index=df_scaled.index)
            if bool(cfg.get("add_distance_features", True)):
                out["dist_to_enrolled_proto"] = dist_enrolled
                out["dist_to_graduate_proto"] = dist_graduate
            if bool(cfg.get("add_margin_feature", True)):
                out["proto_margin"] = (dist_graduate - dist_enrolled).astype(float)
            if bool(cfg.get("add_ratio_features", True)):
                out["proto_ratio"] = safe_divide(dist_enrolled, dist_graduate, default=1.0, eps=eps).clip(lower=0.0, upper=1.0e6)
            if bool(cfg.get("add_cosine_similarity", False)):
                enrolled_norm = max(float(np.linalg.norm(enrolled_proto.to_numpy(dtype=float))), eps)
                graduate_norm = max(float(np.linalg.norm(graduate_proto.to_numpy(dtype=float))), eps)
                sample_norm = np.maximum(np.linalg.norm(df_scaled.to_numpy(dtype=float), axis=1), eps)
                enrolled_cos = np.clip(df_scaled.to_numpy(dtype=float) @ enrolled_proto.to_numpy(dtype=float) / (sample_norm * enrolled_norm), -1.0, 1.0)
                graduate_cos = np.clip(df_scaled.to_numpy(dtype=float) @ graduate_proto.to_numpy(dtype=float) / (sample_norm * graduate_norm), -1.0, 1.0)
                out["cosine_to_enrolled_proto"] = enrolled_cos.astype(float)
                out["cosine_to_graduate_proto"] = graduate_cos.astype(float)
            return _sanitize_feature_frame(out).fillna(0.0)

        transformed = {
            "train": transform_frame(train_scaled),
            "valid": transform_frame(valid_scaled),
            "test": transform_frame(test_scaled),
        }
        report["prototype_source_columns"] = list(source_columns)
        report["prototype_feature_columns"] = list(transformed["train"].columns)
        report["created_feature_count"] = int(len(report["prototype_feature_columns"]))
        return transformed, report
    except Exception as exc:
        report["warning"] = f"prototype_augmentation_failed:{type(exc).__name__}: {exc}"
        if str(cfg.get("on_failure", "disable_and_continue")).strip().lower() == "disable_and_continue":
            report["enabled"] = False
            report["disabled_due_to_failure"] = True
            return empty, report
        raise


def build_stage2_prototype_distance_features(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cfg: dict[str, Any] | None = None,
    enrolled_positive_label: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    robust_cfg = _resolve_robust_prototype_config(feature_cfg)
    if bool(robust_cfg.get("enabled", False)):
        return _build_robust_stage2_prototype_distance_features(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            X_test=X_test,
            feature_cfg=feature_cfg,
            enrolled_positive_label=enrolled_positive_label,
        )
    return _build_legacy_stage2_prototype_distance_features(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        X_test=X_test,
        feature_cfg=feature_cfg,
        enrolled_positive_label=enrolled_positive_label,
    )
