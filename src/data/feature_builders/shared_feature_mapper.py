"""Map dataset-specific features to shared semantic benchmark concepts."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_CONCEPT_CANDIDATES: dict[str, list[str]] = {
    "age_band": ["age_band"],
    "gender": ["gender", "sex"],
    "prior_attainment": ["highest_education", "num_of_prev_attempts", "prior_grade", "entry_score"],
    "registration_delay_days": ["reg_date", "days_to_first_activity", "days_to_first_assessment"],
    "assessment_submission_rate": ["late_submission_rate", "assessment_count"],
    "assessment_score_mean": ["assessment_score_mean", "score_mean"],
    "vle_clicks_total": ["vle_clicks_total", "total_clicks"],
    "vle_active_weeks": ["vle_active_days", "unique_days"],
    "disability_flag": ["disability"],
    "socio_economic_index": ["imd_band", "socio_economic_index"],
}


def _pick_source_column(concept: str, dataset_name: str, df: pd.DataFrame, config_map: dict[str, Any]) -> str | None:
    dataset_map = config_map.get(dataset_name, {})
    explicit = dataset_map.get(concept)
    if explicit and explicit in df.columns:
        return explicit

    for candidate in DEFAULT_CONCEPT_CANDIDATES.get(concept, []):
        if candidate in df.columns:
            return candidate
    return None


def map_to_shared_features(
    features_df: pd.DataFrame,
    shared_schema_config: dict[str, Any],
    dataset_name: str,
    shared_feature_set: str = "minimal_comparison",
    strict: bool = False,
) -> pd.DataFrame:
    """Map one dataset feature table to a shared semantic feature space."""
    sets_cfg = shared_schema_config.get("shared_feature_sets", {})
    if shared_feature_set not in sets_cfg:
        raise ValueError(f"Unknown shared_feature_set '{shared_feature_set}'. Available: {list(sets_cfg)}")

    concepts = list(sets_cfg[shared_feature_set])
    mapping_cfg = shared_schema_config.get("feature_name_map", {})
    output = pd.DataFrame(index=features_df.index)

    for concept in concepts:
        source_col = _pick_source_column(concept, dataset_name, features_df, mapping_cfg)
        if source_col is None:
            if strict:
                raise ValueError(f"Dataset '{dataset_name}' has no source column mapped for concept '{concept}'.")
            output[concept] = pd.NA
        else:
            output[concept] = features_df[source_col]

    schema_cfg = shared_schema_config.get("shared_schema", {})
    entity_col = schema_cfg.get("entity_id")
    outcome_col = schema_cfg.get("outcome_label")
    for col in [entity_col, outcome_col]:
        if col and col in features_df.columns:
            output[col] = features_df[col]

    output["source_dataset"] = dataset_name
    return output
