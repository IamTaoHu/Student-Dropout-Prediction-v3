from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.feature_builders.uct_stage2_advanced_features import build_stage2_interaction_split_data
from src.data.feature_builders.uct_stage2_feature_separation import (
    build_advanced_enrolled_feature_separation_split_data,
)
from src.data.feature_builders.uct_stage2_feature_sharpening import (
    build_stage2_feature_sharpening_split_data,
)
from src.data.feature_builders.uct_stage2_selective_interactions import (
    build_stage2_selective_interaction_split_data,
)
from src.experiment.config_resolution import (
    _resolve_two_stage_stage2_advanced_config,
    _resolve_two_stage_stage2_feature_separation_config,
    _resolve_two_stage_stage2_feature_sharpening_config,
    _resolve_two_stage_stage2_selective_interactions_config,
)
from src.preprocessing.tabular_pipeline import run_tabular_preprocessing

def _prepare_two_stage_stage2_feature_bundle(
    *,
    two_stage_cfg: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    preprocess_cfg: dict[str, Any],
) -> dict[str, Any]:
    feature_cfg = _resolve_two_stage_stage2_feature_sharpening_config(two_stage_cfg)
    advanced_cfg = _resolve_two_stage_stage2_advanced_config(two_stage_cfg)
    feature_separation_cfg = _resolve_two_stage_stage2_feature_separation_config(two_stage_cfg)
    selective_cfg = _resolve_two_stage_stage2_selective_interactions_config(two_stage_cfg)
    requested_groups = list(feature_cfg.get("groups", []))
    interaction_cfg = advanced_cfg.get("interaction_features", {})
    prototype_cfg = advanced_cfg.get("prototype_distance", {})
    feature_separation_groups = list(feature_separation_cfg.get("feature_groups", []))
    requested_interaction_groups = list(interaction_cfg.get("groups", [])) if isinstance(interaction_cfg, dict) else []
    selective_allowlist = list(selective_cfg.get("feature_allowlist", []))
    print(
        "[two_stage][stage2][feature_sharpening] "
        f"enabled={bool(feature_cfg.get('enabled', False))} "
        f"requested_groups={requested_groups if requested_groups else []}"
    )
    print(
        "[two_stage][stage2][feature_separation] "
        f"enabled={bool(feature_separation_cfg.get('enabled', False))} "
        f"strict_mode={bool(feature_separation_cfg.get('strict_mode', False))} "
        f"feature_groups={feature_separation_groups if feature_separation_groups else []} "
        f"create_composite_scores={bool(feature_separation_cfg.get('create_composite_scores', True))}"
    )
    print(
        "[two_stage][stage2][advanced_enrolled_separation] "
        f"enabled={bool(advanced_cfg.get('enabled', False))} "
        f"interaction_enabled={bool(interaction_cfg.get('enabled', False))} "
        f"interaction_groups={requested_interaction_groups if requested_interaction_groups else []} "
        f"prototype_distance_enabled={bool(prototype_cfg.get('enabled', False))} "
        f"prototype_metric_set={list(prototype_cfg.get('metric_set', [])) if isinstance(prototype_cfg, dict) else []}"
    )
    print(
        "[two_stage][stage2][selective_interactions] "
        f"enabled={bool(selective_cfg.get('enabled', False))} "
        f"feature_allowlist={selective_allowlist if selective_allowlist else []}"
    )
    if (
        not bool(feature_cfg.get("enabled", False))
        and not bool(feature_separation_cfg.get("enabled", False))
        and not bool(interaction_cfg.get("enabled", False))
        and not bool(selective_cfg.get("enabled", False))
    ):
        return {
            "enabled": False,
            "requested_groups": requested_groups,
            "feature_separation_groups": feature_separation_groups,
            "advanced_requested_groups": requested_interaction_groups,
            "selective_feature_allowlist": selective_allowlist,
            "report": {
                "enabled": False,
                "requested_groups": requested_groups,
                "feature_separation_enabled": bool(feature_separation_cfg.get("enabled", False)),
                "feature_separation_groups": feature_separation_groups,
                "feature_separation_strict_mode": bool(feature_separation_cfg.get("strict_mode", False)),
                "feature_separation_apply_only_to_stage2": bool(feature_separation_cfg.get("apply_only_to_stage2", True)),
                "advanced_enrolled_separation_enabled": bool(advanced_cfg.get("enabled", False)),
                "interaction_features_enabled": bool(interaction_cfg.get("enabled", False)),
                "prototype_distance_enabled": bool(prototype_cfg.get("enabled", False)),
                "interaction_requested_groups": requested_interaction_groups,
                "selective_interactions_enabled": bool(selective_cfg.get("enabled", False)),
                "selective_feature_allowlist": selective_allowlist,
                "prototype_metric_set": list(prototype_cfg.get("metric_set", [])) if isinstance(prototype_cfg, dict) else [],
                "default_groups": list(feature_cfg.get("default_groups", [])),
                "created_features": [],
                "created_feature_count": 0,
                "created_features_by_group": {},
                "skipped_features_by_group": {},
                "missing_base_columns": [],
                "skipped_features": [],
                "feature_sharpening": {"enabled": False},
                "feature_separation": {"enabled": False},
                "interaction_features": {"enabled": False},
                "selective_interactions": {"enabled": False},
                "prototype_distance": {
                    "enabled": bool(prototype_cfg.get("enabled", False)),
                    "metric_set": list(prototype_cfg.get("metric_set", [])) if isinstance(prototype_cfg, dict) else [],
                },
            },
        }

    preprocess_cfg_stage2 = dict(preprocess_cfg)
    preprocess_cfg_stage2["target_column"] = "target"
    preprocess_cfg_stage2["id_columns"] = []
    combined_stage2_frames: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}

    sharpening_report: dict[str, Any] = {"enabled": False}
    if bool(feature_cfg.get("enabled", False)):
        split_features, sharpening_report = build_stage2_feature_sharpening_split_data(
            splits,
            target_column="target",
            feature_cfg=feature_cfg,
        )
        stage2_artifacts = run_tabular_preprocessing(split_features, preprocess_cfg_stage2)
        for split_name, df_key in (("train", "X_train"), ("valid", "X_valid"), ("test", "X_test")):
            combined_stage2_frames[split_name].append(getattr(stage2_artifacts, df_key).reset_index(drop=True))
        created_groups = {
            group: len(names)
            for group, names in (sharpening_report.get("created_features_by_group", {}) if isinstance(sharpening_report, dict) else {}).items()
        }
        skipped_groups = {
            group: len(names)
            for group, names in (sharpening_report.get("skipped_features_by_group", {}) if isinstance(sharpening_report, dict) else {}).items()
        }
        print(
            "[two_stage][stage2][feature_sharpening] "
            f"created_feature_count={int(sharpening_report.get('created_feature_count', 0))} "
            f"created_features={sharpening_report.get('created_features', [])}"
        )
        print(
            "[two_stage][stage2][feature_sharpening] "
            f"created_by_group={created_groups} "
            f"skipped_by_group={skipped_groups}"
        )

    feature_separation_report: dict[str, Any] = {"enabled": False}
    if bool(feature_separation_cfg.get("enabled", False)):
        feature_separation_splits, feature_separation_report = build_advanced_enrolled_feature_separation_split_data(
            splits,
            target_column="target",
            feature_cfg=feature_separation_cfg,
        )
        feature_separation_artifacts = run_tabular_preprocessing(feature_separation_splits, preprocess_cfg_stage2)
        for split_name, df_key in (("train", "X_train"), ("valid", "X_valid"), ("test", "X_test")):
            combined_stage2_frames[split_name].append(getattr(feature_separation_artifacts, df_key).reset_index(drop=True))
        created_groups = {
            group: len(names)
            for group, names in (feature_separation_report.get("created_features_by_group", {}) if isinstance(feature_separation_report, dict) else {}).items()
        }
        skipped_groups = {
            group: len(names)
            for group, names in (feature_separation_report.get("skipped_features_by_group", {}) if isinstance(feature_separation_report, dict) else {}).items()
        }
        print(
            "[two_stage][stage2][feature_separation] "
            f"enabled={bool(feature_separation_report.get('enabled', False))} "
            f"created_features={feature_separation_report.get('created_features', [])}"
        )
        print(
            "[two_stage][stage2][feature_separation] "
            f"missing_base_columns={feature_separation_report.get('missing_base_columns', [])} "
            f"skipped_features={feature_separation_report.get('skipped_features', [])} "
            f"skipped_by_group={skipped_groups} "
            f"created_by_group={created_groups}"
        )

    interaction_report: dict[str, Any] = {"enabled": False}
    if bool(interaction_cfg.get("enabled", False)):
        interaction_splits, interaction_report = build_stage2_interaction_split_data(
            splits,
            target_column="target",
            feature_cfg=interaction_cfg,
        )
        interaction_artifacts = run_tabular_preprocessing(interaction_splits, preprocess_cfg_stage2)
        for split_name, df_key in (("train", "X_train"), ("valid", "X_valid"), ("test", "X_test")):
            combined_stage2_frames[split_name].append(getattr(interaction_artifacts, df_key).reset_index(drop=True))
        created_groups = {
            group: len(names)
            for group, names in (interaction_report.get("created_features_by_group", {}) if isinstance(interaction_report, dict) else {}).items()
        }
        skipped_groups = {
            group: len(names)
            for group, names in (interaction_report.get("skipped_features_by_group", {}) if isinstance(interaction_report, dict) else {}).items()
        }
        print(
            "[two_stage][stage2][interaction_features] "
            f"created_feature_count={int(interaction_report.get('created_feature_count', 0))} "
            f"created_features={interaction_report.get('created_features', [])}"
        )
        print(
            "[two_stage][stage2][interaction_features] "
            f"created_by_group={created_groups} "
            f"skipped_by_group={skipped_groups}"
        )

    selective_report: dict[str, Any] = {"enabled": False}
    if bool(selective_cfg.get("enabled", False)):
        selective_splits, selective_report = build_stage2_selective_interaction_split_data(
            splits,
            target_column="target",
            feature_cfg=selective_cfg,
        )
        selective_artifacts = run_tabular_preprocessing(selective_splits, preprocess_cfg_stage2)
        for split_name, df_key in (("train", "X_train"), ("valid", "X_valid"), ("test", "X_test")):
            combined_stage2_frames[split_name].append(getattr(selective_artifacts, df_key).reset_index(drop=True))
        print(
            "[v8] created selective interactions: "
            f"{selective_report.get('created_features', [])}"
        )
        print(
            "[v8] skipped existing interaction features: "
            f"{selective_report.get('skipped_existing_features', [])}"
        )

    aggregated_train = pd.concat(combined_stage2_frames["train"], axis=1) if combined_stage2_frames["train"] else pd.DataFrame(index=splits["train"].index)
    aggregated_valid = pd.concat(combined_stage2_frames["valid"], axis=1) if combined_stage2_frames["valid"] else pd.DataFrame(index=splits["valid"].index)
    aggregated_test = pd.concat(combined_stage2_frames["test"], axis=1) if combined_stage2_frames["test"] else pd.DataFrame(index=splits["test"].index)
    combined_created_features = (
        list(sharpening_report.get("created_features", []))
        + list(feature_separation_report.get("created_features", []))
        + list(interaction_report.get("created_features", []))
        + list(selective_report.get("created_features", []))
    )
    report = {
        "enabled": True,
        "requested_groups": requested_groups,
        "feature_separation_groups": feature_separation_groups,
        "interaction_requested_groups": requested_interaction_groups,
        "selective_feature_allowlist": selective_allowlist,
        "feature_separation_enabled": bool(feature_separation_cfg.get("enabled", False)),
        "feature_separation_strict_mode": bool(feature_separation_cfg.get("strict_mode", False)),
        "feature_separation_apply_only_to_stage2": bool(feature_separation_cfg.get("apply_only_to_stage2", True)),
        "advanced_enrolled_separation_enabled": bool(advanced_cfg.get("enabled", False)),
        "interaction_features_enabled": bool(interaction_cfg.get("enabled", False)),
        "selective_interactions_enabled": bool(selective_cfg.get("enabled", False)),
        "prototype_distance_enabled": bool(prototype_cfg.get("enabled", False)),
        "prototype_metric_set": list(prototype_cfg.get("metric_set", [])) if isinstance(prototype_cfg, dict) else [],
        "default_groups": list(feature_cfg.get("default_groups", [])),
        "created_features": combined_created_features,
        "created_feature_count": int(len(combined_created_features)),
        "created_features_by_group": {
            **(sharpening_report.get("created_features_by_group", {}) if isinstance(sharpening_report, dict) else {}),
            **(feature_separation_report.get("created_features_by_group", {}) if isinstance(feature_separation_report, dict) else {}),
            **(interaction_report.get("created_features_by_group", {}) if isinstance(interaction_report, dict) else {}),
            "selective_interactions": list(selective_report.get("created_features", [])),
        },
        "skipped_features_by_group": {
            **(sharpening_report.get("skipped_features_by_group", {}) if isinstance(sharpening_report, dict) else {}),
            **(feature_separation_report.get("skipped_features_by_group", {}) if isinstance(feature_separation_report, dict) else {}),
            **(interaction_report.get("skipped_features_by_group", {}) if isinstance(interaction_report, dict) else {}),
            "selective_interactions": list(selective_report.get("skipped_features", [])),
        },
        "missing_base_columns": list(feature_separation_report.get("missing_base_columns", [])),
        "skipped_features": list(feature_separation_report.get("skipped_features", [])),
        "skipped_existing_features": list(feature_separation_report.get("skipped_existing_features", [])),
        "source_columns": {
            "feature_sharpening": sharpening_report.get("source_columns", {}) if isinstance(sharpening_report, dict) else {},
            "feature_separation": feature_separation_report.get("source_columns", {}) if isinstance(feature_separation_report, dict) else {},
            "interaction_features": interaction_report.get("source_columns", {}) if isinstance(interaction_report, dict) else {},
            "selective_interactions": selective_report.get("source_columns", {}) if isinstance(selective_report, dict) else {},
        },
        "feature_sharpening": sharpening_report,
        "feature_separation": feature_separation_report,
        "interaction_features": interaction_report,
        "selective_interactions": selective_report,
        "prototype_distance": {
            "enabled": bool(prototype_cfg.get("enabled", False)),
            "metric_set": list(prototype_cfg.get("metric_set", [])) if isinstance(prototype_cfg, dict) else [],
        },
    }
    return {
        "enabled": True,
        "requested_groups": requested_groups,
        "feature_separation_groups": feature_separation_groups,
        "advanced_requested_groups": requested_interaction_groups,
        "selective_feature_allowlist": selective_allowlist,
        "report": report,
        "X_train": aggregated_train.reset_index(drop=True),
        "X_valid": aggregated_valid.reset_index(drop=True),
        "X_test": aggregated_test.reset_index(drop=True),
        "feature_names": list(aggregated_train.columns),
    }
