"""Four-class mapping utilities (primarily for full OULAD outcomes)."""

from __future__ import annotations

import pandas as pd


def _normalize_label(value: object) -> str:
    return str(value).strip().lower()


def _default_oulad_four_class_mapping() -> dict[str, int]:
    return {
        "distinction": 0,
        "pass": 1,
        "fail": 2,
        "withdrawn": 3,
    }


def map_four_class_target(
    df: pd.DataFrame,
    source_column: str,
    dataset_name: str,
    mapping: dict[str, int] | None = None,
) -> pd.Series:
    """Map labels to four classes, validating complete and explicit class coverage."""
    if source_column not in df.columns:
        raise KeyError(f"Source target column '{source_column}' not found in DataFrame.")

    ds = dataset_name.strip().lower()
    if mapping is None:
        if ds != "oulad":
            raise ValueError("Default four-class mapping is only defined for OULAD.")
        mapping = _default_oulad_four_class_mapping()

    norm_map = {_normalize_label(k): int(v) for k, v in mapping.items()}
    raw = df[source_column]
    mapped = raw.astype(str).map(lambda x: norm_map.get(_normalize_label(x)))

    unmapped = sorted(raw[mapped.isna()].dropna().astype(str).unique().tolist())
    if unmapped:
        raise ValueError(f"Unmapped labels in '{source_column}' for four-class mapping: {unmapped}")

    observed_classes = sorted(mapped.dropna().astype(int).unique().tolist())
    if len(observed_classes) < 4:
        raise ValueError(
            f"Four-class mapping produced {len(observed_classes)} class(es): {observed_classes}. "
            "Use only when all four classes are present."
        )
    return mapped.astype(int)
