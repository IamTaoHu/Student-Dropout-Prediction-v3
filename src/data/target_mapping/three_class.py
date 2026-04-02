"""Three-class target mapping utilities across supported datasets."""

from __future__ import annotations

import pandas as pd


def _normalize_label(value: object) -> str:
    return str(value).strip().lower()


def _validate(series: pd.Series, mapped: pd.Series, source_column: str) -> None:
    unmapped = sorted(series[mapped.isna()].dropna().astype(str).unique().tolist())
    if unmapped:
        raise ValueError(f"Unmapped labels in '{source_column}' for three-class mapping: {unmapped}")
    unique_classes = sorted(mapped.dropna().astype(int).unique().tolist())
    if len(unique_classes) < 3:
        raise ValueError(
            f"Three-class mapping produced {len(unique_classes)} class(es): {unique_classes}. "
            "Provide a mapping with three distinct classes."
        )


def _default_oulad_three_class_mapping() -> dict[str, int]:
    return {
        "pass": 0,
        "distinction": 0,
        "fail": 1,
        "withdrawn": 2,
    }


def map_three_class_target(
    df: pd.DataFrame,
    source_column: str,
    dataset_name: str,
    mapping: dict[str, int] | None = None,
) -> pd.Series:
    """Map raw labels into a three-class target with strict validation."""
    if source_column not in df.columns:
        raise KeyError(f"Source target column '{source_column}' not found in DataFrame.")

    ds = dataset_name.strip().lower()
    if mapping is None:
        if ds == "oulad":
            mapping = _default_oulad_three_class_mapping()
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' requires an explicit three-class mapping in config."
            )

    norm_map = {_normalize_label(k): int(v) for k, v in mapping.items()}
    raw = df[source_column]
    mapped = raw.astype(str).map(lambda x: norm_map.get(_normalize_label(x)))
    _validate(raw, mapped, source_column)
    return mapped.astype(int)
