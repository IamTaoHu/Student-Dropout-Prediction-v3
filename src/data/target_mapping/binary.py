"""Binary target mapping utilities for UCT Student and OULAD."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _normalize_label(value: object) -> str:
    return str(value).strip().lower()


def _validate_mapping(series: pd.Series, mapped: pd.Series, source_column: str) -> None:
    unknown = sorted(series[mapped.isna()].dropna().astype(str).unique().tolist())
    if unknown:
        raise ValueError(f"Unmapped labels in '{source_column}' for binary mapping: {unknown}")


def _build_default_oulad_binary_mapping() -> dict[str, int]:
    return {
        "withdrawn": 1,
        "fail": 1,
        "pass": 0,
        "distinction": 0,
    }


def _build_mapping_from_label_sets(
    positive_labels: Iterable[str] | None,
    negative_labels: Iterable[str] | None,
) -> dict[str, int]:
    if positive_labels is None or negative_labels is None:
        raise ValueError("Both positive_labels and negative_labels are required when mapping is not provided.")
    mapping: dict[str, int] = {}
    for label in positive_labels:
        mapping[_normalize_label(label)] = 1
    for label in negative_labels:
        mapping[_normalize_label(label)] = 0
    return mapping


def map_binary_target(
    df: pd.DataFrame,
    source_column: str,
    dataset_name: str,
    mapping: dict[str, int] | None = None,
    positive_labels: Iterable[str] | None = None,
    negative_labels: Iterable[str] | None = None,
) -> pd.Series:
    """Map raw outcome labels into binary target values (0/1)."""
    if source_column not in df.columns:
        raise KeyError(f"Source target column '{source_column}' not found in DataFrame.")

    ds = dataset_name.strip().lower()
    if mapping is None:
        if ds == "oulad":
            mapping = _build_default_oulad_binary_mapping()
        elif ds == "uct_student":
            mapping = _build_mapping_from_label_sets(positive_labels, negative_labels)
        else:
            raise ValueError(f"Unsupported dataset for binary mapping: '{dataset_name}'")

    normalized_mapping = {_normalize_label(k): int(v) for k, v in mapping.items()}
    raw = df[source_column]
    mapped = raw.astype(str).map(lambda x: normalized_mapping.get(_normalize_label(x)))
    _validate_mapping(raw, mapped, source_column)
    return mapped.astype(int)
