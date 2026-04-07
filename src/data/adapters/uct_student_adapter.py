"""Normalize UCT / UCI Student data into a schema-aware internal representation."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _to_snake_case(name: str) -> str:
    """Convert a raw column name to snake_case."""
    name = str(name).strip()
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_").lower()


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first matching column from candidates, case-insensitive."""
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def adapt_uct_student_schema(
    raw_tables: dict[str, pd.DataFrame] | pd.DataFrame,
    schema_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize UCT/UCI Student raw data and infer schema components.

    Notes
    -----
    - This adapter accepts either:
      1) a DataFrame, or
      2) a dict[str, DataFrame] containing a 'students' table.
    - The UCI Student dataset commonly used in this project does not include a
      natural student id column. If no valid id column is configured/found,
      a synthetic row id '__row_id__' is generated.
    - The configured outcome column may be provided in raw form (e.g. 'Target').
      After normalization it becomes snake_case (e.g. 'target').
    """
    if isinstance(raw_tables, pd.DataFrame):
        df = raw_tables.copy()
    elif isinstance(raw_tables, dict):
        if "students" not in raw_tables:
            raise KeyError("UCT adapter expects a 'students' table key.")
        df = raw_tables["students"].copy()
    else:
        raise TypeError("raw_tables must be a DataFrame or dict[str, pd.DataFrame].")

    # Normalize all raw column names.
    # Important examples for downstream UCT feature engineering:
    # - "Curricular units 1st sem (approved)" -> "curricular_units_1st_sem_approved"
    # - "Curricular units 2nd sem (enrolled)" -> "curricular_units_2nd_sem_enrolled"
    # - "Curricular units 1st sem (evaluations)" -> "curricular_units_1st_sem_evaluations"
    # - "Curricular units 2nd sem (without evaluations)" -> "curricular_units_2nd_sem_without_evaluations"
    # - "Curricular units 1st sem (grade)" -> "curricular_units_1st_sem_grade"
    rename_map = {col: _to_snake_case(col) for col in df.columns}
    df = df.rename(columns=rename_map)

    # Resolve configured schema fields after normalization.
    configured_target = schema_config.get("outcome_column")
    configured_entity = schema_config.get("entity_id")
    configured_term = schema_config.get("term_column")

    target_column = _to_snake_case(str(configured_target)) if configured_target else None
    id_column = _to_snake_case(str(configured_entity)) if configured_entity else None
    term_column = _to_snake_case(str(configured_term)) if configured_term else None

    # Fallback target inference if config is absent or mismatched.
    if not target_column or target_column not in df.columns:
        target_column = _first_existing(
            df.columns.tolist(),
            ["target", "status", "final_result", "outcome"],
        )

    if not target_column:
        raise ValueError(
            "Unable to infer UCT target column. "
            "Please set schema.outcome_column in dataset config."
        )

    # UCT/UCI often has no natural ID column. Generate one if missing.
    if not id_column or id_column not in df.columns:
        id_column = "__row_id__"
        if id_column not in df.columns:
            df[id_column] = range(len(df))

    # Optional term column.
    if term_column and term_column not in df.columns:
        term_column = None

    numeric_columns = [
        c
        for c in df.columns
        if c not in {target_column, id_column}
        and (term_column is None or c != term_column)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical_columns = [
        c
        for c in df.columns
        if c not in {target_column, id_column}
        and (term_column is None or c != term_column)
        and not pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_columns = numeric_columns + categorical_columns

    return {
        "dataset_name": "uct_student",
        "data": df,
        "dataframe": df,  # compatibility alias
        "id_column": id_column,
        "entity_id": id_column,  # compatibility alias
        "term_column": term_column,
        "target_column": target_column,
        "outcome_column": target_column,  # compatibility alias
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "rename_map": rename_map,
    }
