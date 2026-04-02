"""Normalize OULAD raw tables and validate join-key assumptions."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

EXPECTED_TABLE_KEYS: dict[str, set[str]] = {
    "studentinfo": {"id_student", "code_module", "code_presentation", "final_result"},
    "studentregistration": {"id_student", "code_module", "code_presentation"},
    "studentassessment": {"id_student", "id_assessment"},
    "assessments": {"id_assessment", "code_module", "code_presentation"},
    "studentvle": {"id_student", "id_site", "date"},
    "vle": {"id_site", "code_module", "code_presentation"},
    "courses": {"code_module", "code_presentation"},
}


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_").lower()


def _normalize_table_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: _to_snake_case(col) for col in df.columns}
    out = df.rename(columns=rename_map)
    for col in ("code_module", "code_presentation"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def _match_table_key(normalized_name: str) -> str:
    aliases = {
        "student_info": "studentinfo",
        "studentinfo": "studentinfo",
        "student_registration": "studentregistration",
        "studentregistration": "studentregistration",
        "student_assessment": "studentassessment",
        "studentassessment": "studentassessment",
        "student_vle": "studentvle",
        "studentvle": "studentvle",
        "assessments": "assessments",
        "vle": "vle",
        "courses": "courses",
    }
    if normalized_name in aliases:
        return aliases[normalized_name]
    dense_name = _normalize_table_name(normalized_name)
    if dense_name in aliases:
        return aliases[dense_name]
    return dense_name


def adapt_oulad_schema(raw_tables: dict[str, pd.DataFrame], schema_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize OULAD table names/columns and enforce join key availability."""
    if not isinstance(raw_tables, dict):
        raise TypeError("OULAD adapter expects dict[str, DataFrame] input.")

    normalized_tables: dict[str, pd.DataFrame] = {}
    table_aliases: dict[str, str] = {}
    for original_name, df in raw_tables.items():
        key = _match_table_key(original_name)
        normalized_tables[key] = _normalize_dataframe_columns(df.copy())
        table_aliases[original_name] = key

    missing_required_tables = [k for k in EXPECTED_TABLE_KEYS if k not in normalized_tables]
    if missing_required_tables:
        raise ValueError(f"OULAD adapter missing required table(s): {missing_required_tables}")

    key_warnings: list[str] = []
    for table_name, required_cols in EXPECTED_TABLE_KEYS.items():
        columns = set(normalized_tables[table_name].columns)
        missing_cols = sorted(required_cols - columns)
        if missing_cols:
            raise ValueError(f"OULAD table '{table_name}' missing required column(s): {missing_cols}")
        if normalized_tables[table_name].empty:
            key_warnings.append(f"Table '{table_name}' is empty.")

    join_keys = {
        "enrollment": ["id_student", "code_module", "code_presentation"],
        "assessment_bridge": ["id_assessment"],
        "vle_bridge": ["id_site"],
    }
    target_column = str(schema_config.get("outcome_column", "final_result"))

    return {
        "dataset_name": "oulad",
        "tables": normalized_tables,
        "table_aliases": table_aliases,
        "join_keys": join_keys,
        "target_column": target_column,
        "warnings": key_warnings,
    }
