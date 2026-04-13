"""Minimal paper-style feature builder for the UCI Student dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


UCI_PAPER_STYLE_BASE_COLUMNS: tuple[str, ...] = (
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "previous_qualification_grade",
    "nacionality",
    "mother_s_qualification",
    "father_s_qualification",
    "mother_s_occupation",
    "father_s_occupation",
    "admission_grade",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "age_at_enrollment",
    "international",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
)

UCI_PAPER_STYLE_NUMERIC_COLUMNS: tuple[str, ...] = (
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
)

UCI_PAPER_STYLE_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "marital_status",
    "application_mode",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "nacionality",
    "mother_s_qualification",
    "father_s_qualification",
    "mother_s_occupation",
    "father_s_occupation",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
)


def _normalize_target(series: pd.Series) -> pd.Series:
    values = series.astype(object)
    values = values.where(pd.notna(values), np.nan)
    values = values.map(lambda value: value.strip() if isinstance(value, str) else value)
    values = values.replace({"": np.nan})
    return values.astype(object)


def _normalize_categorical(series: pd.Series) -> pd.Series:
    values = series.astype(object)
    values = values.where(pd.notna(values), np.nan)
    values = values.map(lambda value: value.strip() if isinstance(value, str) else value)
    values = values.replace({"": np.nan})
    return values.astype(object)


def build_uci_student_paper_style_features(
    adapted: dict[str, Any] | pd.DataFrame,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    """Keep only base UCI columns with minimal cleaning and explicit typing."""
    if isinstance(adapted, dict):
        if "data" not in adapted:
            raise KeyError("UCI paper-style feature builder expects adapted schema with 'data' key.")
        df = adapted["data"].copy()
        id_column = adapted.get("id_column")
        target_column = adapted.get("target_column")
    else:
        df = adapted.copy()
        id_column = feature_config.get("id_column")
        target_column = feature_config.get("target_column")

    if target_column and target_column in df.columns:
        df[target_column] = _normalize_target(df[target_column])

    available_base_columns = [column for column in UCI_PAPER_STYLE_BASE_COLUMNS if column in df.columns]
    available_numeric_columns = [column for column in UCI_PAPER_STYLE_NUMERIC_COLUMNS if column in df.columns]
    available_categorical_columns = [
        column for column in UCI_PAPER_STYLE_CATEGORICAL_COLUMNS if column in df.columns
    ]

    selected_columns: list[str] = []
    for column in (id_column, target_column):
        if column and column in df.columns and column not in selected_columns:
            selected_columns.append(column)
    selected_columns.extend([column for column in available_base_columns if column not in selected_columns])

    result = df.loc[:, selected_columns].copy()

    for column in available_numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].astype(float)

    for column in available_categorical_columns:
        result[column] = _normalize_categorical(result[column])
        result[column] = result[column].where(pd.notna(result[column]), np.nan)
        result[column] = result[column].astype(object)

    result = result.where(pd.notna(result), np.nan)

    return result
