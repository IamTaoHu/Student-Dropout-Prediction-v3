"""Stratified split helpers for reproducible train/valid/test creation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic stratified splitting."""

    test_size: float
    validation_size: float
    random_state: int
    stratify_column: str


def _validate_split_config(config: SplitConfig) -> None:
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be in (0, 1).")
    if not 0 <= config.validation_size < 1:
        raise ValueError("validation_size must be in [0, 1).")
    if config.test_size + config.validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1.")


def stratified_train_valid_test_split(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    """Split a labeled DataFrame into train/valid/test while preserving class ratios."""
    _validate_split_config(config)
    if config.stratify_column not in df.columns:
        raise KeyError(f"Stratify column '{config.stratify_column}' not found in DataFrame.")

    stratify_values = df[config.stratify_column]
    train_valid_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_values,
    )

    if config.validation_size == 0:
        return {
            "train": train_valid_df.reset_index(drop=True),
            "valid": pd.DataFrame(columns=df.columns),
            "test": test_df.reset_index(drop=True),
        }

    valid_relative = config.validation_size / (1 - config.test_size)
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_relative,
        random_state=config.random_state,
        stratify=train_valid_df[config.stratify_column],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
