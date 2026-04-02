"""Load raw UCT Student tabular data from config-driven paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_path(path_value: str | Path, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def load_uct_student_dataframe(dataset_config: dict[str, Any], base_dir: Path | None = None) -> pd.DataFrame:
    """Load UCT Student CSV as a single DataFrame with basic validation."""
    paths_cfg = dataset_config.get("paths", {})
    csv_path = paths_cfg.get("raw_file")
    if not csv_path:
        raw_root = paths_cfg.get("raw_root")
        default_file = dataset_config.get("source", {}).get("filename", "uct_student.csv")
        csv_path = str(Path(raw_root) / default_file) if raw_root else default_file

    delimiter = str(dataset_config.get("source", {}).get("delimiter", ","))
    encoding = str(dataset_config.get("source", {}).get("encoding", "utf-8"))
    target_col = dataset_config.get("schema", {}).get("outcome_column")
    if not target_col:
        raise ValueError("UCT dataset config must define schema.outcome_column.")

    resolved_path = _resolve_path(csv_path, base_dir)
    if not resolved_path.exists():
        raise FileNotFoundError(f"UCT Student raw file not found: {resolved_path}")

    df = pd.read_csv(resolved_path, sep=delimiter, encoding=encoding)
    if target_col not in df.columns:
        raise ValueError(
            f"UCT Student target column '{target_col}' missing from raw file. "
            f"Columns available: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("UCT Student raw dataset is empty.")
    return df


def load_uct_student_tables(dataset_config: dict[str, Any], base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Backward-compatible wrapper returning UCT data under a logical table key."""
    return {"students": load_uct_student_dataframe(dataset_config, base_dir=base_dir)}
