"""Load raw OULAD tables from config-driven paths with key validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OULAD_TABLES: dict[str, str] = {
    "assessments": "assessments.csv",
    "vle": "vle.csv",
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentAssessment": "studentAssessment.csv",
    "studentVle": "studentVle.csv",
    "courses": "courses.csv",
}


def _resolve_path(path_value: str | Path, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return (base_dir / path).resolve()


def load_oulad_tables(dataset_config: dict[str, Any], base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all required OULAD source tables and validate required keys."""
    paths_cfg = dataset_config.get("paths", {})
    raw_root = _resolve_path(paths_cfg.get("raw_root", "data/raw/oulad"), base_dir)
    source_cfg = dataset_config.get("source_tables", {})

    if isinstance(source_cfg, list):
        table_to_file = {name: f"{name}.csv" for name in source_cfg}
    elif isinstance(source_cfg, dict):
        table_to_file = {str(k): str(v) for k, v in source_cfg.items()}
    else:
        table_to_file = DEFAULT_OULAD_TABLES.copy()

    required_tables = dataset_config.get("required_tables", list(DEFAULT_OULAD_TABLES.keys()))
    missing_required = [tbl for tbl in required_tables if tbl not in table_to_file]
    if missing_required:
        raise ValueError(f"OULAD config missing required table mapping(s): {missing_required}")

    delimiter = str(dataset_config.get("source", {}).get("delimiter", ","))
    encoding = str(dataset_config.get("source", {}).get("encoding", "utf-8"))

    loaded: dict[str, pd.DataFrame] = {}
    for table_name, filename in table_to_file.items():
        table_path = _resolve_path(raw_root / filename, base_dir=None)
        if not table_path.exists():
            raise FileNotFoundError(f"OULAD source table '{table_name}' not found at: {table_path}")
        loaded[table_name] = pd.read_csv(table_path, sep=delimiter, encoding=encoding)

    missing_loaded = [tbl for tbl in required_tables if tbl not in loaded]
    if missing_loaded:
        raise ValueError(f"Failed to load required OULAD table(s): {missing_loaded}")

    return loaded
