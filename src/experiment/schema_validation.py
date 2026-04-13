from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

def align_feature_schema(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    fill_value: float = np.nan,
) -> pd.DataFrame:
    aligned = target_df.copy()
    for col in reference_df.columns:
        if col not in aligned.columns:
            aligned[col] = fill_value
    aligned = aligned.loc[:, list(reference_df.columns)]
    return aligned


def _duplicate_columns(df: pd.DataFrame) -> list[str]:
    counts = pd.Series(df.columns).value_counts()
    return [str(col) for col, count in counts.items() if int(count) > 1]


def validate_feature_schema(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    context: str,
) -> None:
    duplicate_reference = _duplicate_columns(reference_df)
    duplicate_target = _duplicate_columns(target_df)
    if duplicate_reference or duplicate_target:
        raise ValueError(
            f"{context}: duplicate columns detected. "
            f"reference_duplicates={duplicate_reference}, target_duplicates={duplicate_target}"
        )
    ref_cols = list(reference_df.columns)
    tgt_cols = list(target_df.columns)
    if ref_cols == tgt_cols:
        print(f"[v8] schema validation before {context}: passed")
        return
    ref_set = set(ref_cols)
    tgt_set = set(tgt_cols)
    missing = [col for col in ref_cols if col not in tgt_set]
    extra = [col for col in tgt_cols if col not in ref_set]
    mismatched_positions: list[dict[str, Any]] = []
    for idx, (ref_col, tgt_col) in enumerate(zip(ref_cols, tgt_cols)):
        if ref_col != tgt_col:
            mismatched_positions.append({"position": idx, "reference": ref_col, "target": tgt_col})
        if len(mismatched_positions) >= 10:
            break
    raise ValueError(
        f"{context}: feature schema mismatch. "
        f"missing_columns={missing[:10]}, extra_columns={extra[:10]}, "
        f"mismatched_positions={mismatched_positions}"
    )


def _sanitize_lightgbm_feature_name(name: Any) -> str:
    text = str(name).strip()
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "feature"


def _sanitize_lightgbm_feature_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    sanitized_columns: list[str] = []
    mapping: dict[str, str] = {}
    used: dict[str, int] = {}
    for original in list(df.columns):
        base = _sanitize_lightgbm_feature_name(original)
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used[candidate] = 1
        mapping[str(original)] = candidate
        sanitized_columns.append(candidate)
    out = df.copy()
    out.columns = sanitized_columns
    return out, mapping


def _sanitize_lightgbm_feature_frames(
    *,
    frames: dict[str, pd.DataFrame],
    model_name: str,
    stage_name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if model_name != "lightgbm":
        return frames, {"applied": False, "model": model_name, "stage": stage_name, "mapping": {}}
    if not frames:
        return frames, {"applied": False, "model": model_name, "stage": stage_name, "mapping": {}}
    reference_name, reference_df = next(iter(frames.items()))
    reference_columns = list(reference_df.columns)
    _, mapping = _sanitize_lightgbm_feature_names(reference_df)
    sanitized_columns = [mapping[str(original)] for original in reference_columns]
    sanitized_frames: dict[str, pd.DataFrame] = {}
    for frame_name, frame in frames.items():
        validate_feature_schema(reference_df, frame, context=f"{stage_name}:{frame_name}:lightgbm_feature_name_schema")
        renamed = frame.copy()
        renamed.columns = sanitized_columns
        sanitized_frames[frame_name] = renamed
    print(
        "[lightgbm][feature_names] "
        f"model={model_name} stage={stage_name} reference_frame={reference_name} column_count={len(reference_columns)}"
    )
    return sanitized_frames, {
        "applied": True,
        "model": model_name,
        "stage": stage_name,
        "mapping": mapping,
    }


def _log_duplicate_feature_check(df: pd.DataFrame, *, context: str) -> None:
    duplicates = _duplicate_columns(df)
    if duplicates:
        print(f"[v8] duplicate feature check: failed context={context} duplicates={duplicates}")
        raise ValueError(f"{context}: duplicate feature columns detected: {duplicates}")
    print(f"[v8] duplicate feature check: passed context={context}")
