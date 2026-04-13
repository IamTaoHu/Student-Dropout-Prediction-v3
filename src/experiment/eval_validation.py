from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

def _debug_lengths(*, context: str, **named_arrays: Any) -> dict[str, int | None]:
    lengths: dict[str, int | None] = {}
    for name, value in named_arrays.items():
        if value is None:
            lengths[name] = None
            continue
        try:
            lengths[name] = int(len(value))
        except TypeError:
            arr = np.asarray(value)
            lengths[name] = int(arr.shape[0]) if arr.ndim > 0 else None
    print(f"[debug][lengths] context={context} lengths={lengths}")
    return lengths


def _assert_same_length_arrays(*, context: str, **named_arrays: Any) -> dict[str, int | None]:
    lengths = _debug_lengths(context=context, **named_arrays)
    concrete_lengths = {name: value for name, value in lengths.items() if value is not None}
    if concrete_lengths:
        expected = next(iter(concrete_lengths.values()))
        mismatched = {name: value for name, value in concrete_lengths.items() if value != expected}
        if mismatched:
            raise ValueError(f"{context}: inconsistent lengths detected: {concrete_lengths}")
    return lengths


def _assert_1d_label_vector(arr: Any, *, name: str, context: str) -> np.ndarray:
    arr_np = np.asarray(arr)
    print(
        "[debug][label_vector] "
        f"context={context} name={name} python_type={type(arr).__name__} "
        f"numpy_dtype={arr_np.dtype} ndim={arr_np.ndim} shape={arr_np.shape}"
    )
    if arr_np.ndim != 1:
        raise ValueError(
            f"{context}: invalid {name}; expected 1D hard-label vector but got "
            f"python_type={type(arr).__name__}, dtype={arr_np.dtype}, ndim={arr_np.ndim}, shape={arr_np.shape}."
        )
    return arr_np


def _assert_probability_payload(
    arr: Any,
    *,
    name: str,
    context: str,
    expected_rows: int,
) -> np.ndarray:
    arr_np = np.asarray(arr)
    print(
        "[debug][probability_payload] "
        f"context={context} name={name} python_type={type(arr).__name__} "
        f"numpy_dtype={arr_np.dtype} ndim={arr_np.ndim} shape={arr_np.shape}"
    )
    if arr_np.ndim not in {1, 2}:
        raise ValueError(
            f"{context}: invalid {name}; expected probability payload with ndim 1 or 2 but got "
            f"python_type={type(arr).__name__}, dtype={arr_np.dtype}, ndim={arr_np.ndim}, shape={arr_np.shape}."
        )
    row_count = int(arr_np.shape[0]) if arr_np.ndim >= 1 else 0
    if row_count != int(expected_rows):
        raise ValueError(
            f"{context}: invalid {name}; expected_rows={expected_rows} but got "
            f"dtype={arr_np.dtype}, ndim={arr_np.ndim}, shape={arr_np.shape}."
        )
    print(f"[debug][probability_payload_rank] context={context} name={name} rank={'1D' if arr_np.ndim == 1 else '2D'}")
    return arr_np


def _validate_two_stage_eval_bundle(
    *,
    y_true: pd.Series | np.ndarray | list[Any],
    y_pred: np.ndarray | list[Any],
    y_proba: np.ndarray | list[list[float]] | None = None,
    sample_weight: np.ndarray | list[float] | None = None,
    ids: pd.Series | np.ndarray | list[Any] | None = None,
    split_name: str,
    model_name: str,
) -> dict[str, Any]:
    context = f"[{model_name}] split_semantics='{split_name}'"
    y_true_arr = _assert_1d_label_vector(y_true, name="y_true", context=context)
    y_pred_arr = _assert_1d_label_vector(y_pred, name="y_pred", context=context)
    y_proba_arr = (
        None
        if y_proba is None
        else _assert_probability_payload(
            y_proba,
            name="y_proba",
            context=context,
            expected_rows=int(y_true_arr.shape[0]),
        )
    )
    sample_weight_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    ids_arr = None if ids is None else np.asarray(ids, dtype=object)
    lengths = _assert_same_length_arrays(
        context=context,
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        sample_weight=sample_weight_arr,
        ids=ids_arr,
    )
    return {
        "y_true": pd.Series(y_true_arr),
        "y_pred": np.asarray(y_pred_arr),
        "y_proba": y_proba_arr,
        "sample_weight": sample_weight_arr,
        "ids": ids_arr,
        "lengths": lengths,
    }
