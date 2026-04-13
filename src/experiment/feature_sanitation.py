from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def validate_and_sanitize_feature_matrix(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    model_name: str,
    feature_stage: str,
    sanitation_cfg: dict[str, Any] | None = None,
    extra_frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    cfg = sanitation_cfg if isinstance(sanitation_cfg, dict) else {"enabled": False}
    frame_payload = {
        "train": X_train.copy(),
        "valid": X_valid.copy(),
        "test": X_test.copy(),
    }
    if isinstance(extra_frames, dict):
        for frame_name, frame in extra_frames.items():
            if isinstance(frame, pd.DataFrame):
                frame_payload[frame_name] = frame.copy()

    report: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "model": model_name,
        "feature_stage": feature_stage,
        "replace_inf": bool(cfg.get("replace_inf", True)),
        "imputation_applied": False,
        "pre_sanitize_nan_count": {},
        "pre_sanitize_inf_count": {},
        "final_finite_check": {},
    }
    if not bool(cfg.get("enabled", False)):
        return frame_payload, report

    processed: dict[str, pd.DataFrame] = {}
    for frame_name, frame in frame_payload.items():
        numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
        inf_mask = np.isinf(numeric_frame.to_numpy(dtype=float))
        report["pre_sanitize_inf_count"][frame_name] = int(inf_mask.sum())
        if bool(cfg.get("replace_inf", True)):
            numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
        report["pre_sanitize_nan_count"][frame_name] = int(numeric_frame.isna().sum().sum())
        processed[frame_name] = numeric_frame

    if bool(cfg.get("impute_missing", True)):
        imputer = SimpleImputer(strategy=str(cfg.get("strategy", "median") or "median"))
        imputer.fit(processed["train"])
        report["imputation_applied"] = True
        for frame_name, frame in processed.items():
            processed[frame_name] = pd.DataFrame(
                imputer.transform(frame),
                columns=frame.columns,
                index=frame.index,
            )

    for frame_name, frame in processed.items():
        finite_ok = bool(np.isfinite(frame.to_numpy(dtype=float)).all())
        report["final_finite_check"][frame_name] = finite_ok
        if not finite_ok and bool(cfg.get("fail_if_non_finite_after_impute", True)):
            raise ValueError(
                f"[{model_name}] non-finite values remain after sanitation at {feature_stage} for split='{frame_name}'."
            )

    print(f"[v8] pre-sanitize NaN count: {report['pre_sanitize_nan_count']} model={model_name} stage={feature_stage}")
    print(f"[v8] pre-sanitize Inf count: {report['pre_sanitize_inf_count']} model={model_name} stage={feature_stage}")
    print(f"[v8] imputation applied: {report['imputation_applied']} model={model_name} stage={feature_stage}")
    print(f"[v8] post-sanitize finite check: {report['final_finite_check']} model={model_name} stage={feature_stage}")
    return processed, report
