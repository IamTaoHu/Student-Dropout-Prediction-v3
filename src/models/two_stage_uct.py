"""Two-stage UCT 3-class classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TwoStageUct3ClassClassifier:
    """Wrap two binary models into a single 3-class predictor."""

    stage1_model: Any
    stage2_model: Any
    dropout_label: int
    enrolled_label: int
    graduate_label: int
    decision_mode: str = "soft_fused"
    threshold_stage1: float = 0.5
    stage1_positive_label: int = 1
    stage2_positive_label: int = 1
    class_thresholds: dict[int, float] | None = None

    def __post_init__(self) -> None:
        mode = str(self.decision_mode).strip().lower()
        if mode not in {"soft_fused", "hard_stage1"}:
            raise ValueError("decision_mode must be one of: soft_fused, hard_stage1.")
        self.decision_mode = mode
        if self.threshold_stage1 < 0.0 or self.threshold_stage1 > 1.0:
            raise ValueError("threshold_stage1 must be within [0.0, 1.0].")

        self.classes_ = np.array(
            [int(self.dropout_label), int(self.enrolled_label), int(self.graduate_label)],
            dtype=int,
        )
        self._class_index = {int(label): idx for idx, label in enumerate(self.classes_)}
        self._class_threshold_vector = self._build_class_threshold_vector(self.class_thresholds)
        # Explainability/figure fallback hook.
        self.explainability_model = self.stage2_model

    def _build_class_threshold_vector(self, class_thresholds: dict[int, float] | None) -> np.ndarray:
        if not class_thresholds:
            return np.zeros(len(self.classes_), dtype=float)
        out = np.zeros(len(self.classes_), dtype=float)
        for raw_label, raw_threshold in class_thresholds.items():
            label = int(raw_label)
            if label not in self._class_index:
                continue
            threshold = float(raw_threshold)
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("class_thresholds values must be within [0.0, 1.0].")
            out[self._class_index[label]] = threshold
        return out

    @staticmethod
    def _as_dataframe(X: pd.DataFrame | np.ndarray | list[list[float]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape={arr.shape}.")
        return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])

    @staticmethod
    def _resolve_probability_column(
        proba: np.ndarray,
        classes: np.ndarray | None,
        positive_label: int,
    ) -> np.ndarray:
        if proba.ndim != 2:
            raise ValueError(f"Binary probability output must be 2D, got shape={proba.shape}.")
        if proba.shape[1] < 2:
            raise ValueError(f"Binary probability output must have at least 2 columns, got shape={proba.shape}.")
        if classes is None:
            idx = 1
            return proba[:, idx]

        normalized_classes = [str(v) for v in np.asarray(classes).tolist()]
        token = str(int(positive_label))
        if token in normalized_classes:
            return proba[:, normalized_classes.index(token)]

        # Defensive fallback for estimators that do not expose classes_ in a compatible dtype.
        return proba[:, 1]

    def _stage1_dropout_probability(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(self.stage1_model.predict_proba(X), dtype=float)
        classes = getattr(self.stage1_model, "classes_", None)
        p_dropout = self._resolve_probability_column(
            proba=proba,
            classes=classes,
            positive_label=int(self.stage1_positive_label),
        )
        return np.clip(p_dropout, 0.0, 1.0)

    def _stage2_enrolled_graduate_probability(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        proba = np.asarray(self.stage2_model.predict_proba(X), dtype=float)
        classes = getattr(self.stage2_model, "classes_", None)
        p_graduate_given_non_dropout = self._resolve_probability_column(
            proba=proba,
            classes=classes,
            positive_label=int(self.stage2_positive_label),
        )
        p_graduate_given_non_dropout = np.clip(p_graduate_given_non_dropout, 0.0, 1.0)
        p_enrolled_given_non_dropout = np.clip(1.0 - p_graduate_given_non_dropout, 0.0, 1.0)
        return p_enrolled_given_non_dropout, p_graduate_given_non_dropout

    def _fused_probabilities(self, X_df: pd.DataFrame) -> np.ndarray:
        """Compute fused 3-class probabilities in explicit class order.

        Class order is always: [dropout, enrolled, graduate].
        """
        p_dropout = self._stage1_dropout_probability(X_df)
        p_non_dropout = np.clip(1.0 - p_dropout, 0.0, 1.0)
        p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)

        p_dropout_final = p_dropout
        p_enrolled_final = p_non_dropout * p_enrolled_given_non_dropout
        p_graduate_final = p_non_dropout * p_graduate_given_non_dropout

        out = np.column_stack([p_dropout_final, p_enrolled_final, p_graduate_final]).astype(float)
        if out.ndim != 2 or out.shape[1] != 3:
            raise ValueError(f"Expected fused probabilities with shape (n, 3), got {out.shape}.")
        out = np.clip(out, 0.0, 1.0)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        normalized = out / row_sums
        # Final defensive normalization so each row sums exactly to 1.0.
        normalized[:, -1] = 1.0 - normalized[:, :-1].sum(axis=1)
        normalized = np.clip(normalized, 0.0, 1.0)
        row_sums = normalized.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        return normalized / row_sums

    def predict_proba(self, X: pd.DataFrame | np.ndarray | list[list[float]]) -> np.ndarray:
        X_df = self._as_dataframe(X)
        return self._fused_probabilities(X_df)

    def _predict_soft_thresholded(self, fused_proba: np.ndarray) -> np.ndarray:
        return self.predict_from_fused_probabilities(
            fused_proba=fused_proba,
            classes=self.classes_,
            thresholds=self._class_threshold_vector,
        )

    @staticmethod
    def predict_from_fused_probabilities(
        fused_proba: np.ndarray,
        classes: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        if fused_proba.ndim != 2:
            raise ValueError(f"Expected fused_proba with 2 dimensions, got shape={fused_proba.shape}.")
        classes_arr = np.asarray(classes, dtype=int)
        if classes_arr.ndim != 1 or classes_arr.size != fused_proba.shape[1]:
            raise ValueError("classes must be 1D and match fused_proba column count.")

        if thresholds is None:
            threshold_vec = np.zeros(fused_proba.shape[1], dtype=float)
        else:
            threshold_vec = np.asarray(thresholds, dtype=float).reshape(-1)
            if threshold_vec.size != fused_proba.shape[1]:
                raise ValueError("thresholds must have one value per class.")
            if np.any(threshold_vec < 0.0) or np.any(threshold_vec > 1.0):
                raise ValueError("thresholds must be within [0.0, 1.0].")

        # Fast path: argmax when thresholds are effectively disabled.
        if np.allclose(threshold_vec, 0.0):
            pred_idx = np.argmax(fused_proba, axis=1)
            return classes_arr[pred_idx]

        preds = np.empty(fused_proba.shape[0], dtype=int)
        for row_idx in range(fused_proba.shape[0]):
            row = fused_proba[row_idx]
            eligible = np.flatnonzero(row >= threshold_vec)
            if eligible.size > 0:
                # Deterministic tie-break via stable class order.
                local_best = eligible[np.argmax(row[eligible])]
                preds[row_idx] = int(classes_arr[local_best])
            else:
                preds[row_idx] = int(classes_arr[int(np.argmax(row))])
        return preds

    def predict(self, X: pd.DataFrame | np.ndarray | list[list[float]]) -> np.ndarray:
        X_df = self._as_dataframe(X)
        if self.decision_mode == "hard_stage1":
            p_dropout = self._stage1_dropout_probability(X_df)
            p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
            mapped_stage2 = np.where(
                p_graduate_given_non_dropout >= p_enrolled_given_non_dropout,
                int(self.graduate_label),
                int(self.enrolled_label),
            )
            return np.where(
                p_dropout >= float(self.threshold_stage1),
                int(self.dropout_label),
                mapped_stage2,
            ).astype(int)

        fused_proba = self.predict_proba(X_df)
        return self._predict_soft_thresholded(fused_proba).astype(int)
