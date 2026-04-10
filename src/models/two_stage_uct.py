"""Two-stage UCT 3-class classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Stage2PositiveProbabilityCalibrator:
    """Post-hoc binary probability transform for Stage 2 positive-class probabilities."""

    method: str = "none"
    payload: dict[str, Any] | None = None

    def transform(self, probabilities: np.ndarray | list[float]) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
        payload = self.payload if isinstance(self.payload, dict) else {}
        method = str(self.method).strip().lower()
        if method in {"", "none"}:
            return p
        if method == "temperature_scaling":
            temperature = max(float(payload.get("temperature", 1.0)), 1.0e-6)
            logits = np.log(p / (1.0 - p))
            return 1.0 / (1.0 + np.exp(-(logits / temperature)))
        if method == "sigmoid":
            coef = float(payload.get("coef", 1.0))
            intercept = float(payload.get("intercept", 0.0))
            logits = np.log(p / (1.0 - p))
            return 1.0 / (1.0 + np.exp(-((coef * logits) + intercept)))
        if method == "isotonic":
            thresholds = np.asarray(payload.get("x_thresholds", []), dtype=float)
            values = np.asarray(payload.get("y_thresholds", []), dtype=float)
            if thresholds.size == 0 or values.size == 0 or thresholds.size != values.size:
                return p
            return np.interp(p, thresholds, values, left=values[0], right=values[-1])
        return p


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
    threshold_stage1_low: float | None = None
    threshold_stage1_high: float | None = None
    middle_band_enabled: bool = False
    middle_band_behavior: str = "force_stage2_soft_fusion"
    stage1_positive_label: int = 1
    stage2_positive_label: int = 1
    stage2_positive_target_label: int | None = None
    class_thresholds: dict[int, float] | None = None
    stage2_decision_config: dict[str, Any] | None = None
    stage2_probability_calibrator: Any | None = None
    stage1_feature_columns: list[str] | None = None
    stage2_feature_columns: list[str] | None = None

    def __post_init__(self) -> None:
        mode = str(self.decision_mode).strip().lower()
        alias_map = {
            "soft_fused": "soft_fused",
            "hard_stage1": "hard_routing",
            "hard_routing": "hard_routing",
            "soft_fusion": "soft_fusion",
            "soft_fusion_with_dropout_threshold": "soft_fusion_with_dropout_threshold",
            "soft_fusion_with_middle_band": "soft_fusion_with_middle_band",
            "pure_soft_argmax": "pure_soft_argmax",
        }
        if mode not in alias_map:
            raise ValueError(
                "decision_mode must be one of: soft_fused, hard_stage1, hard_routing, "
                "soft_fusion, soft_fusion_with_dropout_threshold, soft_fusion_with_middle_band, pure_soft_argmax."
            )
        mode = alias_map[mode]
        self.decision_mode = mode
        if self.threshold_stage1 < 0.0 or self.threshold_stage1 > 1.0:
            raise ValueError("threshold_stage1 must be within [0.0, 1.0].")
        if self.threshold_stage1_low is None:
            self.threshold_stage1_low = float(self.threshold_stage1)
        if self.threshold_stage1_high is None:
            self.threshold_stage1_high = float(self.threshold_stage1)
        self.threshold_stage1_low = float(self.threshold_stage1_low)
        self.threshold_stage1_high = float(self.threshold_stage1_high)
        if self.threshold_stage1_low < 0.0 or self.threshold_stage1_low > 1.0:
            raise ValueError("threshold_stage1_low must be within [0.0, 1.0].")
        if self.threshold_stage1_high < 0.0 or self.threshold_stage1_high > 1.0:
            raise ValueError("threshold_stage1_high must be within [0.0, 1.0].")
        if self.threshold_stage1_low > self.threshold_stage1_high:
            raise ValueError("threshold_stage1_low must be <= threshold_stage1_high.")

        self.classes_ = np.array(
            [int(self.dropout_label), int(self.enrolled_label), int(self.graduate_label)],
            dtype=int,
        )
        if self.stage2_positive_target_label is None:
            self.stage2_positive_target_label = int(self.graduate_label)
        self.stage2_positive_target_label = int(self.stage2_positive_target_label)
        if self.stage2_positive_target_label not in {int(self.enrolled_label), int(self.graduate_label)}:
            raise ValueError("stage2_positive_target_label must match enrolled_label or graduate_label.")
        self._class_index = {int(label): idx for idx, label in enumerate(self.classes_)}
        self._class_threshold_vector = self._build_class_threshold_vector(self.class_thresholds)
        self.stage2_decision_config = self._normalize_stage2_decision_config(self.stage2_decision_config)
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
    def _normalize_stage2_decision_config(stage2_decision_config: dict[str, Any] | None) -> dict[str, Any]:
        raw_cfg = stage2_decision_config if isinstance(stage2_decision_config, dict) else {}
        strategy = str(raw_cfg.get("strategy", "argmax")).strip().lower()
        enabled = bool(raw_cfg.get("enabled", False)) and strategy == "enrolled_guarded_threshold"
        threshold = float(raw_cfg.get("enrolled_probability_threshold", 0.5))
        margin_guard = float(raw_cfg.get("graduate_margin_guard", 0.0))
        enrolled_margin = raw_cfg.get("enrolled_margin")
        if enrolled_margin is None:
            enrolled_margin = -float(margin_guard)
        enrolled_margin = float(enrolled_margin)
        dropout_probability_guard = raw_cfg.get("dropout_probability_guard")
        if dropout_probability_guard is None:
            dropout_probability_guard = 1.0
        dropout_probability_guard = float(dropout_probability_guard)
        threshold = min(max(threshold, 0.0), 1.0)
        margin_guard = min(max(margin_guard, 0.0), 1.0)
        enrolled_margin = min(max(enrolled_margin, -1.0), 1.0)
        dropout_probability_guard = min(max(dropout_probability_guard, 0.0), 1.0)
        normalized = dict(raw_cfg)
        normalized.update(
            {
                "enabled": enabled,
                "strategy": strategy if enabled else "argmax",
                "enrolled_probability_threshold": threshold,
                "graduate_margin_guard": margin_guard,
                "enrolled_margin": enrolled_margin,
                "dropout_probability_guard": dropout_probability_guard,
            }
        )
        return normalized

    @staticmethod
    def _as_dataframe(X: pd.DataFrame | np.ndarray | list[list[float]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape={arr.shape}.")
        return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])

    @staticmethod
    def _select_stage_features(
        X: pd.DataFrame,
        feature_columns: list[str] | None,
        *,
        stage_name: str,
    ) -> pd.DataFrame:
        if not feature_columns:
            return X
        missing = [col for col in feature_columns if col not in X.columns]
        if missing:
            preview = missing[:10]
            raise ValueError(
                f"{stage_name} missing required feature columns: {preview}"
                + (" ..." if len(missing) > 10 else "")
            )
        return X.loc[:, feature_columns]

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
        X_stage1 = self._select_stage_features(X, self.stage1_feature_columns, stage_name="stage1")
        proba = np.asarray(self.stage1_model.predict_proba(X_stage1), dtype=float)
        classes = getattr(self.stage1_model, "classes_", None)
        p_dropout = self._resolve_probability_column(
            proba=proba,
            classes=classes,
            positive_label=int(self.stage1_positive_label),
        )
        return np.clip(p_dropout, 0.0, 1.0)

    def _stage2_enrolled_graduate_probability(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_stage2 = self._select_stage_features(X, self.stage2_feature_columns, stage_name="stage2")
        proba = np.asarray(self.stage2_model.predict_proba(X_stage2), dtype=float)
        classes = getattr(self.stage2_model, "classes_", None)
        p_positive_given_non_dropout = self._resolve_probability_column(
            proba=proba,
            classes=classes,
            positive_label=int(self.stage2_positive_label),
        )
        p_positive_given_non_dropout = np.clip(p_positive_given_non_dropout, 0.0, 1.0)
        if self.stage2_probability_calibrator is not None:
            p_positive_given_non_dropout = np.asarray(
                self.stage2_probability_calibrator.transform(p_positive_given_non_dropout),
                dtype=float,
            )
            p_positive_given_non_dropout = np.clip(p_positive_given_non_dropout, 0.0, 1.0)
        p_negative_given_non_dropout = np.clip(1.0 - p_positive_given_non_dropout, 0.0, 1.0)
        if int(self.stage2_positive_target_label) == int(self.enrolled_label):
            p_enrolled_given_non_dropout = p_positive_given_non_dropout
            p_graduate_given_non_dropout = p_negative_given_non_dropout
        else:
            p_enrolled_given_non_dropout = p_negative_given_non_dropout
            p_graduate_given_non_dropout = p_positive_given_non_dropout
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

    def predict_stage_probabilities(
        self,
        X: pd.DataFrame | np.ndarray | list[list[float]],
    ) -> dict[str, np.ndarray]:
        X_df = self._as_dataframe(X)
        p_dropout = self._stage1_dropout_probability(X_df)
        p_non_dropout = np.clip(1.0 - p_dropout, 0.0, 1.0)
        p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
        return {
            "stage1_prob_dropout": p_dropout,
            "stage1_prob_non_dropout": p_non_dropout,
            "stage2_prob_enrolled": p_enrolled_given_non_dropout,
            "stage2_prob_graduate": p_graduate_given_non_dropout,
        }

    @staticmethod
    def apply_stage2_decision_policy(
        base_predictions: np.ndarray,
        *,
        p_enrolled_given_non_dropout: np.ndarray | None,
        p_graduate_given_non_dropout: np.ndarray | None,
        p_dropout: np.ndarray | None = None,
        dropout_label: int,
        enrolled_label: int,
        graduate_label: int,
        stage2_decision_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        preds = np.asarray(base_predictions, dtype=int).copy()
        reasons = np.full(preds.shape[0], "argmax", dtype=object)
        cfg = TwoStageUct3ClassClassifier._normalize_stage2_decision_config(stage2_decision_config)
        if not bool(cfg.get("enabled", False)):
            reasons[preds == int(dropout_label)] = "dropout_preserved"
            return preds, reasons.astype(str)

        if p_enrolled_given_non_dropout is None or p_graduate_given_non_dropout is None:
            reasons[preds == int(dropout_label)] = "dropout_preserved"
            reasons[preds != int(dropout_label)] = "argmax_fallback_no_stage2_probabilities"
            return preds, reasons.astype(str)

        p_enrolled = np.asarray(p_enrolled_given_non_dropout, dtype=float)
        p_graduate = np.asarray(p_graduate_given_non_dropout, dtype=float)
        if p_enrolled.shape != preds.shape or p_graduate.shape != preds.shape:
            reasons[preds == int(dropout_label)] = "dropout_preserved"
            reasons[preds != int(dropout_label)] = "argmax_fallback_invalid_stage2_probability_shape"
            return preds, reasons.astype(str)

        threshold = float(cfg.get("enrolled_probability_threshold", 0.5))
        margin_guard = float(cfg.get("graduate_margin_guard", 0.0))
        enrolled_margin = float(cfg.get("enrolled_margin", -margin_guard))
        dropout_probability_guard = float(cfg.get("dropout_probability_guard", 1.0))
        non_dropout_mask = preds != int(dropout_label)
        dropout_guard_mask = np.ones(preds.shape[0], dtype=bool)
        if p_dropout is not None:
            p_dropout_arr = np.asarray(p_dropout, dtype=float)
            if p_dropout_arr.shape == preds.shape:
                dropout_guard_mask = p_dropout_arr <= dropout_probability_guard
        enrolled_mask = (
            non_dropout_mask
            & (p_enrolled >= threshold)
            & ((p_graduate - p_enrolled) <= margin_guard)
            & ((p_enrolled - p_graduate) >= enrolled_margin)
            & dropout_guard_mask
        )
        graduate_mask = non_dropout_mask & (~enrolled_mask)

        preds[enrolled_mask] = int(enrolled_label)
        preds[graduate_mask] = int(graduate_label)
        reasons[preds == int(dropout_label)] = "dropout_preserved"
        graduate_reason = np.full(int(np.sum(graduate_mask)), "graduate_preserved_by_margin_guard", dtype=object)
        if np.any(graduate_mask):
            graduate_reason = np.where(
                p_enrolled[graduate_mask] < threshold,
                "graduate_preserved_below_enrolled_threshold",
                graduate_reason,
            )
            graduate_reason = np.where(
                (p_enrolled[graduate_mask] >= threshold) & ((p_enrolled[graduate_mask] - p_graduate[graduate_mask]) < enrolled_margin),
                "graduate_preserved_below_enrolled_margin",
                graduate_reason,
            )
            if p_dropout is not None:
                p_dropout_arr = np.asarray(p_dropout, dtype=float)
                if p_dropout_arr.shape == preds.shape:
                    graduate_reason = np.where(
                        (p_enrolled[graduate_mask] >= threshold)
                        & (p_dropout_arr[graduate_mask] > dropout_probability_guard),
                        "graduate_preserved_by_dropout_guard",
                        graduate_reason,
                    )
        reasons[graduate_mask] = graduate_reason
        reasons[enrolled_mask] = np.where(
            p_graduate[enrolled_mask] > p_enrolled[enrolled_mask],
            "enrolled_selected_within_margin_guard",
            "enrolled_selected_by_threshold",
        )
        return preds.astype(int), reasons.astype(str)

    def _predict_soft_thresholded(self, fused_proba: np.ndarray) -> np.ndarray:
        return self.predict_from_fused_probabilities(
            fused_proba=fused_proba,
            classes=self.classes_,
            thresholds=self._class_threshold_vector,
        )

    @staticmethod
    def predict_with_middle_band_from_fused_probabilities(
        fused_proba: np.ndarray,
        classes: np.ndarray,
        *,
        dropout_label: int,
        enrolled_label: int,
        graduate_label: int,
        low_threshold: float,
        high_threshold: float,
        p_enrolled_given_non_dropout: np.ndarray | None = None,
        p_graduate_given_non_dropout: np.ndarray | None = None,
        stage2_decision_config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if low_threshold < 0.0 or low_threshold > 1.0:
            raise ValueError("low_threshold must be within [0.0, 1.0].")
        if high_threshold < 0.0 or high_threshold > 1.0:
            raise ValueError("high_threshold must be within [0.0, 1.0].")
        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be < high_threshold for middle-band routing.")

        classes_arr = np.asarray(classes, dtype=int)
        class_to_index = {int(label): idx for idx, label in enumerate(classes_arr.tolist())}
        try:
            dropout_idx = class_to_index[int(dropout_label)]
            enrolled_idx = class_to_index[int(enrolled_label)]
            graduate_idx = class_to_index[int(graduate_label)]
        except KeyError as exc:
            raise ValueError("classes must include dropout, enrolled, and graduate labels.") from exc

        preds = np.empty(fused_proba.shape[0], dtype=int)
        regions = np.empty(fused_proba.shape[0], dtype=object)
        p_dropout = fused_proba[:, dropout_idx]
        p_enrolled = fused_proba[:, enrolled_idx]
        p_graduate = fused_proba[:, graduate_idx]

        hard_dropout_mask = p_dropout >= float(high_threshold)
        safe_non_dropout_mask = p_dropout <= float(low_threshold)
        middle_band_mask = (~hard_dropout_mask) & (~safe_non_dropout_mask)

        preds[hard_dropout_mask] = int(dropout_label)
        regions[hard_dropout_mask] = "hard_dropout"

        preds[safe_non_dropout_mask] = np.where(
            p_enrolled[safe_non_dropout_mask] >= p_graduate[safe_non_dropout_mask],
            int(enrolled_label),
            int(graduate_label),
        )
        regions[safe_non_dropout_mask] = "safe_non_dropout"

        if np.any(middle_band_mask):
            middle_pred_idx = np.argmax(fused_proba[middle_band_mask], axis=1)
            preds[middle_band_mask] = classes_arr[middle_pred_idx]
            regions[middle_band_mask] = "middle_band"
        preds, _ = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            preds,
            p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
            p_graduate_given_non_dropout=p_graduate_given_non_dropout,
            dropout_label=int(dropout_label),
            enrolled_label=int(enrolled_label),
            graduate_label=int(graduate_label),
            stage2_decision_config=stage2_decision_config,
        )
        return preds, regions.astype(str)

    @staticmethod
    def predict_with_dropout_threshold_from_fused_probabilities(
        fused_proba: np.ndarray,
        classes: np.ndarray,
        *,
        dropout_label: int,
        enrolled_label: int,
        graduate_label: int,
        dropout_threshold: float,
        p_enrolled_given_non_dropout: np.ndarray | None = None,
        p_graduate_given_non_dropout: np.ndarray | None = None,
        stage2_decision_config: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if fused_proba.ndim != 2:
            raise ValueError(f"Expected fused_proba with 2 dimensions, got shape={fused_proba.shape}.")
        if fused_proba.shape[1] != 3:
            raise ValueError("Dropout-threshold prediction requires 3 fused probability columns.")
        if dropout_threshold < 0.0 or dropout_threshold > 1.0:
            raise ValueError("dropout_threshold must be within [0.0, 1.0].")

        classes_arr = np.asarray(classes, dtype=int)
        class_to_index = {int(label): idx for idx, label in enumerate(classes_arr.tolist())}
        try:
            dropout_idx = class_to_index[int(dropout_label)]
            enrolled_idx = class_to_index[int(enrolled_label)]
            graduate_idx = class_to_index[int(graduate_label)]
        except KeyError as exc:
            raise ValueError("classes must include dropout, enrolled, and graduate labels.") from exc

        preds = np.empty(fused_proba.shape[0], dtype=int)
        p_dropout = fused_proba[:, dropout_idx]
        p_enrolled = fused_proba[:, enrolled_idx]
        p_graduate = fused_proba[:, graduate_idx]
        # Threshold override is deliberate: gate dropout first, then choose within non-dropout classes.
        dropout_mask = p_dropout >= float(dropout_threshold)
        preds[dropout_mask] = int(dropout_label)
        preds[~dropout_mask] = np.where(
            p_enrolled[~dropout_mask] >= p_graduate[~dropout_mask],
            int(enrolled_label),
            int(graduate_label),
        )
        preds, _ = TwoStageUct3ClassClassifier.apply_stage2_decision_policy(
            preds,
            p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
            p_graduate_given_non_dropout=p_graduate_given_non_dropout,
            dropout_label=int(dropout_label),
            enrolled_label=int(enrolled_label),
            graduate_label=int(graduate_label),
            stage2_decision_config=stage2_decision_config,
        )
        return preds

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
        if self.decision_mode == "hard_routing":
            p_dropout = self._stage1_dropout_probability(X_df)
            p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
            mapped_stage2 = np.where(
                p_graduate_given_non_dropout >= p_enrolled_given_non_dropout,
                int(self.graduate_label),
                int(self.enrolled_label),
            )
            pred = np.where(
                p_dropout >= float(self.threshold_stage1),
                int(self.dropout_label),
                mapped_stage2,
            ).astype(int)
            pred, _ = self.apply_stage2_decision_policy(
                pred,
                p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
                p_graduate_given_non_dropout=p_graduate_given_non_dropout,
                dropout_label=int(self.dropout_label),
                enrolled_label=int(self.enrolled_label),
                graduate_label=int(self.graduate_label),
                stage2_decision_config=self.stage2_decision_config,
            )
            return pred.astype(int)

        fused_proba = self.predict_proba(X_df)
        if self.decision_mode in {"soft_fusion", "pure_soft_argmax"}:
            pred_idx = np.argmax(fused_proba, axis=1)
            pred = self.classes_[pred_idx].astype(int)
            p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
            pred, _ = self.apply_stage2_decision_policy(
                pred,
                p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
                p_graduate_given_non_dropout=p_graduate_given_non_dropout,
                dropout_label=int(self.dropout_label),
                enrolled_label=int(self.enrolled_label),
                graduate_label=int(self.graduate_label),
                stage2_decision_config=self.stage2_decision_config,
            )
            return pred.astype(int)
        if self.decision_mode == "soft_fusion_with_dropout_threshold":
            p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
            return self.predict_with_dropout_threshold_from_fused_probabilities(
                fused_proba=fused_proba,
                classes=self.classes_,
                dropout_label=int(self.dropout_label),
                enrolled_label=int(self.enrolled_label),
                graduate_label=int(self.graduate_label),
                dropout_threshold=float(self.threshold_stage1),
                p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
                p_graduate_given_non_dropout=p_graduate_given_non_dropout,
                stage2_decision_config=self.stage2_decision_config,
            ).astype(int)
        if self.decision_mode == "soft_fusion_with_middle_band":
            p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
            pred, _ = self.predict_with_middle_band_from_fused_probabilities(
                fused_proba=fused_proba,
                classes=self.classes_,
                dropout_label=int(self.dropout_label),
                enrolled_label=int(self.enrolled_label),
                graduate_label=int(self.graduate_label),
                low_threshold=float(self.threshold_stage1_low),
                high_threshold=float(self.threshold_stage1_high),
                p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
                p_graduate_given_non_dropout=p_graduate_given_non_dropout,
                stage2_decision_config=self.stage2_decision_config,
            )
            return pred.astype(int)
        pred = self._predict_soft_thresholded(fused_proba).astype(int)
        p_enrolled_given_non_dropout, p_graduate_given_non_dropout = self._stage2_enrolled_graduate_probability(X_df)
        pred, _ = self.apply_stage2_decision_policy(
            pred,
            p_enrolled_given_non_dropout=p_enrolled_given_non_dropout,
            p_graduate_given_non_dropout=p_graduate_given_non_dropout,
            dropout_label=int(self.dropout_label),
            enrolled_label=int(self.enrolled_label),
            graduate_label=int(self.graduate_label),
            stage2_decision_config=self.stage2_decision_config,
        )
        return pred.astype(int)
