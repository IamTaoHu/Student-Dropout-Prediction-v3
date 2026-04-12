"""Generate standard experiment visualization artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize


def _as_series(y: pd.Series | np.ndarray | list[Any]) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(np.asarray(y))


def _as_dataframe(X: pd.DataFrame | np.ndarray | list[list[float]]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    arr = np.asarray(X)
    return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])


def _scorer_from_primary_metric(primary_metric: str) -> str:
    metric = (primary_metric or "macro_f1").lower().strip()
    if metric.startswith("test_"):
        metric = metric[len("test_") :]
    if metric.startswith("valid_"):
        metric = metric[len("valid_") :]

    mapping = {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
    }
    return mapping.get(metric, "f1_macro")


def plot_learning_curve(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path,
    experiment_name: str,
    primary_metric: str = "macro_f1",
    random_state: int = 42,
) -> Path:
    """Plot and save learning curve (train vs validation score)."""
    X = _as_dataframe(X_train)
    y = _as_series(y_train)
    scorer = _scorer_from_primary_metric(primary_metric)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=clone(model),
        X=X,
        y=y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring=scorer,
        n_jobs=1,
        shuffle=True,
        random_state=random_state,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Train score")
    plt.plot(train_sizes, valid_mean, marker="o", label="Validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15)
    plt.title(f"{experiment_name} - Learning Curve")
    plt.xlabel("Training size")
    plt.ylabel("Score")
    plt.legend(frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _is_tree_model_for_shap(model: Any) -> bool:
    if hasattr(model, "explainability_model"):
        model = getattr(model, "explainability_model")
    module_name = model.__class__.__module__.lower()
    class_name = model.__class__.__name__.lower()
    return any(token in f"{module_name}.{class_name}" for token in ["xgb", "lgbm", "catboost"])


def _has_shap_dependency() -> bool:
    return importlib.util.find_spec("shap") is not None


def _write_placeholder_figure(output_path: Path, title: str, reason: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.axis("off")
    plt.title(title)
    plt.text(
        0.5,
        0.5,
        f"Unavailable\n{reason}",
        ha="center",
        va="center",
        wrap=True,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _sample_for_shap(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X_df = _as_dataframe(X)
    y_sr = _as_series(y)
    if len(X_df) <= max_samples:
        return X_df, y_sr
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_df), size=max_samples, replace=False)
    idx.sort()
    return X_df.iloc[idx].reset_index(drop=True), y_sr.iloc[idx].reset_index(drop=True)


def _normalize_shap_values(shap_values: Any, n_classes: int | None) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return [arr]

    if arr.ndim == 3:
        if n_classes is not None and arr.shape[0] == n_classes:
            return [arr[i] for i in range(arr.shape[0])]
        if n_classes is not None and arr.shape[-1] == n_classes:
            return [arr[:, :, i] for i in range(arr.shape[-1])]
    return [arr.reshape(arr.shape[0], -1)]


def _get_expected_value_for_class(expected_value: Any, class_idx: int) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        return float(np.asarray(expected_value)[class_idx])
    return float(expected_value)


def plot_shap_beeswarm(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    experiment_name: str,
    max_samples: int = 1000,
    random_state: int = 42,
) -> Path | None:
    """Plot SHAP beeswarm global explanation for tree models."""
    if not _is_tree_model_for_shap(model):
        return None

    try:
        import shap
    except ImportError:
        return None

    X_sample, y_sample = _sample_for_shap(X_test, y_test, max_samples=max_samples, random_state=random_state)
    y_sample = y_sample.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    class_labels = list(getattr(model, "classes_", sorted(y_sample.unique().tolist())))
    class_to_idx = {label: i for i, label in enumerate(class_labels)}
    shap_values_by_class = _normalize_shap_values(shap_values_raw, n_classes=len(class_labels))

    if len(shap_values_by_class) > 1:
        per_row = []
        default_idx = 0
        for i, label in enumerate(y_sample.to_numpy()):
            class_idx = class_to_idx.get(label, default_idx)
            class_idx = min(class_idx, len(shap_values_by_class) - 1)
            per_row.append(shap_values_by_class[class_idx][i])
        shap_values_for_plot = np.vstack(per_row)
    else:
        shap_values_for_plot = shap_values_by_class[0]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, X_sample, show=False)
    plt.title(f"{experiment_name} - SHAP Beeswarm")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_shap_waterfall(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    experiment_name: str,
    max_samples: int = 1000,
    random_state: int = 42,
) -> list[Path]:
    """Plot 1 SHAP waterfall per class for tree models."""
    if not _is_tree_model_for_shap(model):
        return []

    try:
        import shap
    except ImportError:
        return []

    X_sample, y_sample = _sample_for_shap(X_test, y_test, max_samples=max_samples, random_state=random_state)
    y_sample = y_sample.reset_index(drop=True)
    class_labels = sorted(y_sample.unique().tolist())
    model_classes = list(getattr(model, "classes_", class_labels))
    class_to_idx = {label: i for i, label in enumerate(model_classes)}

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    shap_values_by_class = _normalize_shap_values(shap_values_raw, n_classes=len(model_classes))

    saved_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_label in class_labels:
        class_rows = np.where(y_sample.to_numpy() == class_label)[0]
        if len(class_rows) == 0:
            continue
        row_idx = int(class_rows[0])
        class_idx = class_to_idx.get(class_label, 0)
        class_idx = min(class_idx, len(shap_values_by_class) - 1)

        values = shap_values_by_class[class_idx][row_idx]
        base_val = _get_expected_value_for_class(expected_value, class_idx)
        features = X_sample.iloc[row_idx]

        explanation = shap.Explanation(
            values=np.asarray(values),
            base_values=base_val,
            data=features.to_numpy(),
            feature_names=X_sample.columns.tolist(),
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f"{experiment_name} - SHAP Waterfall (class {class_label})")
        plt.tight_layout()
        out_path = output_dir / f"shap_waterfall_class_{class_label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def plot_pr_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray | list[list[float]],
    output_path: Path,
    experiment_name: str,
) -> Path | None:
    """Plot precision-recall curve for binary or multiclass (OvR)."""
    if y_pred_proba is None:
        return None

    y_true = _as_series(y_test).to_numpy()
    proba = np.asarray(y_pred_proba)
    if proba.ndim != 2:
        return None

    class_labels = np.unique(y_true)
    if len(class_labels) < 2:
        return None

    plt.figure(figsize=(8, 6))
    if len(class_labels) == 2 and proba.shape[1] >= 2:
        pos_class = class_labels[-1]
        y_binary = (y_true == pos_class).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, proba[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f"class {pos_class} (AUC={pr_auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=class_labels)
        n_curves = min(y_bin.shape[1], proba.shape[1])
        for i in range(n_curves):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, linewidth=2, label=f"class {class_labels[i]} (AUC={pr_auc:.3f})")

    plt.title(f"{experiment_name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_all_figures(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_proba: np.ndarray | list[list[float]] | None,
    output_dir: str | Path,
    experiment_name: str,
    primary_metric: str = "macro_f1",
    random_state: int = 42,
    include_status: bool = False,
) -> dict[str, Any]:
    """Generate required visualization set for one experiment run."""
    effective_model = getattr(model, "explainability_model", model)
    root = Path(output_dir)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    statuses: dict[str, dict[str, Any]] = {}
    class_labels_for_fallback = sorted(_as_series(y_test).unique().tolist())

    try:
        lc_path = plot_learning_curve(
            model=effective_model,
            X_train=X_train,
            y_train=y_train,
            output_path=figures_dir / "learning_curve.png",
            experiment_name=experiment_name,
            primary_metric=primary_metric,
            random_state=random_state,
        )
        saved["learning_curve"] = str(lc_path)
        statuses["learning_curve"] = {"status": "generated", "path": str(lc_path)}
    except Exception as exc:
        statuses["learning_curve"] = {
            "status": "failed",
            "reason": f"learning_curve_generation_failed: {type(exc).__name__}: {exc}",
        }

    shap_compatible = _is_tree_model_for_shap(model)
    shap_available = _has_shap_dependency()
    if not shap_compatible:
        fallback_reason = "best_model_not_shap_compatible"
        beeswarm_path = _write_placeholder_figure(
            figures_dir / "shap_beeswarm.png",
            f"{experiment_name} - SHAP Beeswarm",
            fallback_reason,
        )
        saved["shap_beeswarm"] = str(beeswarm_path)
        waterfall_paths: list[str] = []
        for class_label in class_labels_for_fallback:
            wf_path = _write_placeholder_figure(
                figures_dir / f"shap_waterfall_class_{class_label}.png",
                f"{experiment_name} - SHAP Waterfall (class {class_label})",
                fallback_reason,
            )
            saved[wf_path.stem] = str(wf_path)
            waterfall_paths.append(str(wf_path))
        statuses["shap_beeswarm"] = {
            "status": "generated",
            "path": str(beeswarm_path),
            "reason": fallback_reason,
        }
        statuses["shap_waterfall"] = {
            "status": "generated",
            "paths": waterfall_paths,
            "reason": fallback_reason,
        }
    elif not shap_available:
        fallback_reason = "dependency_missing: shap"
        beeswarm_path = _write_placeholder_figure(
            figures_dir / "shap_beeswarm.png",
            f"{experiment_name} - SHAP Beeswarm",
            fallback_reason,
        )
        saved["shap_beeswarm"] = str(beeswarm_path)
        waterfall_paths = []
        for class_label in class_labels_for_fallback:
            wf_path = _write_placeholder_figure(
                figures_dir / f"shap_waterfall_class_{class_label}.png",
                f"{experiment_name} - SHAP Waterfall (class {class_label})",
                fallback_reason,
            )
            saved[wf_path.stem] = str(wf_path)
            waterfall_paths.append(str(wf_path))
        statuses["shap_beeswarm"] = {
            "status": "generated",
            "path": str(beeswarm_path),
            "reason": fallback_reason,
        }
        statuses["shap_waterfall"] = {
            "status": "generated",
            "paths": waterfall_paths,
            "reason": fallback_reason,
        }
    else:
        try:
            beeswarm_path = plot_shap_beeswarm(
                model=effective_model,
                X_test=X_test,
                y_test=y_test,
                output_path=figures_dir / "shap_beeswarm.png",
                experiment_name=experiment_name,
                max_samples=1000,
                random_state=random_state,
            )
            if beeswarm_path is not None:
                saved["shap_beeswarm"] = str(beeswarm_path)
                statuses["shap_beeswarm"] = {"status": "generated", "path": str(beeswarm_path)}
            else:
                statuses["shap_beeswarm"] = {
                    "status": "skipped",
                    "reason": "not_applicable_for_model_type",
                }
        except Exception as exc:
            fallback_reason = f"figure_generation_error: {exc}"
            beeswarm_path = _write_placeholder_figure(
                figures_dir / "shap_beeswarm.png",
                f"{experiment_name} - SHAP Beeswarm",
                fallback_reason,
            )
            saved["shap_beeswarm"] = str(beeswarm_path)
            statuses["shap_beeswarm"] = {
                "status": "generated",
                "path": str(beeswarm_path),
                "reason": fallback_reason,
            }

        try:
            waterfall_paths = plot_shap_waterfall(
                model=effective_model,
                X_test=X_test,
                y_test=y_test,
                output_dir=figures_dir,
                experiment_name=experiment_name,
                max_samples=1000,
                random_state=random_state,
            )
            for wf_path in waterfall_paths:
                saved[wf_path.stem] = str(wf_path)
            if waterfall_paths:
                statuses["shap_waterfall"] = {
                    "status": "generated",
                    "paths": [str(p) for p in waterfall_paths],
                }
            else:
                fallback_reason = "not_applicable_for_model_type"
                fallback_paths: list[str] = []
                for class_label in class_labels_for_fallback:
                    wf_path = _write_placeholder_figure(
                        figures_dir / f"shap_waterfall_class_{class_label}.png",
                        f"{experiment_name} - SHAP Waterfall (class {class_label})",
                        fallback_reason,
                    )
                    saved[wf_path.stem] = str(wf_path)
                    fallback_paths.append(str(wf_path))
                statuses["shap_waterfall"] = {
                    "status": "generated",
                    "paths": fallback_paths,
                    "reason": fallback_reason,
                }
        except Exception as exc:
            fallback_reason = f"figure_generation_error: {exc}"
            fallback_paths: list[str] = []
            for class_label in class_labels_for_fallback:
                wf_path = _write_placeholder_figure(
                    figures_dir / f"shap_waterfall_class_{class_label}.png",
                    f"{experiment_name} - SHAP Waterfall (class {class_label})",
                    fallback_reason,
                )
                saved[wf_path.stem] = str(wf_path)
                fallback_paths.append(str(wf_path))
            statuses["shap_waterfall"] = {
                "status": "generated",
                "paths": fallback_paths,
                "reason": fallback_reason,
            }

    try:
        pr_path = plot_pr_curve(
            y_test=y_test,
            y_pred_proba=y_pred_proba,
            output_path=figures_dir / "pr_curve.png",
            experiment_name=experiment_name,
        )
        if pr_path is not None:
            saved["pr_curve"] = str(pr_path)
            statuses["pr_curve"] = {"status": "generated", "path": str(pr_path)}
        else:
            reason = "y_pred_proba_missing" if y_pred_proba is None else "not_applicable_for_model_type"
            statuses["pr_curve"] = {"status": "skipped", "reason": reason}
    except Exception as exc:
        statuses["pr_curve"] = {
            "status": "failed",
            "reason": f"figure_generation_error: {exc}",
        }

    if include_status:
        return {"artifact_paths": saved, "artifact_status": statuses}
    return saved
