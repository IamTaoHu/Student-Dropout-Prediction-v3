# CODEX_GAP_ANALYSIS_UCT_PAPER_ALIGNMENT

## Previous mismatches vs paper-aligned UCT benchmark
- UCT paper-push configs were not present in this checkout.
- No bundled `all` config existed for the required model family.
- UCT 3-class paper outputs were not explicitly mirrored into `runtime_artifacts/` for benchmark-style portability.
- Summary metadata did not always expose split sizes/seed/class-distribution context in one place.
- SVM Optuna space did not tune `class_weight`, and `gamma` was not a numeric search.

## What was fixed
- Added six configs:
  - `exp_bm_uct_3class_paper_push_svm`
  - `exp_bm_uct_3class_paper_push_gb`
  - `exp_bm_uct_3class_paper_push_lgbm`
  - `exp_bm_uct_3class_paper_push_catboost`
  - `exp_bm_uct_3class_paper_push_xgb`
  - `exp_bm_uct_3class_paper_push_all`
- Enforced strict UCT 3-class mapping (`Dropout=0`, `Enrolled=1`, `Graduate=2`) and explicit class order in each config.
- Kept paper-aligned preprocessing in configs: row-drop missing policy, one-hot, standard scaling, train-only Isolation Forest, train-only SMOTE.
- Kept primary objective/ranking as Macro F1 (`metrics.primary: macro_f1`, Optuna `scoring: f1_macro`).
- Added benchmark output mirroring to `runtime_artifacts/` (config-driven) for summary/leaderboard/metrics/predictions/manifest and per-model confusion matrices.
- Extended run summary metadata with seed, split config/sizes, and train class distributions before/after outlier filtering.
- Extended SVM Optuna search to include `C`, numeric `gamma`, and `class_weight`; paper-push SVM is locked to `kernel: rbf` through config override.
- Added focused regression tests for new paper-push configs and 3-class summary/runtime contract behavior.

## Remaining uncertainties
- Exact macro-F1 parity with `article_CSEDU_2026` is still sensitive to dataset version, preprocessing details beyond documented steps, and random-seed/trial-budget variance.
- CatBoost/LightGBM/XGBoost availability still depends on optional backend installation in the local environment.
- SHAP figure generation for non-tree-compatible cases now writes deterministic placeholder figures to preserve contract shape; this is format-compatible but not equivalent to model-native SHAP plots.

## Why Macro F1 may still lag
- Differences in feature engineering scope between this repo and paper implementation.
- Differences in dropped rows after missing-value filtering and resulting class balance.
- Optuna search-space and trial-budget differences (even when aligned nominally).
- Backend/version differences (scikit-learn, xgboost, lightgbm, catboost, shap) affecting convergence and probability calibration.
