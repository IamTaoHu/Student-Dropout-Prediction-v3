# code3.0

Config-driven benchmark framework for student-dropout prediction across UCT Student and OULAD.

## What This Repo Does
- Loads dataset-specific raw data from YAML configs.
- Adapts schemas and builds dataset-specific features.
- Maps targets (binary / 3-class / 4-class where supported).
- Runs stratified split, preprocessing, optional outlier filtering, optional balancing.
- Trains and evaluates multiple classical/boosting models.
- Saves benchmark outputs under `results/<experiment_id>/`.
- Persists best model + preprocessing/runtime artifacts for downstream explainability.

## Installation
```bash
python -m pip install -r requirements.txt
```

Main dependencies include:
- Core: `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `joblib`
- Tuning/imbalance: `optuna`, `imbalanced-learn`
- Boosting backends: `xgboost`, `lightgbm`, `catboost`
- Explainability: `shap`, `lime`
- Reporting: `matplotlib`, `seaborn`

## Main Entry Points
- Build UCT processed features:
```bash
python scripts/build_uct_student_dataset.py --dataset-config configs/datasets/uct_student.yaml
```
- Build OULAD processed features:
```bash
python scripts/build_oulad_dataset.py --dataset-config configs/datasets/oulad.yaml
```
- Run benchmark:
```bash
python scripts/run_experiment.py --experiment-config configs/experiments/exp_bm_uct_binary_paper_style.yaml
python scripts/run_experiment.py --experiment-config configs/experiments/exp_bm_oulad_binary_paper_style.yaml
python scripts/run_experiment.py --experiment-config configs/experiments/exp_bm_oulad_binary_post_leakage_fix.yaml
```
Optional compact summary mode:
```bash
python scripts/run_experiment.py --experiment-config configs/experiments/exp_bm_uct_binary_paper_style.yaml --compact-summary
```
- Run explainability from saved benchmark artifacts:
```bash
python scripts/run_explainability.py --benchmark-summary results/<experiment_id>/benchmark_summary.json
```
Optional:
```bash
python scripts/run_explainability.py --benchmark-summary results/<experiment_id>/benchmark_summary.json --experiment-config configs/experiments/exp_bm_uct_binary_paper_style.yaml
```

## Data Validity Checks
- Validate UCT raw label vocabulary against config mappings:
```bash
python scripts/validate_uct_label_mapping.py --dataset-config configs/datasets/uct_student.yaml
```

Expected UCT raw outcome vocabulary (normalized):
- `dropout`
- `enrolled`
- `graduate`

OULAD post-leakage-fix rebaseline command:
```bash
python scripts/run_experiment.py --experiment-config configs/experiments/exp_bm_oulad_binary_post_leakage_fix.yaml
```

Expected output root for post-fix run:
- `results/exp_bm_oulad_binary_post_leakage_fix/`

## Output Layout
Each benchmark run writes to `results/<experiment_id>/`:
- Mandatory contract outputs (or explicit status in manifest):
- `benchmark_summary.json`
- includes `benchmark_summary_version` / `schema_version` for contract tracking
- `metrics.json`
- `predictions.csv`
- `leaderboard.csv`
- `benchmark_summary.md`
- `runtime_artifacts/`
- `model/`
- `artifact_manifest.json` (machine-readable status for mandatory/optional artifacts)
- Figure attempts for best model:
- `figures/learning_curve.png` status recorded in manifest
- `figures/pr_curve.png` status recorded in manifest
- Confusion matrix artifacts from reporting (`confusion_matrix_<model>.png`) with status recorded in manifest

Optional/capability-dependent outputs:
- `figures/shap_beeswarm.png`
- `figures/shap_waterfall_class_*.png`
- `explainability/` directory and explainability files
- SHAP/LIME/AIME CSV and image outputs

Compact summary behavior:
- In compact mode, large arrays (`y_pred_test`, `y_proba_test`) are omitted from `benchmark_summary.json`.
- Explainability remains compatible through required `artifact_paths` contract keys.

Explainability writes to `results/<experiment_id>/explainability/` and updates the same `artifact_manifest.json`:
- `explanation_report.json`
- `explanation_report.md`
- CSV outputs for SHAP/LIME/AIME where generated
- `aime_similarity.png` when AIME plotting is enabled

Legacy results policy (current):
- Older result folders created before manifest rollout may not include `artifact_manifest.json`.
- Current tooling treats those folders as legacy outputs (no automatic backfill in this round).
- Legacy results without manifest are allowed for historical reference but should not be used as contract-validated artifacts.

## Supported / Experimental
- Supported in CLI: single-dataset experiment configs with `experiment.dataset_config`.
- Not supported in CLI: multi-dataset shared execution via `experiment.datasets` remains intentionally unsupported and fails explicitly with `NotImplementedError`.
- Decision status: keep unsupported for now (no partial implementation in this round).

## Forbidden Feature Column Policy
- Default leakage guard always forbids `final_result`.
- Preprocessing forbidden columns can be extended via:
- experiment config: `preprocessing.forbidden_feature_columns`
- dataset config: `preprocessing.forbidden_feature_columns`
- Source target column is also treated as forbidden when it differs from canonical `target`.

## Validation Workflow (Canonical)
```bash
python -m compileall src scripts
python scripts/run_experiment.py --help
python scripts/build_uct_student_dataset.py --help
python scripts/build_oulad_dataset.py --help
python scripts/run_explainability.py --help
python -m unittest discover -s tests -v
```

Critical regression coverage:
- `tests/test_preprocessing_safeguards.py`
- `tests/test_oulad_paper_features.py`
- `tests/test_smoke_benchmark.py`
- `tests/test_artifact_manifest.py`
- `tests/test_benchmark_contract.py`
- `tests/test_explanation_report.py`
