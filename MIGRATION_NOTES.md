# Migration Notes

This document maps legacy responsibilities into the new `code3.0` architecture.

## Legacy to New Module Mapping

### From `code2.0/src/experiments/ml_opti.py`

- Legacy model-family selection -> `src/models/registry.py`
- Legacy Optuna search logic -> `src/models/search_spaces.py` and later tuner integration
- Legacy train/evaluate flow -> `src/models/train_eval.py`
- Legacy preprocessing sequence -> `src/preprocessing/tabular_pipeline.py`
- Legacy Isolation Forest handling -> `src/preprocessing/outlier.py`
- Legacy SMOTE handling -> `src/preprocessing/balancing.py`

### From legacy OULAD feature scripts

Requested source file `oulad_features.py` was not found by that exact filename in the current workspace. Closest legacy references discovered:
- `code2.0/src/data/oulad_features_v2.py`
- `code2.0/src/data/oulad_features_v3.py`
- related build/audit/test scripts under `code2.0/tools/` and `code2.0/tests/`

Mapped responsibilities:
- raw OULAD table loading -> `src/data/loaders/oulad_loader.py`
- OULAD schema normalization and joins -> `src/data/adapters/oulad_adapter.py`
- OULAD paper-style feature generation -> `src/data/feature_builders/oulad_paper_features.py`
- cross-dataset alignment -> `src/data/feature_builders/shared_feature_mapper.py`

## Notes

- No full logic has been ported yet by design.
- All current modules are minimal typed placeholders with TODO boundaries for incremental migration.
- Config-first structure is now ready for binary and multiclass benchmark expansion.
