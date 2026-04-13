from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from scripts.run_experiment import (
    _build_locked_onehot_vocabulary,
    _build_predefined_uci_feature_splits,
    _drop_rows_with_missing_values,
    _prepare_preprocessing_config,
    _resolve_experiment_feature_config,
    load_yaml,
)
from src.preprocessing.tabular_pipeline import run_tabular_preprocessing


class UciFeatureSpaceDimensionTests(unittest.TestCase):
    def test_locked_train_parquet_vocabulary_recovers_expected_paper_style_feature_space(self) -> None:
        dataset_config_path = Path("configs/datasets/uci_student_presplit_parquet.yaml")
        exp_cfg = load_yaml(Path("configs/experiments/exp_uci_3class_paper_repro_feature_exact.yaml"))
        dataset_cfg = _resolve_experiment_feature_config(
            exp_cfg=exp_cfg,
            dataset_cfg=load_yaml(dataset_config_path),
        )

        try:
            predefined_splits, _dataset_name, id_column, source_target_col, _source_meta = _build_predefined_uci_feature_splits(
                dataset_cfg,
                formulation="three_class",
                target_mapping={"Dropout": 0, "Enrolled": 1, "Graduate": 2},
            )
        except ImportError as exc:
            self.skipTest(f"Parquet engine is unavailable in this environment: {exc}")

        full_pre_split_train_feature_df = predefined_splits["train"].reset_index(drop=True)
        train_feature_df, _train_missing_meta = _drop_rows_with_missing_values(
            predefined_splits["train"],
            exp_cfg.get("preprocessing", {}),
        )
        test_feature_df, _test_missing_meta = _drop_rows_with_missing_values(
            predefined_splits["test"],
            exp_cfg.get("preprocessing", {}),
        )
        reduced_train = train_feature_df.iloc[:400].reset_index(drop=True)

        preprocess_cfg = _prepare_preprocessing_config(
            exp_cfg,
            dataset_cfg,
            id_column=id_column,
            source_target_col=source_target_col,
        )
        locked_vocabulary = _build_locked_onehot_vocabulary(full_pre_split_train_feature_df, preprocess_cfg)
        preprocess_cfg["onehot_categories"] = locked_vocabulary["categories"]

        artifacts = run_tabular_preprocessing(
            {
                "train": reduced_train,
                "valid": pd.DataFrame(columns=reduced_train.columns),
                "test": test_feature_df.head(64).reset_index(drop=True),
            },
            preprocess_cfg,
        )

        self.assertEqual(locked_vocabulary["categorical_column_count"], 17)
        self.assertEqual(locked_vocabulary["numeric_column_count"], 19)
        self.assertEqual(locked_vocabulary["encoded_categorical_feature_count"], 206)
        self.assertEqual(locked_vocabulary["preprocessed_feature_count"], 225)
        self.assertEqual(artifacts.metadata["encoded_categorical_feature_count"], 206)
        self.assertEqual(artifacts.metadata["preprocessed_feature_count"], 225)
        self.assertEqual(artifacts.X_train.shape[1], 225)
        self.assertEqual(artifacts.X_test.shape[1], 225)
        self.assertEqual(
            artifacts.metadata["onehot_column_category_counts"]["father_s_occupation"],
            43,
        )


if __name__ == "__main__":
    unittest.main()
