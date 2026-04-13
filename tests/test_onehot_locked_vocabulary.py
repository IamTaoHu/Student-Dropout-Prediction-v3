from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing.tabular_pipeline import run_tabular_preprocessing


class OneHotLockedVocabularyTests(unittest.TestCase):
    def test_locked_vocabulary_preserves_full_onehot_width_on_reduced_train_subset(self) -> None:
        full_train = pd.DataFrame(
            {
                "student_id": [1, 2, 3, 4, 5],
                "category": ["common_a", "common_b", "rare_c", "rare_d", "rare_e"],
                "status": ["x", "x", "y", "y", "z"],
                "score": [10.0, 11.0, 12.0, 13.0, 14.0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        reduced_train = full_train.iloc[:2].reset_index(drop=True)
        valid = full_train.iloc[2:4].reset_index(drop=True)
        test = full_train.iloc[4:].reset_index(drop=True)
        locked_categories = {
            "category": ["common_a", "common_b", "rare_c", "rare_d", "rare_e"],
            "status": ["x", "y", "z"],
        }
        config = {
            "target_column": "target",
            "id_columns": ["student_id"],
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "scaling": False,
            "onehot": True,
            "onehot_handle_unknown": "ignore",
            "onehot_drop": None,
            "lock_category_vocabulary_from_pre_split_train": True,
            "onehot_categories_source": "categorical_dtype_or_full_pre_split_train",
            "onehot_categories": locked_categories,
        }

        artifacts = run_tabular_preprocessing(
            {
                "train": reduced_train,
                "valid": valid,
                "test": test,
            },
            config,
        )

        self.assertEqual(artifacts.metadata["encoded_categorical_feature_count"], 8)
        self.assertEqual(artifacts.metadata["preprocessed_feature_count"], 9)
        self.assertTrue(bool(artifacts.metadata["onehot_categories_locked"]))
        self.assertEqual(
            artifacts.metadata["onehot_column_category_counts"],
            {"category": 5, "status": 3},
        )
        self.assertEqual(artifacts.X_train.shape[1], 9)
        self.assertEqual(artifacts.X_valid.shape[1], 9)
        self.assertEqual(artifacts.X_test.shape[1], 9)


if __name__ == "__main__":
    unittest.main()
