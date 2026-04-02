"""UCT/UCI target mapping regression tests for strict 3-class benchmark alignment."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.target_mapping.three_class import map_three_class_target


class UctThreeClassMappingTests(unittest.TestCase):
    def test_three_class_mapping_exact_labels(self) -> None:
        df = pd.DataFrame(
            {
                "target": ["Dropout", "Enrolled", "Graduate", "Dropout", "Graduate"],
            }
        )
        mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
        y = map_three_class_target(
            df,
            source_column="target",
            dataset_name="uct_student",
            mapping=mapping,
        )
        self.assertListEqual(y.tolist(), [0, 1, 2, 0, 2])

    def test_three_class_mapping_rejects_unmapped_labels(self) -> None:
        df = pd.DataFrame({"target": ["Dropout", "Enrolled", "Transferred"]})
        mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
        with self.assertRaisesRegex(ValueError, "Unmapped labels"):
            _ = map_three_class_target(
                df,
                source_column="target",
                dataset_name="uct_student",
                mapping=mapping,
            )


if __name__ == "__main__":
    unittest.main()
