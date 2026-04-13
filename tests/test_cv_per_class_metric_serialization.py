from __future__ import annotations

import unittest

from scripts.run_experiment import _add_named_cv_per_class_metrics


class CvPerClassMetricSerializationTests(unittest.TestCase):
    def test_named_cv_per_class_metrics_are_serialized_with_label_tokens(self) -> None:
        metrics: dict[str, float] = {}
        _add_named_cv_per_class_metrics(
            metrics,
            {
                "0": {"precision_mean": 0.61, "recall_mean": 0.62, "f1_mean": 0.63},
                "1": {"precision_mean": 0.51, "recall_mean": 0.52, "f1_mean": 0.53},
                "2": {"precision_mean": 0.71, "recall_mean": 0.72, "f1_mean": 0.73},
            },
            {"0": "Dropout", "1": "Enrolled", "2": "Graduate"},
        )

        self.assertEqual(metrics["cv_precision_dropout"], 0.61)
        self.assertEqual(metrics["cv_recall_enrolled"], 0.52)
        self.assertEqual(metrics["cv_f1_graduate"], 0.73)
        self.assertEqual(metrics["cv_f1_dropout_mean"], 0.63)


if __name__ == "__main__":
    unittest.main()
