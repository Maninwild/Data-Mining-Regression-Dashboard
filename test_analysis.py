"""Regression tests for the shared analysis pipeline."""

import math
import unittest

from analysis import (
    evaluate_models,
    load_dataset,
    mine_frequent_patterns,
    prepare_features,
)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_dataset()
        cls.features, cls.target = prepare_features(cls.frame)

    def test_dataset_and_features_are_valid(self) -> None:
        self.assertEqual(len(self.frame), 1000)
        self.assertEqual(len(self.features), len(self.target))
        self.assertNotIn("defect_id", self.features.columns)
        self.assertNotIn("defect_date", self.features.columns)
        self.assertFalse(self.features.isna().any().any())

    def test_model_metrics_are_reproducible_and_finite(self) -> None:
        metrics, curve, best_k = evaluate_models(self.features, self.target)
        self.assertEqual(set(metrics["model"]), {"Mean baseline", "Linear regression", "Decision tree", "Scaled KNN"})
        self.assertTrue(all(math.isfinite(value) for value in metrics[["r2", "mae", "rmse"]].to_numpy().ravel()))
        self.assertEqual(list(curve["k"]), list(range(1, 16)))
        self.assertIn(best_k, range(1, 16))

    def test_frequent_pattern_mining_produces_results(self) -> None:
        patterns = mine_frequent_patterns(self.frame)
        self.assertFalse(patterns.empty)
        self.assertEqual(list(patterns.columns), ["support", "itemsets", "size"])
        self.assertTrue(patterns["support"].between(0.3, 1).all())

    def test_invalid_minimum_support_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mine_frequent_patterns(self.frame, min_support=0)


if __name__ == "__main__":
    unittest.main()
