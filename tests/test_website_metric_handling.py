import importlib.util
import os
import unittest

import numpy as np

from benchpress.methods.predictors import (
    predict_benchpress_scores,
    predict_logit_bias_als_scores,
)
from website.scripts.add_prediction_intervals import attach_prediction_intervals


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
PREDICTOR_PATH = os.path.join(
    REPO_ROOT, "website", "add-model", "predictor.py")


def load_browser_predictor():
    spec = importlib.util.spec_from_file_location(
        "website_add_model_predictor", PREDICTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebsiteMetricHandlingTests(unittest.TestCase):
    def test_noncanonical_matrix_keeps_heuristic_fallback(self):
        matrix = np.asarray([
            [90.0, 80.0],
            [70.0, np.nan],
            [60.0, 50.0],
        ])
        predicted = predict_benchpress_scores(matrix)
        self.assertEqual(predicted.shape, matrix.shape)
        self.assertTrue(np.isfinite(predicted[1, 1]))

    def test_browser_predictor_matches_package_for_mixed_metric_ranges(self):
        matrix = np.asarray([
            [90.0, 0.10],
            [80.0, np.nan],
            [70.0, 0.30],
        ])
        benchmark_ids = ["accuracy", "edit_distance"]
        metric = {
            "accuracy": {
                "type": "pass_at_1_pct",
                "range": [0, 100],
                "higher_is_better": True,
            },
            "edit_distance": {
                "type": "normalized_edit_distance",
                "range": [0, 1],
                "higher_is_better": False,
            },
        }
        package = predict_logit_bias_als_scores(
            matrix,
            metric=metric,
            benchmark_ids=benchmark_ids,
        )
        browser = load_browser_predictor().predict_benchpress_scores(
            matrix,
            metric_specs=[
                metric[benchmark_id] for benchmark_id in benchmark_ids
            ],
        )
        np.testing.assert_allclose(browser, package, rtol=0.0, atol=1e-12)
        self.assertGreaterEqual(browser[1, 1], 0.0)
        self.assertLessEqual(browser[1, 1], 1.0)

    def test_website_intervals_stay_inside_declared_ranges(self):
        data = {
            "models": [{"id": "a"}, {"id": "b"}],
            "benchmarks": [
                {
                    "id": "edit_distance",
                    "metric": {
                        "type": "normalized_edit_distance",
                        "range": [0, 1],
                        "higher_is_better": False,
                    },
                },
                {
                    "id": "accuracy",
                    "metric": {
                        "type": "pct",
                        "range": [0, 100],
                        "higher_is_better": True,
                    },
                },
            ],
            "observed": [[0.1, 90.0], [0.2, 80.0]],
            "predictions": [[0.05, 95.0], [0.95, 5.0]],
        }
        scores = {
            "test_i": np.asarray([0, 0, 1, 1]),
            "test_j": np.asarray([0, 1, 0, 1]),
            "actual": np.asarray([0.1, 90.0, 0.2, 80.0]),
            "predicted": np.asarray([0.2, 85.0, 0.4, 70.0]),
            "combined_risk_model_uncertainty": np.asarray(
                [1.0, 1.0, 1.0, 1.0]),
            "metadata_json": np.asarray('{"matrix_shape":[2,2]}'),
        }
        results = {
            "setting": {"matrix_shape": [2, 2]},
            "confidence_methods": {
                "combined_risk_model": {
                    "conformal_90_scale_median": 20.0,
                    "conformal_90_interval": {
                        "coverage": 0.9,
                        "median_width": 40.0,
                    },
                },
            },
        }
        attach_prediction_intervals(
            data, scores, results, "scores.npz", "results.json")
        for row in data["prediction_intervals"]:
            self.assertGreaterEqual(row[0][0], 0.0)
            self.assertLessEqual(row[0][1], 1.0)
            self.assertGreaterEqual(row[1][0], 0.0)
            self.assertLessEqual(row[1][1], 100.0)


if __name__ == "__main__":
    unittest.main()
