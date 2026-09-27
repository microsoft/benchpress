import unittest
from unittest.mock import patch

import numpy as np

from benchpress.methods.predictors import (
    predict_benchpress_scores,
    predict_prior_residual_scores,
)
from benchpress.stats import cluster_bootstrap_mean
from analysis.quantization_prior import estimate_scheme_prior, select_correction_weight


class PriorResidualTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[-1.0, np.nan, 2.0], [-2.0, 0.5, np.nan]])
        self.mean = np.full(self.matrix.shape, -0.5)
        self.scale = np.full(self.matrix.shape, 2.0)

    def test_zero_weight_returns_prior_and_does_not_fit(self):
        with patch("benchpress.methods.predictors.complete_bias_als") as fit:
            result = predict_prior_residual_scores(
                self.matrix, self.mean, self.scale, correction_weight=0)
        fit.assert_not_called()
        missing = np.isnan(self.matrix)
        np.testing.assert_array_equal(result[missing], self.mean[missing])
        np.testing.assert_array_equal(result[~missing], self.matrix[~missing])

    def test_residual_scaling_missingness_and_signed_output(self):
        with patch("benchpress.methods.predictors.complete_bias_als",
                   return_value=np.full(self.matrix.shape, -2.0)) as fit:
            result = predict_prior_residual_scores(
                self.matrix, self.mean, self.scale, correction_weight=0.25)
        np.testing.assert_allclose(
            fit.call_args.args[0], (self.matrix - self.mean) / self.scale,
            equal_nan=True)
        self.assertFalse(fit.call_args.kwargs["normalize"])
        np.testing.assert_array_equal(result[np.isnan(self.matrix)], [-1.5, -1.5])
        np.testing.assert_array_equal(result[np.isfinite(self.matrix)],
                                      self.matrix[np.isfinite(self.matrix)])

    def test_real_als_is_deterministic_and_affine_equivariant(self):
        result = predict_prior_residual_scores(self.matrix, self.mean, self.scale)
        repeated = predict_prior_residual_scores(self.matrix, self.mean, self.scale)
        transformed = predict_prior_residual_scores(
            3 * self.matrix + 7, 3 * self.mean + 7, 3 * self.scale)
        np.testing.assert_array_equal(result, repeated)
        np.testing.assert_allclose(transformed, 3 * result + 7, rtol=0, atol=1e-10)
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_prior_and_weight_fail(self):
        for scale in (np.zeros_like(self.scale), -self.scale,
                      np.full(self.scale.shape, np.nan)):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    predict_prior_residual_scores(self.matrix, self.mean, scale)
        for weight in (-0.1, 1.1, np.nan):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    predict_prior_residual_scores(
                        self.matrix, self.mean, self.scale, correction_weight=weight)
        with self.assertRaises(ValueError):
            predict_prior_residual_scores(self.matrix, self.mean[:, :1], self.scale)

    def test_no_observations_require_prior_only(self):
        empty = np.full(self.matrix.shape, np.nan)
        with self.assertRaises(ValueError):
            predict_prior_residual_scores(empty, self.mean, self.scale)
        np.testing.assert_array_equal(
            predict_prior_residual_scores(empty, self.mean, self.scale, 0), self.mean)

    def test_default_predictor_still_routes_to_original_method(self):
        with patch("benchpress.methods.predictors.predict_logit_bias_als_scores",
                   return_value=self.matrix) as original:
            result = predict_benchpress_scores(self.matrix)
        self.assertIs(result, self.matrix)
        original.assert_called_once_with(
            self.matrix, rank=2, lam=0.1, metric=None, benchmark_ids=None)

    def test_prior_moments_ignore_hidden_cells(self):
        mean, scale, moments = estimate_scheme_prior(
            np.array([[1.0, 3.0], [5.0, np.nan]]), np.array(["a", "a"]))
        np.testing.assert_array_equal(mean, np.full((2, 2), 3.0))
        np.testing.assert_allclose(scale, np.sqrt(4 * (1 + 1 / 3)))
        self.assertEqual(moments["a"]["n"], 3)

    def test_inner_selection_never_restores_outer_targets_and_ties_choose_zero(self):
        matrix = np.arange(36 * 6, dtype=float).reshape(36, 6) / 100
        matrix[0, 3:] = np.nan
        schemes = np.full(36, "a")
        calls = []

        def prior_only(train, mean, scale):
            calls.append(train.copy())
            return mean.copy()

        with patch("analysis.quantization_prior.predict_prior_residual_scores",
                   side_effect=prior_only):
            weight, validation = select_correction_weight(matrix, schemes, 0, 4)
        self.assertEqual(weight, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(validation["records"]), 90)
        all_rows = []
        for train, fold in zip(calls, validation["folds"]):
            self.assertTrue(np.isnan(train[0, 3:]).all())
            np.testing.assert_array_equal(train[0, :3], matrix[0, :3])
            rows = [int(row) for row in fold["revealed_columns_by_row"]]
            all_rows.extend(rows)
            for row in rows:
                self.assertEqual(np.isfinite(train[row]).sum(), 3)
        self.assertEqual(len(set(all_rows)), 30)
        self.assertNotIn(0, all_rows)

    def test_cluster_bootstrap_preserves_grouped_constant_effect(self):
        summary = cluster_bootstrap_mean([-2, -2, -2, -2], ["a", "a", "b", "c"])
        self.assertEqual(summary["mean_delta"], -2)
        self.assertEqual(summary["ci_lower"], -2)
        self.assertEqual(summary["ci_upper"], -2)
        self.assertEqual(summary["n_clusters"], 3)
        self.assertEqual(
            summary, cluster_bootstrap_mean([-2, -2, -2, -2], ["a", "a", "b", "c"]))
        with self.assertRaises(ValueError):
            cluster_bootstrap_mean([1], ["a"])


if __name__ == "__main__":
    unittest.main()
