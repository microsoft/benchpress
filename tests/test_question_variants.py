import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import approx_fprime, minimize

from benchpress.methods.completers import (
    complete_bias_als,
    complete_group_bias_als,
    fit_binary_group_low_rank,
)
from benchpress.methods.predictors import (
    fit_question_score_variant,
    predict_group_mean_ridge_scores,
    predict_shrunk_group_residual_scores,
)


class QuestionVariantTests(unittest.TestCase):
    def test_group_offsets_recover_opposing_subtask_skills(self):
        groups = np.repeat([0, 1], 6)
        truth = np.repeat([[1., 4.], [2., 3.], [4., 1.], [3., 2.]], 6, axis=1)
        matrix = truth.copy()
        matrix[-1, [2, 3, 4, 5, 8, 9, 10, 11]] = np.nan
        before = matrix.copy()
        pred = complete_group_bias_als(
            matrix, groups, rank=0, lam=1e-5, n_iter=80, n_ensemble=1)
        np.testing.assert_allclose(pred, truth, atol=1e-4)
        np.testing.assert_array_equal(matrix, before)

    def test_binary_gradient_and_observation_preservation(self):
        matrix = np.array([[0., 1., 0., 1.], [1., 0., 1., 0.],
                           [1., np.nan, 0., np.nan]])
        groups = [0, 0, 1, 1]

        def checked_minimize(fun, x0, **kwargs):
            _, grad = fun(x0)
            numeric = approx_fprime(x0, lambda x: fun(x)[0], epsilon=1e-6)
            np.testing.assert_allclose(grad, numeric, atol=2e-6)
            return minimize(fun, x0, **kwargs)

        with patch('scipy.optimize.minimize', side_effect=checked_minimize):
            pred, records = fit_binary_group_low_rank(
                matrix, groups, rank=1, max_iter=100, n_ensemble=1)
        np.testing.assert_array_equal(pred[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
        self.assertTrue(np.all((pred >= 0) & (pred <= 1)))
        self.assertEqual(len(records), 1)
        repeated, _ = fit_binary_group_low_rank(
            matrix, groups, rank=1, max_iter=100, n_ensemble=1)
        np.testing.assert_array_equal(pred, repeated)

    def test_raw_inverse_preserves_empirical_binary_mean(self):
        matrix = np.array([[0.], [0.], [0.], [100.], [np.nan]])
        with patch('benchpress.methods.predictors.complete_bias_als',
                   return_value=np.zeros_like(matrix)) as fit:
            pred, _ = fit_question_score_variant(matrix, [0], 'raw')
        self.assertAlmostEqual(pred[-1, 0], 25.)
        self.assertTrue(np.isnan(fit.call_args.args[0][-1, 0]))
        np.testing.assert_array_equal(pred[:-1], matrix[:-1])

    def test_grouped_als_is_deterministic_with_irregular_masks(self):
        matrix = np.array([[0., 1., 2., np.nan], [1., np.nan, 3., 2.],
                           [np.nan, 2., 4., 1.]])
        args = dict(column_groups=[0, 0, 1, 1], rank=2, n_iter=4, n_ensemble=2)
        pred = complete_group_bias_als(matrix, **args)
        np.testing.assert_array_equal(pred, complete_group_bias_als(matrix, **args))
        np.testing.assert_array_equal(pred[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
        self.assertTrue(np.isfinite(pred).all())

    def test_invalid_or_unidentified_inputs_fail(self):
        with self.assertRaises(ValueError):
            fit_binary_group_low_rank(np.array([[0., 0.5]]), [0, 0])
        with self.assertRaises(ValueError):
            complete_group_bias_als(np.array([[1., np.nan]]), [0, 0])
        with self.assertRaises(ValueError):
            complete_group_bias_als(np.array([[1., 2.]]), [0])
        with self.assertRaises(ValueError):
            fit_question_score_variant(np.array([[0., 100.]]), [0, 0], 'unknown')

    def test_fitted_values_do_not_change_default_completion(self):
        matrix = np.array([[0., 1., 2.], [1., 2., np.nan]])
        options = dict(rank=1, n_iter=5, n_ensemble=1, normalize=False)
        fitted = complete_bias_als(matrix, preserve_observed=False, **options)
        completed = complete_bias_als(matrix, **options)
        np.testing.assert_array_equal(fitted[np.isnan(matrix)], completed[np.isnan(matrix)])
        np.testing.assert_array_equal(completed[np.isfinite(matrix)], matrix[np.isfinite(matrix)])

    def test_group_mean_ridge_learns_cross_group_relation(self):
        matrix = np.array([[0., 20., 80., 100.], [20., 40., 60., 80.],
                           [40., 60., 40., 60.], [60., 80., 20., 40.],
                           [30., np.nan, 50., np.nan]])
        before = matrix.copy()
        pred, diagnostics = predict_group_mean_ridge_scores(matrix, [0, 0, 1, 1], lam=1e-6)
        np.testing.assert_allclose(pred[-1], [30., 50., 50., 70.], atol=1e-5)
        np.testing.assert_array_equal(matrix, before)
        self.assertEqual(diagnostics['n_reference_rows'], 4)
        with self.assertRaises(ValueError):
            predict_group_mean_ridge_scores(
                np.vstack((matrix, [np.nan, 30., 50., np.nan])), [0, 0, 1, 1])

    def test_shrinkage_uses_only_reference_moments_and_preserves_observations(self):
        matrix = np.array([[0., 100., 0., 100.], [100., 100., 0., 0.],
                           [0., np.nan, 100., np.nan]])
        with patch('benchpress.methods.predictors.complete_bias_als',
                   return_value=np.zeros_like(matrix)) as fit:
            pred, diagnostics = predict_shrunk_group_residual_scores(matrix, [0, 0, 1, 1])
        self.assertFalse(fit.call_args.kwargs['preserve_observed'])
        self.assertEqual([m['n_reference_rows'] for m in diagnostics['group_prior']], [2, 2])
        np.testing.assert_array_equal(pred[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
        self.assertTrue(np.isfinite(pred).all())

    def test_item_components_recover_information_absent_from_group_means(self):
        source = [[z, 40 - z, z, 10., 30., 20.] for z in range(0, 40, 5)]
        matrix = np.array(source + [[30., 10., np.nan, 10., 30., np.nan]])
        groups = [0, 0, 0, 1, 1, 1]
        mean_only, _ = predict_group_mean_ridge_scores(matrix, groups)
        item_aware, diagnostics = predict_group_mean_ridge_scores(matrix, groups, item_rank=1)
        self.assertAlmostEqual(mean_only[-1, 2], 17.5)
        self.assertLess(abs(item_aware[-1, 2] - 30), 0.2)
        self.assertAlmostEqual(item_aware[-1, 5], 20)
        np.testing.assert_array_equal(
            item_aware[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
        self.assertEqual(diagnostics['item_rank'], 1)


if __name__ == '__main__':
    unittest.main()
