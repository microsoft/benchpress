"""Checks for pooled medians bootstrapped by whole model groups."""

import unittest

import numpy as np

from benchpress.stats import cluster_bootstrap_median


class ClusterBootstrapMedianTest(unittest.TestCase):
    def test_unequal_clusters_keep_all_cells(self):
        result = cluster_bootstrap_median([0, 0, 0, 100], ['a', 'a', 'a', 'b'])
        self.assertEqual(result['median'], 0.0)
        self.assertEqual(result['ci_lower'], 0.0)
        self.assertEqual(result['ci_upper'], 100.0)
        self.assertEqual(result['n_clusters'], 2)
        self.assertEqual(result['n_records'], 4)

    def test_constant_errors_have_zero_width(self):
        result = cluster_bootstrap_median([3, 3, 3, 3], ['a', 'a', 'b', 'c'])
        self.assertEqual((result['ci_lower'], result['median'], result['ci_upper']),
                         (3.0, 3.0, 3.0))

    def test_paired_draws_are_reproducible_and_ignore_row_order(self):
        values = np.array([1, 2, 9, 20, 21])
        groups = np.array(['a', 'a', 'b', 'c', 'c'])
        first = cluster_bootstrap_median(values, groups)
        self.assertEqual(first, cluster_bootstrap_median(values[::-1], groups[::-1]))
        shifted = cluster_bootstrap_median(values + 7, groups)
        for key in ('median', 'ci_lower', 'ci_upper'):
            self.assertAlmostEqual(shifted[key], first[key] + 7)

    def test_rejects_invalid_inputs(self):
        for values, groups in [([1], ['a']), ([1, np.nan], ['a', 'b']),
                               ([1, 2], ['a']), ([[1, 2]], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                cluster_bootstrap_median(values, groups)
        for kwargs in ({'n_resamples': 0}, {'confidence': 1}):
            with self.assertRaises(ValueError):
                cluster_bootstrap_median([1, 2], ['a', 'b'], **kwargs)


if __name__ == '__main__':
    unittest.main()
