import unittest

import numpy as np

from benchpress.evaluation_harness import compute_binary_accuracy


class QuestionAccuracyTests(unittest.TestCase):
    def test_strict_threshold_in_normalized_percentage_scores(self):
        scores = np.array([-20., 0., 49.9, 50., 50.1, 100., 120.]) / 100.
        labels = np.array([0, 0, 0, 0, 1, 1, 1])
        result = compute_binary_accuracy(
            labels, scores, np.ones(7, dtype=bool), threshold=0.5)
        self.assertEqual(result, {'accuracy': 100., 'correct': 7, 'n': 7})

    def test_only_hidden_target_entries_count(self):
        labels = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]])
        scores = np.array([[1., 1., 1.], [1., 0.9, 0.8], [0., 0.2, 0.9]])
        mask = np.array([[False, False, False], [False, True, True],
                         [False, True, True]])
        result = compute_binary_accuracy(labels, scores, mask, threshold=0.5)
        self.assertEqual(result, {'accuracy': 25., 'correct': 1, 'n': 4})
        scores[~mask] = np.nan
        self.assertEqual(compute_binary_accuracy(labels, scores, mask, threshold=0.5), result)

    def test_pooling_weights_each_question_equally(self):
        labels = np.array([0, 0, 0, 1])
        scores = np.array([0.1, 0.1, 0.1, 0.1])
        result = compute_binary_accuracy(labels, scores, np.ones(4, bool), threshold=0.5)
        self.assertEqual(result, {'accuracy': 75., 'correct': 3, 'n': 4})

    def test_invalid_selected_entries_are_not_dropped(self):
        for labels, scores in [([0, 2], [0.2, 0.8]), ([0, 1], [0.2, np.nan]),
                               ([0, 1], [np.inf, 0.8])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                compute_binary_accuracy(labels, scores, np.ones(2, bool), threshold=0.5)

    def test_invalid_mask_and_empty_set_fail(self):
        for mask in [[1, 0], np.array([True]), np.array([False, False])]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                compute_binary_accuracy([0, 1], [0.2, 0.8], mask, threshold=0.5)
        with self.assertRaises(ValueError):
            compute_binary_accuracy([0], [0.2], [True], threshold=np.nan)


if __name__ == '__main__':
    unittest.main()
