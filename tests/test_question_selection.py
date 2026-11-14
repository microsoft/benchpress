import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchpress.methods.question_selection import select_profile_anchors
from experiments.sec5_findings.item_level_probes import selection
from experiments.sec5_findings.item_level_probes.run import config_hash


class QuestionSelectionTests(unittest.TestCase):
    def test_distinct_profiles_cover_both_clusters_reproducibly(self):
        profiles = np.array([[0., 0.], [0., 1.], [1., 0.],
                             [99., 100.], [100., 99.], [100., 100.]])
        before = profiles.copy()
        selected = select_profile_anchors(profiles, 2, 42)
        self.assertEqual(sum(selected < 3), 1)
        self.assertEqual(sum(selected >= 3), 1)
        np.testing.assert_array_equal(selected, select_profile_anchors(profiles, 2, 42))
        np.testing.assert_array_equal(profiles, before)

    def test_identical_profiles_still_fill_the_exact_unique_budget(self):
        selected = select_profile_anchors(np.ones((12, 3)), 7, 4)
        np.testing.assert_array_equal(selected, np.arange(7))

    def test_invalid_profile_or_budget_fails(self):
        for profiles, count in [(np.ones((3, 2)), 0), (np.ones((3, 2)), 4),
                                 (np.ones((3, 2)), 1.5), (np.ones(3), 1),
                                 (np.array([[np.nan]]), 1)]:
            with self.subTest(count=count, profiles=profiles), self.assertRaises(ValueError):
                select_profile_anchors(profiles, count, 0)

    def test_paired_scoring_excludes_questions_revealed_by_either_policy(self):
        data = SimpleNamespace(
            answers=np.array([[0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]]),
            col_questions=[np.arange(3), np.arange(3, 6)])
        observed = [np.array([1]), np.array([4])]
        selected = {'allocation': [1, 1], 'seconds': 0.,
                    'selection_hash': config_hash([[1], [4]])}
        config = {
            'base_config_hash': 'base', 'lam': 0.1, 'als': {},
            'ranks': [8, 40], 'code_commit': 'fit',
            'random_sources': {}, 'metric': {'stored_score_scale': 100., 'threshold': 0.5},
        }
        grid = {'seeds': [0], 'budgets': [2]}
        with tempfile.TemporaryDirectory() as root:
            for rank in (8, 40):
                original = dict(base_config_hash='base', rank=rank, methods=['logit'],
                                lam=0.1, als={})
                chash = config_hash(original)
                config['random_sources'][str(rank)] = chash
                path = os.path.join(root, 'variants', chash)
                os.makedirs(path)
                with open(os.path.join(path, 'manifest.json'), 'w') as f:
                    json.dump({'config': original, 'config_hash': chash}, f)
            ctx = SimpleNamespace(
                out=root, chash='base', data=data, held=[1], config=grid,
                subset=lambda values, key: grid[key] if values is None else values)
            args = SimpleNamespace(selection_hash='selected', seeds=None, budgets=None,
                                   code_commit='scorer', ranks=[40])
            units = [{'targets': [1], 'seed': 0, 'budget': 2, 'column': j} for j in (0, 1)]

            def predictions(ctx, path, chash, selected):
                random = '/variants/' in path
                rank = next(r for r in (8, 40) if (
                    config['random_sources'][str(r)] == chash if random else f'/r{r}/' in path))
                pred = [[100., 50., 100., 0., 0., 100.]] if random else [
                    [0., 0., 100., 100., 100., 0.]]
                return np.array(pred), {
                    'rank': rank, 'seed': 0, 'budget': 2,
                    'selection': 'random' if random else 'benchpress_anchors',
                    'selection_hash': config_hash([[1], [4]]),
                    'question_pred_sha256': 'a' * 64, 'seconds': 0.,
                }

            with patch.object(selection, 'Context', return_value=ctx), \
                    patch.object(selection, 'load_manifest', return_value={
                        'config': config, 'config_hash': 'selected'}), \
                    patch.object(selection, 'load_profiles'), \
                    patch.object(selection, 'load_selection', return_value=(observed, selected)), \
                    patch.object(selection, 'load_coresets', return_value=(
                        {'random': [np.array([0]), np.array([0])]}, units)), \
                    patch.object(selection, 'load_question_predictions', side_effect=predictions):
                selection.summarize(args)
            scores = os.path.join(root, 'question_selection', 'selected', 'scores')
            hashes = os.listdir(scores)
            self.assertEqual(len(hashes), 1)
            with open(os.path.join(scores, hashes[0], 'summary.json')) as f:
                summary = json.load(f)
            self.assertEqual(hashes[0], config_hash(summary['config']))
        self.assertEqual(summary['evaluation_masks'][0]['questions'], [2, 5])
        self.assertEqual(summary['config']['ranks'], [40])
        self.assertEqual(summary['config']['fit_code_commit'], 'fit')
        self.assertEqual(set(summary['table']), {'benchpress_r40'})
        self.assertEqual(len(summary['per_seed']), 2)
        for row in summary['per_seed']:
            self.assertEqual(row['n'], 2)
            self.assertEqual(row['accuracy'], 50. if row['selection'] == 'random' else 100.)
        self.assertTrue(all(row['gain_pp'] == 50. for row in summary['paired_gains']))


if __name__ == '__main__':
    unittest.main()
