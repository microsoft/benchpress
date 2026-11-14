import glob
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.sec5_findings.item_level_probes import baselines
from experiments.sec5_findings.item_level_probes.run import hidden_estimate


class QuestionBaselineTests(unittest.TestCase):
    def setUp(self):
        import benchpred

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.answers = np.array([
            [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1]])
        self.ctx = SimpleNamespace(
            out=self.temp.name, chash='base', sel=[0, 1, 2, 3], held=[4, 5],
            config={'seeds': [0], 'budgets': [2], 'selections': ['random', 'anchors']},
            data=SimpleNamespace(answers=self.answers, col_questions=[np.arange(6)]))
        self.ctx.subset = lambda values, key: self.ctx.config[key] if values is None else values
        estimator = benchpred.AIPWPred().fit(self.answers[:4].astype(float), 2, seed=0)
        estimator.compressed_data_indices = np.array([0, 1])
        observed = self.answers[4:, :2]
        mean = (100. * hidden_estimate(estimator.predict(observed), observed, 6)).tolist()
        path = baselines.estimator_path(self.ctx, 0, 2, 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'config_hash': 'base', 'seed': 0, 'budget': 2, 'column': 0,
                'allocation': 2, 'targets': [4, 5], 'gp_irt_anchor_points': [0, 1],
                'coresets': {'random': [0, 1], 'anchors': [0, 1]},
                'pred': {s: {'aipw': mean} for s in ('random', 'anchors')}}, f)
        self.args = SimpleNamespace(
            code_commit='test', seeds=None, budgets=None, selections=None, columns=[0])
        self.model = SimpleNamespace(
            A=np.ones((1, 1, 6)), B=np.arange(-2, 4).reshape(1, 1, 6),
            anchor_points=np.array([0, 1]))

    def run_fit(self):
        with patch.object(baselines, 'Context', return_value=self.ctx), \
                patch.object(baselines, 'benchpred_commit', return_value=baselines.BENCHPRED_COMMIT), \
                patch('benchpred.PIRTPred') as irt:
            irt.return_value.fit.return_value = self.model
            baselines.fit(self.args)
        return irt

    def result(self):
        path = glob.glob(os.path.join(self.temp.name, 'question_baselines', '*',
                                      's0', 'c00', 'random_B2.npz'))[0]
        with np.load(path) as z:
            return {method: z[method].copy() for method in baselines.METHODS}, path

    def test_native_predictions_vary_by_question_and_preserve_observations(self):
        self.run_fit()
        arrays, _ = self.result()
        for pred in arrays.values():
            np.testing.assert_array_equal(pred[:, :2], 100. * self.answers[4:, :2])
            self.assertTrue(np.all(np.ptp(pred[:, 2:], axis=1) > 0))
        self.assertTrue(np.all(np.diff(arrays['gp_irt_items'][:, 2:], axis=1) < 0))

    def test_hidden_target_answers_do_not_enter_predictions(self):
        self.run_fit()
        before, _ = self.result()
        self.ctx.data.answers[4:, 2:] = 1 - self.ctx.data.answers[4:, 2:]
        # A distinct output root forces fresh prediction rather than cache reuse.
        original = self.ctx.out
        self.ctx.out = os.path.join(self.temp.name, 'rerun')
        path = baselines.estimator_path(self.ctx, 0, 2, 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(baselines.estimator_path(SimpleNamespace(out=original), 0, 2, 0)) as f:
            frozen = json.load(f)
        with open(path, 'w') as f:
            json.dump(frozen, f)
        self.run_fit()
        path = glob.glob(os.path.join(self.ctx.out, 'question_baselines', '*',
                                      's0', 'c00', 'random_B2.npz'))[0]
        with np.load(path) as z:
            for method in baselines.METHODS:
                np.testing.assert_array_equal(z[method], before[method])

    def test_resume_checks_raw_checksum_before_skipping(self):
        self.run_fit()
        irt = self.run_fit()
        irt.assert_not_called()
        _, path = self.result()
        with open(path, 'ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_fit()


if __name__ == '__main__':
    unittest.main()
