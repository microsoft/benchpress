"""Information-boundary checks for the fixed-snapshot new-model experiment."""

import numpy as np
import pytest

from experiments.sec5_findings.prospective_update.run_random_reveals import predict_unit


def test_target_hidden_scores_and_new_columns_do_not_enter_fit(monkeypatch):
    import benchpress.methods.predictors as predictors

    reference = np.array([[10., 20., np.nan, 40., 50., 60., 70., 80.],
                          [20., 30., 40., 50., 60., 70., 80., 90.]])
    ids = [f'b{i}' for i in range(8)]
    values = {bid: float(i + 1) for i, bid in enumerate(ids)}
    values['august_only'] = 999.
    seen = []

    def predictor(matrix, **kwargs):
        seen.append(matrix.copy())
        result = np.nan_to_num(matrix.copy(), nan=50.)
        result[-1, :] = np.nanmean(matrix[:-1], axis=0)
        return result

    for name in ('predict_benchpress_scores', 'predict_logit_bias_als_scores',
                 'predict_logit_model_mean_scores'):
        monkeypatch.setattr(predictors, name, predictor)
    first = predict_unit(reference, ids, {}, 'new-model', values, 1, 0)
    revealed = set(first['revealed_benchmarks'])
    assert len(revealed) == 1
    assert len(first['rows']) == 7
    assert all(row['benchmark_id'] in ids for row in first['rows'])
    for matrix in seen:
        np.testing.assert_equal(matrix[:-1], reference)
        assert matrix.shape == (3, 8)
        assert np.isfinite(matrix[-1]).sum() == 1
    altered = {bid: value if bid in revealed else value + 1000
               for bid, value in values.items()}
    seen.clear()
    second = predict_unit(reference, ids, {}, 'new-model', altered, 1, 0)
    assert first['revealed_benchmarks'] == second['revealed_benchmarks']
    for left, right in zip(first['rows'], second['rows']):
        assert left['benchpress'] == right['benchpress']
        assert left['offsets_only'] == right['offsets_only']
        assert left['model_mean'] == right['model_mean']


def test_reveals_are_nested_and_common_test_scores_are_fixed(monkeypatch):
    import benchpress.methods.predictors as predictors

    ids = [f'b{i}' for i in range(16)]
    reference = np.full((2, 16), 50.)
    values = {bid: float(i) for i, bid in enumerate(ids)}
    for name in ('predict_benchpress_scores', 'predict_logit_bias_als_scores',
                 'predict_logit_model_mean_scores'):
        monkeypatch.setattr(predictors, name, lambda matrix, **kwargs: np.full_like(matrix, 50.))
    small = predict_unit(reference, ids, {}, 'new-model', values, 1, 2)
    large = predict_unit(reference, ids, {}, 'new-model', values, 10, 2)
    assert small['revealed_benchmarks'] == large['revealed_benchmarks'][:1]
    assert {r['benchmark_id'] for r in small['rows'] if r['common_test']} == {
        r['benchmark_id'] for r in large['rows']}


def test_insufficient_hidden_scores_are_rejected():
    ids = [f'b{i}' for i in range(6)]
    with pytest.raises(ValueError, match='five hidden'):
        predict_unit(np.full((2, 6), 50.), ids, {}, 'new-model',
                     dict.fromkeys(ids, 50.), 3, 0)
