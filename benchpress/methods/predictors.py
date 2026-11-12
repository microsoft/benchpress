#!/usr/bin/env python3
"""Named score predictors: transform + completion method + inverse transform."""

import numpy as np

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_METRICS,
    col_normalize,
    col_denormalize,
    make_score_predictor,
)
from benchpress.methods.completers import (
    complete_bias_als,
    complete_group_bias_als,
    complete_model_mean,
    fit_binary_group_low_rank,
)


def predict_probit_bias_als_scores(M_train, rank=2, lam=0.1, metric=None,
                                   benchmark_ids=None):
    """Score predictor: Probit transform + Bias ALS completion."""
    predict_fn = make_score_predictor(
        complete_bias_als, 'probit', metric=metric, benchmark_ids=benchmark_ids,
        rank=rank, lam=lam, normalize=False)
    return predict_fn(M_train)


def predict_logit_bias_als_scores(M_train, rank=2, lam=0.1, metric=None,
                                  benchmark_ids=None):
    """Score predictor: Logit transform + Bias ALS completion."""
    predict_fn = make_score_predictor(
        complete_bias_als, 'logit', metric=metric, benchmark_ids=benchmark_ids,
        rank=rank, lam=lam, normalize=False)
    return predict_fn(M_train)


def predict_benchpress_scores(M_train, metric=None, benchmark_ids=None):
    """BenchPress default score predictor: Logit Bias ALS with lambda=0.1 and rank=2."""
    if (metric is None) != (benchmark_ids is None):
        raise ValueError(
            "metric and benchmark_ids must be provided together.")
    if metric is None and benchmark_ids is None:
        if M_train.shape[1] == len(BENCH_IDS):
            metric = BENCH_METRICS
            benchmark_ids = BENCH_IDS
    return predict_logit_bias_als_scores(
        M_train, rank=2, lam=0.1, metric=metric, benchmark_ids=benchmark_ids)


def fit_question_score_variant(M_train, column_groups, method, rank=8, lam=0.1):
    """Return (completed percentage answers, fit diagnostics) from observed data.

    ALS variants keep the existing column standardization and 40-iteration,
    10-initialization schedule. Only the feature transform and grouped offset
    change. Bernoulli fitting operates directly on 0/1 answers.
    """
    if method not in ('raw', 'group_logit', 'group_raw', 'group_binary',
                      'raw_shrink', 'task_mean_ridge'):
        raise ValueError(f'unknown question-score variant: {method}')
    matrix = np.asarray(M_train, dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError('M_train must be a nonempty 2-D matrix')
    if not np.isin(matrix[np.isfinite(matrix)], [0, 100]).all() or np.isinf(matrix).any():
        raise ValueError('observed question scores must be 0 or 100')
    if method == 'raw_shrink':
        return predict_shrunk_group_residual_scores(matrix, column_groups, rank, lam)
    if method == 'task_mean_ridge':
        return predict_group_mean_ridge_scores(matrix, column_groups, lam)
    if method == 'group_binary':
        pred, diagnostics = fit_binary_group_low_rank(
            matrix / 100, column_groups, rank=rank, lam=lam)
        return 100 * pred, {'optimizer': diagnostics}
    ids = [f'q{j}' for j in range(matrix.shape[1])]
    options = {'rank': rank, 'lam': lam}
    if method == 'raw':
        completer = complete_bias_als
        options['normalize'] = False
    else:
        completer = complete_group_bias_als
        options['column_groups'] = column_groups
    predictor = make_score_predictor(
        completer, 'logit' if method == 'group_logit' else 'identity',
        metric={name: {'type': 'pct', 'range': [0, 100]} for name in ids},
        benchmark_ids=ids, **options)
    return predictor(matrix), {'iterations': 40, 'n_ensemble': 10, 'base_seed': 42}


def predict_shrunk_group_residual_scores(matrix, column_groups, rank=8, lam=0.1):
    """Raw ALS plus empirical-Bayes group residuals from fully observed rows.

    Returns linear score predictions and training-only prior moments. The group
    correction targets the missing subset, including its finite-population
    relationship to the observed residual mean. Linear item scores are not
    clipped before aggregation; observed scores are restored exactly.
    """
    matrix = np.asarray(matrix, dtype=float)
    groups = np.asarray(column_groups)
    if groups.shape != (matrix.shape[1],):
        raise ValueError('column_groups must have one label per column')
    observed = np.isfinite(matrix)
    reference = np.flatnonzero(observed.all(1))
    if len(reference) < 2:
        raise ValueError('shrinkage needs at least two fully observed reference rows')
    z, mean, scale = col_normalize(matrix)
    fitted = col_denormalize(complete_bias_als(
        z, rank=rank, lam=lam, normalize=False, preserve_observed=False), mean, scale)
    result = fitted.copy()
    moments = []
    for group in np.unique(groups):
        js = np.flatnonzero(groups == group)
        if len(js) < 2:
            raise ValueError('residual variance needs at least two items per group')
        residual = matrix[:, js] - fitted[:, js]
        reference_residual = residual[reference]
        reference_means = reference_residual.mean(1)
        prior_mean = float(reference_means.mean())
        prior_var = float(reference_means.var(ddof=1))
        within_var = float(reference_residual.var(axis=1, ddof=1).mean())
        n = observed[:, js].sum(1)
        sample_mean = np.divide(np.nansum(residual, axis=1), n,
                                out=np.full(len(n), prior_mean), where=n > 0)
        sampling_var = (1 - n / len(js)) * within_var / np.maximum(n, 1)
        weight = np.divide(prior_var, prior_var + sampling_var,
                           out=np.zeros(len(n)), where=(prior_var + sampling_var) > 0)
        weight[n == 0] = 0
        full_mean = prior_mean + weight * (sample_mean - prior_mean)
        hidden_mean = np.divide(len(js) * full_mean - n * sample_mean, len(js) - n,
                                out=np.zeros(len(n)), where=n < len(js))
        result[:, js] += hidden_mean[:, None]
        moments.append({'group': str(group), 'prior_mean': prior_mean,
                        'prior_variance': prior_var, 'within_variance': within_var,
                        'n_reference_rows': len(reference)})
    result[observed] = matrix[observed]
    return result, {'group_prior': moments, 'item_clipping': False,
                    'iterations': 40, 'n_ensemble': 10, 'base_seed': 42}


def predict_group_mean_ridge_scores(matrix, column_groups, lam=0.1):
    """Predict missing group means from revealed group means with shared ridge.

    Fully observed rows provide supervised examples using the target rows'
    common mask. Training-only feature standardization, intercept, and fixed
    ridge penalty are used. Returns group-mean predictions broadcast over each
    missing group, plus diagnostics; these are not individual-item forecasts.
    """
    from sklearn.linear_model import Ridge

    matrix = np.asarray(matrix, dtype=float)
    groups = np.asarray(column_groups)
    if groups.shape != (matrix.shape[1],) or not np.isfinite(lam) or lam <= 0:
        raise ValueError('invalid column_groups or ridge penalty')
    observed = np.isfinite(matrix)
    reference, target = np.flatnonzero(observed.all(1)), np.flatnonzero(~observed.all(1))
    if len(reference) < 2 or not len(target):
        raise ValueError('ridge needs reference rows and partially observed target rows')
    mask = observed[target[0]]
    if not np.all(observed[target] == mask):
        raise ValueError('target rows must share the same revealed questions')
    features, responses, hidden = [], [], []
    for group in np.unique(groups):
        js = np.flatnonzero(groups == group)
        revealed, missing = js[mask[js]], js[~mask[js]]
        if not len(revealed) or not len(missing):
            raise ValueError('every group needs revealed and hidden questions')
        features.append(matrix[:, revealed].mean(1))
        responses.append(matrix[np.ix_(reference, missing)].mean(1))
        hidden.append(missing)
    features = np.stack(features, axis=1)
    responses = np.stack(responses, axis=1)
    mean, scale = features[reference].mean(0), features[reference].std(0)
    scale[scale < 1e-8] = 1
    features = (features - mean) / scale
    regressor = Ridge(alpha=lam, solver='svd').fit(features[reference], responses)
    estimates = regressor.predict(features[target])
    result = matrix.copy()
    for j, missing in enumerate(hidden):
        result[np.ix_(target, missing)] = estimates[:, j, None]
    return result, {'n_reference_rows': len(reference), 'n_features': features.shape[1],
                    'prediction_target': 'missing group mean, broadcast to its questions',
                    'feature_standardization': 'reference rows only', 'lam': lam}


def predict_prior_residual_scores(M_train, prior_mean, prior_scale,
                                  correction_weight=1.0, rank=2, lam=0.1):
    """Complete signed residuals around a supplied prior, preserving observations.

    All three matrices must have the same nonempty 2-D shape. Only M_train may
    contain NaNs; prior_scale must be finite and strictly positive. Callers own
    training-only estimation of the prior and validation of correction_weight.
    Weight zero returns the prior at missing cells. Weight one adds the full
    Bias ALS correction, fit in prior-standardized units without logit or
    additional column normalization. This is not posterior interval inference.
    """
    matrix = np.asarray(M_train, dtype=float)
    mean = np.asarray(prior_mean, dtype=float)
    scale = np.asarray(prior_scale, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("M_train must be a nonempty 2-D matrix")
    if mean.shape != matrix.shape or scale.shape != matrix.shape:
        raise ValueError("Prior matrices must have the same shape as M_train")
    if np.isinf(matrix).any() or not np.isfinite(mean).all():
        raise ValueError("Scores cannot be infinite and prior_mean must be finite")
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("prior_scale must be finite and strictly positive")
    if not np.isfinite(correction_weight) or not 0 <= correction_weight <= 1:
        raise ValueError("correction_weight must be between zero and one")
    if not isinstance(rank, (int, np.integer)) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be strictly positive")
    observed = np.isfinite(matrix)
    result = mean.copy()
    if correction_weight > 0:
        if not observed.any():
            raise ValueError("Residual fitting requires observed scores")
        standardized = (matrix - mean) / scale
        residual = complete_bias_als(
            standardized, rank=rank, lam=lam, normalize=False)
        if not np.isfinite(residual).all():
            raise ValueError("Bias ALS produced nonfinite prior residuals")
        result += correction_weight * scale * residual
    result[observed] = matrix[observed]
    return result


def predict_logit_model_mean_scores(M_train, metric=None, benchmark_ids=None):
    """Score predictor: Logit transform + row mean completion."""
    if (metric is None) != (benchmark_ids is None):
        raise ValueError(
            "metric and benchmark_ids must be provided together.")
    if metric is None and benchmark_ids is None:
        if M_train.shape[1] == len(BENCH_IDS):
            metric = BENCH_METRICS
            benchmark_ids = BENCH_IDS
    predict_fn = make_score_predictor(
        complete_model_mean, 'logit', metric=metric, benchmark_ids=benchmark_ids)
    return predict_fn(M_train)


def predict_benchmark_median_scores(M_train):
    """No-information baseline: predict every cell with its benchmark column's median.

    The median is computed from the observed entries in `M_train` for each
    column (NaNs are ignored). Columns with no observed scores produce NaN
    predictions for that column.

    This is the canonical "k=0" / no-probe baseline used in the hero figure
    and probe-evaluation curves.
    """
    M = np.asarray(M_train, dtype=float)
    col_medians = np.nanmedian(M, axis=0)  # shape (n_bench,)
    return np.broadcast_to(col_medians, M.shape).copy()
