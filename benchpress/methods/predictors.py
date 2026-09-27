#!/usr/bin/env python3
"""Named score predictors: transform + completion method + inverse transform."""

import numpy as np

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_METRICS,
    make_score_predictor,
)
from benchpress.methods.completers import complete_bias_als, complete_model_mean


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
