#!/usr/bin/env python3
"""Named score predictors: transform + completion method + inverse transform."""

import numpy as np

from benchpress.evaluation_harness import make_score_predictor
from benchpress.methods.completers import complete_bias_als


def predict_probit_bias_als_scores(M_train, rank=2, lam=0.1):
    """Score predictor: Probit transform + Bias ALS completion."""
    predict_fn = make_score_predictor(
        complete_bias_als, 'probit', rank=rank, lam=lam, normalize=False)
    return predict_fn(M_train)


def predict_logit_bias_als_scores(M_train, rank=2, lam=0.1):
    """Score predictor: Logit transform + Bias ALS completion."""
    predict_fn = make_score_predictor(
        complete_bias_als, 'logit', rank=rank, lam=lam, normalize=False)
    return predict_fn(M_train)


def predict_benchpress_scores(M_train):
    """BenchPress default score predictor: Logit Bias ALS with lambda=0.1 and rank=2."""
    return predict_logit_bias_als_scores(M_train, rank=2, lam=0.1)


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

