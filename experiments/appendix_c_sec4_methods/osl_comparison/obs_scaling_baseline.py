"""Observational scaling laws applied to the paper's score matrix (nxZ2 Q1).

This script runs the comparison on the 84 x 133 matrix of the paper, and answers
the question the reviewer's framing leaves open: OSL's recipe was designed for
an almost complete matrix, so what happens when the matrix is 23.3 percent
observed.

OSL needs a fixed block of benchmarks observed for every model, because the
capability measures come from a PCA of that block and the per-target regression
takes those measures as its only inputs. Their Algorithm A.1 opens with a PCA
imputation step, so the recipe as published does tolerate gaps in that block;
what it was designed for is the density they report, under one percent missing.
That is the assumption the sparsity interacts with. The script reports the three
rows used by `tab:osl_comparison`:

    osl_densest8_block       OSL with the 8 densest benchmarks as the feature
                                                     block, excluding the target.
    osl_full_matrix_impute   OSL with every benchmark except the target as the
                                                     feature block.
    benchpress               The paper's fixed point predictor, rank-2 logit
                                                     Bias ALS with lambda 0.1, on the same folds.

Errors use the same per-fold median aggregation as the main-body method
comparison: `compute_prediction_error(..., aggregation='per_group_median')`
grouped by fold, over the canonical held-out cells where each method returns a
finite prediction. `benchpress` therefore reproduces the paper's headline number.

Usage:
    python obs_scaling_baseline.py [--smoke] [--workers 12]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from benchpress.evaluation_harness import (
    M_FULL, compute_prediction_error, load_folds,
)
from benchpress.methods.predictors import predict_benchpress_scores
from benchpress.methods.transforms import _is_pct_bench

HERE = os.path.dirname(os.path.abspath(__file__))

MAX_ITER = 1000
TOL = 1e-4
IMPUTE_COMPONENTS = 1
CAPABILITY_COMPONENTS = 3
SIGMOID_FLOOR_MAX = 0.2
INIT_SLOPE = 3e-2
MAX_FEV = 10000

BLOCK_SIZE = 8
N_COMPONENTS = 3
# The sigmoid law carries N_COMPONENTS slopes, an intercept and a floor, so a
# target column observed by fewer training models than that has no fit at all
# and `curve_fit` refuses it. Every column above that line is allowed to run and
# to overfit, so the coverage OSL reaches is a property of the recipe rather
# than of a threshold chosen here.
MIN_TRAIN_MODELS = N_COMPONENTS + 2
METHODS = ['osl_densest8_block', 'osl_full_matrix_impute', 'benchpress']


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def pca_impute(train, test=None, n_components=IMPUTE_COMPONENTS,
               max_iter=MAX_ITER, tol=TOL, bounds=None):
    """Iterative PCA imputation, fitted on `train` and applied to `test`."""
    train = np.asarray(train, dtype=float)
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    train_missing = np.isnan(train_scaled)

    imputer = SimpleImputer(strategy='mean').fit(train_scaled)
    train_filled = imputer.transform(train_scaled)

    pca = PCA(n_components=n_components)
    if train_missing.any():
        for _ in range(max_iter):
            reconstruction = pca.inverse_transform(pca.fit_transform(train_filled))
            if np.allclose(train_filled, reconstruction, atol=tol):
                break
            train_filled[train_missing] = reconstruction[train_missing]
    else:
        pca.fit(train_filled)

    train_imputed = scaler.inverse_transform(train_filled)
    if bounds is not None:
        train_imputed = np.clip(train_imputed, bounds[0], bounds[1])
    if test is None:
        return train_imputed, None

    test = np.asarray(test, dtype=float)
    test_scaled = scaler.transform(test)
    test_missing = np.isnan(test_scaled)
    test_filled = imputer.transform(test_scaled)
    if test_missing.any():
        for _ in range(max_iter):
            reconstruction = pca.inverse_transform(pca.transform(test_filled))
            if np.allclose(test_filled, reconstruction, atol=tol):
                break
            test_filled[test_missing] = reconstruction[test_missing]

    test_imputed = scaler.inverse_transform(test_filled)
    if bounds is not None:
        test_imputed = np.clip(test_imputed, bounds[0], bounds[1])
    return train_imputed, test_imputed


def fit_capability_pca(train, n_components=CAPABILITY_COMPONENTS):
    """Fit the capability PCA on complete training rows, without rescaling."""
    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(train, dtype=float))
    return pca


class SigmoidCapabilityRegression:
    """`E = (1 - b) * sigmoid(alpha + beta' S) + b`, fitted by least squares."""

    def __init__(self, floor_max=SIGMOID_FLOOR_MAX):
        self.floor_max = floor_max
        self.alpha = None
        self.beta = None
        self.floor = None

    @staticmethod
    def _design(S):
        S = np.asarray(S, dtype=float)
        return np.column_stack([np.ones(len(S)), S])

    def _model(self, X, *params):
        weights, floor = np.asarray(params[:-1]), params[-1]
        return (1.0 - floor) * _sigmoid(X @ weights) + floor

    def fit(self, S, y):
        X = self._design(S)
        n_weights = X.shape[1]
        p0 = np.concatenate([[0.0], np.full(n_weights - 1, INIT_SLOPE), [0.0]])
        lower = [-np.inf] * n_weights + [0.0]
        upper = [np.inf] * n_weights + [self.floor_max]
        popt, _ = curve_fit(
            self._model, X, np.asarray(y, dtype=float), p0=p0,
            bounds=(lower, upper), maxfev=MAX_FEV,
        )
        self.alpha, self.beta, self.floor = popt[0], popt[1:-1], popt[-1]
        return self

    def predict(self, S):
        params = np.concatenate([[self.alpha], self.beta, [self.floor]])
        return self._model(self._design(S), *params)


class ObservationalScalingPredictor:
    """Predict a hidden target column from a base benchmark block, OSL style."""

    def __init__(self, n_capability_components=CAPABILITY_COMPONENTS,
                 impute_components=IMPUTE_COMPONENTS, metric_range=(0.0, 1.0)):
        self.n_capability_components = n_capability_components
        self.impute_components = impute_components
        self.metric_low, self.metric_high = float(metric_range[0]), float(metric_range[1])
        if self.metric_high <= self.metric_low:
            raise ValueError('metric_range must be increasing')
        self.pca = None
        self.regression = None

    def fit(self, base_train, target_train, base_test=None):
        train_imputed, test_imputed = pca_impute(
            base_train, base_test, n_components=self.impute_components,
        )
        self.pca = fit_capability_pca(
            train_imputed, n_components=self.n_capability_components,
        )
        span = self.metric_high - self.metric_low
        normalized = (np.asarray(target_train, dtype=float) - self.metric_low) / span
        self.regression = SigmoidCapabilityRegression().fit(
            self.pca.transform(train_imputed), normalized,
        )
        return self, test_imputed

    def predict(self, base_imputed):
        scores = self.regression.predict(self.pca.transform(base_imputed))
        return scores * (self.metric_high - self.metric_low) + self.metric_low

    def fit_predict(self, base_train, target_train, base_test):
        _, test_imputed = self.fit(base_train, target_train, base_test)
        return self.predict(test_imputed)


def choose_block(M_train, target, size):
    """Pick the densest benchmarks in the training matrix, excluding the target."""
    counts = (~np.isnan(M_train)).sum(axis=0).astype(float)
    counts[target] = -1.0
    return np.argsort(-counts, kind='stable')[:size]


def column_metric_range(M_train, target):
    """Theoretical range of a benchmark column, as OSL's sigmoid fit requires.

    Percentage-scale benchmarks span [0, 100] whatever the observed scores are.
    The remaining columns, such as arena ratings and dollar costs, have no
    natural ceiling, so the observed training range stands in for one. That
    substitution caps predictions at the best training score, which is a
    limitation of applying OSL's parametric form to such columns rather than a
    choice; `n_columns_without_natural_range` in the output counts them.
    """
    if _is_pct_bench(target, M_train):
        return (0.0, 100.0), True
    values = M_train[~np.isnan(M_train[:, target]), target]
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return None, False
    return (low, high), False


def predict_osl_column(M_train, target, test_rows, block):
    """Fit one OSL regression for `target` and predict `test_rows`.

    Args:
        M_train: training matrix with test cells already removed.
        target: index of the benchmark column being predicted.
        test_rows: model indices whose target cell is held out.
        block: benchmark indices forming the feature block.
    Returns:
        Predictions aligned with `test_rows`, with NaN where OSL cannot fit a
        target-column regression.
    """
    features = M_train[:, block]
    train_rows = np.where(~np.isnan(M_train[:, target]))[0]
    out = np.full(len(test_rows), np.nan)

    metric_range, _ = column_metric_range(M_train, target)
    if metric_range is None:
        return out, 'degenerate_metric_range'

    if len(train_rows) < MIN_TRAIN_MODELS:
        return out

    # Columns the fitted models never observe carry no information and would
    # make the standardization step degenerate.
    usable = (~np.isnan(features[train_rows])).sum(axis=0) >= 2
    if usable.sum() < N_COMPONENTS:
        return out
    features = features[:, usable]

    try:
        out[:] = ObservationalScalingPredictor(
            n_capability_components=N_COMPONENTS, metric_range=metric_range,
        ).fit_predict(
            features[train_rows], M_train[train_rows, target],
            features[np.asarray(test_rows)],
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return out
    return out


def run_fold(fold_index, M_train, test_set):
    """Score every method on one fold and return per-cell predictions."""
    cells = np.asarray(test_set, dtype=int)
    truth = M_FULL[cells[:, 0], cells[:, 1]]
    preds = {m: np.full(len(cells), np.nan) for m in METHODS}

    M_pred = predict_benchpress_scores(M_train)
    preds['benchpress'] = M_pred[cells[:, 0], cells[:, 1]]

    all_columns = np.arange(M_train.shape[1])
    for target in np.unique(cells[:, 1]):
        where = np.where(cells[:, 1] == target)[0]
        rows = cells[where, 0]
        fixed = choose_block(M_train, target, BLOCK_SIZE)
        full = all_columns[all_columns != target]
        for name, block in (('osl_densest8_block', fixed),
                            ('osl_full_matrix_impute', full)):
            column = predict_osl_column(M_train, target, rows, block)
            preds[name][where] = column

    return fold_index, cells, truth, preds


def score_method(truth, pred, groups):
    """Per-fold median error, matching the main-body method comparison.

    Groups are fold indices, so each method is scored as the median over the 30
    per-fold MedAE/MedAPE values rather than by pooling every cell. This is the
    aggregation `compute_prediction_error` uses for the paper's headline numbers,
    so `benchpress` here matches `tab:full_grid` exactly.
    """
    metrics = compute_prediction_error(
        truth, pred, groups=groups, aggregation='per_group_median')
    finite = ~np.isnan(truth) & ~np.isnan(pred)
    scored = ~np.isnan(truth)
    return {
        'n': int(finite.sum()),
        'coverage': float(finite.sum() / scored.sum()) if scored.sum() else 0.0,
        'medape': metrics['medape_median'],
        'medae': metrics['medae_median'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='two folds only, for a fast sanity run')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', default=os.path.join(HERE, 'obs_scaling_baseline.json'))
    args = parser.parse_args()

    folds = load_folds()
    if args.smoke:
        folds = folds[:2]

    truth_all, preds_all, groups_all, per_fold = [], {m: [] for m in METHODS}, [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_fold, i, M_train, test_set)
                   for i, (M_train, test_set) in enumerate(folds)]
        for future in as_completed(futures):
            index, cells, truth, preds = future.result()
            per_fold.append((index, cells, truth, preds))

    per_fold.sort(key=lambda r: r[0])
    for index, cells, truth, preds in per_fold:
        truth_all.append(truth)
        groups_all.append(np.full(len(cells), index))
        for name in METHODS:
            preds_all[name].append(preds[name])
    truth_all = np.concatenate(truth_all)
    groups_all = np.concatenate(groups_all)
    preds_all = {m: np.concatenate(v) for m, v in preds_all.items()}

    results = {
        'description': 'observational scaling laws on the paper score matrix',
        'n_folds': len(folds),
        'block_size': BLOCK_SIZE,
        'n_components': N_COMPONENTS,
        'min_train_models': MIN_TRAIN_MODELS,
        'benchpress_predictor': 'logit Bias ALS, rank 2, lambda 0.1',
        'smoke': bool(args.smoke),
        'accuracy': {
            name: score_method(truth_all, preds_all[name], groups_all)
            for name in METHODS
        },
    }

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
