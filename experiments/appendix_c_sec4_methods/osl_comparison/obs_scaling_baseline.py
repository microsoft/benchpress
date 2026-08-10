"""Observational scaling laws applied to the paper's score matrix (nxZ2 Q1).

The companion script `obs_scaling_setup.py` runs both predictors on OSL's data.
This one runs the comparison the other way round, on the 84 x 133 matrix of the
paper, and answers the question the reviewer's framing leaves open: OSL's recipe
was designed for an almost complete matrix, so what happens when the matrix is
23.3 percent observed.

OSL needs a fixed block of benchmarks observed for every model, because the
capability measures come from a PCA of that block and the per-target regression
takes those measures as its only inputs. Their Algorithm A.1 opens with a PCA
imputation step, so the recipe as published does tolerate gaps in that block;
what it was designed for is the density they report, under one percent missing.
That is the assumption the sparsity interacts with, so the script reports two
separate things.

Applicability. How many models have a complete block of the T densest
benchmarks, and how many benchmarks have enough observed models to fit a
regression at all. No predictor is involved; this is a property of the matrix.

Accuracy. Five predictors on the canonical shared folds:

  osl            OSL as published, meaning Algorithm A.1 including its PCA
                 imputation. The feature block is the 8 densest benchmarks in
                 the training matrix, excluding the target. This is the arm to
                 compare against.
  osl_no_impute  The same block with the imputation step removed, so a model is
                 scored only when it observes the whole block. This is not their
                 recipe; it is a diagnostic that isolates how much of OSL's
                 reach on this matrix is carried by imputation alone.
  osl_full       The most generous variant: every benchmark except the target
                 enters the block, then PCA imputation. Further from what OSL
                 describes, but it removes any doubt that the 8 benchmark block
                 was chosen unfavourably.
  osl_impute_r*  Algorithm A.1 step 1 alone, run over the whole matrix and read
                 as a completer: no capability PCA and no sigmoid regression.
                 This is what `base_llm_eval_pca_scaling.ipynb` does, and it is
                 the arm to quote when the question is about building on their
                 imputation rather than about their full recipe. The suffix is
                 the reconstruction rank; their notebooks fix it at 1, and the
                 rest of `IMPUTE_COMPONENT_GRID` is swept so that the arm is not
                 held to a rank that happens to suit their matrix.
  benchpress     The paper's fixed point predictor, rank-2 logit Bias ALS with
                 lambda 0.1, on the same folds.

Errors are reported twice: pooled over each method's own covered cells, and
pooled over the cells every scoring method reaches, so that a method declining
the hardest cells is not credited with an accuracy advantage.

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

from benchpress.evaluation_harness import (
    BENCH_IDS, BENCH_NAMES, M_FULL, MODEL_IDS, OBSERVED,
    compute_prediction_error, load_folds,
)
from benchpress.methods.predictors import predict_benchpress_scores
from benchpress.methods.transforms import _is_pct_bench

from observational_scaling import ObservationalScalingPredictor, pca_impute

HERE = os.path.dirname(os.path.abspath(__file__))

BLOCK_SIZE = 8
N_COMPONENTS = 3
# The sigmoid law carries N_COMPONENTS slopes, an intercept and a floor, so a
# target column observed by fewer training models than that has no fit at all
# and `curve_fit` refuses it. Every column above that line is allowed to run and
# to overfit, so the coverage OSL reaches is a property of the recipe rather
# than of a threshold chosen here.
MIN_TRAIN_MODELS = N_COMPONENTS + 2
DENSITY_GRID = [3, 5, 8, 10, 12, 15, 20]
COVERAGE_GRID = [8, 10, 15, 20, 30, 47]
# Reconstruction ranks for the imputation-only arm. 1 is what OSL's notebooks
# fix; the rest are swept so that the arm is reported at its best rank rather
# than at the one their matrix happened to need. Selecting that rank on the test
# cells favours the arm, which is the direction to err in for a baseline.
IMPUTE_COMPONENT_GRID = [1, 2, 3, 5]

COLUMN_METHODS = ['osl', 'osl_no_impute', 'osl_full']
IMPUTE_METHODS = ['osl_impute_r%d' % k for k in IMPUTE_COMPONENT_GRID]
METHODS = COLUMN_METHODS + IMPUTE_METHODS + ['benchpress']


def applicability():
    """Count how much of the matrix OSL's complete-block requirement admits."""
    per_bench = OBSERVED.sum(axis=0)
    order = np.argsort(-per_bench, kind='stable')

    complete_block = []
    for size in DENSITY_GRID:
        block = order[:size]
        n = int(OBSERVED[:, block].all(axis=1).sum())
        complete_block.append({
            'block_size': size,
            'n_models_with_complete_block': n,
            'frac_models': round(n / OBSERVED.shape[0], 4),
        })

    enough_models = [{
        'min_observed_models': k,
        'n_benchmarks': int((per_bench >= k).sum()),
        'frac_benchmarks': round(float((per_bench >= k).mean()), 4),
    } for k in COVERAGE_GRID]

    return {
        'n_models': int(OBSERVED.shape[0]),
        'n_benchmarks': int(OBSERVED.shape[1]),
        'observed_cells': int(OBSERVED.sum()),
        'fill_rate': round(float(OBSERVED.mean()), 4),
        'densest_benchmarks': [
            {'id': BENCH_IDS[j], 'name': BENCH_NAMES[BENCH_IDS[j]],
             'n_models': int(per_bench[j])}
            for j in order[:BLOCK_SIZE]
        ],
        'models_with_complete_block': complete_block,
        'benchmarks_with_enough_models': enough_models,
    }


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


def osl_column(M_train, target, test_rows, block, complete_only):
    """Fit one OSL regression for `target` and predict `test_rows`.

    Args:
        M_train: training matrix with test cells already removed.
        target: index of the benchmark column being predicted.
        test_rows: model indices whose target cell is held out.
        block: benchmark indices forming the feature block.
        complete_only: when True, restrict both the fitted models and the
            predicted models to those observing every benchmark in the block,
            which is what OSL requires when its imputation is not used.

    Returns:
        A pair `(predictions, reason)`. `predictions` is aligned with
        `test_rows` and holds NaN where the model cannot be scored. `reason`
        names the gate that stopped the fit, or 'ok' when it ran, so that a
        method reporting no coverage can be attributed to a stated requirement
        of the recipe rather than to a silent failure.
    """
    features = M_train[:, block]
    train_rows = np.where(~np.isnan(M_train[:, target]))[0]
    out = np.full(len(test_rows), np.nan)

    metric_range, _ = column_metric_range(M_train, target)
    if metric_range is None:
        return out, 'degenerate_metric_range'

    if complete_only:
        complete = ~np.isnan(features).any(axis=1)
        train_rows = train_rows[complete[train_rows]]
        keep = complete[test_rows]
    else:
        keep = np.ones(len(test_rows), dtype=bool)

    if len(train_rows) < MIN_TRAIN_MODELS:
        return out, ('too_few_models_with_complete_block' if complete_only
                     else 'fewer_train_models_than_free_parameters')
    if not keep.any():
        return out, 'no_test_model_with_complete_block'

    rows = np.asarray(test_rows)[keep]
    # Columns the fitted models never observe carry no information and would
    # make the standardization step degenerate.
    usable = (~np.isnan(features[train_rows])).sum(axis=0) >= 2
    if usable.sum() < N_COMPONENTS:
        return out, 'too_few_usable_features'
    features = features[:, usable]

    try:
        out[keep] = ObservationalScalingPredictor(
            n_capability_components=N_COMPONENTS, metric_range=metric_range,
        ).fit_predict(features[train_rows], M_train[train_rows, target], features[rows])
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return out, 'fit_failed'
    return out, 'ok'


def run_fold(fold_index, M_train, test_set, methods):
    """Score every method on one fold and return per-cell predictions."""
    cells = np.asarray(test_set, dtype=int)
    truth = M_FULL[cells[:, 0], cells[:, 1]]
    preds = {m: np.full(len(cells), np.nan) for m in methods}
    reasons = {m: [] for m in methods if m in COLUMN_METHODS}

    if 'benchpress' in methods:
        M_pred = predict_benchpress_scores(M_train)
        preds['benchpress'] = M_pred[cells[:, 0], cells[:, 1]]

    impute_ranks = [k for k in IMPUTE_COMPONENT_GRID
                    if 'osl_impute_r%d' % k in methods]
    if impute_ranks:
        # OSL's imputation clips to the metric range. Their notebooks pass one
        # [0, 1] pair because every column of their matrix is an accuracy; here
        # the columns carry different scales, so the limit comes per column from
        # the same range rule the sigmoid fit uses.
        limits = np.empty((2, M_train.shape[1]))
        for target in range(M_train.shape[1]):
            span, _ = column_metric_range(M_train, target)
            limits[:, target] = (-np.inf, np.inf) if span is None else span
        for rank in impute_ranks:
            imputed, _ = pca_impute(M_train, n_components=rank, bounds=limits)
            preds['osl_impute_r%d' % rank] = imputed[cells[:, 0], cells[:, 1]]

    all_columns = np.arange(M_train.shape[1])
    for target in np.unique(cells[:, 1]):
        where = np.where(cells[:, 1] == target)[0]
        rows = cells[where, 0]
        fixed = choose_block(M_train, target, BLOCK_SIZE)
        full = all_columns[all_columns != target]
        for name, block, complete_only in (('osl', fixed, False),
                                           ('osl_no_impute', fixed, True),
                                           ('osl_full', full, False)):
            if name in methods:
                column, reason = osl_column(M_train, target, rows, block, complete_only)
                preds[name][where] = column
                reasons[name].append(reason)

    return fold_index, cells, truth, preds, reasons


def summarise(truth, preds, methods):
    """Pool errors per method and again on the cells every method reaches.

    A method that declines to predict the hardest cells would otherwise look
    accurate for the wrong reason, so `common_cells` restricts every method to
    the intersection of their coverage and is the only fair head-to-head. A
    method that reaches no cell at all is excluded from that intersection,
    since including it would empty the comparison; its `coverage` is the
    result to read for it.
    """
    scoring = [m for m in methods if np.isfinite(preds[m]).any()]
    common = np.isfinite(truth)
    for name in scoring:
        common = common & np.isfinite(preds[name])
    out = {'common_support': {'methods': scoring,
                              'n_cells': int(common.sum()),
                              'frac_cells': round(float(common.mean()), 4)}}
    for name in methods:
        pred = preds[name]
        covered = np.isfinite(pred) & np.isfinite(truth)
        out[name] = {
            'coverage': round(float(covered.mean()), 4),
            'n_predicted': int(covered.sum()),
            'n_cells': int(len(truth)),
            'own_coverage_cells': compute_prediction_error(truth[covered], pred[covered]),
            'common_cells': compute_prediction_error(truth[common], pred[common]),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='two folds only, for a fast sanity run')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--methods', nargs='*', default=METHODS, choices=METHODS)
    parser.add_argument('--out', default=os.path.join(HERE, 'obs_scaling_baseline.json'))
    args = parser.parse_args()

    folds = load_folds()
    if args.smoke:
        folds = folds[:2]

    truth_all, preds_all, per_fold = [], {m: [] for m in args.methods}, []
    reason_counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_fold, i, M_train, test_set, args.methods)
                   for i, (M_train, test_set) in enumerate(folds)]
        for future in as_completed(futures):
            index, cells, truth, preds, reasons = future.result()
            per_fold.append((index, cells, truth, preds))
            for name, values in reasons.items():
                counter = reason_counts.setdefault(name, {})
                for reason in values:
                    counter[reason] = counter.get(reason, 0) + 1

    per_fold.sort(key=lambda r: r[0])
    for _, _, truth, preds in per_fold:
        truth_all.append(truth)
        for name in args.methods:
            preds_all[name].append(preds[name])
    truth_all = np.concatenate(truth_all)
    preds_all = {m: np.concatenate(v) for m, v in preds_all.items()}

    results = {
        'description': 'observational scaling laws on the paper score matrix',
        'n_folds': len(folds),
        'block_size': BLOCK_SIZE,
        'n_components': N_COMPONENTS,
        'min_train_models': MIN_TRAIN_MODELS,
        'impute_component_grid': IMPUTE_COMPONENT_GRID,
        'benchpress_predictor': 'logit Bias ALS, rank 2, lambda 0.1',
        'smoke': bool(args.smoke),
        'applicability': applicability(),
        'column_fit_outcomes': reason_counts,
        'accuracy': summarise(truth_all, preds_all, args.methods),
    }

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    np.savez_compressed(
        args.out.replace('.json', '_raw.npz'),
        truth=truth_all,
        fold=np.concatenate([np.full(len(t), i) for i, _, t, _ in per_fold]),
        cells=np.concatenate([c for _, c, _, _ in per_fold]),
        model_ids=np.array(MODEL_IDS),
        bench_ids=np.array(BENCH_IDS),
        **{m: preds_all[m] for m in args.methods},
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
