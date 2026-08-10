"""PC1/PC2 loadings of the score matrix (rebuttal nxZ2 Q3).

The reviewer asks why the leading direction should be read as general
capability, noting that almost any signal direction would put frontier models
at one end and small models at the other, and that the diffuseness argument of
Observational Scaling Laws Sec. 3.2 is stronger.

The matrix is 77% missing, so a plain SVD needs the missing cells filled, and
filling them with the column mean would fabricate most of the structure.
Instead we complete the matrix with the paper's own rank-2 Bias ALS predictor
(fit on observed cells only), standardize columns, and take the SVD.

Three statistics are reported:

1. Diffuseness of PC1 over the 133 benchmarks: sign agreement, the spread of
   the loading magnitudes against the uniform value 1/sqrt(133), and how many
   benchmarks are needed to reach half of the squared loading.
2. Whether PC2, a genuine signal direction, also orders models by capability.
   Capability is the naive ranking: each model's mean z-scored observed score.
3. A completion-free check: per benchmark, the Spearman correlation between its
   observed scores and the PC1 model score, over observed cells only.

Usage:
    python pc_loadings.py [--out results.json]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

from benchpress.evaluation_harness import (
    M_FULL,
    OBSERVED,
    MODEL_IDS,
    BENCH_IDS,
    make_score_predictor,
)
from benchpress.methods.completers import complete_bias_als

TOP_K = 5


def extremes(values, names, k=TOP_K):
    order = np.argsort(-np.asarray(values))
    top = [(names[i], round(float(values[i]), 4)) for i in order[:k]]
    bottom = [(names[i], round(float(values[i]), 4)) for i in order[::-1][:k]]
    return {'top': top, 'bottom': bottom}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'),
    )
    args = parser.parse_args()

    predict = make_score_predictor(
        complete_bias_als, 'logit', rank=2, lam=0.1, normalize=False
    )
    M_hat = predict(M_FULL)
    if not np.isfinite(M_hat).all():
        raise ValueError('completion left non-finite cells')

    M_z = (M_hat - M_hat.mean(axis=0)) / M_hat.std(axis=0)
    U, s, Vt = np.linalg.svd(M_z, full_matrices=False)
    variance = s ** 2 / (s ** 2).sum()

    observed_z = np.where(OBSERVED, M_FULL, np.nan)
    observed_z = (observed_z - np.nanmean(observed_z, axis=0)) / np.nanstd(observed_z, axis=0)
    capability = np.nanmean(observed_z, axis=1)

    n_bench = len(BENCH_IDS)
    out = {
        'shape': [len(MODEL_IDS), n_bench],
        'observed_cells': int(OBSERVED.sum()),
        'completion': 'rank-2 Bias ALS, logit transform, lam=0.1, fit on observed cells',
        'components': [],
    }

    pc1_model_score = None
    for pc in (0, 1):
        model_load = U[:, pc] * s[pc]
        bench_load = Vt[pc, :].copy()
        # SVD sign is arbitrary; orient so that a positive loading means "scores higher".
        sign = -1.0 if bench_load.sum() < 0 else 1.0
        model_load, bench_load = model_load * sign, bench_load * sign
        if pc == 0:
            pc1_model_score = model_load.copy()

        squared = np.sort(bench_load ** 2)[::-1]
        cumulative = np.cumsum(squared) / squared.sum()
        out['components'].append({
            'pc': pc + 1,
            'variance_share': round(float(variance[pc]), 4),
            'models': extremes(model_load, MODEL_IDS),
            'benchmarks': extremes(bench_load, BENCH_IDS),
            'benchmark_sign_agreement': round(
                float(max((bench_load > 0).mean(), (bench_load < 0).mean())), 4
            ),
            'benchmark_abs_loading': {
                'min': round(float(np.abs(bench_load).min()), 4),
                'median': round(float(np.median(np.abs(bench_load))), 4),
                'max': round(float(np.abs(bench_load).max()), 4),
                'uniform': round(float(1 / np.sqrt(n_bench)), 4),
            },
            'benchmarks_for_half_squared_loading': int(np.searchsorted(cumulative, 0.5) + 1),
            'spearman_with_capability': round(
                float(spearmanr(model_load, capability).correlation), 4
            ),
        })

    rhos, rho_names = [], []
    for j in range(n_bench):
        mask = OBSERVED[:, j]
        if mask.sum() < 5:
            continue
        rho = spearmanr(M_FULL[mask, j], pc1_model_score[mask]).correlation
        if np.isfinite(rho):
            rhos.append(float(rho))
            rho_names.append(BENCH_IDS[j])
    rhos = np.array(rhos)
    out['completion_free_check'] = {
        'description': (
            'per benchmark, Spearman between its observed scores and the PC1 model '
            'score, computed on observed cells only'
        ),
        'n_benchmarks': int(len(rhos)),
        'n_positive': int((rhos > 0).sum()),
        'n_above_half': int((rhos > 0.5).sum()),
        'quartiles': [round(float(v), 4) for v in np.percentile(rhos, [0, 25, 50, 75, 100])],
        'negative': sorted(
            [(rho_names[i], round(rhos[i], 4)) for i in np.where(rhos <= 0)[0]],
            key=lambda kv: kv[1],
        ),
    }

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
