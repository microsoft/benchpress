#!/usr/bin/env python3
"""
Pairwise OLS in logit+zscore space
===================================
For every ordered pair (j, j'), fit univariate OLS in logit+zscore space,
inverse-transform predictions back to raw scores, then report Pearson r
(in logit+zscore space) along with MedAE and MedAPE.

Output: pairwise_ols_stats.json
"""
import numpy as np
import os, sys, time
from scipy import stats as sp_stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from benchpress.all_methods import (
    M_FULL, N_MODELS, N_BENCH,
    BENCH_IDS, BENCH_NAMES, BENCH_CATS,
    _is_pct_bench, _to_logit, _from_logit,
)
from benchpress.evaluation_harness import col_normalize, col_denormalize, compute_prediction_error
from benchpress.io_utils import write_json

MIN_SHARED = 5


def run():
    is_pct = np.array([_is_pct_bench(j, M_FULL) for j in range(N_BENCH)])

    # Pre-compute logit-transformed columns
    M_logit = M_FULL.copy()
    for j in range(N_BENCH):
        if is_pct[j]:
            valid = np.isfinite(M_FULL[:, j])
            M_logit[valid, j] = _to_logit(M_FULL[valid, j])

    # Column-wise z-score using benchpress shared util (full-column stats, NaN-safe)
    M_z, col_mean, col_std = col_normalize(M_logit)

    # Per-benchmark best-neighbor stats
    # For each target j, find predictor j' that maximizes R²
    best = {}  # j -> {best_j2, r, medae, medape, n_shared}

    for j_target in range(N_BENCH):
        best_r2 = -np.inf
        best_info = None

        for j_pred in range(N_BENCH):
            if j_pred == j_target:
                continue

            # Shared models
            mask = np.isfinite(M_FULL[:, j_target]) & np.isfinite(M_FULL[:, j_pred])
            n_shared = mask.sum()
            if n_shared < MIN_SHARED:
                continue

            x_z = M_z[mask, j_pred]
            y_z = M_z[mask, j_target]
            if not (np.all(np.isfinite(x_z)) and np.all(np.isfinite(y_z))):
                continue
            if np.nanstd(x_z) < 1e-12 or np.nanstd(y_z) < 1e-12:
                continue

            # OLS in z-score space: y_z = a + b * x_z
            b, a, _, _, _ = sp_stats.linregress(x_z, y_z)
            y_z_hat = a + b * x_z

            # Inverse z-score (per-column, benchmark-wise)
            y_logit_hat = y_z_hat * col_std[j_target] + col_mean[j_target]

            # Inverse logit (if pct benchmark)
            if is_pct[j_target]:
                y_hat = _from_logit(y_logit_hat)
            else:
                y_hat = y_logit_hat

            y_true = M_FULL[mask, j_target]

            # Pearson r in logit+zscore space (matches OLS fit space)
            r_res = sp_stats.pearsonr(x_z, y_z)
            r = float(r_res.statistic) if np.isfinite(r_res.statistic) else 0.0
            r2 = r ** 2

            # Canonical metrics (vector mode): medape/medae in raw score space.
            m = compute_prediction_error(np.asarray(y_true), np.asarray(y_hat))
            medae = float(m['medae'])
            medape = float(m['medape'])

            if r2 > best_r2:
                best_r2 = r2
                best_info = {
                    'j_pred': j_pred,
                    'r': round(r, 3),
                    'r2': round(r2, 3),
                    'medae': round(medae, 2),
                    'medape': round(medape, 1),
                    'n_shared': int(n_shared),
                    'raw_predictions': {
                        'model_indices': mask.nonzero()[0].tolist(),
                        'actuals': [round(float(v), 6) for v in y_true],
                        'preds': [round(float(v), 6) for v in y_hat],
                    },
                }

        best[j_target] = best_info

    # Build output
    out = {}
    for j in range(N_BENCH):
        info = best[j]
        if info is None:
            continue
        name = BENCH_NAMES.get(BENCH_IDS[j]) or BENCH_IDS[j]
        neighbor_name = BENCH_NAMES.get(BENCH_IDS[info['j_pred']]) or BENCH_IDS[info['j_pred']]
        out[name] = {
            'best_neighbor': neighbor_name,
            'max_r': info['r'],
            'max_r2': info['r2'],
            'medape': info['medape'],
            'medae': info['medae'],
            'n_shared': info['n_shared'],
            'category': BENCH_CATS[j],
            'raw_predictions': info.get('raw_predictions', {}),
        }

    # Print ranked
    sorted_items = sorted(out.items(), key=lambda x: x[1]['max_r'], reverse=True)
    print(f"{'Target':30s}  {'Best Neighbor':30s}  {'r':>5s}  {'MedAPE':>7s}  {'MedAE':>6s}  {'n':>4s}")
    print('-' * 88)
    for name, info in sorted_items:
        print(f"{name:30s}  {info['best_neighbor']:30s}  {info['max_r']:5.3f}  {info['medape']:6.1f}%  {info['medae']:6.2f}  {info['n_shared']:4d}")

    # Summary
    rs = [v['max_r'] for v in out.values()]
    for t in [0.80, 0.85, 0.90, 0.95]:
        cnt = sum(1 for r in rs if r >= t)
        print(f"  r ≥ {t}: {cnt}/{len(rs)}")
    print(f"  Median best-neighbor r: {np.median(rs):.3f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'pairwise_ols_stats.json')
    write_json(out_path, out, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    t0 = time.time()
    run()
    print(f"Done in {time.time() - t0:.1f}s")
