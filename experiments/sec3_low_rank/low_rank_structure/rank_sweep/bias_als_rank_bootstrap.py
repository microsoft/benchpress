#!/usr/bin/env python
"""Summarize the logit Bias ALS rank sweep with model-cluster bootstrap intervals.

Reads the `logit_bias_als` raw predictions written by `run.py --methods logit_bias_als`
and writes pooled MedAE/MedAPE per rank plus bootstrap intervals for the
difference in pooled MedAE between each rank and the reference rank.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rank_arrays(results):
    entries = results['logit_bias_als']
    ranks = sorted(entries, key=int)
    arrays = {}
    keys = None
    for rank in ranks:
        rows = entries[rank]['raw_predictions']
        cur_keys = np.array([[r['seed'], r['fold'], r['model_idx'], r['benchmark_idx']] for r in rows])
        if keys is None:
            keys = cur_keys
        elif not np.array_equal(keys, cur_keys):
            raise ValueError(f'rank {rank} raw predictions are not aligned with rank {ranks[0]}')
        arrays[rank] = (np.array([r['true'] for r in rows], float), np.array([r['pred'] for r in rows], float))
    return ranks, keys, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=os.path.join(HERE, 'results.json'))
    parser.add_argument('--out', default=os.path.join(HERE, 'bias_als_rank_bootstrap.json'))
    parser.add_argument('--reference-rank', type=int, default=2)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ranks, keys, arrays = load_rank_arrays(json.load(open(args.results)))
    models = keys[:, 2]
    unique_models = np.unique(models)
    rows_by_model = [np.flatnonzero(models == m) for m in unique_models]
    abs_err = {r: np.abs(p - t) for r, (t, p) in arrays.items()}
    rng = np.random.default_rng(args.seed)
    boot_rows = [
        np.concatenate([rows_by_model[i] for i in rng.integers(0, len(unique_models), len(unique_models))])
        for _ in range(args.n_boot)
    ]
    ref = str(args.reference_rank)
    summary = {
        'config': {
            'results': os.path.relpath(args.results, HERE),
            'reference_rank': args.reference_rank,
            'n_boot': args.n_boot,
            'seed': args.seed,
            'bootstrap_unit': 'model',
            'n_models': int(len(unique_models)),
            'n_predictions': int(len(models)),
        },
        'by_rank': {},
    }
    for rank in ranks:
        t, p = arrays[rank]
        valid = t != 0
        diffs = [np.median(abs_err[rank][rows]) - np.median(abs_err[ref][rows]) for rows in boot_rows]
        summary['by_rank'][rank] = {
            'pooled_medae': float(np.median(abs_err[rank])),
            'pooled_medape': float(np.median(abs_err[rank][valid] / np.abs(t[valid]) * 100)),
            'medae_minus_reference': float(np.median(abs_err[rank]) - np.median(abs_err[ref])),
            'medae_minus_reference_ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    for rank, row in summary['by_rank'].items():
        lo, hi = row['medae_minus_reference_ci95']
        print(f"rank {rank:>2}: MedAE {row['pooled_medae']:.3f}  MedAPE {row['pooled_medape']:.2f}  "
              f"diff vs rank {ref} {row['medae_minus_reference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == '__main__':
    main()
