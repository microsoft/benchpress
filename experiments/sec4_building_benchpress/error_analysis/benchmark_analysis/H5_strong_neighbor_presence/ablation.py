#!/usr/bin/env python3
"""H5 ablation: strong-neighbor presence.

For each target benchmark j, identify all benchmarks whose Pearson |r| with j
in the OBSERVED matrix is >= threshold; mask those entire columns from the
training matrix and re-run BenchPress on j. Compare against the no-mask baseline.
Sweep thresholds {0.95, 0.90, 0.85}; 5 seeds per threshold.

Symmetric counterpart to §5.2 H5 (peer presence).
"""
import os, sys, time
import numpy as np
from benchpress.stats import wilcoxon_per_benchmark

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import (
    SEED, M_FULL, N_BENCH, OBSERVED, BENCH_IDS,
    predict_benchpress_scores, compute_prediction_error,
    holdout_half_per_benchmark, mask_cells, mask_columns, pairwise_benchmark_corr,
    write_json_next_to,
)

N_SEEDS = 5
THRESHOLDS = [0.95, 0.90, 0.85]
METRICS = ["medape", "medae"]


def ablation_h6_strong_neighbor():
    rng = np.random.RandomState(SEED + 6000)
    print("[H6] computing pairwise |r| matrix...")
    t0 = time.time()
    R, _ = pairwise_benchmark_corr()
    print(f"  done in {time.time() - t0:.1f}s")

    # Cache predict_benchpress_scores per (j, threshold, seed) is too much; instead recompute per (j, seed)
    # since baseline depends on test_idx (different per seed). But threshold-mask doesn't depend on
    # seed, so we can cache the column-removal pattern.
    pred_cache = {}

    def get_pred(j: int, thr: float | None, test_idx, seed_key):
        # cache per (j, thr, seed_key) because test cells are masked (seed-dependent)
        key = (j, thr, seed_key)
        if key in pred_cache:
            return pred_cache[key]
        M_masked = mask_cells((idx, j) for idx in test_idx)  # hide held-out test cells
        n_strong = 0
        if thr is not None:
            strong = np.where(R[j] >= thr)[0]
            strong = strong[strong != j]  # exclude self
            M_masked = mask_columns(strong, base_matrix=M_masked)
            n_strong = int(len(strong))
        P = predict_benchpress_scores(M_masked)
        pred_cache[key] = (P, n_strong)
        return pred_cache[key]

    records = []
    t0 = time.time()
    for j in range(N_BENCH):
        obs_rows = np.where(OBSERVED[:, j])[0]
        if len(obs_rows) < 6:
            continue
        for s in range(N_SEEDS):
            test_idx, _ = holdout_half_per_benchmark(j, rng, min_test=3)
            true = M_FULL[test_idx, j]

            # Baseline: only mask test cells
            P_base, _ = get_pred(j, None, test_idx, s)
            pb = P_base[test_idx, j]
            valid = np.isfinite(pb)
            if valid.sum() < 2:
                continue
            mb = compute_prediction_error(true[valid], pb[valid])

            for thr in THRESHOLDS:
                P_treat, n_strong = get_pred(j, thr, test_idx, s)
                pt = P_treat[test_idx, j]
                v2 = valid & np.isfinite(pt)
                if v2.sum() < 2:
                    continue
                mt = compute_prediction_error(true[v2], pt[v2])

                rec = {
                    "bench_id": BENCH_IDS[j], "seed": s, "threshold": thr,
                    "n_strong_neighbors": n_strong, "n_test": int(v2.sum()),
                }
                # Recompute base metrics on the same v2 mask for paired comparison.
                mb_paired = compute_prediction_error(true[v2], pb[v2])
                for m in METRICS:
                    rec[f"base_{m}"] = float(mb_paired[m])
                    rec[f"treat_{m}"] = float(mt[m])
                    rec[f"delta_{m}"] = float(mt[m]) - float(mb_paired[m])

                rec["raw_base"] = [
                    [int(test_idx[vi]), int(j),
                     round(float(true[vi]), 6), round(float(pb[vi]), 6)]
                    for vi in range(len(test_idx)) if v2[vi]
                ]
                rec["raw_treat"] = [
                    [int(test_idx[vi]), int(j),
                     round(float(true[vi]), 6), round(float(pt[vi]), 6)]
                    for vi in range(len(test_idx)) if v2[vi]
                ]
                records.append(rec)
        print(f"  H6 [{j+1:2d}/{N_BENCH}] {BENCH_IDS[j]:30s}  ({time.time() - t0:.0f}s, cache={len(pred_cache)})")

    wilcoxon = {}
    for thr in THRESHOLDS:
        threshold_records = [r for r in records if r["threshold"] == thr]
        wilcoxon[str(thr)] = wilcoxon_per_benchmark(
            threshold_records,
            METRICS,
            drop_zeros_for_test=True,
            invalid_median=0.0,
            invalid_p=1.0,
        )

    return {"records": records, "wilcoxon": wilcoxon, "thresholds": THRESHOLDS}


def main():
    np.random.seed(SEED)
    out = ablation_h6_strong_neighbor()
    out["hypothesis"] = "H6"
    out["intervention"] = "mask_strong_neighbor_columns"
    out_path = write_json_next_to(__file__, out, filename="ablation_results.json", indent=2)
    print(f"[saved] {out_path} ({len(out['records'])} records)")
    print("\n== Wilcoxon summary ==")
    for thr_str, metrics in out["wilcoxon"].items():
        print(f"  threshold={thr_str}")
        for m, v in metrics.items():
            print(f"    {m:10s}  Δmedian={v['median_delta']:+.3f}  p={v['p_value']:.4f}  n={v['n']}")


if __name__ == "__main__":
    main()
