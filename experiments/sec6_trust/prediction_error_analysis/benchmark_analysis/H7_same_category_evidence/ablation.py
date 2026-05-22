#!/usr/bin/env python3
"""H7: Same-category evidence — ablation only.

For each target benchmark j, mask all benchmarks sharing j's category from the
training matrix and re-run BenchPress on j's hide-half held-out cells.
Compare against the no-mask baseline. 10 seeds, paired Wilcoxon across benchmarks.
"""
import os, sys, time
import numpy as np
from benchpress.stats import wilcoxon_per_benchmark

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import (
    SEED, M_FULL, N_BENCH, OBSERVED, BENCH_IDS, BENCH_CATS,
    predict_benchpress_scores, compute_prediction_error,
    holdout_half_per_benchmark, mask_cells, mask_columns,
    write_json_next_to,
)

# This is cheaper than the other ablations because it has one category-mask treatment, so it
# uses 10 seeds for stability while the heavier ablations use 5 seeds.
N_SEEDS = 10
METRICS = ["medape", "medae"]


def ablation_h8_category():
    cats = list(BENCH_CATS)
    rng = np.random.RandomState(SEED)
    records = []
    t0 = time.time()
    for j in range(N_BENCH):
        obs_rows = np.where(OBSERVED[:, j])[0]
        if len(obs_rows) < 6:
            continue
        same_cat_cols = [c for c in range(N_BENCH) if c != j and cats[c] == cats[j]]
        if len(same_cat_cols) == 0:
            continue
        for s in range(N_SEEDS):
            test_idx, _ = holdout_half_per_benchmark(j, rng, min_test=0)
            true = M_FULL[test_idx, j]

            M_base = mask_cells((idx, j) for idx in test_idx)
            P_base = predict_benchpress_scores(M_base)

            M_treat = mask_columns(same_cat_cols, base_matrix=M_base)
            P_treat = predict_benchpress_scores(M_treat)

            pb = P_base[test_idx, j]; pt = P_treat[test_idx, j]
            valid = np.isfinite(pb) & np.isfinite(pt)
            if valid.sum() < 3:
                continue
            mb = compute_prediction_error(true[valid], pb[valid])
            mt = compute_prediction_error(true[valid], pt[valid])

            rec = {
                "bench_id": BENCH_IDS[j], "seed": s, "category": cats[j],
                "n_same_cat_removed": len(same_cat_cols), "n_eval": int(valid.sum()),
            }
            for m in METRICS:
                rec[f"base_{m}"] = float(mb[m])
                rec[f"treat_{m}"] = float(mt[m])
                rec[f"delta_{m}"] = float(mt[m]) - float(mb[m])
            rec["raw_base"] = [
                [int(test_idx[vi]), int(j),
                 round(float(true[vi]), 6), round(float(pb[vi]), 6)]
                for vi in range(len(test_idx)) if valid[vi]
            ]
            rec["raw_treat"] = [
                [int(test_idx[vi]), int(j),
                 round(float(true[vi]), 6), round(float(pt[vi]), 6)]
                for vi in range(len(test_idx)) if valid[vi]
            ]
            records.append(rec)
        print(f"  H8 [{j+1:2d}/{N_BENCH}] {BENCH_IDS[j]:30s}  ({time.time() - t0:.0f}s)")

    wilcoxon = wilcoxon_per_benchmark(
        records,
        METRICS,
        drop_zeros_for_test=True,
        invalid_median=0.0,
        invalid_p=1.0,
    )

    return {"records": records, "wilcoxon": wilcoxon}


def main():
    np.random.seed(SEED)
    out = ablation_h8_category()
    out["hypothesis"] = "H8"
    out["intervention"] = "mask_same_category_columns"
    out_path = write_json_next_to(__file__, out, filename="ablation_results.json", indent=2)
    print(f"[saved] {out_path} ({len(out['records'])} records)")
    print("\n== Wilcoxon summary ==")
    for m, v in out["wilcoxon"].items():
        print(f"    {m:10s}  Δmedian={v['median_delta']:+.3f}  p={v['p_value']:.4f}  n={v['n']}")


if __name__ == "__main__":
    main()
