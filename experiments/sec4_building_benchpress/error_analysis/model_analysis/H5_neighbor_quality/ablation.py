#!/usr/bin/env python3
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import (
    SEED, N_SEEDS, OBSERVED, NEIGHBOR_THRESH, H6_MASK_FRAC,
    M_FULL, N_MODELS, predict_benchpress_scores,
    pairwise_abs_r, model_metrics,
    run_baseline, paired_wilcoxon,
    MODEL_IDS, MODEL_PROVIDERS,
    write_json_next_to,
)

def run_h5(R, seeds):
    model_a = {i: [] for i in range(N_MODELS)}
    model_p = {i: [] for i in range(N_MODELS)}
    raw_s, raw_i, raw_j, raw_a, raw_p = [], [], [], [], []

    for seed in seeds:
        rng = np.random.RandomState(seed * 2000)
        hide_masks = {}
        for i in range(N_MODELS):
            obs_j = np.where(OBSERVED[i])[0]
            if len(obs_j) < 4:
                continue
            rng.shuffle(obs_j)
            hide_masks[i] = obs_j[:len(obs_j)//2].tolist()

        for i in range(N_MODELS):
            if i not in hide_masks:
                continue
            M_train = M_FULL.copy()
            for j in hide_masks[i]:
                M_train[i, j] = np.nan
            neighbours = np.where(R[i] >= NEIGHBOR_THRESH)[0]
            for nb in neighbours:
                M_train[nb, :] = np.nan
            M_pred = predict_benchpress_scores(M_train)
            for j in hide_masks[i]:
                a, p = M_FULL[i, j], M_pred[i, j]
                if np.isfinite(a) and np.isfinite(p):
                    model_a[i].append(a); model_p[i].append(p)
                    raw_s.append(seed); raw_i.append(i); raw_j.append(int(j))
                    raw_a.append(round(float(a), 6)); raw_p.append(round(float(p), 6))

    out = {}
    for i in range(N_MODELS):
        m = model_metrics(model_a[i], model_p[i])
        if m is not None:
            out[i] = m
    raw_preds = {"seeds": raw_s, "models": raw_i, "benchmarks": raw_j, "actuals": raw_a, "preds": raw_p}
    return out, raw_preds

def main():
    np.random.seed(SEED)
    seeds = list(range(N_SEEDS))
    t0 = time.time()
    print("[H5] Computing pairwise |r|...")
    R = pairwise_abs_r()
    print("[H5] Running baseline...")
    base, base_raw = run_baseline(seeds)
    print(f"     Baseline: {len(base)} models")
    print(f"[H5] Remove |r|>={NEIGHBOR_THRESH} neighbours...")
    h5, h5_raw = run_h5(R, seeds)
    print(f"     H5: {len(h5)} models")
    metrics = ["medape", "medae"]
    hyp = {m: paired_wilcoxon(base, h5, m) for m in metrics}
    result = {"hypothesis": "H5_neighbor_quality", "neighbor_thresh": NEIGHBOR_THRESH,
              "tests": hyp,
              "raw_predictions": {"baseline": base_raw, "ablation": h5_raw}}
    write_json_next_to(__file__, result, indent=2)
    print(f"[H5] Done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
