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

def run_h8_fraction(frac, seeds):
    model_a = {i: [] for i in range(N_MODELS)}
    model_p = {i: [] for i in range(N_MODELS)}
    raw_s, raw_i, raw_j, raw_a, raw_p = [], [], [], [], []

    for seed in seeds:
        rng = np.random.RandomState(seed * 2000)
        M_train = M_FULL.copy()
        test_cells = {i: [] for i in range(N_MODELS)}

        for i in range(N_MODELS):
            obs_j = np.where(OBSERVED[i])[0]
            if len(obs_j) < 4:
                continue
            rng.shuffle(obs_j)
            n_hide = max(1, int(len(obs_j) * frac))
            for j in obs_j[:n_hide]:
                M_train[i, j] = np.nan
                test_cells[i].append(j)

        M_pred = predict_benchpress_scores(M_train)

        for i in range(N_MODELS):
            for j in test_cells[i]:
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
    base, base_raw = run_baseline(seeds)
    h8_25, h8_25_raw = run_h8_fraction(0.25, seeds)
    h8_75, h8_75_raw = run_h8_fraction(0.75, seeds)
    metrics = ["medape", "medae"]
    tests_25 = {m: paired_wilcoxon(base, h8_25, m) for m in metrics}
    tests_75 = {m: paired_wilcoxon(base, h8_75, m) for m in metrics}
    result = {"hypothesis": "H8_observation_count",
              "hide_25pct": {"tests": tests_25},
              "hide_75pct": {"tests": tests_75},
              "raw_predictions": {"baseline": base_raw,
                                  "hide_25pct": h8_25_raw, "hide_75pct": h8_75_raw}}
    write_json_next_to(__file__, result, indent=2)
    print(f"[H8] Done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
