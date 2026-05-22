#!/usr/bin/env python3
"""Shared utilities for §4 model-level error hypotheses.

Provides:
  - M_FULL, OBSERVED, MODEL_IDS, MODEL_NAMES, etc. from benchpress
  - benchpress_half_holdout_per_model() — hide-half per model (N_SEEDS seeds; requires >=4 observed scores)
  - build_rows() — per-model feature rows
  - univariate_spearman() — feature × target Spearman correlations
  - pairwise_abs_r() — N×N |Pearson r| matrix
  - model_metrics() — per-model metrics from (actual, pred) lists
  - run_baseline() — standard hide-half per-model for ablation experiments
  - paired_wilcoxon() — imported shared paired Wilcoxon signed-rank test
"""
import os
import time

import numpy as np
from scipy import stats as sp_stats
from benchpress.stats import paired_wilcoxon_by_metric as paired_wilcoxon

from benchpress.all_methods import (
    MODEL_IDS, MODEL_NAMES, MODEL_PROVIDERS, MODEL_PARAMS, M_FULL,
    N_MODELS, N_BENCH, MODEL_REASONING,
    predict_benchpress_scores,
)
from benchpress.evaluation_harness import col_normalize, compute_prediction_error, holdout_half_per_model, rank2_r2
from benchpress.io_utils import load_json, write_json_next_to

SEED = 42
N_SEEDS = 10
HERE = os.path.dirname(os.path.abspath(__file__))
OBSERVED = ~np.isnan(M_FULL)
NEIGHBOR_THRESH = 0.95
H6_MASK_FRAC = 0.5

A_FEATURES = ["log_params", "is_reasoning", "med_true", "rank2_R2"]
TARGETS = ["medape", "medae"]


# ── Core computations ─────────────────────────────────────────────────




def pairwise_abs_r() -> np.ndarray:
    """N_MODELS × N_MODELS matrix of pairwise |Pearson r| on raw scores."""
    R = np.zeros((N_MODELS, N_MODELS))
    for i in range(N_MODELS):
        obs_i = np.where(OBSERVED[i])[0]
        set_i = set(obs_i.tolist())
        for j in range(i + 1, N_MODELS):
            obs_j = np.where(OBSERVED[j])[0]
            shared = sorted(set_i & set(obs_j.tolist()))
            if len(shared) < 3:
                continue
            r = np.corrcoef(M_FULL[i, shared], M_FULL[j, shared])[0, 1]
            R[i, j] = R[j, i] = abs(r)
    return R


def model_metrics(actuals, preds):
    """Compute per-model metrics from (actual, pred) lists.

    Metrics are undefined for fewer than 3 held-out points, so those cases are
    excluded from per-model aggregates rather than assigned synthetic values.
    """
    if len(actuals) < 3:
        return None
    a, p = np.array(actuals), np.array(preds)
    m = compute_prediction_error(a, p)
    return {
        "medape": float(m["medape"]),
        "medae": float(m["medae"]),
    }


# ── Hide-half holdout per model ──────────────────────────────────────

def benchpress_half_holdout_per_model():
    """Run BenchPress hide-half holdout: for each seed, hide half of each model's
    observed scores, predict with BenchPress, aggregate per-model metrics.

    Note: previously named `benchpress_loo_per_model` — a misnomer, this is
    hide-half (n_hide = len(obs)//2), not leave-one-out.
    Models with fewer than 4 observed scores are skipped before holdout so that
    at least two train and two test points can exist.
    """
    model_actuals = {i: [] for i in range(N_MODELS)}
    model_preds = {i: [] for i in range(N_MODELS)}
    raw_seeds, raw_models, raw_benchmarks, raw_actuals, raw_preds = [], [], [], [], []

    t0 = time.time()
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed * 2000)
        M_train, test_cells = holdout_half_per_model(rng, min_obs=4)

        M_pred = predict_benchpress_scores(M_train)

        for i in range(N_MODELS):
            for j in test_cells[i]:
                a, p = M_FULL[i, j], M_pred[i, j]
                if np.isfinite(a) and np.isfinite(p):
                    model_actuals[i].append(a)
                    model_preds[i].append(p)
                    raw_seeds.append(seed)
                    raw_models.append(i)
                    raw_benchmarks.append(int(j))
                    raw_actuals.append(round(float(a), 6))
                    raw_preds.append(round(float(p), 6))

        if (seed + 1) % 5 == 0:
            print(f"    Seed {seed}/{N_SEEDS} done ({time.time()-t0:.0f}s)")

    per_model = {}
    for i in range(N_MODELS):
        if len(model_actuals[i]) < 3:
            continue
        actual = np.array(model_actuals[i])
        pred = np.array(model_preds[i])
        m = compute_prediction_error(actual, pred)
        per_model[MODEL_IDS[i]] = {
            "medape": float(m["medape"]),
            "medae": float(m["medae"]),
        }

    print(f"    BenchPress hide-half done: {len(per_model)} models, {time.time()-t0:.0f}s")
    raw_predictions = {
        "seeds": raw_seeds, "models": raw_models, "benchmarks": raw_benchmarks,
        "actuals": raw_actuals, "preds": raw_preds,
    }
    return per_model, raw_predictions


def build_rows(loo: dict) -> list:
    r2_vec = rank2_r2(M_FULL, axis=1)
    providers = np.asarray(MODEL_PROVIDERS)

    rows = []
    for i, mid in enumerate(MODEL_IDS):
        if mid not in loo:
            continue
        name = MODEL_NAMES.get(mid, mid) if isinstance(MODEL_NAMES, dict) else mid
        stat = loo[mid]
        row = M_FULL[i]
        obs = row[~np.isnan(row)]
        if len(obs) < 2:
            continue
        params = float(MODEL_PARAMS[i]) if np.isfinite(MODEL_PARAMS[i]) else float("nan")
        log_params = float(np.log10(params)) if np.isfinite(params) and params > 0 else float("nan")
        rows.append({
            "id":           mid,
            "name":         name,
            "provider":     str(providers[i]),
            "medape":       stat["medape"],
            "medae":        stat["medae"],
            "log_params":   log_params,
            "is_reasoning": int(bool(MODEL_REASONING[i])),
            "med_true":     float(np.median(obs)),
            "rank2_R2":     float(r2_vec[i]),
            "n_obs":        int(len(obs)),
            "params":       params,
        })
    return rows


def univariate_spearman(rows: list, target: str, features: list) -> dict:
    y = np.array([r[target] for r in rows])
    out = {}
    for f in features:
        x = np.array([r[f] for r in rows])
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            out[f] = {"rho": float("nan"), "p": float("nan"), "n": int(mask.sum())}
            continue
        rho, p = sp_stats.spearmanr(x[mask], y[mask])
        out[f] = {"rho": float(rho), "p": float(p), "n": int(mask.sum())}
    return out


# ── Ablation baseline ─────────────────────────────────────────────────

def run_baseline(seeds: list) -> tuple:
    """Standard BenchPress hide-half. Returns (per_model_metrics, raw_preds)."""
    model_a = {i: [] for i in range(N_MODELS)}
    model_p = {i: [] for i in range(N_MODELS)}
    raw_s, raw_i, raw_j, raw_a, raw_p = [], [], [], [], []

    for seed in seeds:
        rng = np.random.RandomState(seed * 2000)
        M_train, test_cells = holdout_half_per_model(rng, min_obs=4)

        M_pred = predict_benchpress_scores(M_train)

        for i in range(N_MODELS):
            for j in test_cells[i]:
                a, p = M_FULL[i, j], M_pred[i, j]
                if np.isfinite(a) and np.isfinite(p):
                    model_a[i].append(a)
                    model_p[i].append(p)
                    raw_s.append(seed); raw_i.append(i); raw_j.append(int(j))
                    raw_a.append(round(float(a), 6)); raw_p.append(round(float(p), 6))

    out = {}
    for i in range(N_MODELS):
        m = model_metrics(model_a[i], model_p[i])
        if m is not None:
            out[i] = m
    raw_predictions = {"seeds": raw_s, "models": raw_i, "benchmarks": raw_j, "actuals": raw_a, "preds": raw_p}
    return out, raw_predictions

