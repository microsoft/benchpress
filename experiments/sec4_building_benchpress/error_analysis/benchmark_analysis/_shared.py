"""Shared utilities for §4 benchmark-level error hypotheses."""
import os

import numpy as np
from scipy import stats as sp_stats

from benchpress.all_methods import (
    BENCH_IDS, BENCH_NAMES, BENCH_CATS, M_FULL, N_BENCH, predict_benchpress_scores,
)
from benchpress.evaluation_harness import (
    compute_prediction_error,
    holdout_half_per_benchmark,
    keep_only_benchmark_rows,
    mask_cells,
    mask_columns,
    rank2_r2,
)
from benchpress.io_utils import load_json, write_json_next_to

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREDICTIONS_BY_BENCHMARK_REL = (
    "benchpress/evaluation/default_predictions/benchpress_default/by_benchmark.json"
)
DEFAULT_PREDICTIONS_BY_BENCHMARK = os.path.normpath(os.path.join(
    HERE,
    "..", "..", "..", "..",
    DEFAULT_PREDICTIONS_BY_BENCHMARK_REL,
))
OBSERVED = ~np.isnan(M_FULL)

FEATURES = ["rank2_R2", "n_obs", "best_neighbor_r", "n_shared", "med_score", "std_score", "n_same_cat"]
TARGETS = ["medape", "medae"]





def univariate_spearman(rows: list[dict], target: str, features: list[str] = None) -> dict:
    """Spearman ρ + p-value for each feature vs target."""
    if features is None:
        features = FEATURES
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


def load_benchpress_default_errors() -> dict:
    """Per-benchmark BenchPress default prediction errors from §4.2 folds.

    Returns: {bench_id: {medape, medae, n}}
    """
    d = load_json(DEFAULT_PREDICTIONS_BY_BENCHMARK)
    return {
        r["bench_id"]: {
            "medape": float(r["medape"]),
            "medae": float(r["medae"]),
            "n": int(r["n"]),
        }
        for r in d["benchmarks"]
    }


def load_benchpress_baseline() -> dict:
    """Backward-compatible alias for the canonical BenchPress default errors."""
    return load_benchpress_default_errors()


def pairwise_benchmark_corr(min_shared: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise benchmark |Pearson r| and shared-count matrices from M_FULL."""
    corr = np.full((N_BENCH, N_BENCH), np.nan, dtype=float)
    shared = np.zeros((N_BENCH, N_BENCH), dtype=int)
    for a in range(N_BENCH):
        col_a = M_FULL[:, a]
        obs_a = np.isfinite(col_a)
        for b in range(a + 1, N_BENCH):
            col_b = M_FULL[:, b]
            mask = obs_a & np.isfinite(col_b)
            n = int(mask.sum())
            shared[a, b] = shared[b, a] = n
            if n < min_shared:
                continue
            r, _ = sp_stats.pearsonr(col_a[mask], col_b[mask])
            if np.isfinite(r):
                corr[a, b] = corr[b, a] = abs(float(r))
    return corr, shared


def find_best_neighbor(j: int) -> int | None:
    """Return benchmark index of the most correlated observed-score neighbor."""
    corr, _ = pairwise_benchmark_corr()
    row = corr[j].copy()
    row[j] = np.nan
    if not np.any(np.isfinite(row)):
        return None
    return int(np.nanargmax(row))


def best_neighbor_features(j: int, corr: np.ndarray, shared: np.ndarray) -> dict:
    """Best-neighbor correlation features for benchmark j, computed locally."""
    row = corr[j].copy()
    row[j] = np.nan
    if not np.any(np.isfinite(row)):
        return {
            "best_neighbor": None,
            "best_neighbor_r": float("nan"),
            "n_shared": 0,
        }
    jj = int(np.nanargmax(row))
    return {
        "best_neighbor": BENCH_IDS[jj],
        "best_neighbor_r": float(row[jj]),
        "n_shared": int(shared[j, jj]),
    }


def build_rows() -> list[dict]:
    """Build per-benchmark feature rows from the canonical §4.2 default predictions."""
    import time
    bp_base = load_benchpress_default_errors()
    r2_vec = rank2_r2(M_FULL, axis=0)
    corr, shared = pairwise_benchmark_corr()

    from collections import Counter
    cats = list(BENCH_CATS)
    cat_counts = Counter(cats)

    rows = []
    t0 = time.time()
    for j, bid in enumerate(BENCH_IDS):
        name = BENCH_NAMES.get(bid, bid)
        bp = bp_base.get(bid)
        if bp is None:
            continue
        col = M_FULL[:, j]
        obs = col[~np.isnan(col)]
        if len(obs) < 2:
            continue
        neighbor = best_neighbor_features(j, corr, shared)

        rows.append({
            "id": bid,
            "name": name,
            "medape":    bp["medape"],
            "medae":     bp["medae"],
            "rank2_R2":  float(r2_vec[j]),
            "n_obs":     int(len(obs)),
            "best_neighbor": neighbor["best_neighbor"],
            "best_neighbor_r": neighbor["best_neighbor_r"],
            "n_shared":  neighbor["n_shared"],
            "med_score": float(np.median(obs)),
            "std_score": float(np.std(obs, ddof=1)),
            "n_same_cat": int(cat_counts[cats[j]] - 1),
        })
        elapsed = time.time() - t0
        print(f"  build_rows [{j+1:2d}/{N_BENCH}] {bid:30s} ({elapsed:.0f}s)")
    return rows
