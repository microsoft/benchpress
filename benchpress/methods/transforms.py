#!/usr/bin/env python3
"""Score-space transforms and transform/z-score pipeline for BenchPress."""

import numpy as np

from benchpress.evaluation_harness import col_normalize

NON_PCT_BENCHMARKS = {
    'chatbot_arena_elo': 'Elo rating (~1000-1500)',
    'codeforces_rating': 'rating (~800-2200)',
    'aa_intelligence_index': 'index (0-100+)',
    'gdpval_aa_elo': 'Elo rating',
    'swelancer_freelance_dollars': 'dollars',
    'vending_bench_2': 'task-specific score',
}


def benchmark_scale(bench_id):
    """Human-readable score scale for a benchmark id."""
    return NON_PCT_BENCHMARKS.get(bench_id, '0-100%')


def clamp_score_for_benchmark(score, bench_id):
    """Clamp scalar predictions to the benchmark's valid score range."""
    if bench_id == 'chatbot_arena_elo':
        return float(np.clip(score, 800, 1600))
    if bench_id in {'codeforces_rating', 'gdpval_aa_elo'}:
        return float(np.clip(score, 0, 2800))
    if bench_id == 'swelancer_freelance_dollars':
        return float(max(score, 0))
    if bench_id not in NON_PCT_BENCHMARKS:
        return float(np.clip(score, 0, 100))
    return float(score)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_pct_bench(j, M):
    """Heuristic: benchmark j uses a percentage scale [0,100]."""
    vals = M[~np.isnan(M[:, j]), j]
    if len(vals) == 0:
        return False
    return vals.min() >= -1 and vals.max() <= 101

def _to_logit(x, eps=0.5):
    """Convert percentage [0,100] → logit space. Clips to [eps, 100-eps] first."""
    p = np.clip(x, eps, 100 - eps) / 100.0
    return np.log(p / (1 - p))

def _from_logit(z):
    """Convert logit → percentage [0,100]."""
    return 100.0 / (1 + np.exp(-z))

def _to_probit(x, eps=0.5):
    """Convert percentage [0,100] → probit (inverse normal CDF) space."""
    from scipy.stats import norm
    p = np.clip(np.asarray(x, dtype=float), eps, 100 - eps) / 100.0
    return norm.ppf(p)

def _from_probit(z):
    """Convert probit → percentage [0,100]."""
    from scipy.stats import norm
    return norm.cdf(np.asarray(z, dtype=float)) * 100.0

def _to_log(x):
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0))

def _from_log(z):
    return np.expm1(np.asarray(z, dtype=float))

def _to_asinh(x, scale=50.0):
    return np.arcsinh(np.asarray(x, dtype=float) / scale)

def _from_asinh(z, scale=50.0):
    return np.sinh(np.asarray(z, dtype=float)) * scale

def _to_sqrt(x):
    return np.sqrt(np.maximum(np.asarray(x, dtype=float), 0))

def _from_sqrt(z):
    return np.asarray(z, dtype=float) ** 2

def _to_raw(x):
    return np.asarray(x, dtype=float)

def _from_raw(x):
    return np.asarray(x, dtype=float)

# Quantile transform (stateful: stores stats in module-level dict)
_quantile_stats = {}

def _to_quantile(x):
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return x
    ranks = rankdata(x)
    _quantile_stats['sorted'] = np.sort(x)
    _quantile_stats['n'] = n
    return ranks / (n + 1)

def _from_quantile(z):
    z = np.asarray(z, dtype=float)
    sorted_vals = _quantile_stats.get('sorted', None)
    n = _quantile_stats.get('n', 1)
    if sorted_vals is None:
        return z * 100.0
    result = np.full_like(z, np.nan)
    finite = np.isfinite(z)
    if finite.any():
        zf = z[finite]
        idx = np.clip(zf * (n + 1) - 1, 0, n - 1)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        frac = idx - lo
        result[finite] = sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
    return result

# Registry: name → (to_fn, from_fn, pct_only)
TRANSFORMS = {
    'identity':  (_to_raw,      _from_raw,      False),
    'log':       (_to_log,      _from_log,      False),
    'logit':     (_to_logit,    _from_logit,    True),
    'asinh':     (_to_asinh,    _from_asinh,    True),
    'sqrt':      (_to_sqrt,     _from_sqrt,     True),
    'probit':    (_to_probit,   _from_probit,   True),
    'quantile':  (_to_quantile, _from_quantile, False),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORM PIPELINE: raw → feature transform → z-score (and inverse)
# ══════════════════════════════════════════════════════════════════════════════

def apply_transform(M, to_fn, pct_only):
    """Transform observed entries, then per-column z-score standardize.

    Pipeline: raw → optional feature transform → z-score.
    Returns (M_z, obs, is_pct, col_mu, col_std).
    """
    obs = ~np.isnan(M)
    n_bench = M.shape[1]
    is_pct = np.array([_is_pct_bench(j, M) for j in range(n_bench)])
    M_t = M.copy()
    for j in range(n_bench):
        if (not pct_only) or is_pct[j]:
            valid = obs[:, j]
            M_t[valid, j] = to_fn(M[valid, j])
    M_z, col_mu, col_std = col_normalize(M_t)
    return M_z, obs, is_pct, col_mu, col_std

def invert_transform(M_pred, M_train, to_fn, from_fn, pct_only, obs, is_pct,
                     col_mu, col_std):
    """Invert z-score then feature transform on predicted (missing) entries.

    Pipeline: z-score⁻¹ → transform⁻¹ → clip [0,100] for pct benchmarks.
    Re-runs to_fn per column to restore stateful transform state (e.g., quantile).
    """
    M_out = M_train.copy()
    n_models, n_bench = M_train.shape
    for j in range(n_bench):
        should_transform = (not pct_only) or is_pct[j]
        # Re-run forward transform to set any side-effect state (e.g., quantile stats)
        if should_transform:
            valid = obs[:, j]
            if valid.any():
                to_fn(M_train[valid, j])
        for i in range(n_models):
            if obs[i, j]:
                continue
            val = M_pred[i, j]
            if not np.isfinite(val):
                continue
            val = val * col_std[j] + col_mu[j]
            if should_transform:
                val = from_fn(val)
            # Clip percentage benchmarks to [0, 100]
            if is_pct[j]:
                val = np.clip(val, 0, 100)
            M_out[i, j] = val
    return M_out
