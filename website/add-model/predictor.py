"""
Self-contained BenchPress predictor for in-browser execution via Pyodide.

Mirrors the current BenchPress recipe in benchpress/methods/predictors.py:

    predict_benchpress_scores(M) = make_score_predictor(
        complete_bias_als, transform='logit', rank=2, lam=0.1, normalize=False
    )(M)

Pipeline per call:
    raw scores (NaN for missing)
        -> per-column logit (only on percent benchmarks; others identity)
        -> per-column z-score
        -> Bias ALS (rank=2, lam=0.1, n_iter=40, n_ensemble=10)
        -> z-score^{-1}
        -> logit^{-1}  (clip to [0,100] for percent benchmarks)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------

def _is_pct_bench(j, M):
    vals = M[~np.isnan(M[:, j]), j]
    if len(vals) == 0:
        return False
    return vals.min() >= -1 and vals.max() <= 101


def _to_logit(x, eps=0.5):
    p = np.clip(np.asarray(x, dtype=float), eps, 100 - eps) / 100.0
    return np.log(p / (1 - p))


def _from_logit(z):
    return 100.0 / (1 + np.exp(-np.asarray(z, dtype=float)))


# --------------------------------------------------------------------------
# Column z-score
# --------------------------------------------------------------------------

def col_stats(M):
    cm = np.nanmean(M, axis=0)
    cs = np.nanstd(M, axis=0)
    cs[cs < 1e-8] = 1.0
    return cm, cs


def col_normalize(M):
    cm, cs = col_stats(M)
    M_norm = (M - cm) / cs
    M_norm[np.isnan(M)] = np.nan
    return M_norm, cm, cs


# --------------------------------------------------------------------------
# Bias ALS completion method (matches complete_bias_als with normalize=False, in z-scored space)
# --------------------------------------------------------------------------

def _bias_als_zspace(M_z, rank=2, lam=0.1, n_iter=40,
                    n_ensemble=10, base_seed=42):
    """ALS on z-scored matrix. Z_hat = mu + a_i + c_j + U_i . V_j."""
    obs = ~np.isnan(M_z)
    n_models, n_bench = M_z.shape

    row_obs = [np.where(obs[i])[0] for i in range(n_models)]
    col_obs = [np.where(obs[:, j])[0] for j in range(n_bench)]
    row_vals = [M_z[i, row_obs[i]] for i in range(n_models)]
    col_vals = [M_z[col_obs[j], j] for j in range(n_bench)]

    obs_ij = np.argwhere(obs)
    obs_vals = M_z[obs]
    z_global_mean = float(np.mean(obs_vals)) if obs_vals.size else 0.0
    eye_r1 = np.eye(rank + 1) * lam

    def _run_one(seed):
        rng = np.random.RandomState(seed)
        mu = z_global_mean
        a = np.zeros(n_models)
        c = np.zeros(n_bench)
        U = rng.normal(0.0, 0.01, size=(n_models, rank))
        V = rng.normal(0.0, 0.01, size=(n_bench, rank))

        for _ in range(n_iter):
            # Block A: per-row update of (a_i, U_i)
            for i in range(n_models):
                js = row_obs[i]
                if js.size == 0:
                    continue
                r = row_vals[i] - mu - c[js]
                X = np.column_stack([np.ones(js.size), V[js]])
                A_mat = X.T @ X + eye_r1
                b_vec = X.T @ r
                sol = np.linalg.solve(A_mat, b_vec)
                a[i] = sol[0]
                U[i] = sol[1:]

            # Block B: per-col update of (c_j, V_j)
            for j in range(n_bench):
                is_ = col_obs[j]
                if is_.size == 0:
                    continue
                r = col_vals[j] - mu - a[is_]
                Y = np.column_stack([np.ones(is_.size), U[is_]])
                A_mat = Y.T @ Y + eye_r1
                b_vec = Y.T @ r
                sol = np.linalg.solve(A_mat, b_vec)
                c[j] = sol[0]
                V[j] = sol[1:]

            # mu update: residual mean over Omega (no reg)
            UV_obs = np.einsum('ij,ij->i',
                               U[obs_ij[:, 0]], V[obs_ij[:, 1]])
            mu = float(np.mean(obs_vals - a[obs_ij[:, 0]]
                               - c[obs_ij[:, 1]] - UV_obs))

        return mu + a[:, None] + c[None, :] + U @ V.T

    Z_bar = np.zeros_like(M_z)
    for s in range(n_ensemble):
        Z_bar += _run_one(base_seed + s)
    Z_bar /= n_ensemble
    return Z_bar


# --------------------------------------------------------------------------
# Pipeline: logit + z-score + ALS + invert
# --------------------------------------------------------------------------

def predict_benchpress_scores(M_train, rank=2, lam=0.1):
    """BenchPress default score predictor: Logit Bias ALS with lam=0.1 and rank=2."""
    obs = ~np.isnan(M_train)
    n_models, n_bench = M_train.shape
    is_pct = np.array([_is_pct_bench(j, M_train) for j in range(n_bench)])

    # Forward: logit per pct-col
    M_t = M_train.copy()
    for j in range(n_bench):
        if is_pct[j]:
            valid = obs[:, j]
            M_t[valid, j] = _to_logit(M_train[valid, j])

    # Per-column z-score
    M_z, cm, cs = col_normalize(M_t)

    # Solve in z-space
    Z_pred = _bias_als_zspace(M_z, rank=rank, lam=lam)

    # Invert
    M_out = M_train.copy()
    for j in range(n_bench):
        for i in range(n_models):
            if obs[i, j]:
                continue
            v = Z_pred[i, j]
            if not np.isfinite(v):
                continue
            v = v * cs[j] + cm[j]
            if is_pct[j]:
                v = _from_logit(v)
                v = float(np.clip(v, 0, 100))
            M_out[i, j] = v
    return M_out


# --------------------------------------------------------------------------
# Add-model entry point (called from app.js)
# --------------------------------------------------------------------------

def predict_new_model(M_list, new_row_scores):
    """Append a new row with `new_row_scores` and return the predicted row.

    M_list:  n_models x n_bench list-of-lists, None for missing.
    new_row_scores: dict {bench_idx_str: float}.
    Returns: list of n_bench predicted floats for the new model.
    """
    n_models = len(M_list)
    n_bench = len(M_list[0])
    M = np.full((n_models, n_bench), np.nan, dtype=float)
    for i in range(n_models):
        row = M_list[i]
        for j in range(n_bench):
            v = row[j]
            if v is not None:
                M[i, j] = float(v)

    new_row = np.full((1, n_bench), np.nan)
    for j_str, v in new_row_scores.items():
        j = int(j_str)
        new_row[0, j] = float(v)

    M_aug = np.vstack([M, new_row])
    P = predict_benchpress_scores(M_aug)
    return P[-1].tolist()
