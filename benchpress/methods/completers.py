#!/usr/bin/env python3
"""Completion methods for BenchPress matrix completion."""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from benchpress.evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_PROVIDERS, MODEL_REASONING,
    MODEL_OPEN, MODEL_PARAMS, MODEL_ACTIVE, BENCH_CATS,
    col_normalize, col_denormalize,
)

# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK MEAN
# ══════════════════════════════════════════════════════════════════════════════

def complete_benchmark_mean(M_train):
    """Complete missing cells with the column (benchmark) mean."""
    M_pred = M_train.copy()
    col_mean = np.nanmean(M_train, axis=0)
    for j in range(N_BENCH):
        mask = np.isnan(M_pred[:, j])
        M_pred[mask, j] = col_mean[j]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL MEAN
# ══════════════════════════════════════════════════════════════════════════════

def complete_model_mean(M_train):
    """Complete missing cells with the row (model) mean."""
    M_pred = M_train.copy()
    obs = ~np.isnan(M_train)
    n_models, n_bench = M_train.shape
    row_sum = np.nansum(M_train, axis=1)
    row_count = obs.sum(axis=1)
    row_mean = np.divide(
        row_sum,
        row_count,
        out=np.full(n_models, np.nan),
        where=row_count > 0,
    )
    col_sum = np.nansum(M_train, axis=0)
    col_count = obs.sum(axis=0)
    col_mean = np.divide(
        col_sum,
        col_count,
        out=np.full(n_bench, np.nan),
        where=col_count > 0,
    )

    for i in range(n_models):
        for j in range(n_bench):
            if np.isnan(M_pred[i, j]):
                M_pred[i, j] = row_mean[i] if np.isfinite(row_mean[i]) else col_mean[j]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-KNN
# ══════════════════════════════════════════════════════════════════════════════

def complete_model_knn(M_train, k=5):
    """Complete missing cells from nearest models by RMSE over shared benchmarks."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    for i in range(M_train.shape[0]):
        missing_j = np.where(np.isnan(M_train[i]))[0]
        if len(missing_j) == 0:
            continue
        obs_i = obs[i]
        dists = []
        for i2 in range(M_train.shape[0]):
            if i2 == i:
                continue
            shared = obs_i & obs[i2]
            if shared.sum() < 3:
                continue
            d = np.sqrt(np.mean((M_train[i, shared] - M_train[i2, shared]) ** 2))
            dists.append((i2, d))
        dists.sort(key=lambda x: x[1])
        neighbors = [d[0] for d in dists[:k]]

        for j in missing_j:
            vals = [M_train[i2, j] for i2 in neighbors if obs[i2, j]]
            if vals:
                M_pred[i, j] = np.mean(vals)
            else:
                M_pred[i, j] = np.nanmean(M_train[:, j])
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  BENCH-KNN (transposed)
# ══════════════════════════════════════════════════════════════════════════════

def complete_bench_knn(M_train, k=5):
    """For each missing (i,j), find k benchmarks most correlated with j, predict from model i's scores on them.
    
    Iterates benchmark-first: for each benchmark j, precompute top-k most correlated
    benchmarks (from ALL benchmarks), then for each model i with missing j, use those
    neighbors if observed.
    """
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()
    _, cm, cs = col_normalize(M_train)

    for j in range(N_BENCH):
        corrs = []
        for j2 in range(N_BENCH):
            if j2 == j:
                continue
            shared = obs[:, j] & obs[:, j2]
            if shared.sum() < 5:
                continue
            r = np.corrcoef(M_train[shared, j], M_train[shared, j2])[0, 1]
            if np.isfinite(r):
                corrs.append((j2, r))
        corrs.sort(key=lambda x: -x[1])
        top_j = [c[0] for c in corrs[:k]]

        for i in range(N_MODELS):
            if obs[i, j]:
                continue
            vals, weights, source_cols = [], [], []
            for j2 in top_j:
                if obs[i, j2]:
                    vals.append(M_train[i, j2])
                    r2 = [c[1] for c in corrs if c[0] == j2][0]
                    weights.append(max(r2, 0.01))
                    source_cols.append(j2)
            if vals:
                weights = np.array(weights)
                weights /= weights.sum()
                pred_z = 0.0
                for w, v, j2 in zip(weights, vals, source_cols):
                    pred_z += w * (v - cm[j2]) / cs[j2]
                M_pred[i, j] = cm[j] + pred_z * cs[j]
            else:
                M_pred[i, j] = cm[j]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  BENCHREG (current best from v8)
# ══════════════════════════════════════════════════════════════════════════════

def complete_benchreg(M_train, top_k=5, min_r2=0.2):
    """Predict each benchmark from top_k most correlated benchmarks using linear regression."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()

    for j in range(N_BENCH):
        targets_obs = np.where(obs[:, j])[0]
        if len(targets_obs) < 5:
            continue
        correlations = []
        for j2 in range(N_BENCH):
            if j2 == j:
                continue
            shared = obs[:, j] & obs[:, j2]
            if shared.sum() < 5:
                correlations.append((j2, -1))
                continue
            x, y = M_train[shared, j2], M_train[shared, j]
            ss_tot = np.sum((y - y.mean())**2)
            if ss_tot < 1e-10:
                correlations.append((j2, -1))
                continue
            cov = np.sum((x - x.mean()) * (y - y.mean()))
            var_x = np.sum((x - x.mean())**2)
            if var_x < 1e-10:
                correlations.append((j2, -1))
                continue
            slope = cov / var_x
            intercept = y.mean() - slope * x.mean()
            y_hat = slope * x + intercept
            ss_res = np.sum((y - y_hat)**2)
            r2 = 1 - ss_res / ss_tot
            correlations.append((j2, r2))
        correlations.sort(key=lambda x: -x[1])
        best = [(j2, r2) for j2, r2 in correlations[:top_k] if r2 >= min_r2]
        if not best:
            continue
        for i in range(N_MODELS):
            if not np.isnan(M_train[i, j]):
                continue
            preds, weights = [], []
            for j2, r2 in best:
                if np.isnan(M_train[i, j2]):
                    continue
                shared = obs[:, j] & obs[:, j2]
                if shared.sum() < 5:
                    continue
                x, y = M_train[shared, j2], M_train[shared, j]
                cov = np.sum((x - x.mean()) * (y - y.mean()))
                var_x = np.sum((x - x.mean())**2)
                if var_x < 1e-10:
                    continue
                slope = cov / var_x
                intercept = y.mean() - slope * x.mean()
                preds.append(slope * M_train[i, j2] + intercept)
                weights.append(r2)
            if preds:
                M_pred[i, j] = np.average(preds, weights=weights)
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  MODELREG (row-wise counterpart to BenchReg)
# ══════════════════════════════════════════════════════════════════════════════

def complete_modelreg(M_train, top_k=5, min_r2=0.2):
    """Predict each model from top_k most correlated models using linear regression."""
    obs = ~np.isnan(M_train)
    M_pred = M_train.copy()
    n_models, n_bench = M_train.shape

    for i in range(n_models):
        targets_obs = np.where(obs[i])[0]
        if len(targets_obs) < 5:
            continue
        correlations = []
        for i2 in range(n_models):
            if i2 == i:
                continue
            shared = obs[i] & obs[i2]
            if shared.sum() < 5:
                correlations.append((i2, -1))
                continue
            x, y = M_train[i2, shared], M_train[i, shared]
            ss_tot = np.sum((y - y.mean())**2)
            if ss_tot < 1e-10:
                correlations.append((i2, -1))
                continue
            cov = np.sum((x - x.mean()) * (y - y.mean()))
            var_x = np.sum((x - x.mean())**2)
            if var_x < 1e-10:
                correlations.append((i2, -1))
                continue
            slope = cov / var_x
            intercept = y.mean() - slope * x.mean()
            y_hat = slope * x + intercept
            ss_res = np.sum((y - y_hat)**2)
            r2 = 1 - ss_res / ss_tot
            correlations.append((i2, r2))
        correlations.sort(key=lambda x: -x[1])
        best = [(i2, r2) for i2, r2 in correlations[:top_k] if r2 >= min_r2]
        if not best:
            continue
        for j in range(n_bench):
            if not np.isnan(M_train[i, j]):
                continue
            preds, weights = [], []
            for i2, r2 in best:
                if np.isnan(M_train[i2, j]):
                    continue
                shared = obs[i] & obs[i2]
                if shared.sum() < 5:
                    continue
                x, y = M_train[i2, shared], M_train[i, shared]
                cov = np.sum((x - x.mean()) * (y - y.mean()))
                var_x = np.sum((x - x.mean())**2)
                if var_x < 1e-10:
                    continue
                slope = cov / var_x
                intercept = y.mean() - slope * x.mean()
                preds.append(slope * M_train[i2, j] + intercept)
                weights.append(r2)
            if preds:
                M_pred[i, j] = np.average(preds, weights=weights)
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  M1: SOFT-IMPUTE
# ══════════════════════════════════════════════════════════════════════════════

def complete_soft_impute(M_train, rank=5, max_iter=100, tol=1e-4, normalize=True):
    """Soft-Impute: iterate SVD completion until convergence.
    
    Args:
        normalize: If True (default), internally z-scores before Soft-Impute (standalone mode).
                   If False, operates directly on input matrix (pipeline mode).
    """
    obs = ~np.isnan(M_train)

    if normalize:
        M_work, cm, cs = col_normalize(M_train)
    else:
        M_work = M_train.copy()
        cm, cs = None, None

    # Initialize missing with column mean
    col_mean = np.nanmean(M_work, axis=0)
    M_imp = M_work.copy()
    for j in range(M_work.shape[1]):
        M_imp[np.isnan(M_imp[:, j]), j] = col_mean[j] if np.isfinite(col_mean[j]) else 0

    converged = False
    for it in range(max_iter):
        M_old = M_imp.copy()
        try:
            U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        M_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        M_imp = np.where(obs, M_work, M_approx)
        M_imp[np.isnan(M_imp)] = 0
        diff = np.sqrt(np.mean((M_imp - M_old)**2))
        rel_diff = diff / (np.sqrt(np.mean(M_old**2)) + 1e-12)
        if rel_diff < tol:
            converged = True
            break
    if not converged:
        warnings.warn(f"Soft-Impute rank-{rank} did not converge after {max_iter} iters "
                       f"(final rel_diff={rel_diff:.3e}, tol={tol:.0e})")

    M_pred = col_denormalize(M_imp, cm, cs) if normalize else M_imp.copy()
    M_pred[obs] = M_train[obs]
    return M_pred
# ══════════════════════════════════════════════════════════════════════════════
#  M1b: Bias-decomposed rank-r ALS (Dimitris §14.15, 2026-04-25)
# ══════════════════════════════════════════════════════════════════════════════

def complete_bias_als(M_train, rank=2, lam=0.1, n_iter=40,
                     n_ensemble=10, base_seed=42, normalize=True):
    """Bias-decomposed low-rank completion via ALS, ensemble averaged.

    Model:  Z_hat_ij = mu + a_i + c_j + U_i . V_j
        - mu        : global mean
        - a_i       : per-model bias (n_models,)
        - c_j       : per-benchmark bias (n_bench,)
        - U_i, V_j  : rank-r latent factors

    Loss on observed entries Omega:
        sum (Z_ij - mu - a_i - c_j - U_i.V_j)^2
        + lam * (||a||^2 + ||c||^2 + ||U||_F^2 + ||V||_F^2)

    Optimizer: alternating ridge regression. Each block solves an (r+1)-dim
    linear system per row/column in closed form. mu updated as residual mean
    (no regularization, per spec).

    Ensemble: average n_ensemble independent random initializations.

    Args:
        rank        : latent dimension (Dimitris uses 2)
        lam         : L2 regularization strength on a, c, U, V
        n_iter      : ALS iterations per init (Dimitris uses ~40)
        n_ensemble  : number of random inits to average
        base_seed   : ensemble seeds = base_seed + 0..n_ensemble-1
        normalize   : if True, internally col-normalize (standalone mode);
                      if False, operate directly (pipeline mode)
    """
    obs = ~np.isnan(M_train)
    n_models, n_bench = M_train.shape

    if normalize:
        M_work, cm, cs = col_normalize(M_train)
    else:
        M_work = M_train.copy()
        cm, cs = None, None

    # Pre-compute per-row / per-col observed indices and values
    obs_work = ~np.isnan(M_work)
    row_obs = [np.where(obs_work[i])[0] for i in range(n_models)]
    col_obs = [np.where(obs_work[:, j])[0] for j in range(n_bench)]
    row_vals = [M_work[i, row_obs[i]] for i in range(n_models)]
    col_vals = [M_work[col_obs[j], j] for j in range(n_bench)]

    obs_ij = np.argwhere(obs_work)  # (n_obs, 2)
    obs_vals = M_work[obs_work]
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
            # Block A: fix (c, V), update (a_i, U_i) per row
            for i in range(n_models):
                js = row_obs[i]
                if js.size == 0:
                    continue
                # residual r_ij = Z_ij - mu - c_j  (a_i + U_i.V_j to fit)
                r = row_vals[i] - mu - c[js]
                X = np.column_stack([np.ones(js.size), V[js]])  # (|S|, r+1)
                # ridge: (X^T X + lam I) [a_i; U_i] = X^T r
                A_mat = X.T @ X + eye_r1
                b_vec = X.T @ r
                sol = np.linalg.solve(A_mat, b_vec)
                a[i] = sol[0]
                U[i] = sol[1:]

            # Block B: fix (a, U), update (c_j, V_j) per col
            for j in range(n_bench):
                is_ = col_obs[j]
                if is_.size == 0:
                    continue
                r = col_vals[j] - mu - a[is_]
                Y = np.column_stack([np.ones(is_.size), U[is_]])  # (|S|, r+1)
                A_mat = Y.T @ Y + eye_r1
                b_vec = Y.T @ r
                sol = np.linalg.solve(A_mat, b_vec)
                c[j] = sol[0]
                V[j] = sol[1:]

            # mu update: residual mean over Omega (no reg per spec)
            UV_obs = np.einsum('ij,ij->i', U[obs_ij[:, 0]], V[obs_ij[:, 1]])
            mu = float(np.mean(obs_vals - a[obs_ij[:, 0]] - c[obs_ij[:, 1]] - UV_obs))

        # Reconstruct full matrix
        Z_hat = mu + a[:, None] + c[None, :] + U @ V.T
        return Z_hat

    Z_bar = np.zeros_like(M_work)
    for s in range(n_ensemble):
        Z_bar += _run_one(base_seed + s)
    Z_bar /= n_ensemble

    M_pred = col_denormalize(Z_bar, cm, cs) if normalize else Z_bar.copy()
    M_pred[obs] = M_train[obs]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  M2: Nuclear Norm (proximal gradient with soft-thresholding)
# ══════════════════════════════════════════════════════════════════════════════

def complete_nuclear_norm(M_train, lam=1.0, max_iter=200, lr=0.1, normalize=True):
    """Nuclear norm minimization via proximal gradient descent.
    
    Args:
        normalize: If True (default), internally z-scores (standalone mode).
                   If False, operates directly on input matrix (pipeline mode).
    """
    obs = ~np.isnan(M_train)

    if normalize:
        M_work, cm, cs = col_normalize(M_train)
    else:
        M_work = M_train.copy()
        cm, cs = None, None

    # Initialize missing values with column means (not zeros)
    col_mean = np.nanmean(M_work, axis=0)
    M_imp = M_work.copy()
    nan_mask = np.isnan(M_imp)
    M_imp[nan_mask] = np.broadcast_to(col_mean, M_imp.shape)[nan_mask]
    obs_work = ~np.isnan(M_work)

    for it in range(max_iter):
        grad = np.zeros_like(M_imp)
        grad[obs_work] = M_imp[obs_work] - M_work[obs_work]
        M_tmp = M_imp - lr * grad
        try:
            U, s, Vt = np.linalg.svd(M_tmp, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        s_thresh = np.maximum(s - lam * lr, 0)
        M_imp = U @ np.diag(s_thresh) @ Vt

    M_pred = col_denormalize(M_imp, cm, cs) if normalize else M_imp.copy()
    M_pred[obs] = M_train[obs]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  M3: NMF with masked loss
# ══════════════════════════════════════════════════════════════════════════════

def complete_nmf(M_train, rank=5, max_iter=500, lr=0.0005, reg=0.01, normalize=True):
    """NMF on non-negative data with masked loss.
    
    Args:
        normalize: If True (default), internally z-scores before NMF (standalone mode).
                   If False, operates directly on input matrix (pipeline mode).
    """
    obs = ~np.isnan(M_train)

    if normalize:
        M_work, cm, cs = col_normalize(M_train)
    else:
        M_work = M_train.copy()
        cm, cs = None, None

    # Shift to non-negative
    col_min = np.nanmin(M_work, axis=0)
    shift = np.where(col_min < 0, -col_min + 0.1, 0)
    M_shifted = M_work.copy()
    for j in range(M_work.shape[1]):
        valid = ~np.isnan(M_shifted[:, j])
        M_shifted[valid, j] += shift[j]
    M_shifted[np.isnan(M_shifted)] = 0

    n_models, n_bench = M_train.shape
    rng = np.random.RandomState(42)
    scale = np.sqrt(np.nanmean(M_shifted[obs]) / rank + 0.01)
    W = np.abs(rng.randn(n_models, rank)) * scale + 0.1
    H = np.abs(rng.randn(rank, n_bench)) * scale + 0.1

    for it in range(max_iter):
        M_approx = W @ H
        err = np.zeros_like(M_shifted)
        err[obs] = M_approx[obs] - M_shifted[obs]
        W = np.maximum(W - lr * (err @ H.T + reg * W), 1e-10)
        H = np.maximum(H - lr * (W.T @ err + reg * H), 1e-10)

    M_pred = W @ H
    for j in range(n_bench):
        M_pred[:, j] -= shift[j]
    if normalize:
        M_pred = col_denormalize(M_pred, cm, cs)
    M_pred[obs] = M_train[obs]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  M4: PMF (Probabilistic Matrix Factorization via MAP)
# ══════════════════════════════════════════════════════════════════════════════

def complete_pmf(M_train, rank=5, max_iter=300, lr=0.001, reg=0.1, normalize=True):
    """PMF with L2 regularization (MAP estimation).
    
    Args:
        normalize: If True (default), internally z-scores (standalone mode).
                   If False, operates directly on input matrix (pipeline mode).
    """
    obs = ~np.isnan(M_train)
    n_models, n_bench = M_train.shape

    if normalize:
        M_work, cm, cs = col_normalize(M_train)
        M_work[np.isnan(M_work)] = 0
    else:
        M_work = M_train.copy()
        M_work[np.isnan(M_work)] = 0
        cm, cs = None, None

    rng = np.random.RandomState(42)
    U = rng.randn(n_models, rank) * 0.1
    V = rng.randn(n_bench, rank) * 0.1

    for it in range(max_iter):
        M_approx = U @ V.T
        err = np.zeros_like(M_work)
        err[obs] = M_approx[obs] - M_work[obs]
        U -= lr * (err @ V + reg * U)
        V -= lr * (err.T @ U + reg * V)

    M_pred = U @ V.T
    if normalize:
        M_pred = col_denormalize(M_pred, cm, cs)
    M_pred[obs] = M_train[obs]
    return M_pred

# ══════════════════════════════════════════════════════════════════════════════
#  M5: MLP (2-layer neural network)
# ══════════════════════════════════════════════════════════════════════════════

def complete_mlp(M_train, hidden=32, epochs=500, lr=1e-3, n_seeds=3):
    """2-layer MLP autoencoder for matrix completion. Average over n_seeds for stability."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return np.full_like(M_train, np.nan)

    obs = ~np.isnan(M_train)
    n_models, n_bench = M_train.shape
    M_norm, col_mean, col_std = col_normalize(M_train)

    # Replace NaN with 0 for torch input
    M_norm_filled = np.where(obs, M_norm, 0)
    X = torch.tensor(M_norm_filled, dtype=torch.float32)
    mask = torch.tensor(obs, dtype=torch.float32)

    preds_all = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(
            nn.Linear(n_bench, hidden), nn.ReLU(),
            nn.Linear(hidden, n_bench),
        )
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        for _ in range(epochs):
            out = net(X)
            loss = ((out - X) ** 2 * mask).sum() / mask.sum()
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            pred_norm = net(X).numpy()
        preds_all.append(col_denormalize(pred_norm, col_mean, col_std))

    M_pred = np.mean(preds_all, axis=0)
    M_pred[obs] = M_train[obs]
    return M_pred
