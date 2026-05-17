#!/usr/bin/env python3
"""BenchPress confidence methods and interval calibration primitives."""

import math
import os
import pickle
import warnings

import numpy as np
from scipy import stats as sp_stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from benchpress.evaluation_harness import (
    M_FULL,
    compute_prediction_error,
    load_folds,
    make_score_predictor,
)
from benchpress.methods.completers import (
    complete_bias_als,
    complete_model_knn,
    complete_soft_impute,
)

SEED = 42
DEFAULT_CONFIDENCE_METHODS = [
    "disagreement",
    "structural_support",
    "combined_risk_model",
]
CONFIDENCE_METHODS = {
    "disagreement": "Ensemble-spread uncertainty model",
    "structural_support": "Matrix-support uncertainty model",
    "combined_risk_model": "Hybrid uncertainty model",
}
DEFAULT_HP_VARIANTS = [
    ("logit", complete_bias_als, {"rank": 2, "lam": 0.01, "normalize": False}),
    ("logit", complete_bias_als, {"rank": 2, "lam": 0.1, "normalize": False}),
    ("logit", complete_bias_als, {"rank": 2, "lam": 1.0, "normalize": False}),
]
DEFAULT_STRONG_METHODS = [
    ("probit", complete_bias_als, {"rank": 2, "lam": 0.1, "normalize": False}),
    ("quantile", complete_bias_als, {"rank": 2, "lam": 0.1, "normalize": False}),
    ("identity", complete_bias_als, {"rank": 2, "lam": 0.1, "normalize": False}),
    ("quantile", complete_soft_impute, {"rank": 2, "normalize": False}),
    ("logit", complete_soft_impute, {"rank": 2, "normalize": False}),
    ("probit", complete_soft_impute, {"rank": 2, "normalize": False}),
    ("asinh", complete_bias_als, {"rank": 2, "lam": 0.1, "normalize": False}),
    ("sqrt", complete_bias_als, {"rank": 2, "lam": 0.1, "normalize": False}),
    ("identity", complete_soft_impute, {"rank": 2, "normalize": False}),
    ("logit", complete_model_knn, {"k": 10}),
    ("probit", complete_model_knn, {"k": 10}),
    ("identity", complete_model_knn, {"k": 10}),
]


def spearman_uncertainty_error(actual, predicted, uncertainty):
    """Spearman correlation between uncertainty and realized absolute error."""
    abs_err = np.abs(np.asarray(predicted) - np.asarray(actual))
    uncertainty = np.asarray(uncertainty)
    valid = np.isfinite(abs_err) & np.isfinite(uncertainty)
    if valid.sum() < 3:
        return float("nan")
    rho, _ = sp_stats.spearmanr(uncertainty[valid], abs_err[valid])
    return float(rho) if np.isfinite(rho) else float("nan")


def coverage_width(actual, lower, upper):
    """Coverage and median interval width for interval predictions."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return {"coverage": float("nan"), "median_width": float("nan"),
                "median_relative_width": float("nan"), "n": 0}
    covered = (actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid])
    widths = upper[valid] - lower[valid]
    denom = np.abs(actual[valid])
    rel_valid = denom > 1e-8
    median_rel = (
        float(np.median(widths[rel_valid] / denom[rel_valid]) * 100.0)
        if np.any(rel_valid)
        else float("nan")
    )
    return {
        "coverage": float(np.mean(covered)),
        "median_width": float(np.median(widths)),
        "median_relative_width": median_rel,
        "n": int(valid.sum()),
    }


def conformal_interval(actual, predicted, uncertainty, fold_id, ci=0.90):
    """Leave-fold-out conformal scaling for raw uncertainty scores."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    fold_id = np.asarray(fold_id, dtype=int)
    scale = np.full_like(predicted, np.nan, dtype=float)
    eps = 1e-8
    for fold in np.unique(fold_id):
        cal = fold_id != fold
        valid = (cal & np.isfinite(actual) & np.isfinite(predicted)
                 & np.isfinite(uncertainty) & (uncertainty > eps))
        if valid.sum() < 5:
            continue
        ratio = np.abs(predicted[valid] - actual[valid]) / uncertainty[valid]
        scale[fold_id == fold] = float(np.quantile(ratio, ci))
    lower = predicted - scale * uncertainty
    upper = predicted + scale * uncertainty
    return lower, upper, scale


def risk_coverage_curve(actual, predicted, uncertainty):
    """Prediction error after keeping the most confident fraction of cells."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(uncertainty)
    order = np.argsort(uncertainty[valid])
    a = actual[valid][order]
    p = predicted[valid][order]
    rows = []
    for frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
        n_keep = max(1, int(math.ceil(frac * len(a))))
        metrics = compute_prediction_error(a[:n_keep], p[:n_keep])
        rows.append({
            "kept_fraction": float(frac),
            "n": int(n_keep),
            "medape": metrics["medape"],
            "medae": metrics["medae"],
        })
    return rows


def uncertainty_tercile_errors(actual, predicted, uncertainty):
    """MedAPE/MedAE in low-, medium-, and high-uncertainty terciles."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(uncertainty)
    if valid.sum() < 3:
        return []
    order = np.argsort(uncertainty[valid])
    splits = np.array_split(order, 3)
    labels = ["low_uncertainty", "medium_uncertainty", "high_uncertainty"]
    a = actual[valid]
    p = predicted[valid]
    rows = []
    for label, idx in zip(labels, splits):
        metrics = compute_prediction_error(a[idx], p[idx])
        rows.append({
            "bin": label,
            "n": int(len(idx)),
            "medape": metrics["medape"],
            "medae": metrics["medae"],
        })
    return rows


def normal_interval(predicted, uncertainty, ci=0.90):
    """Normal-theory interval from a raw uncertainty score."""
    z = 1.6448536269514722 if ci == 0.90 else sp_stats.norm.ppf(0.5 + ci / 2.0)
    return predicted - z * uncertainty, predicted + z * uncertainty


def mad_uncertainty(stack):
    """Robust spread estimate across a stack of prediction methods."""
    median = np.nanmedian(stack, axis=0)
    return 1.4826 * np.nanmedian(np.abs(stack - median[None, :]), axis=0)


def stack_features(stack, target_pred):
    """Features used by the ensemble-spread confidence method."""
    center = np.nanmedian(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    mad = mad_uncertainty(stack)
    delta = np.abs(target_pred - center)
    span = np.nanpercentile(stack, 90, axis=0) - np.nanpercentile(stack, 10, axis=0)
    return {
        "std": std,
        "mad": mad,
        "delta_to_median": delta,
        "p90_p10_span": span,
    }


def confidence_feature_sets(hp_features, strong_features, structural_features):
    """Return feature dictionaries for all three BenchPress confidence methods."""
    disagreement_features = {
        **{f"hp_{k}": v for k, v in hp_features.items()},
        **{f"strong_{k}": v for k, v in strong_features.items()},
    }
    combined_features = {
        **{f"structural_{k}": v for k, v in structural_features.items()},
        **disagreement_features,
    }
    return {
        "disagreement": disagreement_features,
        "structural_support": structural_features,
        "combined_risk_model": combined_features,
    }


def feature_matrix(feature_dict):
    """Turn named nonnegative confidence features into an MLP design matrix."""
    feature_names = sorted(feature_dict)
    X = np.column_stack([
        np.log1p(np.maximum(np.asarray(feature_dict[name], dtype=float), 0.0))
        for name in feature_names
    ])
    return X, feature_names


def fit_mlp_predict(X_train, y_train, X_test, hidden_layers, seed):
    """Fit one MLP risk model and predict held-out risk."""
    scaler, model = fit_mlp_model(X_train, y_train, hidden_layers, seed)
    return model.predict(scaler.transform(X_test))


def fit_mlp_model(X_train, y_train, hidden_layers, seed):
    """Fit and return the scaler/model pair for a confidence MLP."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=3e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train)
    return scaler, model


def select_mlp_config(X, y, fold_id, train_mask, hidden_grid, seed=SEED):
    """Choose the confidence MLP width/depth using training-fold validation."""
    inner_train = train_mask & ((fold_id % 5) != 0)
    inner_val = train_mask & ((fold_id % 5) == 0)
    if inner_val.sum() < 50 or inner_train.sum() < X.shape[1] + 50:
        inner_train = train_mask & ((fold_id % 3) != 0)
        inner_val = train_mask & ((fold_id % 3) == 0)
    if inner_val.sum() < 50 or inner_train.sum() < X.shape[1] + 50:
        return hidden_grid[0]

    scores = []
    for idx, hidden_layers in enumerate(hidden_grid):
        pred = fit_mlp_predict(
            X[inner_train], y[inner_train], X[inner_val],
            hidden_layers, seed=seed + idx)
        score = compute_prediction_error(y[inner_val], pred)["medae"]
        scores.append((score, hidden_layers))
    scores.sort(key=lambda row: row[0])
    return scores[0][1]


def leave_fold_mlp_error_calibrator(actual, predicted, fold_id, feature_dict,
                                    folds_to_run=None, label="mlp", seed=SEED):
    """Cross-fit an MLP that predicts log absolute error from confidence features."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    fold_id = np.asarray(fold_id, dtype=int)
    X, feature_names = feature_matrix(feature_dict)
    y = np.log1p(np.abs(predicted - actual))
    out = np.full(len(y), np.nan, dtype=float)
    hidden_grid = [(16,), (32,), (64, 32)]
    selected = {}

    valid_all = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    folds = np.unique(fold_id)
    if folds_to_run is not None:
        folds_to_run = {int(f) for f in folds_to_run}
        folds = np.asarray([f for f in folds if int(f) in folds_to_run], dtype=int)
    for fold in folds:
        print(f"[{label}] fold {int(fold)} start", flush=True)
        train = (fold_id != fold) & valid_all
        test = (fold_id == fold) & np.all(np.isfinite(X), axis=1)
        if train.sum() < X.shape[1] + 50 or test.sum() == 0:
            continue
        hidden_layers = select_mlp_config(X, y, fold_id, train, hidden_grid, seed=seed)
        selected[str(int(fold))] = list(hidden_layers)
        pred = fit_mlp_predict(
            X[train], y[train], X[test],
            hidden_layers, seed=seed + 1000 + int(fold))
        out[test] = np.expm1(pred)
        print(f"[{label}] fold {int(fold)} done hidden={hidden_layers}", flush=True)
    return np.maximum(out, 0.0), feature_names, selected


def best_axis_correlations(M, axis):
    """Return each row/column's strongest same-axis neighbor in the training matrix."""
    arr = np.asarray(M, dtype=float)
    if axis == 0:
        arr = arr.T
    n_items = arr.shape[0]
    best_corr = np.zeros(n_items, dtype=float)
    best_overlap = np.zeros(n_items, dtype=float)
    for a in range(n_items):
        xa = arr[a]
        for b in range(a + 1, n_items):
            xb = arr[b]
            mask = np.isfinite(xa) & np.isfinite(xb)
            overlap = int(mask.sum())
            if overlap < 3:
                continue
            va = xa[mask]
            vb = xb[mask]
            if np.std(va) <= 1e-12 or np.std(vb) <= 1e-12:
                continue
            corr = abs(float(np.corrcoef(va, vb)[0, 1]))
            if not np.isfinite(corr):
                continue
            if corr > best_corr[a]:
                best_corr[a] = corr
                best_overlap[a] = overlap
            if corr > best_corr[b]:
                best_corr[b] = corr
                best_overlap[b] = overlap
    return best_corr, best_overlap


def _safe_nanmedian(arr, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmedian(arr, axis=axis)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_nanstd(arr, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanstd(arr, axis=axis)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def structural_support_features(reference, folds=None):
    """Features used by the matrix-support confidence method."""
    if folds is None:
        folds = load_folds(n_seeds=10, n_folds=3, base_seed=42, min_scores=1)
    feature_names = [
        "row_obs_count",
        "col_obs_count",
        "row_median_score",
        "col_median_score",
        "col_score_dispersion",
        "row_best_peer_abs_corr",
        "row_best_peer_overlap",
        "col_best_neighbor_abs_corr",
    ]
    features = {
        name: np.full(len(reference["actual"]), np.nan, dtype=float)
        for name in feature_names
    }

    cursor = 0
    for fold_idx, (M_train, test_set) in enumerate(folds):
        M_train = np.asarray(M_train, dtype=float)
        row_obs = np.isfinite(M_train).sum(axis=1).astype(float)
        col_obs = np.isfinite(M_train).sum(axis=0).astype(float)
        row_median = _safe_nanmedian(M_train, axis=1)
        col_median = _safe_nanmedian(M_train, axis=0)
        col_std = _safe_nanstd(M_train, axis=0)
        row_best_corr, row_best_overlap = best_axis_correlations(M_train, axis=1)
        col_best_corr, _ = best_axis_correlations(M_train, axis=0)

        n = len(test_set)
        for local_idx, (i, j) in enumerate(test_set):
            idx = cursor + local_idx
            if int(reference["fold_id"][idx]) != fold_idx:
                raise ValueError("Fold order mismatch in structural feature computation")
            if int(reference["test_i"][idx]) != i or int(reference["test_j"][idx]) != j:
                raise ValueError("Test-cell order mismatch in structural feature computation")
            features["row_obs_count"][idx] = row_obs[i]
            features["col_obs_count"][idx] = col_obs[j]
            features["row_median_score"][idx] = row_median[i]
            features["col_median_score"][idx] = col_median[j]
            features["col_score_dispersion"][idx] = col_std[j]
            features["row_best_peer_abs_corr"][idx] = row_best_corr[i]
            features["row_best_peer_overlap"][idx] = row_best_overlap[i]
            features["col_best_neighbor_abs_corr"][idx] = col_best_corr[j]
        cursor += n

    return features


def structural_support_features_for_cells(M_train, cells):
    """Matrix-support features for arbitrary cells in a deployment matrix."""
    M_train = np.asarray(M_train, dtype=float)
    cells = [(int(i), int(j)) for i, j in cells]
    row_obs = np.isfinite(M_train).sum(axis=1).astype(float)
    col_obs = np.isfinite(M_train).sum(axis=0).astype(float)
    row_median = _safe_nanmedian(M_train, axis=1)
    col_median = _safe_nanmedian(M_train, axis=0)
    col_std = _safe_nanstd(M_train, axis=0)
    row_best_corr, row_best_overlap = best_axis_correlations(M_train, axis=1)
    col_best_corr, _ = best_axis_correlations(M_train, axis=0)
    return {
        "row_obs_count": np.asarray([row_obs[i] for i, _ in cells], dtype=float),
        "col_obs_count": np.asarray([col_obs[j] for _, j in cells], dtype=float),
        "row_median_score": np.asarray([row_median[i] for i, _ in cells], dtype=float),
        "col_median_score": np.asarray([col_median[j] for _, j in cells], dtype=float),
        "col_score_dispersion": np.asarray([col_std[j] for _, j in cells], dtype=float),
        "row_best_peer_abs_corr": np.asarray(
            [row_best_corr[i] for i, _ in cells], dtype=float),
        "row_best_peer_overlap": np.asarray(
            [row_best_overlap[i] for i, _ in cells], dtype=float),
        "col_best_neighbor_abs_corr": np.asarray(
            [col_best_corr[j] for _, j in cells], dtype=float),
    }


def default_confidence_artifact_path():
    """Default persisted calibrator location shipped/created with the package."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "evaluation",
        "default_confidence",
        "benchpress_default",
        "calibrator.pkl",
    )


def _predict_with_spec(M_train, spec):
    transform, completer, kwargs = spec
    return make_score_predictor(completer, transform, **kwargs)(M_train)


def _prediction_stack(M_train, specs):
    return np.stack([_predict_with_spec(M_train, spec) for spec in specs], axis=0)


def default_confidence_features(M_train, target_pred=None, cells=None):
    """Build the three BenchPress confidence feature sets for deployment cells."""
    M_train = np.asarray(M_train, dtype=float)
    if target_pred is None:
        target_pred = _predict_with_spec(M_train, DEFAULT_HP_VARIANTS[1])
    if cells is None:
        cell_arr = np.argwhere(~np.isfinite(M_train))
        cells = [(int(i), int(j)) for i, j in cell_arr]
    else:
        cells = [(int(i), int(j)) for i, j in cells]
    if not cells:
        empty = np.asarray([], dtype=float)
        return target_pred, {
            name: {feature: empty for feature in []}
            for name in DEFAULT_CONFIDENCE_METHODS
        }, cells

    rows = np.asarray([i for i, _ in cells], dtype=int)
    cols = np.asarray([j for _, j in cells], dtype=int)
    hp_stack = _prediction_stack(M_train, DEFAULT_HP_VARIANTS)[:, rows, cols]
    strong_stack = _prediction_stack(M_train, DEFAULT_STRONG_METHODS)[:, rows, cols]
    target_values = target_pred[rows, cols]
    hp_features = stack_features(hp_stack, target_values)
    strong_features = stack_features(strong_stack, target_values)
    structural_features = structural_support_features_for_cells(M_train, cells)
    return target_pred, confidence_feature_sets(
        hp_features, strong_features, structural_features), cells


def _training_records(M, folds):
    fold_ids, rows, cols, actual, predicted = [], [], [], [], []
    strong_parts = []
    structural_parts = []
    for fold_id, (M_train, test_set) in enumerate(folds):
        target_pred, feature_sets, cells = default_confidence_features(
            M_train, cells=test_set)
        cell_rows = np.asarray([i for i, _ in cells], dtype=int)
        cell_cols = np.asarray([j for _, j in cells], dtype=int)
        fold_ids.append(np.full(len(cells), fold_id, dtype=int))
        rows.append(cell_rows)
        cols.append(cell_cols)
        actual.append(np.asarray([M[i, j] for i, j in cells], dtype=float))
        predicted.append(target_pred[cell_rows, cell_cols])
        strong_parts.append(feature_sets["disagreement"])
        structural_parts.append(feature_sets["structural_support"])

    def concat_dict(parts):
        keys = sorted(parts[0])
        return {key: np.concatenate([part[key] for part in parts]) for key in keys}

    hp_strong = concat_dict(strong_parts)
    structural = concat_dict(structural_parts)
    return {
        "fold_id": np.concatenate(fold_ids),
        "test_i": np.concatenate(rows),
        "test_j": np.concatenate(cols),
        "actual": np.concatenate(actual),
        "predicted": np.concatenate(predicted),
        "feature_sets": {
            "disagreement": hp_strong,
            "structural_support": structural,
            "combined_risk_model": {
                **{f"structural_{k}": v for k, v in structural.items()},
                **hp_strong,
            },
        },
    }


def _fit_final_confidence_model(actual, predicted, fold_id, feature_dict,
                                crossfit_uncertainty, seed=SEED, ci=0.90):
    X, feature_names = feature_matrix(feature_dict)
    y = np.log1p(np.abs(np.asarray(predicted, dtype=float)
                        - np.asarray(actual, dtype=float)))
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    hidden_grid = [(16,), (32,), (64, 32)]
    hidden_layers = select_mlp_config(
        X, y, np.asarray(fold_id, dtype=int), valid, hidden_grid, seed=seed)
    scaler, model = fit_mlp_model(X[valid], y[valid], hidden_layers, seed=seed + 2000)
    unc = np.asarray(crossfit_uncertainty, dtype=float)
    ratio_valid = valid & np.isfinite(unc) & (unc > 1e-8)
    scale = float(np.quantile(np.abs(predicted[ratio_valid] - actual[ratio_valid])
                              / unc[ratio_valid], ci))
    return {
        "feature_names": feature_names,
        "hidden_layers": list(hidden_layers),
        "scaler": scaler,
        "model": model,
        "conformal_ci": float(ci),
        "conformal_scale": scale,
    }


def train_default_confidence_calibrator(M=None, folds=None, methods=None,
                                        artifact_path=None, seed=SEED):
    """Train and persist the default BenchPress confidence calibrator."""
    M = M_FULL if M is None else np.asarray(M, dtype=float)
    if folds is None:
        folds = load_folds(n_seeds=10, n_folds=3, base_seed=42, min_scores=1)
    methods = DEFAULT_CONFIDENCE_METHODS if methods is None else list(methods)
    records = _training_records(M, folds)
    calibrators = {}
    crossfit_selected = {}
    for method in methods:
        uncertainty, feature_names, selected = leave_fold_mlp_error_calibrator(
            records["actual"],
            records["predicted"],
            records["fold_id"],
            records["feature_sets"][method],
            label=method,
            seed=seed,
        )
        crossfit_selected[method] = selected
        calibrators[method] = _fit_final_confidence_model(
            records["actual"],
            records["predicted"],
            records["fold_id"],
            records["feature_sets"][method],
            uncertainty,
            seed=seed,
            ci=0.90,
        )
        calibrators[method]["crossfit_feature_names"] = feature_names

    artifact = {
        "version": 1,
        "matrix_shape": list(M.shape),
        "seed": int(seed),
        "methods": methods,
        "calibrators": calibrators,
        "crossfit_selected_hidden_layers_by_fold": crossfit_selected,
    }
    if artifact_path is None:
        artifact_path = default_confidence_artifact_path()
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    tmp_path = f"{artifact_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(artifact, f)
    os.replace(tmp_path, artifact_path)
    return artifact


def load_or_train_default_confidence_calibrator(artifact_path=None,
                                                train_if_missing=True):
    """Load the default confidence artifact, training and saving it if missing."""
    if artifact_path is None:
        artifact_path = default_confidence_artifact_path()
    if os.path.exists(artifact_path):
        with open(artifact_path, "rb") as f:
            return pickle.load(f)
    if not train_if_missing:
        raise FileNotFoundError(
            f"BenchPress confidence artifact not found: {artifact_path}. "
            "Call with train_if_missing=True or run train_default_confidence_calibrator()."
        )
    return train_default_confidence_calibrator(artifact_path=artifact_path)


def predict_confidence_intervals(M_train, M_pred=None, artifact=None,
                                 artifact_path=None, method="combined_risk_model",
                                 train_if_missing=True, cells=None):
    """Predict uncertainty and conformal intervals for missing/deployment cells."""
    if artifact is None:
        artifact = load_or_train_default_confidence_calibrator(
            artifact_path=artifact_path, train_if_missing=train_if_missing)
    if method not in artifact["calibrators"]:
        raise ValueError(f"Unknown confidence method {method!r}; available: "
                         f"{sorted(artifact['calibrators'])}")
    target_pred, feature_sets, cells = default_confidence_features(
        M_train, target_pred=M_pred, cells=cells)
    cal = artifact["calibrators"][method]
    feature_dict = feature_sets[method]
    if not cells:
        empty = np.asarray([], dtype=float)
        return {
            "method": method,
            "cells": [],
            "predicted": empty,
            "uncertainty": empty,
            "lower": empty,
            "upper": empty,
            "confidence_level": float(cal["conformal_ci"]),
            "artifact": artifact,
        }
    X = np.column_stack([
        np.log1p(np.maximum(np.asarray(feature_dict[name], dtype=float), 0.0))
        for name in cal["feature_names"]
    ])
    uncertainty = np.expm1(cal["model"].predict(cal["scaler"].transform(X)))
    uncertainty = np.maximum(uncertainty, 0.0)
    rows = np.asarray([i for i, _ in cells], dtype=int)
    cols = np.asarray([j for _, j in cells], dtype=int)
    predicted = target_pred[rows, cols]
    width = uncertainty * float(cal["conformal_scale"])
    return {
        "method": method,
        "cells": cells,
        "predicted": predicted,
        "uncertainty": uncertainty,
        "lower": predicted - width,
        "upper": predicted + width,
        "confidence_level": float(cal["conformal_ci"]),
        "artifact": artifact,
    }


def summarize_confidence_method(name, actual, predicted, fold_id, uncertainty,
                                lower=None, upper=None):
    """Aggregate risk ranking and interval metrics for a confidence method."""
    out = {
        "spearman_uncertainty_abs_error": spearman_uncertainty_error(
            actual, predicted, uncertainty),
        "risk_coverage_curve": risk_coverage_curve(actual, predicted, uncertainty),
        "uncertainty_terciles": uncertainty_tercile_errors(
            actual, predicted, uncertainty),
    }

    raw_lower, raw_upper = normal_interval(predicted, uncertainty, ci=0.90)
    out["normal_90_interval"] = coverage_width(actual, raw_lower, raw_upper)

    conf_lower, conf_upper, scale = conformal_interval(
        actual, predicted, uncertainty, fold_id, ci=0.90)
    out["conformal_90_interval"] = coverage_width(actual, conf_lower, conf_upper)
    out["conformal_90_scale_median"] = (
        float(np.nanmedian(scale)) if np.any(np.isfinite(scale)) else float("nan")
    )

    if lower is not None and upper is not None:
        out["native_90_interval"] = coverage_width(actual, lower, upper)

    return name, out
