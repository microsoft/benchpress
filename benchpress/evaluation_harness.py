#!/usr/bin/env python3
"""Evaluation harness for LLM benchmark matrix completion."""

import numpy as np
import sys, warnings, os
from collections import defaultdict
from benchpress.io_utils import load_json, safe_token, write_json, write_json_atomic

warnings.filterwarnings('ignore')
from benchpress.build_benchmark_matrix import MODELS, BENCHMARKS, DATA

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

MODEL_IDS   = [m[0] for m in MODELS]
BENCH_IDS   = [b[0] for b in BENCHMARKS]
MODEL_NAMES = {m[0]: m[1] for m in MODELS}
BENCH_NAMES = {b[0]: b[1] for b in BENCHMARKS}
MODEL_IDX   = {m: i for i, m in enumerate(MODEL_IDS)}
BENCH_IDX   = {b: i for i, b in enumerate(BENCH_IDS)}
N_MODELS    = len(MODEL_IDS)
N_BENCH     = len(BENCH_IDS)

# Metadata arrays
MODEL_PROVIDERS  = np.array([m[2] for m in MODELS])
MODEL_REASONING  = np.array([m[7] if len(m) > 7 else False for m in MODELS], dtype=bool)
MODEL_OPEN       = np.array([m[8] if len(m) > 8 else False for m in MODELS], dtype=bool)
MODEL_PARAMS     = np.array([m[4] if len(m) > 4 and m[4] is not None else np.nan for m in MODELS], dtype=float)
MODEL_ACTIVE     = np.array([m[5] if len(m) > 5 and m[5] is not None else np.nan for m in MODELS], dtype=float)
BENCH_CATS       = np.array([b[2] for b in BENCHMARKS])

# Build full matrix
M_FULL = np.full((N_MODELS, N_BENCH), np.nan)
for mid, bid, score, url in DATA:
    if mid in MODEL_IDX and bid in BENCH_IDX:
        M_FULL[MODEL_IDX[mid], BENCH_IDX[bid]] = score

OBSERVED = ~np.isnan(M_FULL)


def matrix_summary():
    observed = int(OBSERVED.sum())
    total = N_MODELS * N_BENCH
    obs_per_model = OBSERVED.sum(axis=1)
    return {
        "n_models": N_MODELS,
        "n_bench": N_BENCH,
        "observed": observed,
        "fill_pct": observed / total * 100,
        "obs_per_model_min": int(obs_per_model.min()),
        "obs_per_model_median": float(np.median(obs_per_model)),
        "obs_per_model_max": int(obs_per_model.max()),
    }


def format_matrix_summary():
    summary = matrix_summary()
    return (
        f"Matrix: {summary['n_models']}x{summary['n_bench']}, "
        f"observed: {summary['observed']}, fill: {summary['fill_pct']:.1f}%\n"
        "Obs/model: "
        f"min={summary['obs_per_model_min']}, "
        f"median={summary['obs_per_model_median']:.0f}, "
        f"max={summary['obs_per_model_max']}"
    )


def print_matrix_summary():
    print(format_matrix_summary())

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

EPS_ZERO = 1e-6  # |actual| threshold below which APE is dropped

# ──────────────────────────────────────────────────────────────────────────────
# Prediction-error API — the only shared helper for MedAE/MedAPE in BenchPress.
#
# Two modes (dispatched by `actual` ndim):
#
#   1) Vector mode — actual is 1D, predicted is 1D (same length).
#
#   2) Matrix mode — actual is 2D (M_actual). predicted is either:
#         • 2D ndarray   (single M_pred shared across all groups), OR
#         • dict[g, 2D]  (different M_pred per group; groups REQUIRED).
#      test_set REQUIRED: list of (i, j) cells to score on.
#
# Aggregation:
#   • 'pool'             — all cells/groups pooled into one batch.
#                          One set of prediction-error metrics.
#   • 'per_group_median' — groups REQUIRED. Per-group → median. Returns *_median.
#
# `groups` parameter:
#   • Vector mode: 1D array of length len(actual), labeling each cell.
#   • Matrix mode: 1D array of length len(test_set), labeling each test cell.
#   • Required when predicted is a dict (matrix mode multi-group).
#
# Return field names depend on aggregation:
#   'pool'             → {n, medae, medape}
#   'per_group_median' → {n_groups, medae_median, medape_median,
#                         per_group: {g -> {n, medae, medape}}}
#
# Errors:
#   • Matrix mode + dict predicted + aggregation='pool' → ValueError.
#     Pooling is ambiguous when M_pred varies per group. Flatten predictions
#     first if a pooled score is desired.
# ──────────────────────────────────────────────────────────────────────────────

_VALID_AGG = ('pool', 'per_group_median')


def compute_prediction_error(actual, predicted, test_set=None, groups=None,
                             aggregation='pool', label=""):
    if aggregation not in _VALID_AGG:
        raise ValueError(f"aggregation must be one of {_VALID_AGG}, "
                         f"got {aggregation!r}")
    if aggregation == 'per_group_median' and groups is None:
        raise ValueError(f"aggregation={aggregation!r} requires groups")

    actual_arr = np.asarray(actual)
    is_dict_pred = isinstance(predicted, dict)

    if actual_arr.ndim == 1:
        # Vector mode
        if is_dict_pred:
            raise ValueError("Vector mode does not accept dict `predicted`.")
        if test_set is not None:
            raise ValueError("Vector mode does not accept `test_set`.")
        predicted_arr = np.asarray(predicted)
        if predicted_arr.shape != actual_arr.shape:
            raise ValueError(
                f"Vector mode shape mismatch: {actual_arr.shape} vs "
                f"{predicted_arr.shape}")
        return _vector_mode(actual_arr.astype(float),
                            predicted_arr.astype(float),
                            groups, aggregation)

    if actual_arr.ndim == 2:
        # Matrix mode
        if test_set is None:
            raise ValueError("Matrix mode requires test_set=[(i,j), ...].")
        if is_dict_pred:
            if groups is None:
                raise ValueError(
                    "Matrix mode + dict `predicted` requires `groups`.")
            if aggregation == 'pool':
                raise ValueError(
                    "Matrix mode + dict `predicted` does not support "
                    "aggregation='pool' when M_pred varies per group. Flatten "
                    "actual/predicted first if a pooled score is desired.")
            return _matrix_mode_dict(actual_arr.astype(float), predicted,
                                     test_set, groups, aggregation)
        predicted_arr = np.asarray(predicted)
        if predicted_arr.ndim != 2 or predicted_arr.shape != actual_arr.shape:
            raise ValueError(
                f"Matrix mode shape mismatch: {actual_arr.shape} vs "
                f"{predicted_arr.shape}")
        return _matrix_mode_single(actual_arr.astype(float),
                                   predicted_arr.astype(float),
                                   test_set, groups, aggregation)

    raise ValueError(f"actual.ndim must be 1 or 2, got {actual_arr.ndim}")


# ─── Pool-mode primitives (one call → one set of metrics) ────────────────────

def _vector_pool_metrics(actual, predicted):
    """Vector-mode pool metrics on 1-D arrays."""
    valid = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return {'n': 0, 'medape': float('nan'), 'medae': float('nan')}

    abs_err = np.abs(predicted - actual)
    nonzero = np.abs(actual) > EPS_ZERO
    ape = np.full(len(actual), np.nan)
    ape[nonzero] = abs_err[nonzero] / np.abs(actual[nonzero])
    ape_valid = ape[~np.isnan(ape)]

    medape  = float(np.median(ape_valid) * 100) if len(ape_valid) > 0 else float('nan')
    medae   = float(np.median(abs_err))

    return {'n': int(len(actual)), 'medape': medape, 'medae': medae}


def _matrix_pool_metrics(M_actual, M_pred, test_set):
    """Matrix-mode pool: gather test cells and compute prediction error."""
    a_list, p_list = [], []
    for i, j in test_set:
        ai, pi = M_actual[i, j], M_pred[i, j]
        if np.isfinite(ai) and np.isfinite(pi):
            a_list.append(float(ai))
            p_list.append(float(pi))
    return _vector_pool_metrics(np.asarray(a_list), np.asarray(p_list))


# ─── Mode dispatchers (handle aggregation) ───────────────────────────────────

def _vector_mode(actual, predicted, groups, aggregation):
    if aggregation == 'pool':
        return _vector_pool_metrics(actual, predicted)
    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(actual):
        raise ValueError(f"groups length ({len(groups_arr)}) must match "
                         f"actual length ({len(actual)})")
    per_group = {}
    for g in _unique_groups(groups_arr):
        mask = np.array([_eq(x, g) for x in groups_arr])
        m = _vector_pool_metrics(actual[mask], predicted[mask])
        per_group[g] = {'n': m['n'], 'medae': m['medae'], 'medape': m['medape']}
    return _aggregate_per_group(per_group, aggregation)


def _matrix_mode_single(M_actual, M_pred, test_set, groups, aggregation):
    if aggregation == 'pool':
        return _matrix_pool_metrics(M_actual, M_pred, test_set)
    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(test_set):
        raise ValueError(f"groups length ({len(groups_arr)}) must match "
                         f"test_set length ({len(test_set)})")
    test_set_list = list(test_set)
    per_group = {}
    for g in _unique_groups(groups_arr):
        ts_g = [test_set_list[k] for k in range(len(groups_arr))
                if _eq(groups_arr[k], g)]
        m = _matrix_pool_metrics(M_actual, M_pred, ts_g)
        per_group[g] = {'n': m['n'], 'medae': m['medae'], 'medape': m['medape']}
    return _aggregate_per_group(per_group, aggregation)


def _matrix_mode_dict(M_actual, preds_dict, test_set, groups, aggregation):
    """Each group has its own M_pred. aggregation is per_group_* (validated)."""
    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(test_set):
        raise ValueError(f"groups length ({len(groups_arr)}) must match "
                         f"test_set length ({len(test_set)})")
    test_set_list = list(test_set)
    per_group = {}
    for g in _unique_groups(groups_arr):
        if g not in preds_dict:
            raise ValueError(
                f"groups contains {g!r} but `predicted` dict has no such key")
        M_pred_g = np.asarray(preds_dict[g], dtype=float)
        if M_pred_g.shape != M_actual.shape:
            raise ValueError(f"preds_dict[{g!r}] shape {M_pred_g.shape} != "
                             f"M_actual shape {M_actual.shape}")
        ts_g = [test_set_list[k] for k in range(len(groups_arr))
                if _eq(groups_arr[k], g)]
        m = _matrix_pool_metrics(M_actual, M_pred_g, ts_g)
        per_group[g] = {'n': m['n'], 'medae': m['medae'], 'medape': m['medape']}
    return _aggregate_per_group(per_group, aggregation)


# ─── Group helpers ───────────────────────────────────────────────────────────

def _eq(a, b):
    """Group-id equality with numpy-scalar tolerance."""
    av = a.item() if hasattr(a, 'item') else a
    bv = b.item() if hasattr(b, 'item') else b
    return av == bv


def _unique_groups(arr):
    """Stable unique preserving first-occurrence order. Returns python scalars."""
    seen, out = set(), []
    for v in arr:
        key = v.item() if hasattr(v, 'item') else v
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _aggregate_per_group(per_group, aggregation):
    medaes  = [v['medae']  for v in per_group.values() if not np.isnan(v['medae'])]
    medapes = [v['medape'] for v in per_group.values() if not np.isnan(v['medape'])]
    return {
        'n_groups': len(per_group),
        'medae_median':  float(np.median(medaes))  if medaes  else float('nan'),
        'medape_median': float(np.median(medapes)) if medapes else float('nan'),
        'per_group':     per_group,
    }


def print_metrics(m, label=""):
    """Pretty-print a pool-mode metrics dict."""
    if m.get('n', 0) == 0:
        print(f"  {label}: NO PREDICTIONS")
        return
    print(f"  {label}: MedAPE={m['medape']:.1f}%  MedAE={m['medae']:.1f}  (n={m['n']})")


def benchmark_mean_median_metric(key="medape", n_seeds=10, n_folds=3, base_seed=42, min_scores=1):
    """Median fold error for predicting each held-out cell by its benchmark mean."""
    vals = []
    for M_train, test_cells in load_folds(
        n_seeds=n_seeds,
        n_folds=n_folds,
        base_seed=base_seed,
        min_scores=min_scores,
    ):
        M_pred = M_FULL.copy()
        for j in range(M_train.shape[1]):
            observed = M_train[:, j]
            observed = observed[np.isfinite(observed)]
            if len(observed) == 0:
                continue
            M_pred[:, j] = float(np.mean(observed))
        metrics = compute_prediction_error(
            M_FULL, M_pred, test_set=test_cells, aggregation="pool"
        )
        if np.isfinite(metrics.get(key, np.nan)):
            vals.append(metrics[key])
    return float(np.median(vals)) if vals else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Ranking-accuracy API — pairwise ordering over groups with a holdout mask.
#
# For each group (e.g. one benchmark leaderboard), enumerate all unordered cell
# pairs. A pair is scored iff:
#   • both actual and predicted values are finite;
#   • at least one of the two cells is marked held out;
#   • the true values are not tied;
#   • abs(true difference) >= margin.
#
# Accuracy is the fraction of scored pairs whose predicted ordering matches the
# true ordering. Predicted ties count as incorrect and are reported separately.
# ──────────────────────────────────────────────────────────────────────────────

def compute_ranking_accuracy(actual, predicted, heldout, groups=None,
                             margin=0.0, aggregation='pool', label=""):
    if aggregation not in _VALID_AGG:
        raise ValueError(f"aggregation must be one of {_VALID_AGG}, "
                         f"got {aggregation!r}")
    if aggregation == 'per_group_median' and groups is None:
        raise ValueError(f"aggregation={aggregation!r} requires groups")

    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    heldout_arr = np.asarray(heldout, dtype=bool)
    if actual_arr.ndim != 1 or predicted_arr.ndim != 1 or heldout_arr.ndim != 1:
        raise ValueError("actual, predicted, and heldout must be 1-D arrays")
    if predicted_arr.shape != actual_arr.shape or heldout_arr.shape != actual_arr.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual_arr.shape}, "
            f"predicted={predicted_arr.shape}, heldout={heldout_arr.shape}")

    margin = float(margin)
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    if groups is None:
        return _ranking_pool_accuracy(actual_arr, predicted_arr, heldout_arr, margin)

    groups_arr = np.asarray(groups)
    if len(groups_arr) != len(actual_arr):
        raise ValueError(f"groups length ({len(groups_arr)}) must match "
                         f"actual length ({len(actual_arr)})")

    per_group = {}
    for g in _unique_groups(groups_arr):
        mask = np.array([_eq(x, g) for x in groups_arr])
        per_group[g] = _ranking_pool_accuracy(
            actual_arr[mask], predicted_arr[mask], heldout_arr[mask], margin)

    if aggregation == 'per_group_median':
        accuracies = [v['accuracy'] for v in per_group.values()
                      if not np.isnan(v['accuracy'])]
        return {
            'n_groups': len(per_group),
            'n_pairs': int(sum(v['n_pairs'] for v in per_group.values())),
            'n_correct': int(sum(v['n_correct'] for v in per_group.values())),
            'n_predicted_ties': int(sum(v['n_predicted_ties']
                                        for v in per_group.values())),
            'accuracy_median': (
                float(np.median(accuracies)) if accuracies else float('nan')
            ),
            'per_group': per_group,
        }

    total_pairs = int(sum(v['n_pairs'] for v in per_group.values()))
    total_correct = int(sum(v['n_correct'] for v in per_group.values()))
    total_ties = int(sum(v['n_predicted_ties'] for v in per_group.values()))
    return {
        'n_pairs': total_pairs,
        'n_correct': total_correct,
        'n_predicted_ties': total_ties,
        'accuracy': (
            float(total_correct / total_pairs) if total_pairs else float('nan')
        ),
    }


def _ranking_pool_accuracy(actual, predicted, heldout, margin):
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted, heldout = actual[valid], predicted[valid], heldout[valid]

    if len(actual) < 2:
        return {
            'n_pairs': 0,
            'n_correct': 0,
            'n_predicted_ties': 0,
            'accuracy': float('nan'),
        }

    true_diffs = actual[:, None] - actual[None, :]
    pred_diffs = predicted[:, None] - predicted[None, :]
    upper = np.triu(np.ones((len(actual), len(actual)), dtype=bool), k=1)
    has_holdout = heldout[:, None] | heldout[None, :]
    comparable = (
        upper
        & has_holdout
        & (true_diffs != 0)
        & (np.abs(true_diffs) >= margin)
    )

    total = int(comparable.sum())
    if total == 0:
        return {
            'n_pairs': 0,
            'n_correct': 0,
            'n_predicted_ties': 0,
            'accuracy': float('nan'),
        }

    true_sign = np.sign(true_diffs[comparable])
    pred_sign = np.sign(pred_diffs[comparable])
    correct = int((true_sign == pred_sign).sum())
    pred_ties = int((pred_sign == 0).sum())
    return {
        'n_pairs': total,
        'n_correct': correct,
        'n_predicted_ties': pred_ties,
        'accuracy': float(correct / total),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  HOLDOUT STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def holdout_per_model(min_scores=1, n_folds=3, seed=42):
    """Strategy B: K-fold CV within each model's scores.
    
    For each model with ≥min_scores observed, split its scores into n_folds disjoint subsets.
    Each fold uses one subset as test. Folds are disjoint within each model.
    """
    rng = np.random.RandomState(seed)
    
    # Pre-compute per-model fold assignments
    model_fold_assignments = []  # model_fold_assignments[i] = shuffled list of benchmark indices
    for i in range(N_MODELS):
        obs_j = list(np.where(OBSERVED[i])[0])
        if len(obs_j) >= min_scores:
            rng.shuffle(obs_j)
            model_fold_assignments.append(obs_j)
        else:
            model_fold_assignments.append([])
    
    folds = []
    for k in range(n_folds):
        M_train = M_FULL.copy()
        test_set = []
        for i in range(N_MODELS):
            obs_j = model_fold_assignments[i]
            if len(obs_j) == 0:
                continue
            fold_size = max(1, len(obs_j) // n_folds)
            start = k * fold_size
            end = start + fold_size if k < n_folds - 1 else len(obs_j)
            if start >= len(obs_j):
                continue
            hidden = obs_j[start:end]
            for j in hidden:
                M_train[i, j] = np.nan
                test_set.append((i, j))
        folds.append((M_train, test_set))
    return folds


def holdout_half_per_model(rng, min_obs=4):
    """Hide half of each model's observed scores.

    For each model with ≥min_obs observed cells, hide n_hide=len(obs)//2
    of them. RNG state is advanced by the caller's RandomState (one
    rng.shuffle per qualifying model) so callers control reproducibility.

    Returns:
        M_train: matrix with hidden cells set to NaN
        test_cells: dict {model_idx: [bench_idx, ...]} of hidden cells
    """
    M_train = M_FULL.copy()
    test_cells = {i: [] for i in range(N_MODELS)}
    for i in range(N_MODELS):
        obs_j = np.where(OBSERVED[i])[0]
        if len(obs_j) < min_obs:
            continue
        rng.shuffle(obs_j)
        n_hide = len(obs_j) // 2
        for j in obs_j[:n_hide]:
            M_train[i, j] = np.nan
            test_cells[i].append(j)
    return M_train, test_cells


def holdout_half_per_benchmark(j, rng, min_test=3):
    """Hide half of one benchmark column j's observed model rows.

    For benchmark column j, take its observed rows, shuffle (via caller's
    rng so RNG state advances identically to inline code), and split into
    test (first n_test) / train (rest). n_test = max(min_test, len(obs)//2).

    Used by §4 error-analysis benchmark H5/H6/H7/H8 ablations. Caller is
    responsible for filtering benchmarks with too few observations.

    Returns:
        test_idx: array of test-row model indices
        train_idx: array of train-row model indices
    """
    obs_rows = np.where(OBSERVED[:, j])[0]
    perm = rng.permutation(len(obs_rows))
    n_test = max(min_test, len(obs_rows) // 2)
    test_idx = obs_rows[perm[:n_test]]
    train_idx = obs_rows[perm[n_test:]]
    return test_idx, train_idx


def observed_benchmarks_by_model(model_limit=None):
    """Observed benchmark indices for each model row, in matrix order."""
    n_models = N_MODELS if model_limit is None else min(N_MODELS, int(model_limit))
    targets = []
    for i in range(N_MODELS):
        if i < n_models:
            targets.append(np.where(OBSERVED[i])[0].tolist())
        else:
            targets.append([])
    return targets


def load_benchmark_allowlist(path, benchmark_ids=BENCH_IDS, label='Benchmark allowlist'):
    """Load a JSON benchmark-id allowlist and return (index set, id list)."""
    if path is None:
        return None, None
    payload = load_json(path)
    ids = payload.get('benchmark_ids') if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{label} {path} must contain a non-empty benchmark-id list")
    unknown = [bid for bid in ids if bid not in benchmark_ids]
    if unknown:
        raise ValueError(f"{label} {path} contains unknown benchmark IDs: {unknown}")
    duplicates = sorted({bid for bid in ids if ids.count(bid) > 1})
    if duplicates:
        raise ValueError(f"{label} {path} contains duplicate benchmark IDs: {duplicates}")
    return {benchmark_ids.index(bid) for bid in ids}, ids


def pack_probe_predictions(predictions):
    """Compact (model, benchmark, true, pred) tuples into JSON-friendly lists."""
    if not predictions:
        return {'i': [], 'j': [], 'true': [], 'pred': []}
    arr = np.asarray(predictions, dtype=object)
    return {
        'i': [int(x) for x in arr[:, 0]],
        'j': [int(x) for x in arr[:, 1]],
        'true': [float(x) for x in arr[:, 2]],
        'pred': [float(x) for x in arr[:, 3]],
    }


def probe_candidate_cache_path(cache_root, step, benchmark_id):
    """Path for one cached greedy probe-candidate evaluation."""
    return os.path.join(
        cache_root,
        f"step_{int(step):03d}",
        f"{safe_token(benchmark_id)}.json.gz",
    )


def probe_mask_from_indices(probe_indices):
    """Boolean benchmark mask for a probe-column set."""
    probe_mask = np.zeros(N_BENCH, dtype=bool)
    for j in probe_indices:
        probe_mask[int(j)] = True
    return probe_mask


def mask_model_to_probe_columns(model_idx, probe_indices, base_matrix=None):
    """Mask one model row so only probe benchmark columns remain visible.

    Used by the all-known-cell probe protocol: every other model remains fully
    observed, while the target model keeps only cells in the probe columns.
    """
    M_train = M_FULL.copy() if base_matrix is None else np.array(base_matrix, copy=True)
    probe_mask = probe_mask_from_indices(probe_indices)
    M_train[int(model_idx), ~probe_mask] = np.nan
    return M_train, probe_mask


PROBE_SCORE_KEY = {
    'medape': 'medape',
    'medae': 'medae',
}


def evaluate_probe_set(probe_indices, predict_fn, metric='medape',
                       target_by_model=None, base_matrix=None):
    """Evaluate an all-known-cell probe set.

    For each target model, only the selected probe columns stay visible in that
    model's row; every other model remains fully observed. Observed probe cells
    are exact (`pred=true`), and unrevealed observed cells are predicted by
    `predict_fn`.
    """
    if metric not in PROBE_SCORE_KEY:
        raise ValueError(f"Unknown probe metric: {metric!r}")

    probe_set = set(int(p) for p in probe_indices)
    if target_by_model is None:
        target_by_model = observed_benchmarks_by_model()

    predictions = []
    for i in range(N_MODELS):
        target_js = target_by_model[i]
        if not target_js:
            continue

        non_probe_targets = [j for j in target_js if j not in probe_set]
        probe_targets = [j for j in target_js if j in probe_set]

        if non_probe_targets:
            M_train, _ = mask_model_to_probe_columns(
                i, probe_set, base_matrix=base_matrix,
            )
            M_pred = predict_fn(M_train)
            for j in non_probe_targets:
                predictions.append((i, j, float(M_FULL[i, j]), float(M_pred[i, j])))

        for j_known in probe_targets:
            true = float(M_FULL[i, j_known])
            predictions.append((i, j_known, true, true))

    actual = np.array([p[2] for p in predictions])
    predicted = np.array([p[3] for p in predictions])
    metrics = compute_prediction_error(actual, predicted, aggregation='pool')
    score_field = PROBE_SCORE_KEY[metric]
    score = float(metrics[score_field]) if np.isfinite(metrics[score_field]) else float('inf')
    return predictions, {
        'medape': metrics['medape'],
        'medae': metrics['medae'],
        'n': metrics['n'],
    }, score


def mask_cells(cell_indices, base_matrix=None):
    """Return a matrix copy with selected (model, benchmark) cells hidden."""
    M_train = M_FULL.copy() if base_matrix is None else np.array(base_matrix, copy=True)
    for i, j in cell_indices:
        M_train[int(i), int(j)] = np.nan
    return M_train


def mask_columns(column_indices, base_matrix=None):
    """Return a matrix copy with entire benchmark columns hidden."""
    M_train = M_FULL.copy() if base_matrix is None else np.array(base_matrix, copy=True)
    cols = [int(j) for j in column_indices]
    if cols:
        M_train[:, cols] = np.nan
    return M_train


def keep_only_benchmark_rows(bench_idx, row_indices, base_matrix=None):
    """Hide a benchmark column, then restore the selected model rows.

    This is the benchmark-side observation-count primitive: the target column is
    unavailable except for the explicitly retained training rows.
    """
    M_train = M_FULL.copy() if base_matrix is None else np.array(base_matrix, copy=True)
    j = int(bench_idx)
    keep_idx = np.asarray(row_indices, dtype=int)
    M_train[:, j] = np.nan
    M_train[keep_idx, j] = M_FULL[keep_idx, j]
    return M_train


def random_global_probe_set(k, seed_idx, base_seed=42):
    """Global random probe-prefix columns for §5.1 random baseline."""
    seed = int(base_seed) + int(seed_idx)
    rng = np.random.RandomState(seed * 100000)
    bench_perm = np.arange(N_BENCH)
    rng.shuffle(bench_perm)
    return set(int(j) for j in bench_perm[:min(int(k), N_BENCH)])


def random_model_keep_k_split(model_idx, k, seed_idx, base_seed=42):
    """Legacy Figure 1 per-model random keep-k split.

    Returns (kept_set, masked_list). The masked list preserves the historical
    shuffled order so old raw-prediction row order remains unchanged.
    """
    obs_j = np.where(OBSERVED[int(model_idx)])[0]
    if len(obs_j) <= int(k):
        return set(), []
    seed = int(base_seed) + int(seed_idx)
    rng = np.random.RandomState(seed * 100000 + int(k) * 1000 + int(model_idx))
    obs_j_perm = obs_j.copy()
    rng.shuffle(obs_j_perm)
    keep = set(obs_j_perm[:int(k)].tolist())
    masked = [int(j) for j in obs_j_perm if int(j) not in keep]
    return keep, masked


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY: Column normalization (used by some baselines in all_methods.py)
# ══════════════════════════════════════════════════════════════════════════════

def col_stats(M):
    """Column means and stds ignoring NaN."""
    col_mean = np.nanmean(M, axis=0)
    col_std  = np.nanstd(M, axis=0)
    col_std[col_std < 1e-8] = 1.0
    return col_mean, col_std

def col_normalize(M):
    """Z-score normalize columns."""
    cm, cs = col_stats(M)
    M_norm = (M - cm) / cs
    M_norm[np.isnan(M)] = np.nan
    return M_norm, cm, cs

def col_denormalize(M_norm, cm, cs):
    return M_norm * cs + cm


def rank2_r2(M, axis):
    """R² of the best rank-2 reconstruction on z-scored, zero-imputed M.

    Pipeline: column-z-score → impute NaN with 0 → SVD → keep top-2 components
    → measure 1 - resid_ss/total_ss along the requested axis.

    axis=0 → per-column (per-benchmark) R², shape (N_BENCH,)
    axis=1 → per-row    (per-model)     R², shape (N_MODELS,)
    """
    M_z, _, _ = col_normalize(M)
    M_imp = np.where(np.isnan(M_z), 0.0, M_z)
    U, s, Vt = np.linalg.svd(M_imp, full_matrices=False)
    M_hat = U[:, :2] @ np.diag(s[:2]) @ Vt[:2, :]
    total_ss = (M_imp ** 2).sum(axis=axis)
    resid_ss = ((M_imp - M_hat) ** 2).sum(axis=axis)
    return 1.0 - resid_ss / np.where(total_ss > 1e-12, total_ss, 1.0)

# ══════════════════════════════════════════════════════════════════════════════
#  SCORE PREDICTOR (transform + z-score + completion wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def make_score_predictor(completion_fn, transform_name, **completion_kwargs):
    """Wrap a completion method with the transform + z-score pipeline.

    Args:
        completion_fn: callable(M_z, **kwargs) -> M_pred_z (operates in z-scored space)
        transform_name: key in TRANSFORMS dict (e.g., 'logit', 'probit', 'none')
        **completion_kwargs: kwargs passed to completion_fn

    Returns:
        predict_fn(M_train) -> M_pred in original scale
    """
    from benchpress.methods.transforms import TRANSFORMS, apply_transform, invert_transform
    to_fn, from_fn, pct_only = TRANSFORMS[transform_name]

    def predict_fn(M_train):
        M_z, obs, is_pct, col_mu, col_std = apply_transform(M_train, to_fn, pct_only)
        M_pred_z = completion_fn(M_z, **completion_kwargs)
        return invert_transform(M_pred_z, M_train, to_fn, from_fn, pct_only, obs, is_pct,
                                col_mu, col_std)
    return predict_fn

FOLDS_DIR = os.path.join(os.path.dirname(__file__), 'evaluation', 'folds')

def _folds_path(n_seeds, n_folds, base_seed, min_scores):
    tag = f"s{n_seeds}_f{n_folds}_bs{base_seed}_ms{min_scores}"
    return os.path.join(FOLDS_DIR, f"folds_{tag}.json")

def save_folds(n_seeds=10, n_folds=3, base_seed=42, min_scores=1):
    """Generate K-fold CV splits and persist test-cell indices to JSON."""
    path = _folds_path(n_seeds, n_folds, base_seed, min_scores)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    records = []
    for idx in range(n_seeds):
        seed = base_seed + idx
        fold_list = holdout_per_model(
            min_scores=min_scores,
            n_folds=n_folds, seed=seed,
        )
        for fold_idx, (_, test_set) in enumerate(fold_list):
            records.append({
                'seed': int(seed),
                'fold': int(fold_idx),
                'test_cells': [[int(i), int(j)] for i, j in test_set],
            })

    write_json(path, {
        'n_seeds': n_seeds, 'n_folds': n_folds,
        'base_seed': base_seed,
        'min_scores': min_scores,
        'n_models': N_MODELS, 'n_bench': N_BENCH,
        'folds': records,
    })
    print(f"Saved {len(records)} folds → {path}")
    return path

def load_folds(n_seeds=10, n_folds=3, base_seed=42, min_scores=1):
    """Load persisted folds. Returns list of (M_train, test_set) like generate_folds().

    If the file doesn't exist, generates and saves first.
    """
    path = _folds_path(n_seeds, n_folds, base_seed, min_scores)
    if not os.path.exists(path):
        print(f"Folds file not found, generating: {path}")
        save_folds(n_seeds, n_folds, base_seed, min_scores)

    data = load_json(path)

    assert data['n_models'] == N_MODELS and data['n_bench'] == N_BENCH, \
        f"Matrix size mismatch: saved {data['n_models']}×{data['n_bench']}, current {N_MODELS}×{N_BENCH}. Re-run save_folds()."

    folds = []
    for rec in data['folds']:
        test_set = [(i, j) for i, j in rec['test_cells']]
        M_train = M_FULL.copy()
        for i, j in test_set:
            M_train[i, j] = np.nan
        folds.append((M_train, test_set))
    return folds
