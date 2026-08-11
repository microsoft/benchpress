#!/usr/bin/env python
"""Rank sweep under model holdout, with bootstrap confidence intervals.

Reviewer rdwq asks two things about Fig. 3 (held-out MedAPE vs. rank for raw-
and logit-space Soft-Impute):

  * report it on held-out models, ideally the 20% most recent ones, so the
    rank-2 minimum cannot come from copying a near-duplicate sibling row that
    stayed in the training set;
  * report confidence intervals.

Both are answered here by sweeping the same two Soft-Impute variants and the
same rank grid as the paper figure under two cell-selection protocols:

  cell   Canonical per-model folds (the submission protocol). Every model keeps
         two thirds of its observed scores in the training matrix, so a model
         released later than the target is still visible.

  model  Block holdout by release date. The newest `holdout_frac` of models are
         removed from the training matrix entirely. Targets are then predicted
         one at a time: the training matrix holds the older models fully
         observed plus two thirds of that single target's scores, and no other
         held-out model. This is stricter than the deployment setting of
         Sec. 5, where models released before the target are all available.

Confidence intervals come from a cluster bootstrap over models (`cell`) or over
targets (`model`), because held-out cells of one model are strongly dependent.
Every rank is scored on the same resampled clusters, so the script also reports
the paired difference against the best rank; the paired interval is the one
that answers "is rank 2 really better", since marginal intervals across ranks
overlap by construction.

Outputs are written next to this file unless --outdir is given.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Each worker fits small dense SVDs, so a multi-threaded BLAS only adds
# spin-wait contention once several workers run at once. Must precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from benchpress.build_benchmark_matrix import MODELS
from benchpress.evaluation_harness import (
    BENCH_IDS,
    M_FULL,
    MODEL_IDS,
    MODEL_IDX,
    MODEL_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    compute_prediction_error,
    load_folds,
    make_score_predictor,
)
from benchpress.io_utils import write_json_atomic, write_npz_compressed_atomic
from benchpress.methods.completers import complete_soft_impute

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_NAME = "rank_sweep_holdout_models.json"
MD_NAME = "rank_sweep_holdout_models.md"
RAW_NAME = "rank_sweep_holdout_models_raw.npz"

DEFAULT_RANKS = tuple(range(1, 11))
DEFAULT_N_SEEDS = 10
DEFAULT_N_FOLDS = 3
DEFAULT_BASE_SEED = 42
DEFAULT_MIN_SCORES = 1
DEFAULT_HOLDOUT_FRAC = 0.20
DEFAULT_N_BOOT = 2000
DEFAULT_CI = 0.95

METHODS = ("identity_svd", "logit_svd")
METHOD_LABELS = {
    "identity_svd": "Raw-space Soft-Impute",
    "logit_svd": "Logit-space Soft-Impute",
}

MODEL_RELEASE_DATES = {
    m[0]: m[3] for m in MODELS if len(m) > 3 and m[3] and m[0] in MODEL_IDX
}


def complete(M_train, method, rank):
    if method == "identity_svd":
        return complete_soft_impute(M_train, rank=rank)
    if method == "logit_svd":
        return make_score_predictor(
            complete_soft_impute, "logit", rank=rank, normalize=False
        )(M_train)
    raise ValueError(f"unknown method {method!r}")


def newest_models(holdout_frac):
    """The most recently released models, as a fraction of those with dates."""
    dated = sorted(MODEL_RELEASE_DATES.items(), key=lambda kv: (kv[1], kv[0]))
    n_hold = int(round(holdout_frac * len(dated)))
    if n_hold < 1:
        raise ValueError(f"holdout_frac={holdout_frac} selects no model")
    return [mid for mid, _ in dated[-n_hold:]]


def target_folds(model_id, n_folds, seed):
    """Split one model's observed benchmark columns into n_folds test blocks."""
    obs_j = np.where(OBSERVED[MODEL_IDX[model_id]])[0].astype(int)
    rng = np.random.RandomState(seed)
    shuffled = obs_j.copy()
    rng.shuffle(shuffled)
    return [np.sort(block) for block in np.array_split(shuffled, n_folds)]


def cell_units(n_seeds, n_folds, base_seed, min_scores, limit=None):
    """(M_train, test cells) for the canonical protocol, tagged by model.

    Always loads the canonical fold artifact and only subsets it, so a reduced
    run never writes a new fold file into the package.
    """
    folds = load_folds(
        n_seeds=n_seeds, n_folds=n_folds, base_seed=base_seed, min_scores=min_scores
    )
    if limit is not None:
        folds = folds[:limit]
    units = []
    for fold_idx, (M_train, test_set) in enumerate(folds):
        cells = np.array(test_set, dtype=int).reshape(-1, 2)
        units.append((f"fold{fold_idx}", M_train, cells, cells[:, 0]))
    return units


def model_units(holdout_frac, n_seeds, n_folds, base_seed):
    """(M_train, test cells) for the block-holdout protocol, tagged by target."""
    held = newest_models(holdout_frac)
    held_rows = np.array([MODEL_IDX[mid] for mid in held], dtype=int)
    train_rows = np.setdiff1d(np.arange(N_MODELS), held_rows)

    M_base = np.full_like(M_FULL, np.nan, dtype=float)
    M_base[train_rows] = M_FULL[train_rows]
    train_has_column = np.isfinite(M_base).any(axis=0)

    units, dropped = [], 0
    for mid in held:
        row = MODEL_IDX[mid]
        for seed_idx in range(n_seeds):
            for fold_idx, test_j in enumerate(
                target_folds(mid, n_folds, base_seed + seed_idx)
            ):
                keep = test_j[train_has_column[test_j]]
                dropped += len(test_j) - len(keep)
                if len(keep) == 0:
                    continue
                M_train = M_base.copy()
                M_train[row] = M_FULL[row]
                M_train[row, test_j] = np.nan
                cells = np.column_stack([np.full(len(keep), row), keep])
                units.append(
                    (f"{mid}__s{seed_idx}__f{fold_idx}", M_train, cells,
                     np.full(len(keep), row))
                )
    return units, held, train_rows, dropped


def score_unit(args):
    """Predict one unit for every (method, rank); returns flat records."""
    key, M_train, cells, clusters, ranks = args
    out = []
    for method in METHODS:
        for rank in ranks:
            M_pred = complete(M_train, method, rank)
            for (i, j), c in zip(cells, clusters):
                true, pred = M_FULL[i, j], M_pred[i, j]
                if np.isfinite(true) and np.isfinite(pred):
                    out.append((method, int(rank), int(c), int(i), int(j),
                                float(true), float(pred)))
    return key, out


def run_protocol(units, ranks, workers):
    payload = [(k, m, c, cl, ranks) for k, m, c, cl in units]
    records = []
    if workers <= 1:
        for item in payload:
            records.extend(score_unit(item)[1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_unit, item): item[0] for item in payload}
            for n, fut in enumerate(as_completed(futures), 1):
                records.extend(fut.result()[1])
                if n % 25 == 0 or n == len(payload):
                    print(f"    {n}/{len(payload)} units", flush=True)
    return records


def bootstrap(records, ranks, n_boot, ci, seed):
    """Cluster bootstrap over models; paired across ranks within a replicate."""
    methods = sorted({r[0] for r in records})
    by_key = {}
    for method, rank, cluster, _i, _j, true, pred in records:
        by_key.setdefault((method, rank), {}).setdefault(cluster, []).append((true, pred))

    clusters = sorted({r[2] for r in records})
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(clusters), size=(n_boot, len(clusters)))
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    summary = {}
    for method in methods:
        point, boots = {}, {}
        for rank in ranks:
            per_cluster = by_key.get((method, rank), {})
            packed = [np.array(per_cluster.get(c, []), dtype=float).reshape(-1, 2)
                      for c in clusters]
            allc = np.vstack([p for p in packed if len(p)])
            point[rank] = compute_prediction_error(allc[:, 0], allc[:, 1])
            reps = np.empty((n_boot, 2))
            for b in range(n_boot):
                sample = np.vstack([packed[idx] for idx in draws[b] if len(packed[idx])])
                m = compute_prediction_error(sample[:, 0], sample[:, 1])
                reps[b] = (m["medape"], m["medae"])
            boots[rank] = reps

        best = min(ranks, key=lambda r: point[r]["medape"])
        rows = []
        for rank in ranks:
            reps = boots[rank]
            delta = boots[best][:, 0] - reps[:, 0]
            rows.append({
                "rank": int(rank),
                "medape": round(float(point[rank]["medape"]), 4),
                "medape_ci": [round(float(v), 4)
                              for v in np.percentile(reps[:, 0], [lo_q, hi_q])],
                "medae": round(float(point[rank]["medae"]), 4),
                "medae_ci": [round(float(v), 4)
                             for v in np.percentile(reps[:, 1], [lo_q, hi_q])],
                "medape_delta_vs_best": round(float(point[best]["medape"]
                                                    - point[rank]["medape"]), 4),
                "medape_delta_ci": [round(float(v), 4)
                                    for v in np.percentile(delta, [lo_q, hi_q])],
                "delta_excludes_zero": bool(
                    np.percentile(delta, hi_q) < 0 or np.percentile(delta, lo_q) > 0
                ),
            })
        summary[method] = {"best_rank": int(best), "n_clusters": len(clusters),
                           "rows": rows}
    return summary


def markdown(results, ci):
    pct = f"{int(round(100 * ci))}%"
    out = ["# Fig. 3 rank sweep: model holdout and confidence intervals", ""]
    for pkey, block in results["protocols"].items():
        out += [f"## Protocol `{pkey}` ({block['description']})", ""]
        for method, summ in block["summary"].items():
            out += [
                f"**{METHOD_LABELS[method]}** (best rank {summ['best_rank']}; "
                f"{summ['n_clusters']} bootstrap clusters)", "",
                f"| Rank | MedAPE (%) | {pct} CI | MedAPE at best rank minus this rank | {pct} CI |",
                "|---:|---:|:---|---:|:---|",
            ]
            for r in summ["rows"]:
                out.append(
                    f"| {r['rank']} | {r['medape']:.2f} | "
                    f"[{r['medape_ci'][0]:.2f}, {r['medape_ci'][1]:.2f}] | "
                    f"{r['medape_delta_vs_best']:+.2f} | "
                    f"[{r['medape_delta_ci'][0]:+.2f}, {r['medape_delta_ci'][1]:+.2f}] |"
                )
            out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=SCRIPT_DIR)
    p.add_argument("--ranks", default=",".join(str(r) for r in DEFAULT_RANKS))
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--min-scores", type=int, default=DEFAULT_MIN_SCORES)
    p.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--ci", type=float, default=DEFAULT_CI)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--protocols", default="cell,model")
    p.add_argument("--smoke", action="store_true",
                   help="1 seed, ranks 1-3, 100 bootstrap replicates")
    args = p.parse_args()

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    model_seeds, n_boot, cell_fold_limit = args.n_seeds, args.n_boot, None
    if args.smoke:
        ranks, model_seeds, n_boot, cell_fold_limit = [1, 2, 3], 1, 100, 3

    protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    results = {
        "config": {
            "ranks": ranks,
            "cell_fold_seeds": args.n_seeds,
            "cell_fold_limit": cell_fold_limit,
            "model_seeds": model_seeds,
            "n_folds": args.n_folds,
            "base_seed": args.base_seed,
            "holdout_frac": args.holdout_frac,
            "n_boot": n_boot,
            "ci": args.ci,
            "matrix_shape": [int(N_MODELS), int(N_BENCH)],
            "methods": list(METHODS),
        },
        "protocols": {},
    }
    raw = {}

    for pkey in protocols:
        if pkey == "cell":
            units = cell_units(
                args.n_seeds, args.n_folds, args.base_seed, args.min_scores,
                limit=cell_fold_limit,
            )
            meta = {
                "description": "canonical per-model folds, all models in training",
                "n_units": len(units),
            }
        elif pkey == "model":
            units, held, train_rows, dropped = model_units(
                args.holdout_frac, model_seeds, args.n_folds, args.base_seed
            )
            meta = {
                "description": (
                    f"newest {int(round(100 * args.holdout_frac))}% of models held out "
                    "as a block; each target predicted alone"
                ),
                "n_units": len(units),
                "n_held_out_models": len(held),
                "n_train_models": int(len(train_rows)),
                "held_out_models": [MODEL_NAMES[m] for m in held],
                "held_out_release_dates": {m: MODEL_RELEASE_DATES[m] for m in held},
                "test_cells_dropped_no_train_column": int(dropped),
            }
        else:
            raise ValueError(f"unknown protocol {pkey!r}")

        print(f"[{pkey}] {meta['description']}: {len(units)} units", flush=True)
        records = run_protocol(units, ranks, args.workers)
        meta["n_records"] = len(records)
        meta["summary"] = bootstrap(records, ranks, n_boot, args.ci, args.base_seed)
        results["protocols"][pkey] = meta

        arr = np.array(
            [(m, r, c, i, j, t, pd) for m, r, c, i, j, t, pd in records],
            dtype=[("method", "U16"), ("rank", "i4"), ("cluster", "i4"),
                   ("model_idx", "i4"), ("benchmark_idx", "i4"),
                   ("true", "f8"), ("pred", "f8")],
        )
        for name in arr.dtype.names:
            raw[f"{pkey}__{name}"] = arr[name]

    raw["model_ids"] = np.array(MODEL_IDS)
    raw["benchmark_ids"] = np.array(BENCH_IDS)

    os.makedirs(args.outdir, exist_ok=True)
    write_json_atomic(os.path.join(args.outdir, JSON_NAME), results, indent=2)
    write_npz_compressed_atomic(os.path.join(args.outdir, RAW_NAME), **raw)
    md = markdown(results, args.ci)
    with open(os.path.join(args.outdir, MD_NAME), "w") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
