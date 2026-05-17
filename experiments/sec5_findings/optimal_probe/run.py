#!/usr/bin/env python3
"""
Greedy probe-set selection for BenchPress evaluation design.

For a probe set P and a target known cell (i, j):
    * Start from M_FULL (all observed cells visible).
    * For model i, mask every non-probe cell — i.e. the only cells of model i
       left visible are those in P.
    * If j ∈ P, the score is known and its prediction is the true value.
    * Otherwise, run predict_benchpress_scores on that masked matrix and read M_pred[i, j].
    * Record (i, j, true, pred).

The active §5.1 setting evaluates the fixed universe of all known observed
cells at every probe-set size. Within a single probe-set evaluation, predictions
for model i's unrevealed target cells are amortized: one BenchPress call per
model gives predictions for all non-probe target benchmarks at once. Revealed
probe cells contribute zero error.

Greedy criterion is selected by ``--metric`` and computed via
``benchpress.evaluation_harness.compute_prediction_error``:
    * medape (default): pooled MedAPE over the selected target cells.
    * medae           : pooled MedAE over the selected target cells.
The candidate minimising this score is picked each step.

**Raw predictions for every candidate evaluated at every step are saved** so
that score-error summaries (e.g. max-per-bench or MedAE) can be recomputed later
without re-running the experiment.

CLI:
    python run.py                                           # all benchmarks as candidates
    python run.py --candidate-allowlist user_cheap.json      # explicit candidate set
    python run.py --metric medae                            # appendix sensitivity
    python run.py --max-steps 10 --workers 48               # override defaults
    # Resume: re-running the same configuration / --out picks up where it stopped.
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (
    OBSERVED, N_MODELS, N_BENCH,
    BENCH_IDS, BENCH_NAMES, evaluate_probe_set,
    load_benchmark_allowlist, pack_probe_predictions,
    probe_candidate_cache_path,
)
from benchpress.all_methods import predict_benchpress_scores
from benchpress.io_utils import load_json, safe_token, write_json_atomic
from benchpress.shard_utils import short_text_hash

SEED = 42
EVAL_PROTOCOL = 'all_known_probe_cells_zero_error_v1'
np.random.seed(SEED)
random.seed(SEED)

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_STEPS = 10        # greedy selects the first MAX_STEPS probes


SCORE_UNIT = {
    'medape': '%',
    'medae':  '',
}


# ─────────────────────────────────────────────────────────────────────────────
#  Worker pool
# ─────────────────────────────────────────────────────────────────────────────

def _init_worker(seed):
    np.random.seed(seed)
    random.seed(seed)


def _eval_one(args):
    probe_set, cand_j, metric = args
    predictions, metrics, score = evaluate_probe_set(
        probe_set, predict_benchpress_scores, metric=metric,
    )
    return cand_j, predictions, metrics, score


# ─────────────────────────────────────────────────────────────────────────────
#  Main greedy loop
# ─────────────────────────────────────────────────────────────────────────────

def _cache_root(out_path, metric, candidate_source, candidate_allowlist_ids):
    stem = os.path.basename(out_path)
    if stem.endswith('.json.gz'):
        stem = stem[:-len('.json.gz')]
    else:
        stem = os.path.splitext(stem)[0]
    name = (
        f"{safe_token(stem)}__metric-{safe_token(metric)}"
        f"__candidates-{safe_token(candidate_source)}"
        f"__protocol-{safe_token(EVAL_PROTOCOL)}"
    )
    if candidate_allowlist_ids is not None:
        digest = short_text_hash('\n'.join(candidate_allowlist_ids), n=12)
        name += f"__allowlist-{digest}"
    return os.path.join(RESULTS_DIR, '.candidate_cache', name)


def _candidate_record(cand_j, predictions, metrics, score):
    record = {
        'benchmark_id': BENCH_IDS[cand_j],
        'score': float(score) if np.isfinite(score) else float('inf'),
        'medape': float(metrics['medape']) if np.isfinite(metrics['medape']) else None,
        'medae': float(metrics['medae']) if np.isfinite(metrics['medae']) else None,
        'predictions': pack_probe_predictions(predictions),
    }
    return record


def _load_candidate_cache(path, expected_benchmark_id, expected_probe_set):
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    record = payload.get('record', {})
    protocol = payload.get('eval_protocol')
    if protocol != EVAL_PROTOCOL:
        raise RuntimeError(
            f"Candidate cache protocol mismatch in {path}: expected "
            f"{EVAL_PROTOCOL}, found {protocol}. Delete the step cache or use "
            "a different output file."
        )
    if record.get('benchmark_id') != expected_benchmark_id:
        raise RuntimeError(
            f"Candidate cache mismatch in {path}: expected "
            f"{expected_benchmark_id}, found {record.get('benchmark_id')}"
        )
    probe_set = payload.get('probe_set_before_candidate')
    if probe_set != expected_probe_set:
        raise RuntimeError(
            f"Candidate cache mismatch in {path}: expected probe set "
            f"{expected_probe_set}, found {probe_set}. Delete the step cache "
            "or use a different output file."
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS,
                        help=f'Number of greedy steps (default {MAX_STEPS}).')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='Number of parallel worker processes.')
    parser.add_argument('--metric', type=str, default='medape',
                        choices=['medape', 'medae'],
                        help=('Greedy objective: pooled MedAPE or pooled MedAE via '
                              'benchpress.evaluation_harness.compute_prediction_error.'))
    parser.add_argument('--out', type=str, default=None,
                        help='Output JSON filename under results/. '
                             'Defaults to greedy_<metric>.json.')
    parser.add_argument('--candidate-limit', type=int, default=None,
                        help='Debug/smoke-test only: keep the first N candidates.')
    parser.add_argument('--candidate-allowlist', type=str, default=None,
                        help=('Optional JSON file listing the complete benchmark ID '
                              'set allowed as candidate probes.'))
    args = parser.parse_args()

    if args.out is None:
        suffix = '_candidates_allowlist' if args.candidate_allowlist else '_candidates_all'
        args.out = f'greedy_{args.metric}{suffix}.json.gz'

    candidate_allowlist, candidate_allowlist_ids = load_benchmark_allowlist(
        args.candidate_allowlist, label='Candidate allowlist',
    )

    print(f"Matrix: {N_MODELS}×{N_BENCH}, observed={int(OBSERVED.sum())}")

    n_targets = int(OBSERVED.sum())
    if candidate_allowlist is not None:
        candidates = [BENCH_IDS.index(bid) for bid in candidate_allowlist_ids]
        if not candidates:
            raise RuntimeError("Candidate allowlist produced zero candidate benchmarks")
        candidate_source = 'allowlist'
    else:
        candidates = list(range(N_BENCH))
        candidate_source = 'all'
    if args.candidate_limit is not None:
        candidates = candidates[:args.candidate_limit]
    print(f"  Target cells: all observed ({n_targets} cells)")
    if candidate_allowlist is not None:
        print(f"  Candidate allowlist: {args.candidate_allowlist} ({len(candidates)} benchmarks)")
    else:
        print(f"  Candidate set: all benchmarks ({len(candidates)} benchmarks)")
    print(f"  Workers: {args.workers}")

    selected = []
    remaining = list(candidates)
    trajectory = []

    # Resume support: if output exists, pick up from its trajectory.
    # Guard against resuming under a different metric — the score field would
    # then mix two different objectives, silently breaking greedy selection.
    out_path = os.path.join(RESULTS_DIR, args.out)
    if os.path.exists(out_path):
        prev = load_json(out_path)
        prev_config = prev.get('config', {})
        expected_resume = {
            'metric': args.metric,
            'candidate_source': candidate_source,
            'candidate_allowlist_path': (
                os.path.relpath(args.candidate_allowlist, REPO_ROOT)
                if args.candidate_allowlist else None
            ),
            'candidate_allowlist_ids': candidate_allowlist_ids,
            'candidate_limit': args.candidate_limit,
            'n_target_cells': n_targets,
            'n_candidates': len(candidates),
            'eval_protocol': EVAL_PROTOCOL,
        }
        prev_resume = {k: prev_config.get(k) for k in expected_resume}
        if prev_resume != expected_resume:
            raise SystemExit(
                f"Refusing to resume {out_path}: existing config {prev_resume} "
                f"does not match requested config {expected_resume}. Pick a "
                f"different --out or delete the file."
            )
        if prev.get('trajectory'):
            print(f"\nResuming from {out_path} "
                  f"({len(prev['trajectory'])} steps already done).")
            trajectory = prev['trajectory']
            selected = [BENCH_IDS.index(s['added_benchmark']) for s in trajectory]
            remaining = [j for j in candidates if j not in selected]

    t_all = time.time()
    cache_root = _cache_root(
        out_path, args.metric, candidate_source, candidate_allowlist_ids,
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(SEED,),
    ) as pool:
        for step in range(len(selected) + 1, args.max_steps + 1):
            if not remaining:
                break
            print(f"\n--- Step {step}/{args.max_steps} ({len(remaining)} candidates) ---")
            step_t0 = time.time()

            best_score = float('inf')
            best_j = None
            candidate_results = {}
            selected_ids = [BENCH_IDS[j] for j in selected]

            futures = {}
            for cand_j in remaining:
                cache_path = probe_candidate_cache_path(cache_root, step, BENCH_IDS[cand_j])
                cached = _load_candidate_cache(cache_path, BENCH_IDS[cand_j], selected_ids)
                if cached is not None:
                    candidate_results[BENCH_IDS[cand_j]] = cached
                    continue
                future = pool.submit(
                    _eval_one,
                    (selected + [cand_j], cand_j, args.metric),
                )
                futures[future] = cand_j

            for future in as_completed(futures):
                cand_j, predictions, metrics, score = future.result()
                record = _candidate_record(cand_j, predictions, metrics, score)
                write_json_atomic(
                    probe_candidate_cache_path(cache_root, step, BENCH_IDS[cand_j]),
                    {
                        'step': step,
                        'probe_set_before_candidate': selected_ids,
                        'metric': args.metric,
                        'candidate_source': candidate_source,
                        'eval_protocol': EVAL_PROTOCOL,
                        'record': record,
                    },
                )
                candidate_results[BENCH_IDS[cand_j]] = record

            if len(candidate_results) != len(remaining):
                raise RuntimeError(
                    f"Step {step} has {len(candidate_results)} candidate results, "
                    f"expected {len(remaining)}"
                )

            for cand_j in remaining:
                record = candidate_results[BENCH_IDS[cand_j]]
                score = record['score']
                if score < best_score:
                    best_score = score
                    best_j = cand_j
                    best_record = record

            selected.append(best_j)
            remaining.remove(best_j)

            step_record = {
                'step': step,
                'added_benchmark': BENCH_IDS[best_j],
                'added_benchmark_name': BENCH_NAMES.get(BENCH_IDS[best_j], BENCH_IDS[best_j]),
                'score': best_score,
                'medape': best_record['medape'],
                'medae': best_record['medae'],
                'probe_set': [BENCH_IDS[j] for j in selected],
                'elapsed_s': time.time() - step_t0,
                'candidate_results': candidate_results,
            }
            trajectory.append(step_record)
            unit = SCORE_UNIT[args.metric]
            print(f"  → Added {BENCH_IDS[best_j]:30s} score={best_score:.3f}{unit}  "
                  f"[{time.time() - step_t0:.1f}s]")

            output = {
                'config': {
                    'n_models': N_MODELS,
                    'n_bench': N_BENCH,
                    'n_observed': int(OBSERVED.sum()),
                    'n_target_cells': n_targets,
                    'n_candidates': len(candidates),
                    'bench_ids': BENCH_IDS,
                    'eval_scope': 'all_observed_cells',
                    'candidate_source': candidate_source,
                    'candidate_allowlist_path': (
                        os.path.relpath(args.candidate_allowlist, REPO_ROOT)
                        if args.candidate_allowlist else None
                    ),
                    'candidate_allowlist_ids': candidate_allowlist_ids,
                    'candidate_limit': args.candidate_limit,
                    'cell_masking': (
                        'For model i, keep probe cells visible and mask non-probe cells. '
                        'Probe target cells are known and stored with pred=true; '
                        'non-probe target cells are predicted by BenchPress.'
                    ),
                    'eval_protocol': EVAL_PROTOCOL,
                    'metric': args.metric,
                    'prediction_engine': 'predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)',
                    'seed': SEED,
                    'workers': args.workers,
                    'candidate_cache_dir': os.path.relpath(cache_root, SCRIPT_DIR),
                },
                'trajectory': trajectory,
            }
            write_json_atomic(out_path, output, indent=2)

    print(f"\nSaved → {out_path}")
    print(f"Total time: {time.time() - t_all:.1f}s")


if __name__ == '__main__':
    main()
