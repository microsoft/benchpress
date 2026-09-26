#!/usr/bin/env python3
"""Score held-out model baselines on stored probe-validation cells."""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_METRICS,
    BENCH_NAMES,
    M_FULL,
    MODEL_IDS,
    MODEL_IDX,
    MODEL_NAMES,
    N_BENCH,
    N_MODELS,
    OBSERVED,
    base_matrix_for_isolated_probe_target,
    benchmark_metric_identity_sha256,
    compute_prediction_error,
    matrix_identity_sha256,
)
from benchpress.io_utils import load_json, write_json_atomic
from benchpress.methods.predictors import predict_logit_model_mean_scores
from benchpress.methods.transforms import _is_pct_bench


PROTOCOL = 'heldout_logit_model_mean_baseline_v1'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DEFAULT_OUT = os.path.join(RESULTS_DIR, 'heldout_logit_model_mean_baselines.json.gz')

IS_PCT_BENCH = np.array([
    _is_pct_bench(j, M_FULL, metric=BENCH_METRICS, benchmark_ids=BENCH_IDS)
    for j in range(N_BENCH)
], dtype=bool)


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def _manifest():
    return {
        'git_commit': _git_commit(),
        'matrix_identity_sha256': matrix_identity_sha256(M_FULL),
        'benchmark_metric_identity_sha256': benchmark_metric_identity_sha256(
            BENCH_METRICS, BENCH_IDS,
        ),
        'matrix_shape': [int(N_MODELS), int(N_BENCH)],
        'n_observed': int(OBSERVED.sum()),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _metric_value(metrics, key):
    value = metrics.get(key)
    return _finite_or_none(value)


def _unpack_predictions(payload):
    predictions = payload.get('predictions')
    if not isinstance(predictions, dict):
        raise ValueError("validation_non_probe.predictions must be a packed dict")
    required = ['i', 'j', 'true', 'pred']
    missing = [key for key in required if key not in predictions]
    if missing:
        raise ValueError(f"prediction payload missing keys: {missing}")
    n = len(predictions['i'])
    for key in required:
        if len(predictions[key]) != n:
            raise ValueError(
                f"prediction payload length mismatch for {key}: "
                f"{len(predictions[key])} != {n}"
            )
    rows = []
    for idx in range(n):
        rows.append({
            'i': int(predictions['i'][idx]),
            'j': int(predictions['j'][idx]),
            'true': float(predictions['true'][idx]),
            'pred': float(predictions['pred'][idx]),
        })
    return rows


def _method_label(config):
    candidate_source = config.get('candidate_source')
    if candidate_source == 'all':
        return 'cost_unaware_greedy'
    if candidate_source == 'allowlist':
        return 'cost_aware_greedy'
    return str(candidate_source or 'unknown_greedy')


def _as_indices(ids, index, kind):
    unknown = [item for item in ids if item not in index]
    if unknown:
        raise ValueError(f"Unknown {kind} ids in result file: {unknown[:10]}")
    return [int(index[item]) for item in ids]


def _prediction_metrics(records, pred_key, pct_only=False):
    selected = [
        record for record in records
        if (not pct_only) or bool(record['is_percentage_benchmark'])
    ]
    actual = np.array([record['true'] for record in selected], dtype=float)
    predicted = np.array([
        np.nan if record[pred_key] is None else record[pred_key]
        for record in selected
    ], dtype=float)
    metrics = compute_prediction_error(actual, predicted, aggregation='pool')

    valid = np.isfinite(actual) & np.isfinite(predicted)
    abs_err = np.abs(predicted[valid] - actual[valid])
    if len(abs_err) == 0:
        p90 = None
        frac_gt_10 = None
    else:
        p90 = float(np.percentile(abs_err, 90))
        frac_gt_10 = float(np.mean(abs_err > 10.0))
    return {
        'n': int(metrics['n']),
        'medae': _metric_value(metrics, 'medae'),
        'medape': _metric_value(metrics, 'medape'),
        'p90_abs_error': p90,
        'frac_abs_error_gt_10': frac_gt_10,
    }


def _per_target_metrics(records):
    by_target = defaultdict(list)
    for record in records:
        by_target[record['model_id']].append(record)

    per_target = []
    n_bp_better = 0
    n_compared = 0
    for model_id in sorted(by_target):
        target_records = by_target[model_id]
        true = np.array([record['true'] for record in target_records], dtype=float)
        bp_pred = np.array([
            np.nan if record['benchpress_pred'] is None else record['benchpress_pred']
            for record in target_records
        ], dtype=float)
        mm_pred = np.array([
            np.nan if record['logit_model_mean_pred'] is None
            else record['logit_model_mean_pred']
            for record in target_records
        ], dtype=float)
        bp_metrics = compute_prediction_error(true, bp_pred, aggregation='pool')
        mm_metrics = compute_prediction_error(true, mm_pred, aggregation='pool')
        bp_medae = _metric_value(bp_metrics, 'medae')
        mm_medae = _metric_value(mm_metrics, 'medae')
        bp_better = None
        if bp_medae is not None and mm_medae is not None:
            bp_better = bool(bp_medae < mm_medae)
            n_compared += 1
            if bp_better:
                n_bp_better += 1
        per_target.append({
            'model_id': model_id,
            'model_name': MODEL_NAMES.get(model_id, model_id),
            'model_index': int(target_records[0]['model_index']),
            'n_cells': int(len(target_records)),
            'n_revealed_probe_cells': int(target_records[0]['n_revealed_probe_cells']),
            'revealed_probe_ids': target_records[0]['revealed_probe_ids'],
            'benchpress_medae': bp_medae,
            'logit_model_mean_medae': mm_medae,
            'benchpress_medae_lt_model_mean': bp_better,
        })
    win_fraction = None if n_compared == 0 else float(n_bp_better / n_compared)
    return {
        'n_targets': int(len(by_target)),
        'n_targets_compared': int(n_compared),
        'benchpress_medae_lt_model_mean_fraction': win_fraction,
        'per_target': per_target,
    }


def _summarize_step(records):
    target_payload = _per_target_metrics(records)
    return {
        'benchpress': {
            **_prediction_metrics(records, 'benchpress_pred', pct_only=False),
            'percentage_benchmarks': _prediction_metrics(
                records, 'benchpress_pred', pct_only=True,
            ),
        },
        'logit_model_mean': {
            **_prediction_metrics(records, 'logit_model_mean_pred', pct_only=False),
            'percentage_benchmarks': _prediction_metrics(
                records, 'logit_model_mean_pred', pct_only=True,
            ),
        },
        'paired_target_comparison': {
            'n_targets': target_payload['n_targets'],
            'n_targets_compared': target_payload['n_targets_compared'],
            'benchpress_medae_lt_model_mean_fraction': (
                target_payload['benchpress_medae_lt_model_mean_fraction']
            ),
        },
        'per_target': target_payload['per_target'],
    }


def _score_validation_file(path):
    data = load_json(path)
    config = data.get('config', {})
    result_bench_ids = config.get('bench_ids')
    if result_bench_ids is not None and result_bench_ids != BENCH_IDS:
        raise ValueError(f"{path}: benchmark ids do not match the loaded matrix")

    train_model_ids = config.get('train_model_ids')
    validation_model_ids = config.get('validation_model_ids')
    if not train_model_ids or not validation_model_ids:
        raise ValueError(f"{path}: missing train/validation model ids")
    train_model_indices = _as_indices(train_model_ids, MODEL_IDX, 'model')

    source_file = os.path.basename(path)
    source_abspath = os.path.abspath(path)
    split_seed = int(config.get('seed'))
    train_fraction = float(config.get('train_fraction'))
    method = _method_label(config)

    raw_cells = []
    summaries = []
    for step in data.get('trajectory', []):
        k = int(step.get('step', len(step.get('probe_set', []))))
        probe_ids = list(step['probe_set'])
        probe_indices = _as_indices(
            probe_ids,
            {bench_id: idx for idx, bench_id in enumerate(BENCH_IDS)},
            'benchmark',
        )
        prediction_rows = _unpack_predictions(step['validation_non_probe'])
        rows_by_model = defaultdict(list)
        for row in prediction_rows:
            rows_by_model[int(row['i'])].append(row)

        step_records = []
        for model_index in sorted(rows_by_model):
            revealed_probe_indices = [
                int(j) for j in probe_indices
                if OBSERVED[int(model_index), int(j)]
            ]
            revealed_probe_ids = [BENCH_IDS[j] for j in revealed_probe_indices]
            M_train = base_matrix_for_isolated_probe_target(
                model_index,
                probe_indices,
                train_model_indices,
            )
            M_model_mean = predict_logit_model_mean_scores(
                M_train, metric=BENCH_METRICS, benchmark_ids=BENCH_IDS,
            )
            model_id = MODEL_IDS[int(model_index)]
            for row in rows_by_model[model_index]:
                bench_index = int(row['j'])
                bench_id = BENCH_IDS[bench_index]
                true = float(row['true'])
                bp_pred = _finite_or_none(row['pred'])
                mm_pred = _finite_or_none(M_model_mean[int(model_index), bench_index])
                cell = {
                    'source_file': source_file,
                    'source_path': source_abspath,
                    'method': method,
                    'split_seed': split_seed,
                    'train_fraction': train_fraction,
                    'k': k,
                    'probe_set': probe_ids,
                    'model_index': int(model_index),
                    'model_id': model_id,
                    'model_name': MODEL_NAMES.get(model_id, model_id),
                    'benchmark_index': bench_index,
                    'benchmark_id': bench_id,
                    'benchmark_name': BENCH_NAMES.get(bench_id, bench_id),
                    'is_percentage_benchmark': bool(IS_PCT_BENCH[bench_index]),
                    'true': true,
                    'benchpress_pred': bp_pred,
                    'logit_model_mean_pred': mm_pred,
                    'benchpress_abs_error': (
                        None if bp_pred is None else abs(bp_pred - true)
                    ),
                    'logit_model_mean_abs_error': (
                        None if mm_pred is None else abs(mm_pred - true)
                    ),
                    'n_revealed_probe_cells': int(len(revealed_probe_indices)),
                    'revealed_probe_ids': revealed_probe_ids,
                }
                raw_cells.append(cell)
                step_records.append(cell)

        summary = {
            'source_file': source_file,
            'source_path': source_abspath,
            'method': method,
            'split_seed': split_seed,
            'train_fraction': train_fraction,
            'k': k,
            'probe_set': probe_ids,
            'n_cells_in_prediction_payload': int(len(prediction_rows)),
            **_summarize_step(step_records),
        }
        summaries.append(summary)
    return {
        'source': {
            'path': source_abspath,
            'file': source_file,
            'method': method,
            'split_seed': split_seed,
            'train_fraction': train_fraction,
            'n_train_models': int(config.get('n_train_models', len(train_model_ids))),
            'n_validation_models': int(
                config.get('n_validation_models', len(validation_model_ids))
            ),
            'validation_model_ids': validation_model_ids,
            'source_manifest': data.get('manifest', {}),
        },
        'raw_cells': raw_cells,
        'summaries': summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'validation_results',
        nargs='+',
        help='One or more model_split_validation*.json[.gz] files.',
    )
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--indent', type=int, default=None)
    args = parser.parse_args()

    all_raw_cells = []
    all_summaries = []
    sources = []
    for path in args.validation_results:
        result = _score_validation_file(path)
        sources.append(result['source'])
        all_raw_cells.extend(result['raw_cells'])
        all_summaries.extend(result['summaries'])

    payload = {
        'config': {
            'protocol': PROTOCOL,
            'input_validation_results': [os.path.abspath(p) for p in args.validation_results],
            'prediction_engine': (
                'BenchPress from stored validation_non_probe predictions; '
                'baseline from predict_logit_model_mean_scores on the same '
                'isolated held-out target matrix and scored on exactly the '
                'stored validation_non_probe cells.'
            ),
            'metrics': {
                'medae': 'median absolute error in score points; lower is better',
                'medape': 'median absolute percentage error; lower is better',
                'p90_abs_error': '90th percentile absolute error in score points',
                'frac_abs_error_gt_10': (
                    'fraction of scored cells with absolute error above 10 points'
                ),
                'percentage_benchmarks': (
                    'same-cell subset where benchpress.methods.transforms._is_pct_bench '
                    'is true under the loaded benchmark metric metadata'
                ),
            },
        },
        'manifest': _manifest(),
        'sources': sources,
        'summary': all_summaries,
        'raw_cells': all_raw_cells,
    }
    write_json_atomic(args.out, payload, indent=args.indent)
    print(f"Saved -> {args.out}")


if __name__ == '__main__':
    main()
