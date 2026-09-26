#!/usr/bin/env python
"""Compute inter-source score variation from stored candidate values."""

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'results.json')
PAPER_BENCHPRESS_MEDAE = 4.6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-json', default=None,
                        help='Score-matrix JSON. Defaults to BENCHPRESS_DATA or the package data file.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--status', default='verified,verified_third_party')
    return parser.parse_args()


def data_json_path(path_arg):
    if path_arg:
        return path_arg
    env_path = os.environ.get('BENCHPRESS_DATA')
    if env_path:
        return env_path
    return os.path.join(ROOT, 'benchpress', 'data', 'llm_benchmark_data.json')


def load_harness(data_json):
    os.environ['BENCHPRESS_DATA'] = data_json
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from benchpress import evaluation_harness as eh
    from benchpress.io_utils import write_json
    return eh, write_json


def current_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def as_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def distinct_values(values):
    seen = []
    for value in values:
        if value is None:
            continue
        if not any(abs(value - old) <= 1e-12 for old in seen):
            seen.append(value)
    return seen


def benchmark_metric_type(bench_meta):
    if not isinstance(bench_meta, dict):
        return 'unknown'
    canonical = bench_meta.get('canonical_setting') or {}
    if canonical.get('metric_type') is not None:
        return str(canonical.get('metric_type'))
    metric = bench_meta.get('metric')
    if isinstance(metric, dict) and metric.get('type') is not None:
        return str(metric.get('type'))
    if metric is not None:
        text = str(metric).lower()
        if '%' in text or 'percent' in text:
            return 'pct'
        if 'dollar' in text or '$' in text:
            return 'dollars'
        if 'elo' in text:
            return 'elo'
        return text
    return 'unknown'


def add_values(summary, cell_record):
    summary['n_cells'] += 1
    summary['n_alternatives'] += cell_record['n_alternatives']
    if cell_record['chosen_vs_alternative_cell_median'] is not None:
        summary['chosen_cell_medians'].append(cell_record['chosen_vs_alternative_cell_median'])
    if cell_record['alternative_pair_cell_median'] is not None:
        summary['alternative_cell_medians'].append(cell_record['alternative_pair_cell_median'])
    summary['chosen_diffs'].extend(cell_record['chosen_vs_alternative_diffs'])
    summary['alternative_diffs'].extend(cell_record['alternative_pair_diffs'])


def summarize_values(summary):
    def med(values):
        return float(np.median(values)) if values else None
    return {
        'n_cells': int(summary['n_cells']),
        'n_alternatives': int(summary['n_alternatives']),
        'chosen_vs_alternative': {
            'n_pairs': len(summary['chosen_diffs']),
            'median_cell_median_abs_diff': med(summary['chosen_cell_medians']),
            'median_pooled_abs_diff': med(summary['chosen_diffs']),
            'ratio_to_benchpress_medae_4p6': (
                med(summary['chosen_cell_medians']) / PAPER_BENCHPRESS_MEDAE
                if med(summary['chosen_cell_medians']) is not None else None
            ),
        },
        'alternative_vs_alternative': {
            'n_pairs': len(summary['alternative_diffs']),
            'median_cell_median_abs_diff': med(summary['alternative_cell_medians']),
            'median_pooled_abs_diff': med(summary['alternative_diffs']),
            'ratio_to_benchpress_medae_4p6': (
                med(summary['alternative_cell_medians']) / PAPER_BENCHPRESS_MEDAE
                if med(summary['alternative_cell_medians']) is not None else None
            ),
        },
    }


def empty_summary():
    return {
        'n_cells': 0,
        'n_alternatives': 0,
        'chosen_cell_medians': [],
        'alternative_cell_medians': [],
        'chosen_diffs': [],
        'alternative_diffs': [],
    }


def build_cell_record(score_row, benchmark_meta, in_final_matrix):
    chosen = as_float(score_row.get('score'))
    alternatives = []
    for candidate in score_row.get('candidates') or []:
        if isinstance(candidate, dict):
            value = as_float(candidate.get('score'))
        else:
            value = as_float(candidate)
        if value is not None:
            alternatives.append(value)
    if chosen is None or not alternatives:
        return None
    all_distinct = distinct_values([chosen] + alternatives)
    if len(all_distinct) < 2:
        return None
    distinct_alternatives = distinct_values(alternatives)
    chosen_diffs = [abs(chosen - value) for value in distinct_alternatives
                    if abs(chosen - value) > 1e-12]
    alternative_diffs = [abs(a - b) for a, b in itertools.combinations(distinct_alternatives, 2)
                         if abs(a - b) > 1e-12]
    if not chosen_diffs and not alternative_diffs:
        return None
    bench_id = score_row.get('benchmark_id')
    return {
        'model_id': score_row.get('model_id'),
        'benchmark_id': bench_id,
        'benchmark_category': benchmark_meta.get(bench_id, {}).get('category', 'unknown'),
        'benchmark_metric_type': benchmark_metric_type(benchmark_meta.get(bench_id, {})),
        'source_type': score_row.get('source_type', 'unknown'),
        'audit_status': score_row.get('audit_status'),
        'in_final_matrix': bool(in_final_matrix),
        'chosen_score': chosen,
        'n_alternatives': len(alternatives),
        'n_distinct_values_including_chosen': len(all_distinct),
        'chosen_vs_alternative_diffs': chosen_diffs,
        'chosen_vs_alternative_cell_median': (
            float(np.median(chosen_diffs)) if chosen_diffs else None
        ),
        'alternative_pair_diffs': alternative_diffs,
        'alternative_pair_cell_median': (
            float(np.median(alternative_diffs)) if alternative_diffs else None
        ),
    }


def summarize_records(records):
    overall = empty_summary()
    by_metric_type = defaultdict(empty_summary)
    by_category = defaultdict(empty_summary)
    for record in records:
        add_values(overall, record)
        add_values(by_metric_type[record['benchmark_metric_type']], record)
        add_values(by_category[record['benchmark_category']], record)
    category_rows = {
        key: summarize_values(value)
        for key, value in sorted(by_category.items())
        if value['n_cells'] >= 5
    }
    return {
        'overall': summarize_values(overall),
        'by_metric_type': {
            key: summarize_values(value)
            for key, value in sorted(by_metric_type.items())
        },
        'by_category_min_5_cells': category_rows,
    }


def main():
    args = parse_args()
    path = data_json_path(args.data_json)
    eh, write_json = load_harness(path)
    with open(path) as f:
        data = json.load(f)
    statuses = {part.strip() for part in args.status.split(',') if part.strip()}
    benchmark_meta = {row['id']: row for row in data['benchmarks']}
    final_cells = set()
    for model_id in eh.MODEL_IDS:
        for bench_id in eh.BENCH_IDS:
            i = eh.MODEL_IDX[model_id]
            j = eh.BENCH_IDX[bench_id]
            if np.isfinite(eh.M_FULL[i, j]):
                final_cells.add((model_id, bench_id))

    all_records = []
    final_records = []
    for row in data['scores']:
        if statuses and row.get('audit_status') not in statuses:
            continue
        key = (row.get('model_id'), row.get('benchmark_id'))
        record = build_cell_record(row, benchmark_meta, key in final_cells)
        if record is None:
            continue
        all_records.append(record)
        if record['in_final_matrix']:
            final_records.append(record)

    result = {
        'protocol': {
            'data_json_file': os.path.basename(path),
            'status_filter': sorted(statuses),
            'definition': (
                'A cell contributes when the primary score plus stored candidates contain at least two distinct numeric values. '
                'Chosen-vs-alternative diffs compare the stored primary score to each distinct alternative value. '
                'Alternative-vs-alternative diffs compare distinct stored alternatives within the same cell.'
            ),
            'benchpress_reference_medae': PAPER_BENCHPRESS_MEDAE,
            'matrix': {
                'n_models': int(eh.N_MODELS),
                'n_benchmarks': int(eh.N_BENCH),
                'n_observed': int(np.isfinite(eh.M_FULL).sum()),
                'matrix_identity_sha256': eh.matrix_identity_sha256(eh.M_FULL),
            },
            'git_commit': current_commit(),
        },
        'all_verified_cells_with_candidates': summarize_records(all_records),
        'final_matrix_cells_with_candidates': summarize_records(final_records),
        'cell_records': {
            'all_verified_count': len(all_records),
            'final_matrix_count': len(final_records),
            'final_matrix_records': final_records,
        },
    }
    write_json(args.output, result, indent=2, sort_keys=True)
    print(f"WROTE {args.output}")


if __name__ == '__main__':
    main()
