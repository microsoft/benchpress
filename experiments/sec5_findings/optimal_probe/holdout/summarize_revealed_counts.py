"""Compare cached predictors by revealed count on fixed models and test cells."""

import argparse
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import compute_prediction_error
from benchpress.io_utils import load_json, write_json_atomic

PREDICTORS = ('benchpress', 'logit_offsets_only', 'logit_model_mean')


def summarize_revealed_counts(raw_cells, counts):
    """Select earliest prefixes at exact counts; score their common hidden cells."""
    units = defaultdict(dict)
    for row in raw_cells:
        unit = (row['method'], row['model_id'], int(row['k']))
        benchmark = row['benchmark_id']
        if benchmark in units[unit]:
            raise ValueError(f'Duplicate cached cell: {unit}, {benchmark}')
        if benchmark in row['probe_set']:
            raise ValueError(f'Revealed cell included as a test target: {unit}')
        if len(row['revealed_probe_ids']) != row['n_revealed_probe_cells']:
            raise ValueError(f'Inconsistent revealed count: {unit}')
        if not all(row[f'{name}_pred'] is not None and
                   np.isfinite(row[f'{name}_pred']) for name in PREDICTORS):
            raise ValueError(f'Missing predictor output: {unit}, {benchmark}')
        units[unit][benchmark] = row

    selected = defaultdict(dict)
    for (ordering, model, k), cells in sorted(units.items()):
        observed_counts = {r['n_revealed_probe_cells'] for r in cells.values()}
        if len(observed_counts) != 1:
            raise ValueError(f'Inconsistent count within target: {ordering}, {model}, {k}')
        count = observed_counts.pop()
        if count in counts:
            selected[(ordering, model)].setdefault(count, (k, cells))

    groups = []
    for ordering in sorted({key[0] for key in units}):
        retained = []
        excluded = []
        rows_by_count = {count: [] for count in counts}
        per_model = {count: [] for count in counts}
        for (target_ordering, model), states in sorted(selected.items()):
            if target_ordering != ordering:
                continue
            if not all(count in states for count in counts):
                excluded.append(model)
                continue
            common = set.intersection(*(set(states[count][1]) for count in counts))
            if not common:
                raise ValueError(f'No common hidden cells: {ordering}, {model}')
            if common != set(states[counts[-1]][1]):
                raise ValueError(f'Test cells are not nested: {ordering}, {model}')
            benchmarks = sorted(common)
            truth = [states[counts[-1]][1][b]['true'] for b in benchmarks]
            retained.append({
                'model_id': model,
                'source_k_by_count': {str(c): states[c][0] for c in counts},
                'test_benchmark_ids': benchmarks,
                'n_test_cells': len(benchmarks),
            })
            for count in counts:
                rows = [states[count][1][b] for b in benchmarks]
                if [r['true'] for r in rows] != truth:
                    raise ValueError(f'Test labels changed: {ordering}, {model}')
                rows_by_count[count].extend(rows)
                metrics = {
                    name: compute_prediction_error(
                        np.asarray(truth, dtype=float),
                        np.asarray([r[f'{name}_pred'] for r in rows], dtype=float),
                        aggregation='pool',
                    ) for name in PREDICTORS
                }
                per_model[count].append({
                    'model_id': model,
                    **{name: float(metrics[name]['medae']) for name in PREDICTORS},
                })
        summaries = []
        for count in counts:
            rows = rows_by_count[count]
            if not rows:
                raise ValueError(f'Empty matched cohort: {ordering}')
            metrics = {
                name: compute_prediction_error(
                    np.asarray([r['true'] for r in rows], dtype=float),
                    np.asarray([r[f'{name}_pred'] for r in rows], dtype=float),
                    aggregation='pool',
                ) for name in PREDICTORS
            }
            summaries.append({
                'revealed_scores': count,
                'n_models': len(retained),
                'n_test_cells': len(rows),
                'metrics': {name: {
                    'medae': float(metrics[name]['medae']),
                    'medape': float(metrics[name]['medape']),
                    'n': int(metrics[name]['n']),
                } for name in PREDICTORS},
                'benchpress_model_wins': {name: sum(
                    row['benchpress'] < row[name] for row in per_model[count]
                ) for name in PREDICTORS[1:]},
                'per_model_medae': per_model[count],
            })
        groups.append({
            'ordering': ordering,
            'retained_models': retained,
            'excluded_model_ids': excluded,
            'summary': summaries,
        })
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=os.path.join(
        HERE, 'results', 'heldout_baselines_s42.json.gz'))
    parser.add_argument('--max-revealed', type=int, default=3)
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.max_revealed < 1:
        parser.error('--max-revealed must be positive')
    counts = tuple(range(1, args.max_revealed + 1))
    if args.out is None:
        suffix = '' if args.max_revealed == 3 else f'_through{args.max_revealed}'
        args.out = os.path.join(HERE, 'results', f'heldout_revealed_counts_s42{suffix}.json')
    source = load_json(args.input)
    with open(args.input, 'rb') as handle:
        source_hash = hashlib.sha256(handle.read()).hexdigest()
    payload = {
        'protocol': 'heldout_fixed_cells_revealed_counts_v1',
        'revealed_counts': list(counts),
        'selection': (
            'Within each ordering, retain models with cached predictions at every '
            f'exact count {", ".join(map(str, counts))}. Use the earliest requested-probe prefix reaching '
            'each count. Evaluate every predictor at every count on the intersection '
            'of those hidden test cells, once per model-benchmark pair. Cohort '
            'selection uses availability only, never prediction errors.'
        ),
        'source': os.path.relpath(args.input, HERE),
        'source_sha256': source_hash,
        'source_manifest': source['manifest'],
        'groups': summarize_revealed_counts(source['raw_cells'], counts),
    }
    write_json_atomic(args.out, payload, indent=2)
    for group in payload['groups']:
        for row in group['summary']:
            print(group['ordering'], row['revealed_scores'], row['n_models'],
                  row['n_test_cells'],
                  {name: round(row['metrics'][name]['medae'], 4) for name in PREDICTORS},
                  row['benchpress_model_wins'])


if __name__ == '__main__':
    main()
