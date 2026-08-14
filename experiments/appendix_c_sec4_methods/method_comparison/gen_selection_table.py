#!/usr/bin/env python3
"""Generate the Appendix C.2 validation-error leaderboard."""

import glob
import json
import os

import numpy as np

from benchpress.table_utils import format_hyperparameters

SEC4_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'sec4_building_benchpress', 'method_comparison',
)
INNER_SCORES_DIR = os.path.normpath(os.path.join(SEC4_DIR, 'inner_scores'))
MANIFEST_PATH = os.path.normpath(os.path.join(SEC4_DIR, 'manifest.json'))
TOP_N = 15
EXPECTED_SHARDS = 329

TRANSFORM_NAMES = {
    'identity': 'Identity',
    'log': 'Log',
    'logit': 'Logit',
    'asinh': 'Arcsinh',
    'sqrt': 'Square root',
    'probit': 'Probit',
    'quantile': 'Quantile',
}

METHOD_NAMES = {
    'Benchmark Mean': 'Bench-Mean',
    'Model Mean': 'Model-Mean',
    'Bench-KNN': 'Bench-KNN',
    'Model-KNN': 'Model-KNN',
    'BenchReg': 'BenchReg',
    'ModelReg': 'ModelReg',
    'Soft-Impute': 'Soft-Impute',
    'Bias ALS': 'Bias ALS',
    'NMF': 'NMF',
    'PMF': 'PMF',
    'Nuclear Norm': 'Nuclear',
    'MLP': 'MLP',
}


def highlight_benchpress(row, *cells):
    """Highlight the selected BenchPress configuration."""
    hp = row['hp']
    is_selected = (
        row['transform'] == 'logit'
        and row['method'] == 'Bias ALS'
        and hp.get('lam') == 0.1
        and hp.get('rank') == 2
    )
    if not is_selected:
        return cells
    return tuple(rf'\cBP\textbf{{{cell}}}' for cell in cells)


def load_validation_rows():
    """Aggregate the main sweep's configurations on inner-validation cells."""
    paths = sorted(glob.glob(os.path.join(INNER_SCORES_DIR, '*.npz')))
    if len(paths) != EXPECTED_SHARDS:
        raise RuntimeError(
            f'expected {EXPECTED_SHARDS} validation shards in '
            f'{INNER_SCORES_DIR}, found {len(paths)}')

    with open(MANIFEST_PATH) as file:
        completed = json.load(file)['completed']
    manifest_rows = {row['shard_id']: row for row in completed}
    if len(manifest_rows) != EXPECTED_SHARDS:
        raise RuntimeError(
            f'expected {EXPECTED_SHARDS} completed configurations in '
            f'{MANIFEST_PATH}, found {len(manifest_rows)}')

    rows = []
    seen_ids = set()
    expected_columns = [
        'outer_idx', 'inner_idx', 'medape', 'medae', 'coverage', 'n_cells',
    ]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            columns = data['columns'].tolist()
            metrics = data['rows'].astype(float)
            metadata = json.loads(str(data['metadata_json']))
        if columns != expected_columns:
            raise RuntimeError(f'unexpected columns in {path}: {columns}')
        shard_id = metadata['shard_id']
        if shard_id not in manifest_rows:
            raise RuntimeError(
                f'{path} is not part of the main 329-configuration sweep')
        manifest_row = manifest_rows[shard_id]
        for key in ('transform', 'method', 'hp'):
            if metadata[key] != manifest_row[key]:
                raise RuntimeError(
                    f'{key} mismatch for {shard_id}: '
                    f"{metadata[key]!r} != {manifest_row[key]!r}")
        seen_ids.add(shard_id)

        medape = metrics[:, columns.index('medape')]
        medae = metrics[:, columns.index('medae')]
        coverage = metrics[:, columns.index('coverage')]
        n_cells = metrics[:, columns.index('n_cells')]
        total_cells = float(np.sum(n_cells))
        rows.append({
            'transform': metadata['transform'],
            'method': metadata['method'],
            'hp': metadata['hp'],
            'medape': float(np.nanmedian(medape)),
            'medae': float(np.nanmedian(medae)),
            'coverage': (
                float(np.sum(coverage * n_cells) / total_cells)
                if total_cells else 0.0
            ),
        })
    missing_ids = set(manifest_rows) - seen_ids
    if missing_ids:
        raise RuntimeError(
            f'missing validation shards for {len(missing_ids)} main-sweep '
            f'configurations, e.g. {sorted(missing_ids)[:5]}')
    return rows


def gen_table():
    """Render Table 13 in the same layout as the test-error leaderboard."""
    rows = load_validation_rows()
    finite_medape = [row for row in rows if np.isfinite(row['medape'])]
    finite_medae = [row for row in rows if np.isfinite(row['medae'])]
    top_medape = sorted(finite_medape, key=lambda row: row['medape'])[:TOP_N]
    top_medae = sorted(finite_medae, key=lambda row: row['medae'])[:TOP_N]

    lines = [
        r'\begin{tabular}{@{}rlllr@{\hspace{12pt}}rlllr@{}}',
        r'\toprule',
        r'\multicolumn{5}{c}{\textbf{Validation MedAPE (\%) $\downarrow$}} &',
        r'\multicolumn{5}{c}{\textbf{Validation MedAE $\downarrow$}} \\',
        r'\cmidrule(r){1-5} \cmidrule(l){6-10}',
        r'\# & Transform & Method & Hyperparameter & Value &',
        r'\# & Transform & Method & Hyperparameter & Value \\',
        r'\midrule',
    ]

    for rank, (medape_row, medae_row) in enumerate(
            zip(top_medape, top_medae), 1):
        left = [
            TRANSFORM_NAMES.get(
                medape_row['transform'], medape_row['transform']),
            METHOD_NAMES.get(medape_row['method'], medape_row['method']),
            format_hyperparameters(medape_row['hp']),
            f"{medape_row['medape']:.1f} "
            f"({medape_row['coverage']:.0%})".replace('%', r'\%'),
        ]
        right = [
            TRANSFORM_NAMES.get(
                medae_row['transform'], medae_row['transform']),
            METHOD_NAMES.get(medae_row['method'], medae_row['method']),
            format_hyperparameters(medae_row['hp']),
            f"{medae_row['medae']:.2f} "
            f"({medae_row['coverage']:.0%})".replace('%', r'\%'),
        ]
        left = highlight_benchpress(medape_row, *left)
        right = highlight_benchpress(medae_row, *right)
        lines.append(
            f'{rank} & {" & ".join(left)} & '
            f'{rank} & {" & ".join(right)} \\\\')

    lines.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(lines)


if __name__ == '__main__':
    print(gen_table())
