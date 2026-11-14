#!/usr/bin/env python
"""Method-only variants on the frozen question-level evaluation artifacts."""

import argparse
import json
import os
import time

import numpy as np

from run import (
    BP_LAM, DEFAULT_DATA, DEFAULT_OUT, ESTIMATORS, Context, benchpress_path,
    compute_prediction_error, config_hash, int_list, load_benchpress_manifest,
    load_coresets, load_question_predictions, read_unit, sha256_file,
)
from benchpress.io_utils import write_json_atomic, write_npz_compressed_atomic
from benchpress.evaluation_harness import compute_binary_accuracy
from benchpress.methods.predictors import fit_question_score_variant


METHODS = ['raw', 'group_logit', 'group_raw', 'group_binary',
           'raw_shrink', 'task_mean_ridge', 'task_item_pcr']
RANK = 8
LABELS = {
    'logit': 'BenchPress (logit ALS)',
    'raw': 'Raw ALS',
    'group_logit': 'Logit ALS + subtask offsets',
    'group_raw': 'Raw ALS + subtask offsets',
    'group_binary': 'Bernoulli low rank + subtask offsets',
    'raw_shrink': 'Raw ALS + shrunk subtask residuals',
    'task_mean_ridge': 'Subtask-mean ridge',
    'task_item_pcr': 'Subtask-mean ridge + rank-8 item residuals',
}


def fit(args):
    ctx = Context(args, check_commit=False)
    methods = METHODS if args.methods is None else args.methods
    config = {
        'base_config_hash': ctx.chash, 'code_commit': args.code_commit,
        'methods': methods, 'rank': args.rank, 'lam': BP_LAM,
        'als': {'n_iter': 40, 'n_ensemble': 10, 'base_seed': 42,
                'standardize_columns': True},
        'binary': {'max_iter': 400, 'n_ensemble': 10, 'base_seed': 42,
                   'ftol': 1e-7, 'gtol': 1e-4},
        'task_mean_ridge': {'alpha': BP_LAM, 'rank': None,
                            'n_features': len(ctx.data.column_ids)},
        'task_item_pcr': {'alpha': BP_LAM, 'item_rank': args.rank,
                          'feature_residualization': 'training group means'},
        'fit': 'pooled low-rank fits on allowed answers; '
               'supervised ridge on fully observed training rows',
    }
    chash = config_hash(config)
    root = os.path.join(ctx.out, 'variants', chash)
    manifest_path = os.path.join(root, 'manifest.json')
    manifest = {'config': config, 'config_hash': chash}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f) != manifest:
                raise ValueError('variant manifest mismatch')
    else:
        write_json_atomic(manifest_path, manifest)
    train = 100. * ctx.data.answers[ctx.sel].astype(float)
    for seed in ctx.subset(args.seeds, 'seeds'):
        for budget in ctx.subset(args.budgets, 'budgets'):
            coresets, _ = load_coresets(ctx, seed, budget)
            for selection in ctx.subset(args.selections, 'selections'):
                observed = [q[c] for q, c in zip(ctx.data.col_questions, coresets[selection])]
                hidden = [np.setdiff1d(q, o) for q, o in zip(ctx.data.col_questions, observed)]
                revealed = np.concatenate(observed)
                held = np.full((len(ctx.held), ctx.data.answers.shape[1]), np.nan)
                held[:, revealed] = 100. * ctx.data.answers[np.ix_(ctx.held, revealed)]
                matrix = np.vstack((train, held))
                mask_hash = config_hash({'targets': ctx.held,
                                         'observed': [o.tolist() for o in observed]})
                for method in methods:
                    path = os.path.join(root, method, selection, f's{seed}', f'B{budget}.json')
                    old = read_unit(path, chash)
                    if old is not None:
                        if sha256_file(path[:-5] + '.npz') != old['question_pred_sha256']:
                            raise ValueError(f'corrupt prediction artifact: {path}')
                        print(f'cached {method} {selection} s={seed} B={budget}', flush=True)
                        continue
                    start = time.time()
                    print(f'start {method} {selection} s={seed} B={budget}', flush=True)
                    completed, diagnostics = fit_question_score_variant(
                        matrix, ctx.data.question_column, method, rank=args.rank, lam=BP_LAM)
                    if not np.isfinite(completed).all():
                        raise ValueError(f'{method} returned nonfinite predictions')
                    if not np.array_equal(completed[np.isfinite(matrix)], matrix[np.isfinite(matrix)]):
                        raise ValueError(f'{method} changed observed answers')
                    pred = completed[len(ctx.sel):]
                    artifact = path[:-5] + '.npz'
                    write_npz_compressed_atomic(artifact, pred=pred,
                                                targets=np.asarray(ctx.held),
                                                revealed=revealed)
                    write_json_atomic(path, {
                        'config_hash': chash, 'code_commit': args.code_commit,
                        'method': method, 'selection': selection, 'seed': seed,
                        'budget': budget, 'rank': None if method == 'task_mean_ridge' else args.rank,
                        'lam': BP_LAM,
                        'targets': ctx.held, 'mask_hash': mask_hash,
                        'questions_used': int(len(revealed)),
                        'pred': np.stack([pred[:, h].mean(1) for h in hidden], axis=1).tolist(),
                        'question_pred_sha256': sha256_file(artifact),
                        'seconds': time.time() - start, 'diagnostics': diagnostics,
                    })
                    print(f'done {method} {selection} s={seed} B={budget} '
                          f'seconds={time.time() - start:.1f} config={chash}', flush=True)


def summarize(args):
    ctx = Context(args, check_commit=False)
    root = os.path.join(ctx.out, 'variants', args.variant_hash)
    with open(os.path.join(root, 'manifest.json')) as f:
        manifest = json.load(f)
    config = manifest['config']
    if (config_hash(config) != args.variant_hash
            or manifest['config_hash'] != args.variant_hash
            or config['base_config_hash'] != ctx.chash):
        raise ValueError('variant provenance mismatch')
    with open(os.path.join(ctx.out, 'summary.json')) as f:
        reference = json.load(f)
    bp = load_benchpress_manifest(ctx)
    if reference['config_hash'] != ctx.chash or reference['benchpress_config_hash'] != bp['config_hash']:
        raise ValueError('reference summary provenance mismatch')
    reference_rows = {(r['selection'], r['method'], r['seed'], r['budget']): r
                      for r in reference['per_seed']}
    original_methods = [f'benchpress_r{r}' for r in bp['config']['ranks']] + list(ESTIMATORS)
    selected_methods = config['methods'] if args.methods is None else args.methods
    if set(selected_methods) - set(config['methods']):
        raise ValueError('requested methods are not in the variant manifest')
    methods = original_methods + ['variant_' + m for m in selected_methods]
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    selections = ctx.subset(args.selections, 'selections')
    complete = (set(selected_methods) == set(config['methods'])
                and set(seeds) == set(ctx.config['seeds'])
                and set(budgets) == set(ctx.config['budgets'])
                and set(selections) == set(ctx.config['selections']))
    records, optimizer_records = [], []
    for seed in seeds:
        for budget in budgets:
            coresets, units = load_coresets(ctx, seed, budget)
            for selection in selections:
                observed = [q[c] for q, c in zip(ctx.data.col_questions, coresets[selection])]
                hidden = [np.setdiff1d(q, o) for q, o in zip(ctx.data.col_questions, observed)]
                truth = np.stack([100. * ctx.data.answers[np.ix_(ctx.held, h)].mean(1)
                                  for h in hidden], axis=1)
                for method in methods:
                    if method.startswith('variant_'):
                        path = os.path.join(root, method[8:], selection, f's{seed}', f'B{budget}.json')
                        question_pred, unit = load_question_predictions(
                            ctx, path, args.variant_hash, observed)
                        pred = np.stack([question_pred[:, h].mean(1) for h in hidden], axis=1)
                        optimizer_records.append({
                            'method': method, 'selection': selection, 'seed': seed,
                            'budget': budget, 'seconds': unit['seconds'],
                            'diagnostics': unit['diagnostics']})
                    elif method.startswith('benchpress_r'):
                        path = benchpress_path(ctx, selection, int(method[12:]), seed, budget)
                        unit = read_unit(path, bp['config_hash'])
                        if unit is None or unit['targets'] != ctx.held:
                            raise ValueError(f'missing original predictions: {path}')
                        pred = np.asarray(unit['pred'])
                    else:
                        pred = np.array([u['pred'][selection][method] for u in units]).T
                    if pred.shape != truth.shape or not np.isfinite(pred).all():
                        raise ValueError(f'invalid prediction shape/values: {method}')
                    error = compute_prediction_error(truth.ravel(), pred.ravel(), aggregation='pool')
                    if error['n'] != truth.size:
                        raise ValueError('evaluation dropped held-out cells')
                    if method in original_methods:
                        old = reference_rows[selection, method, seed, budget]
                        if abs(error['medae'] - old['medae']) > 1e-12 or error['n'] != old['n']:
                            raise ValueError(f'frozen evaluation changed: {method}')
                    records.append({
                        'selection': selection, 'method': method, 'seed': seed,
                        'budget': budget, 'medae': error['medae'], 'n': error['n'],
                        'questions_used': int(sum(len(o) for o in observed))})
    table = {}
    for selection in selections:
        for method in methods:
            for budget in budgets:
                rows = [r for r in records if (r['selection'], r['method'], r['budget'])
                        == (selection, method, budget)]
                values = np.asarray([r['medae'] for r in rows])
                table.setdefault(selection, {}).setdefault(method, {})[str(budget)] = {
                    'medae_mean': float(values.mean()),
                    'medae_std': float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    'n_seeds': len(rows), 'n_cells': rows[0]['n'],
                    'questions_used_mean': float(np.mean([r['questions_used'] for r in rows]))}
    name = 'summary.json' if complete else 'pilot_summary.json'
    write_json_atomic(os.path.join(root, name), {
        'config': config, 'config_hash': args.variant_hash, 'base_config_hash': ctx.chash,
        'summarize_commit': args.code_commit, 'reference_summary_sha256': sha256_file(
            os.path.join(ctx.out, 'summary.json')),
        'frozen_reference_reproduced': True, 'complete_grid': complete,
        'selected_methods': selected_methods, 'selected_seeds': seeds,
        'selected_budgets': budgets, 'selected_selections': selections, 'per_seed': records,
        'fit_diagnostics': optimizer_records, 'table': table})
    for selection, method_rows in table.items():
        for method, cells in method_rows.items():
            print(selection, method, ' '.join(
                f'{b}:{cells[str(b)]["medae_mean"]:.3f}' for b in budgets))
    print(f'wrote {root}/{name}; complete_grid={complete}')


def summarize_accuracy(args):
    ctx = Context(args, check_commit=False)
    bp = load_benchpress_manifest(ctx)
    with open(os.path.join(ctx.out, 'summary.json')) as f:
        reference = json.load(f)
    if reference['config_hash'] != ctx.chash or reference['benchpress_config_hash'] != bp['config_hash']:
        raise ValueError('reference summary provenance mismatch')
    sources = {}
    for chash in args.variant_hashes:
        manifest_path = os.path.join(ctx.out, 'variants', chash, 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        cfg = manifest['config']
        if (config_hash(cfg) != chash or manifest['config_hash'] != chash
                or cfg['base_config_hash'] != ctx.chash):
            raise ValueError(f'variant provenance mismatch: {manifest_path}')
        for method in cfg['methods']:
            if method != 'logit' or cfg['rank'] not in (8, 40):
                raise ValueError('accuracy recovery expects original BenchPress ranks 8 and 40')
            label = f'benchpress_r{cfg["rank"]}'
            if label in sources:
                raise ValueError(f'duplicate prediction source for {label}')
            sources[label] = {'config_hash': chash, 'method': method}
    expected = {'benchpress_r8', 'benchpress_r40'}
    if set(sources) != expected:
        raise ValueError(f'accuracy requires exactly these fitted methods: {sorted(expected)}')
    metric = {'name': 'binary_accuracy', 'threshold': 0.5, 'comparison': '>',
              'tie_prediction': 0, 'stored_score_scale': 100., 'units': 'percent',
              'aggregation': 'pooled hidden model-question pairs',
              'subtask_constant_methods': []}
    config = {'base_config_hash': ctx.chash, 'code_commit': args.code_commit,
              'sources': sources, 'metric': metric,
              'reference_summary_sha256': sha256_file(os.path.join(ctx.out, 'summary.json'))}
    chash = config_hash(config)
    root = os.path.join(ctx.out, 'accuracy', chash)
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    selections = ctx.subset(args.selections, 'selections')
    methods = ['benchpress_r8', 'benchpress_r40']
    complete = all(set(values) == set(ctx.config[key]) for key, values in
                   [('seeds', seeds), ('budgets', budgets), ('selections', selections)])
    records = []
    truth = ctx.data.answers[ctx.held]
    for seed in seeds:
        for budget in budgets:
            coresets, units = load_coresets(ctx, seed, budget)
            for j, unit in enumerate(units):
                if (unit['targets'] != ctx.held or unit['column'] != j
                        or unit['seed'] != seed or unit['budget'] != budget):
                    raise ValueError('estimator targets or setting mismatch')
            for selection in selections:
                observed = [q[c] for q, c in zip(ctx.data.col_questions, coresets[selection])]
                revealed = np.concatenate(observed)
                test_mask = np.ones(truth.shape, dtype=bool)
                test_mask[:, revealed] = False
                mask_hash = config_hash({'targets': ctx.held,
                                         'observed': [o.tolist() for o in observed]})
                for method in methods:
                    source = sources[method]
                    path = os.path.join(ctx.out, 'variants', source['config_hash'],
                                        source['method'], selection, f's{seed}', f'B{budget}.json')
                    pred, unit = load_question_predictions(
                        ctx, path, source['config_hash'], observed)
                    if (unit['method'], unit['selection'], unit['seed'], unit['budget']) != (
                            source['method'], selection, seed, budget):
                        raise ValueError(f'prediction setting mismatch: {path}')
                    source_hash = unit['question_pred_sha256']
                    if method == 'benchpress_r8':
                        original = read_unit(
                            benchpress_path(ctx, selection, 8, seed, budget), bp['config_hash'])
                        if original is None or original['targets'] != ctx.held:
                            raise ValueError('missing frozen rank-8 reference')
                        np.testing.assert_allclose(unit['pred'], original['pred'],
                                                   rtol=0, atol=1e-12)
                    result = compute_binary_accuracy(
                        truth, pred / metric['stored_score_scale'], test_mask,
                        threshold=metric['threshold'])
                    if result['n'] != len(ctx.held) * (truth.shape[1] - len(revealed)):
                        raise ValueError('accuracy denominator or revealed-question set mismatch')
                    records.append(dict(
                        result, selection=selection, method=method, seed=seed, budget=budget,
                        questions_used=len(revealed), mask_hash=mask_hash,
                        prediction_sha256=source_hash, subtask_constant=False))
                print(f'accuracy {selection} s={seed} B={budget}: {len(methods)} methods', flush=True)
    if len({(r['selection'], r['method'], r['seed'], r['budget']) for r in records}) != len(records):
        raise ValueError('duplicate accuracy settings')
    table = {}
    for selection in selections:
        for method in methods:
            for budget in budgets:
                rows = [r for r in records if (r['selection'], r['method'], r['budget'])
                        == (selection, method, budget)]
                values = np.asarray([r['accuracy'] for r in rows])
                table.setdefault(selection, {}).setdefault(method, {})[str(budget)] = {
                    'accuracy_mean': float(values.mean()),
                    'accuracy_std': float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    'n_seeds': len(rows), 'n_cells_min': min(r['n'] for r in rows),
                    'n_cells_max': max(r['n'] for r in rows),
                    'questions_used_mean': float(np.mean([r['questions_used'] for r in rows]))}
    name = 'summary.json' if complete else 'pilot_summary.json'
    write_json_atomic(os.path.join(root, name), {
        'config': config, 'config_hash': chash, 'complete_grid': complete,
        'frozen_rank8_reproduced': True, 'n_models': len(ctx.held),
        'n_questions': truth.shape[1], 'per_seed': records, 'table': table})
    print(f'wrote {root}/{name}; complete_grid={complete}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'summarize', 'accuracy'])
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--methods', nargs='+', choices=['logit'] + METHODS)
    parser.add_argument('--rank', type=int, default=RANK)
    parser.add_argument('--selections', type=lambda s: s.split(','))
    parser.add_argument('--seeds', type=int_list)
    parser.add_argument('--budgets', type=int_list)
    parser.add_argument('--variant-hash')
    parser.add_argument('--variant-hashes', nargs='+')
    args = parser.parse_args()
    if args.rank < 1:
        parser.error('--rank must be positive')
    if args.stage == 'summarize' and not args.variant_hash:
        parser.error('summarize requires --variant-hash')
    if args.stage == 'accuracy' and not args.variant_hashes:
        parser.error('accuracy requires --variant-hashes')
    {'fit': fit, 'summarize': summarize, 'accuracy': summarize_accuracy}[args.stage](args)


if __name__ == '__main__':
    main()
