"""BenchPress profile anchors versus cached random question selection."""

import argparse
import json
import os
import platform
import time
from importlib import metadata

import numpy as np

from benchpress.evaluation_harness import compute_binary_accuracy
from benchpress.io_utils import write_json_atomic, write_npz_compressed_atomic
from benchpress.methods.predictors import predict_logit_bias_als_scores
from benchpress.methods.question_selection import PROFILE_CLUSTERING, select_profile_anchors
from experiments.sec5_findings.item_level_probes.run import (
    BP_LAM, Context, DEFAULT_DATA, DEFAULT_OUT, PCT_SPEC, allocate, config_hash,
    int_list, load_coresets, load_question_predictions, predict, read_unit, sha256_file,
)

RANKS = [8, 40]
RANDOM_SOURCES = {8: '8e7e521f9645ce06', 40: '577115b7edd126a2'}
RUNTIME = {'python': platform.python_version(), **{
    name: metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}}


def load_manifest(ctx, chash):
    path = os.path.join(ctx.out, 'question_selection', chash, 'manifest.json')
    with open(path) as f:
        manifest = json.load(f)
    if (config_hash(manifest['config']) != chash
            or manifest['config_hash'] != chash
            or manifest['config']['base_config_hash'] != ctx.chash):
        raise ValueError(f'selection manifest mismatch: {path}')
    return manifest


def load_profiles(ctx, root, chash):
    unit = read_unit(os.path.join(root, 'profiles.json'), chash)
    raw = os.path.join(root, 'profiles.npz')
    if unit is None or sha256_file(raw) != unit['profiles_sha256']:
        raise ValueError('training profile artifact is missing or corrupt')
    with np.load(raw) as z:
        np.testing.assert_array_equal(z['training_models'], ctx.sel)
        np.testing.assert_array_equal(z['questions'], np.arange(ctx.data.answers.shape[1]))
        values = z['profiles']
    if (values.shape != (len(ctx.sel), ctx.data.answers.shape[1])
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 100))):
        raise ValueError('invalid fitted training response profiles')
    return values, unit


def load_selection(ctx, root, chash, seed, budget):
    path = os.path.join(root, 'coresets', f's{seed}', f'B{budget}.json')
    unit = read_unit(path, chash)
    profile = read_unit(os.path.join(root, 'profiles.json'), chash)
    if (unit is None or profile is None or unit['seed'] != seed
            or unit['budget'] != budget
            or unit['profiles_sha256'] != profile['profiles_sha256']):
        raise ValueError(f'missing or mismatched question selection: {path}')
    allocation = allocate(budget, ctx.data.sizes)
    np.testing.assert_array_equal(unit['allocation'], allocation)
    if len(unit['observed']) != len(allocation):
        raise ValueError('selection must cover every subtask')
    observed = [np.asarray(ids, dtype=int) for ids in unit['observed']]
    for questions, ids, count in zip(ctx.data.col_questions, observed, allocation):
        if (len(ids) != count or len(np.unique(ids)) != count
                or not np.isin(ids, questions).all()):
            raise ValueError('invalid representative IDs or subtask budget')
    if config_hash(unit['observed']) != unit['selection_hash']:
        raise ValueError('selected question checksum mismatch')
    return observed, unit


def profiles(args):
    ctx = Context(args, check_commit=False)
    config = {
        'comparison': 'question_selection', 'base_config_hash': ctx.chash,
        'code_commit': args.code_commit, 'runtime': RUNTIME,
        'selector': {
            'rank': 40, 'training_models': ctx.sel,
            'features': 'fitted training-model response profiles in percentage units',
            'preserve_observed': False, 'clustering': PROFILE_CLUSTERING,
            'seed': '1000 * selection_seed + subtask',
            'representative': 'nearest unused question per fitted center; index breaks ties',
        },
        'ranks': args.ranks, 'lam': BP_LAM,
        'als': {'n_iter': 40, 'n_ensemble': 10, 'base_seed': 42,
                'standardize_columns': True},
        'fit': 'pooled training rows and held-out rows with only selected answers revealed',
        'random_sources': {str(rank): value for rank, value in RANDOM_SOURCES.items()},
        'metric': {'name': 'binary_accuracy', 'threshold': 0.5, 'comparison': '>',
                   'tie_prediction': 0, 'stored_score_scale': 100., 'units': 'percent',
                   'aggregation': 'pooled held-out model-question pairs',
                   'mask': 'questions unrevealed by either policy, per seed and budget'},
    }
    chash = config_hash(config)
    root = os.path.join(ctx.out, 'question_selection', chash)
    manifest = {'config': config, 'config_hash': chash}
    path = os.path.join(root, 'manifest.json')
    if os.path.exists(path):
        if load_manifest(ctx, chash) != manifest:
            raise ValueError('profile configuration changed')
    else:
        write_json_atomic(path, manifest)
    if os.path.exists(os.path.join(root, 'profiles.json')):
        load_profiles(ctx, root, chash)
        print(f'cached profiles config={chash}', flush=True)
        return
    start = time.time()
    print(f'start training-only profiles rank=40 config={chash}', flush=True)
    train = 100. * ctx.data.answers[ctx.sel].astype(float)
    ids = [f'q{j}' for j in range(train.shape[1])]
    values = predict_logit_bias_als_scores(
        train, rank=40, lam=BP_LAM, metric={q: PCT_SPEC for q in ids},
        benchmark_ids=ids, preserve_observed=False)
    if not np.isfinite(values).all() or np.array_equal(values, train):
        raise ValueError('selector requires fitted profiles, not restored observed answers')
    raw = os.path.join(root, 'profiles.npz')
    write_npz_compressed_atomic(raw, profiles=values, training_models=np.asarray(ctx.sel),
                                questions=np.arange(train.shape[1]))
    write_json_atomic(os.path.join(root, 'profiles.json'), {
        'config_hash': chash, 'profiles_sha256': sha256_file(raw),
        'seconds': time.time() - start, 'training_models': ctx.sel,
        'n_questions': train.shape[1], 'code_commit': args.code_commit,
    })
    load_profiles(ctx, root, chash)
    print(f'done profiles config={chash} seconds={time.time() - start:.1f}', flush=True)


def select(args):
    ctx = Context(args, check_commit=False)
    manifest = load_manifest(ctx, args.selection_hash)
    if (manifest['config']['runtime'] != RUNTIME
            or manifest['config']['code_commit'] != args.code_commit):
        raise ValueError('selection must use the profile source and runtime')
    root = os.path.join(ctx.out, 'question_selection', args.selection_hash)
    values, profile = load_profiles(ctx, root, args.selection_hash)
    for seed in ctx.subset(args.seeds, 'seeds'):
        for budget in ctx.subset(args.budgets, 'budgets'):
            path = os.path.join(root, 'coresets', f's{seed}', f'B{budget}.json')
            if os.path.exists(path):
                load_selection(ctx, root, args.selection_hash, seed, budget)
                print(f'cached selection s={seed} B={budget}', flush=True)
                continue
            start = time.time()
            allocation = allocate(budget, ctx.data.sizes)
            observed, times = [], []
            for j, (questions, count) in enumerate(zip(ctx.data.col_questions, allocation)):
                t0 = time.time()
                local = select_profile_anchors(values[:, questions].T, int(count), 1000 * seed + j)
                observed.append(questions[local].tolist())
                times.append(time.time() - t0)
                print(f'selected s={seed} B={budget} c={j} n={count} seconds={times[-1]:.1f}',
                      flush=True)
            write_json_atomic(path, {
                'config_hash': args.selection_hash, 'seed': seed, 'budget': budget,
                'allocation': allocation.tolist(), 'observed': observed,
                'selection_hash': config_hash(observed),
                'profiles_sha256': profile['profiles_sha256'],
                'seconds': time.time() - start, 'subtask_seconds': times,
            })
            load_selection(ctx, root, args.selection_hash, seed, budget)


def fit(args):
    ctx = Context(args, check_commit=False)
    manifest = load_manifest(ctx, args.selection_hash)
    if (manifest['config']['runtime'] != RUNTIME
            or manifest['config']['code_commit'] != args.code_commit):
        raise ValueError('fitting must use the profile source and runtime')
    if not set(args.ranks).issubset(manifest['config']['ranks']):
        raise ValueError('requested ranks are not in the fitted configuration')
    root = os.path.join(ctx.out, 'question_selection', args.selection_hash)
    for seed in ctx.subset(args.seeds, 'seeds'):
        for budget in ctx.subset(args.budgets, 'budgets'):
            observed, selection = load_selection(ctx, root, args.selection_hash, seed, budget)
            revealed = np.concatenate(observed)
            held = np.full((len(ctx.held), ctx.data.answers.shape[1]), np.nan)
            held[:, revealed] = 100. * ctx.data.answers[np.ix_(ctx.held, revealed)]
            matrix = np.vstack((100. * ctx.data.answers[ctx.sel], held))
            hidden = [np.setdiff1d(q, o) for q, o in zip(ctx.data.col_questions, observed)]
            for rank in args.ranks:
                path = os.path.join(root, 'predictions', f'r{rank}', f's{seed}', f'B{budget}.json')
                if os.path.exists(path):
                    _, unit = load_question_predictions(ctx, path, args.selection_hash, observed)
                    if (unit['rank'], unit['seed'], unit['budget'], unit['selection_hash']) != (
                            rank, seed, budget, selection['selection_hash']):
                        raise ValueError('cached selected-question fit has the wrong setting')
                    print(f'cached selected fit r={rank} s={seed} B={budget}', flush=True)
                    continue
                start = time.time()
                print(f'start selected fit r={rank} s={seed} B={budget}', flush=True)
                completed = predict(matrix, rank, BP_LAM)
                np.testing.assert_array_equal(
                    completed[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
                pred = completed[len(ctx.sel):]
                raw = path[:-5] + '.npz'
                write_npz_compressed_atomic(
                    raw, pred=pred, targets=np.asarray(ctx.held), revealed=revealed)
                write_json_atomic(path, {
                    'config_hash': args.selection_hash, 'code_commit': args.code_commit,
                    'selection': 'benchpress_anchors', 'rank': rank, 'seed': seed, 'budget': budget,
                    'selection_hash': selection['selection_hash'], 'targets': ctx.held,
                    'mask_hash': config_hash({'targets': ctx.held,
                                              'observed': [o.tolist() for o in observed]}),
                    'question_pred_sha256': sha256_file(raw), 'questions_used': len(revealed),
                    'pred': np.stack([pred[:, h].mean(1) for h in hidden], axis=1).tolist(),
                    'seconds': time.time() - start,
                })
                load_question_predictions(ctx, path, args.selection_hash, observed)
                print(f'done selected fit r={rank} s={seed} B={budget} '
                      f'seconds={time.time() - start:.1f}', flush=True)


def summarize(args):
    ctx = Context(args, check_commit=False)
    manifest = load_manifest(ctx, args.selection_hash)
    root = os.path.join(ctx.out, 'question_selection', args.selection_hash)
    load_profiles(ctx, root, args.selection_hash)
    config = manifest['config']
    ranks = sorted(args.ranks)
    if not ranks or not set(ranks).issubset(config['ranks']):
        raise ValueError('requested ranks are not in the fitted configuration')
    for rank in ranks:
        chash = config['random_sources'][str(rank)]
        with open(os.path.join(ctx.out, 'variants', chash, 'manifest.json')) as f:
            original = json.load(f)
        cfg = original['config']
        if (original['config_hash'] != chash or config_hash(cfg) != chash
                or cfg['base_config_hash'] != ctx.chash or cfg['rank'] != rank
                or cfg['methods'] != ['logit'] or cfg['lam'] != config['lam']
                or cfg['als'] != config['als']):
            raise ValueError('random reference differs from the fixed predictor settings')
    seeds = sorted(ctx.subset(args.seeds, 'seeds'))
    budgets = sorted(ctx.subset(args.budgets, 'budgets'))
    score_config = dict(
        config, ranks=ranks, seeds=seeds, budgets=budgets,
        source_config_hash=args.selection_hash, fit_code_commit=config['code_commit'],
        code_commit=args.code_commit, scoring_runtime=RUNTIME,
        random_sources={str(r): config['random_sources'][str(r)] for r in ranks})
    score_hash = config_hash(score_config)
    records, pairs, masks = [], [], []
    truth = ctx.data.answers[ctx.held]
    for seed in seeds:
        for budget in budgets:
            observed, selection = load_selection(ctx, root, args.selection_hash, seed, budget)
            coresets, units = load_coresets(ctx, seed, budget)
            for j, unit in enumerate(units):
                if (unit['targets'], unit['seed'], unit['budget'], unit['column']) != (
                        ctx.held, seed, budget, j):
                    raise ValueError('random reference selection setting mismatch')
            random_observed = [q[c] for q, c in zip(ctx.data.col_questions, coresets['random'])]
            np.testing.assert_array_equal([len(q) for q in random_observed], selection['allocation'])
            excluded = np.union1d(np.concatenate(observed), np.concatenate(random_observed))
            common = np.setdiff1d(np.arange(truth.shape[1]), excluded)
            mask_hash = config_hash({'targets': ctx.held, 'questions': common.tolist()})
            mask = np.ones(truth.shape, dtype=bool)
            mask[:, excluded] = False
            masks.append({
                'seed': seed, 'budget': budget, 'mask_hash': mask_hash,
                'questions': common.tolist(), 'targets': ctx.held,
                'per_subtask': [int(np.isin(q, common).sum()) for q in ctx.data.col_questions],
                'selection_seconds': selection['seconds'],
            })
            for rank in ranks:
                scores = {}
                for policy in ('random', 'benchpress_anchors'):
                    if policy == 'random':
                        chash = config['random_sources'][str(rank)]
                        path = os.path.join(ctx.out, 'variants', chash, 'logit', 'random',
                                            f's{seed}', f'B{budget}.json')
                        selected = random_observed
                    else:
                        chash = args.selection_hash
                        path = os.path.join(root, 'predictions', f'r{rank}',
                                            f's{seed}', f'B{budget}.json')
                        selected = observed
                    pred, unit = load_question_predictions(ctx, path, chash, selected)
                    if (unit['rank'], unit['seed'], unit['budget'], unit['selection']) != (
                            rank, seed, budget, policy):
                        raise ValueError(f'prediction setting mismatch: {path}')
                    if policy == 'benchpress_anchors' and unit['selection_hash'] != selection['selection_hash']:
                        raise ValueError('fit used a different selected set')
                    result = compute_binary_accuracy(
                        truth, pred / config['metric']['stored_score_scale'], mask,
                        threshold=config['metric']['threshold'])
                    if result['n'] != len(ctx.held) * len(common):
                        raise ValueError('paired evaluation denominator mismatch')
                    records.append(dict(
                        result, rank=rank, selection=policy, seed=seed, budget=budget,
                        mask_hash=mask_hash, prediction_sha256=unit['question_pred_sha256'],
                        fit_seconds=unit['seconds'], questions_used=sum(map(len, selected))))
                    scores[policy] = result['accuracy']
                pairs.append({'rank': rank, 'seed': seed, 'budget': budget, 'n': result['n'],
                              'gain_pp': scores['benchpress_anchors'] - scores['random'],
                              'mask_hash': mask_hash})
            print(f'scored s={seed} B={budget} common_questions={len(common)}', flush=True)
    if len(records) != len({(r['rank'], r['selection'], r['seed'], r['budget']) for r in records}):
        raise ValueError('duplicate paired accuracy rows')
    table, gains = {}, {}
    for rank in ranks:
        method = f'benchpress_r{rank}'
        for budget in budgets:
            for policy in ('random', 'benchpress_anchors'):
                rows = [r for r in records if (r['rank'], r['budget'], r['selection'])
                        == (rank, budget, policy)]
                values = np.array([r['accuracy'] for r in rows])
                table.setdefault(method, {}).setdefault(policy, {})[str(budget)] = {
                    'accuracy_mean': float(values.mean()),
                    'accuracy_std': float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    'n_seeds': len(rows), 'n_cells_min': min(r['n'] for r in rows),
                    'n_cells_max': max(r['n'] for r in rows),
                }
            values = np.array([r['gain_pp'] for r in pairs
                               if (r['rank'], r['budget']) == (rank, budget)])
            gains.setdefault(method, {})[str(budget)] = {
                'gain_mean_pp': float(values.mean()),
                'gain_std_pp': float(values.std(ddof=1)) if len(values) > 1 else 0.,
                'n_seeds': len(values),
            }
    complete = (set(seeds) == set(ctx.config['seeds'])
                and set(budgets) == set(ctx.config['budgets']))
    name = 'summary.json' if complete else 'pilot_summary.json'
    output = os.path.join(root, 'scores', score_hash, name)
    write_json_atomic(output, {
        'config': score_config, 'config_hash': score_hash, 'complete_grid': complete,
        'n_models': len(ctx.held), 'n_questions': truth.shape[1],
        'per_seed': records, 'paired_gains': pairs, 'table': table, 'gains': gains,
        'evaluation_masks': masks,
    })
    print(f'wrote {output}; complete_grid={complete}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['profiles', 'select', 'fit', 'summarize'])
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--selection-hash')
    parser.add_argument('--seeds', type=int_list)
    parser.add_argument('--budgets', type=int_list)
    parser.add_argument('--ranks', nargs='+', type=int, choices=RANKS, default=[40])
    args = parser.parse_args()
    if args.stage != 'profiles' and not args.selection_hash:
        parser.error('this stage requires --selection-hash')
    {'profiles': profiles, 'select': select, 'fit': fit, 'summarize': summarize}[args.stage](args)


if __name__ == '__main__':
    main()
