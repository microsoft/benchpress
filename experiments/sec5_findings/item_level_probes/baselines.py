"""Recover the question-level predictors inside GP-IRT and AIPW."""

import argparse
import hashlib
import json
import os
import platform
import time
from importlib import metadata

import numpy as np

from benchpress.evaluation_harness import compute_binary_accuracy
from benchpress.io_utils import write_json_atomic, write_npz_compressed_atomic
from experiments.sec5_findings.item_level_probes.run import (
    BENCHPRED_COMMIT, Context, DEFAULT_DATA, DEFAULT_OUT, benchpred_commit,
    config_hash, estimator_path, int_list, load_coresets,
    read_unit, sha256_file,
)

METHODS = ('gp_irt_items', 'aipw_items')
BP_METHODS = ('benchpress_r8', 'benchpress_r40')


def load_predictions(ctx, path, chash, seed, budget, column, selection):
    """Validate one subtask's raw predictions and return its arrays and metadata."""
    unit = read_unit(path, chash)
    if unit is None:
        raise FileNotFoundError(path)
    if (unit['seed'], unit['budget'], unit['column'], unit['selection']) != (
            seed, budget, column, selection):
        raise ValueError(f'prediction setting mismatch: {path}')
    raw = os.path.splitext(path)[0] + '.npz'
    if sha256_file(raw) != unit['prediction_sha256']:
        raise ValueError(f'prediction checksum mismatch: {raw}')
    frozen = read_unit(estimator_path(ctx, seed, budget, column), ctx.chash)
    if frozen is None or sha256_file(estimator_path(ctx, seed, budget, column)) != unit['estimator_sha256']:
        raise ValueError(f'frozen estimator mismatch: {path}')
    questions = ctx.data.col_questions[column]
    observed = np.asarray(frozen['coresets'][selection], dtype=int)
    with np.load(raw) as z:
        np.testing.assert_array_equal(z['targets'], ctx.held)
        np.testing.assert_array_equal(z['questions'], questions)
        np.testing.assert_array_equal(z['observed'], observed)
        predictions = {method: z[method] for method in METHODS}
    for pred in predictions.values():
        if pred.shape != (len(ctx.held), len(questions)) or not np.isfinite(pred).all():
            raise ValueError(f'invalid question predictions: {path}')
        np.testing.assert_array_equal(
            pred[:, observed], 100. * ctx.data.answers[np.ix_(ctx.held, questions[observed])])
    return predictions, unit


def fit(args):
    import benchpred
    from benchpred.tiny_bench import estimate_ability_parameters, item_curve
    from sklearn.linear_model import Ridge

    if benchpred_commit() != BENCHPRED_COMMIT:
        raise ValueError('benchpred revision differs from the frozen experiment')
    ctx = Context(args, check_commit=False)
    config = {
        'base_config_hash': ctx.chash, 'code_commit': args.code_commit,
        'benchpred_commit': BENCHPRED_COMMIT, 'methods': list(METHODS),
        'irt': {'model_type': 'multidim_2pl', 'dims': 10, 'epochs': 2000,
                'priors': 'hierarchical', 'lr': .1, 'deterministic': True,
                'device': 'cpu', 'dropout': .5, 'hidden': 100, 'log_every': 200},
        'ridge': {'alpha': 10, 'fit_intercept': True},
        'seed': '1000 * selection seed + column',
        'versions': {p: metadata.version(p) for p in
                     ('numpy', 'scipy', 'scikit-learn', 'torch', 'pyro-ppl', 'benchpred')},
        'python': platform.python_version(),
    }
    chash = config_hash(config)
    root = os.path.join(ctx.out, 'question_baselines', chash)
    write_json_atomic(os.path.join(root, 'manifest.json'), {'config': config, 'config_hash': chash})
    for seed in ctx.subset(args.seeds, 'seeds'):
        for column in args.columns:
            if not 0 <= column < len(ctx.data.col_questions):
                raise ValueError(f'invalid column {column}')
            folder = os.path.join(root, f's{seed}', f'c{column:02d}')
            pending = []
            for budget in ctx.subset(args.budgets, 'budgets'):
                for selection in ctx.subset(args.selections, 'selections'):
                    path = os.path.join(folder, f'{selection}_B{budget}.json')
                    if os.path.exists(path):
                        load_predictions(ctx, path, chash, seed, budget, column, selection)
                    else:
                        pending.append((budget, selection, path))
            if not pending:
                continue
            questions = ctx.data.col_questions[column]
            source = ctx.data.answers[np.ix_(ctx.sel, questions)].astype(float)
            first = read_unit(estimator_path(ctx, seed, ctx.config['budgets'][0], column), ctx.chash)
            if first is None:
                raise ValueError('missing frozen IRT anchors')
            model_path = os.path.join(folder, 'irt.npz')
            model_record = read_unit(os.path.join(folder, 'irt.json'), chash)
            if model_record is not None:
                if sha256_file(model_path) != model_record['sha256']:
                    raise ValueError('IRT parameter checksum mismatch')
                with np.load(model_path) as z:
                    A, B = z['A'], z['B']
            else:
                start = time.time()
                model = benchpred.PIRTPred().fit(
                    source, first['allocation'], seed=1000 * seed + column, **config['irt'])
                np.testing.assert_array_equal(model.anchor_points, first['gp_irt_anchor_points'])
                A, B = model.A, model.B
                write_npz_compressed_atomic(model_path, A=A, B=B)
                write_json_atomic(os.path.join(folder, 'irt.json'), {
                    'config_hash': chash, 'seed': seed, 'column': column,
                    'sha256': sha256_file(model_path), 'seconds': time.time() - start,
                    'frozen_initial_anchors_reproduced': True})
            for budget, selection, path in pending:
                start = time.time()
                frozen_path = estimator_path(ctx, seed, budget, column)
                frozen = read_unit(frozen_path, ctx.chash)
                if frozen is None or frozen['targets'] != ctx.held:
                    raise ValueError(f'missing or mismatched estimator: {frozen_path}')
                observed = np.asarray(frozen['coresets'][selection], dtype=int)
                scores = ctx.data.answers[np.ix_(ctx.held, questions[observed])].astype(float)
                hidden = np.setdiff1d(np.arange(len(questions)), observed)
                ridge = Ridge(**config['ridge']).fit(source[:, observed].T, scores.T)
                ridge_pred = ridge.predict(source.T).T
                np.testing.assert_allclose(
                    100. * ridge_pred[:, hidden].mean(1), frozen['pred'][selection]['aipw'],
                    rtol=0, atol=1e-8)
                # GP-IRT fits ability on the original anchor list, including repeats.
                anchors = (np.asarray(frozen['gp_irt_anchor_points'], dtype=int)
                           if selection == 'anchors' else observed)
                responses = ctx.data.answers[np.ix_(ctx.held, questions[anchors])]
                irt_pred = np.vstack([
                    item_curve(estimate_ability_parameters(y, A[:, :, anchors], B[:, :, anchors]),
                               A, B).reshape(-1) for y in responses])
                predictions = {'gp_irt_items': 100. * irt_pred, 'aipw_items': 100. * ridge_pred}
                for pred in predictions.values():
                    if pred.shape != (len(ctx.held), len(questions)) or not np.isfinite(pred).all():
                        raise ValueError('invalid native question predictions')
                    pred[:, observed] = 100. * scores
                raw = os.path.splitext(path)[0] + '.npz'
                write_npz_compressed_atomic(
                    raw, **predictions, observed=observed, targets=np.asarray(ctx.held),
                    questions=questions)
                write_json_atomic(path, {
                    'config_hash': chash, 'seed': seed, 'budget': budget,
                    'column': column, 'selection': selection, 'seconds': time.time() - start,
                    'prediction_sha256': sha256_file(raw), 'estimator_sha256': sha256_file(frozen_path),
                    'irt_sha256': sha256_file(model_path), 'aipw_hidden_mean_reproduced': True})
                print(f'baseline items s={seed} c={column} {selection} B={budget}: {chash}', flush=True)


def summarize(args):
    ctx = Context(args, check_commit=False)
    root = os.path.join(ctx.out, 'question_baselines', args.baseline_hash)
    with open(os.path.join(root, 'manifest.json')) as f:
        manifest = json.load(f)
    if (manifest['config_hash'] != args.baseline_hash
            or config_hash(manifest['config']) != args.baseline_hash
            or manifest['config']['base_config_hash'] != ctx.chash):
        raise ValueError('baseline manifest mismatch')
    with open(args.benchpress_summary) as f:
        bp = json.load(f)
    metric = bp['config']['metric']
    if (not bp['complete_grid'] or not bp['frozen_rank8_reproduced']
            or bp['config_hash'] != config_hash(bp['config'])
            or bp['config']['base_config_hash'] != ctx.chash
            or bp['config']['reference_summary_sha256'] != sha256_file(os.path.join(ctx.out, 'summary.json'))
            or bp['n_models'] != len(ctx.held) or bp['n_questions'] != ctx.data.answers.shape[1]
            or any(metric[k] != v for k, v in {
                'name': 'binary_accuracy', 'threshold': .5, 'comparison': '>',
                'tie_prediction': 0, 'stored_score_scale': 100., 'units': 'percent',
                'aggregation': 'pooled hidden model-question pairs'}.items())):
        raise ValueError('BenchPress accuracy provenance or metric mismatch')
    records = [dict(r) for r in bp['per_seed'] if r['method'] in BP_METHODS]
    keys = [(s, seed, b) for s in ctx.config['selections']
            for seed in ctx.config['seeds'] for b in ctx.config['budgets']]
    if len(records) != 2 * len(keys) or {
            (r['selection'], r['seed'], r['budget'], r['method']) for r in records} != {
            (*key, m) for key in keys for m in BP_METHODS}:
        raise ValueError('incomplete BenchPress reference rows')
    truth = ctx.data.answers[ctx.held]
    for selection, seed, budget in keys:
        coresets, _ = load_coresets(ctx, seed, budget)
        observed = [q[c] for q, c in zip(ctx.data.col_questions, coresets[selection])]
        mask_hash = config_hash({'targets': ctx.held, 'observed': [q.tolist() for q in observed]})
        revealed = np.concatenate(observed)
        mask = np.ones(truth.shape, dtype=bool)
        mask[:, revealed] = False
        predictions = {method: np.empty(truth.shape) for method in METHODS}
        checksums = []
        for column, questions in enumerate(ctx.data.col_questions):
            path = os.path.join(root, f's{seed}', f'c{column:02d}', f'{selection}_B{budget}.json')
            arrays, unit = load_predictions(
                ctx, path, args.baseline_hash, seed, budget, column, selection)
            checksums.append(unit['prediction_sha256'])
            for method in METHODS:
                predictions[method][:, questions] = arrays[method]
        for row in records:
            if (row['selection'], row['seed'], row['budget']) == (selection, seed, budget):
                if row['mask_hash'] != mask_hash or row['n'] != int(mask.sum()):
                    raise ValueError('BenchPress and baseline hidden sets differ')
        digest = hashlib.sha256(json.dumps(checksums).encode()).hexdigest()
        for method, pred in predictions.items():
            result = compute_binary_accuracy(truth, pred / 100., mask, threshold=.5)
            records.append(dict(result, selection=selection, method=method, seed=seed, budget=budget,
                                questions_used=len(revealed), mask_hash=mask_hash,
                                prediction_sha256=digest, subtask_constant=False))
    table = {}
    for selection in ctx.config['selections']:
        for method in BP_METHODS + METHODS:
            for budget in ctx.config['budgets']:
                rows = [r for r in records if (r['selection'], r['method'], r['budget'])
                        == (selection, method, budget)]
                values = np.asarray([r['accuracy'] for r in rows])
                table.setdefault(selection, {}).setdefault(method, {})[str(budget)] = {
                    'accuracy_mean': float(values.mean()), 'accuracy_std': float(values.std(ddof=1)),
                    'n_seeds': len(rows), 'n_cells_min': min(r['n'] for r in rows),
                    'n_cells_max': max(r['n'] for r in rows),
                    'questions_used_mean': float(np.mean([r['questions_used'] for r in rows]))}
    config = {
        'base_config_hash': ctx.chash, 'code_commit': args.code_commit,
        'baseline_config_hash': args.baseline_hash,
        'benchpress_summary_sha256': sha256_file(args.benchpress_summary),
        'benchpress_config_hash': bp['config_hash'],
        'metric': dict(metric, subtask_constant_methods=[]),
        'baseline_scope': 'question-level IRT/ridge predictors; aggregate GP-IRT/AIPW corrections excluded',
    }
    chash = config_hash(config)
    path = os.path.join(ctx.out, 'accuracy', chash, 'summary.json')
    write_json_atomic(path, dict(
        config=config, config_hash=chash, complete_grid=True, frozen_rank8_reproduced=True,
        n_models=len(ctx.held), n_questions=truth.shape[1], per_seed=records, table=table))
    print(f'wrote {path}: {len(records)} records', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'summarize'])
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--seeds', type=int_list)
    parser.add_argument('--budgets', type=int_list)
    parser.add_argument('--selections', type=lambda s: s.split(','))
    parser.add_argument('--columns', type=int_list)
    parser.add_argument('--baseline-hash')
    parser.add_argument('--benchpress-summary')
    args = parser.parse_args()
    if args.stage == 'fit' and not args.columns:
        parser.error('fit requires --columns')
    if args.stage == 'summarize' and not (args.baseline_hash and args.benchpress_summary):
        parser.error('summarize requires --baseline-hash and --benchpress-summary')
    if args.stage == 'summarize' and any(v is not None for v in (
            args.columns, args.seeds, args.budgets, args.selections)):
        parser.error('summarize requires the complete frozen grid; subset flags apply only to fit')
    (fit if args.stage == 'fit' else summarize)(args)


if __name__ == '__main__':
    main()
