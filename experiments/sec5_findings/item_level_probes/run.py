#!/usr/bin/env python
"""Question-level budgets on the Open LLM Leaderboard v2 (Appendix question_level_budgets).

All methods spend the same budget B of evaluated questions on each held-out
model and predict that model's 37 subtask accuracies (percent):

  subtasks   BenchPress revealing whole subtasks, chosen greedily by five-fold
             CV MedAE on the selection models among subtasks that fit the
             remaining budget.
  questions  BenchPress on the model-by-question matrix: B random questions
             split equally across subtasks, answers scored 0/100, subtask
             score = mean of observed and predicted answers.
  random_sampling, random_sampling_learn, aipw, gp_irt
             the per-subtask estimators of Zhang, Dorner and Hardt (2025)
             (package `benchpred`), with the same per-subtask allocation.

Subcommands (run in this order):
  init        write <out>/manifest.json (config + config_hash)
  select      greedy subtask selection per budget (CV cache, resumable)
  subtasks    held-out predictions of `subtasks`
  questions   held-out predictions of `questions` (one unit per seed/B/target)
  estimators  held-out predictions of benchpred methods (unit per method/seed/B/column)
  summarize   MedAE over all held-out cells and over unrevealed cells

Every unit is written atomically as JSON carrying the config_hash; existing
units are skipped, units from another config fail fast.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.evaluation_harness import compute_prediction_error  # noqa: E402
from benchpress.methods.predictors import predict_benchpress_scores  # noqa: E402

PCT_SPEC = {'type': 'pct', 'range': [0, 100]}
BUDGETS = [200, 400, 800, 1600, 3200, 6400]
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
N_FOLDS = 5
FOLD_SEED = 42
BENCHPRED_REPO = 'https://github.com/socialfoundations/benchmark-prediction'
BENCHPRED_COMMIT = '5b7a23e32ee738c69b4c408cb9746429a7fd0043'
ESTIMATORS = {
    'random_sampling': 'RandomSampling',
    'random_sampling_learn': 'RandomSamplingAndLearn',
    'aipw': 'AIPWPred',
    'gp_irt': 'GPIRTPred',
}
METHODS = ['subtasks', 'questions'] + list(ESTIMATORS)
DEFAULT_DATA = os.path.join(HERE, 'results', 'data', 'oll_items.npz')
DEFAULT_OUT = os.path.join(HERE, 'results')


# ─── I/O ─────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config):
    blob = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_unit(path, chash):
    """Return the unit at path, None if absent; fail on a foreign config."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        unit = json.load(f)
    if unit.get('config_hash') != chash:
        raise SystemExit(f'{path} has config_hash {unit.get("config_hash")}, '
                         f'expected {chash}; remove it or use another --out-dir')
    return unit


def benchpred_commit():
    try:
        direct = json.loads(metadata.distribution('benchpred').read_text('direct_url.json'))
    except (metadata.PackageNotFoundError, TypeError, ValueError):
        return None
    return direct.get('vcs_info', {}).get('commit_id')


# ─── Data, split, allocation ─────────────────────────────────────────────────

class Data:
    def __init__(self, path):
        z = np.load(path)
        self.answers = z['answers']
        self.question_column = z['question_column'].astype(int)
        self.column_ids = [str(c) for c in z['column_ids']]
        self.model_ids = [str(m) for m in z['model_ids']]
        n_cols = len(self.column_ids)
        if set(np.unique(self.answers).tolist()) - {0, 1}:
            raise ValueError('answers must be 0/1')
        if not np.all(np.diff(self.question_column) >= 0):
            raise ValueError('question_column must be contiguous blocks')
        self.col_questions = [np.flatnonzero(self.question_column == j)
                              for j in range(n_cols)]
        self.sizes = np.array([len(q) for q in self.col_questions])
        self.truth = np.stack(
            [100.0 * self.answers[:, q].mean(1) for q in self.col_questions], axis=1)


def split_models(n_models):
    """Mirror benchpress.evaluation_harness.split_models_for_probe_validation."""
    perm = np.random.RandomState(SPLIT_SEED).permutation(n_models)
    n_train = int(round(n_models * TRAIN_FRACTION))
    return sorted(int(i) for i in perm[:n_train]), sorted(int(i) for i in perm[n_train:])


def cv_folds(n_sel):
    perm = np.random.RandomState(FOLD_SEED).permutation(n_sel)
    return [np.sort(f) for f in np.array_split(perm, N_FOLDS)]


def allocate(budget, sizes):
    """Equal split of budget across subtasks, each capped at size - 1."""
    caps = np.asarray(sizes) - 1
    if budget > caps.sum():
        raise ValueError(f'budget {budget} exceeds capacity {caps.sum()}')
    k = np.zeros(len(caps), dtype=int)
    remaining = budget
    while True:
        open_cols = np.flatnonzero(k < caps)
        if len(open_cols) == 0:
            break
        share = remaining // len(open_cols)
        if share == 0:
            break
        add = np.minimum(share, caps[open_cols] - k[open_cols])
        k[open_cols] += add
        remaining -= int(add.sum())
    for j in np.flatnonzero(k < caps)[:remaining]:
        k[j] += 1
    if k.sum() != budget:
        raise AssertionError(f'allocation sums to {k.sum()}, not {budget}')
    return k


def predict(matrix):
    ids = [f'c{j}' for j in range(matrix.shape[1])]
    pred = predict_benchpress_scores(
        matrix, metric={i: PCT_SPEC for i in ids}, benchmark_ids=ids)
    if not np.isfinite(pred).all():
        raise AssertionError('BenchPress returned non-finite predictions')
    return pred


# ─── Context shared by subcommands ───────────────────────────────────────────

class Context:
    def __init__(self, args, need_manifest=True, check_commit=True):
        self.out = os.path.abspath(args.out_dir)
        self.data_path = os.path.abspath(args.data)
        self.data = Data(self.data_path)
        self.sel, self.held = split_models(len(self.data.model_ids))
        if not need_manifest:
            return
        path = os.path.join(self.out, 'manifest.json')
        if not os.path.exists(path):
            raise SystemExit(f'missing {path}; run `init` first')
        with open(path) as f:
            self.manifest = json.load(f)
        self.config = self.manifest['config']
        self.chash = self.manifest['config_hash']
        if config_hash(self.config) != self.chash:
            raise SystemExit('manifest config_hash does not match its config')
        if sha256_file(self.data_path) != self.config['data_sha256']:
            raise SystemExit(f'{self.data_path} sha256 differs from manifest')
        if check_commit and args.code_commit != self.config['code_commit']:
            raise SystemExit(f'--code-commit {args.code_commit} differs from '
                             f'manifest {self.config["code_commit"]}')
        if self.sel != self.config['selection_models'] or self.held != self.config['heldout_models']:
            raise SystemExit('model split differs from manifest')

    def subset(self, values, key):
        allowed = self.config[key]
        values = allowed if values is None else values
        bad = sorted(set(values) - set(allowed))
        if bad:
            raise SystemExit(f'{key} {bad} not in manifest {allowed}')
        return values

    def unit(self, obj):
        return dict(obj, config_hash=self.chash, code_commit=self.config['code_commit'])


# ─── init ────────────────────────────────────────────────────────────────────

def cmd_init(args):
    ctx = Context(args, need_manifest=False)
    config = {
        'data_sha256': sha256_file(ctx.data_path),
        'code_commit': args.code_commit,
        'benchpred_repo': BENCHPRED_REPO,
        'benchpred_commit': BENCHPRED_COMMIT,
        'budgets': BUDGETS,
        'seeds': SEEDS,
        'split_seed': SPLIT_SEED,
        'train_fraction': TRAIN_FRACTION,
        'n_folds': N_FOLDS,
        'fold_seed': FOLD_SEED,
        'coreset_seed': '1000 * seed + column',
        'predictor': 'benchpress.methods.predictors.predict_benchpress_scores',
        'methods': METHODS,
        'column_ids': ctx.data.column_ids,
        'column_sizes': ctx.data.sizes.tolist(),
        'selection_models': ctx.sel,
        'heldout_models': ctx.held,
    }
    manifest = {'config': config, 'config_hash': config_hash(config),
                'n_models': len(ctx.data.model_ids),
                'n_questions': int(ctx.data.answers.shape[1])}
    path = os.path.join(ctx.out, 'manifest.json')
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old['config_hash'] != manifest['config_hash']:
            raise SystemExit(f'{path} exists with another config; use another --out-dir')
        print(f'manifest already present: {manifest["config_hash"]}')
        return
    write_json_atomic(path, manifest)
    print(f'wrote {path}: config_hash {manifest["config_hash"]}')


# ─── select: greedy subtask selection by CV MedAE ────────────────────────────

_CV = {}


def _cv_init(truth_sel, folds):
    _CV['truth'] = truth_sel
    _CV['folds'] = folds


def cv_medae(probes):
    """Five-fold CV MedAE on selection models; revealed cells count as exact."""
    truth, folds = _CV['truth'], _CV['folds']
    hidden = np.ones(truth.shape[1], dtype=bool)
    hidden[list(probes)] = False
    predicted = truth.copy()
    for rows in folds:
        matrix = truth.copy()
        matrix[np.ix_(rows, np.flatnonzero(hidden))] = np.nan
        pred = predict(matrix)
        predicted[np.ix_(rows, np.flatnonzero(hidden))] = pred[np.ix_(rows, np.flatnonzero(hidden))]
    return compute_prediction_error(truth.ravel(), predicted.ravel(), aggregation='pool')['medae']


def cmd_select(args):
    ctx = Context(args)
    budgets = ctx.subset(args.budgets, 'budgets')
    sizes = ctx.data.sizes
    truth_sel = ctx.data.truth[ctx.sel]
    folds = cv_folds(len(ctx.sel))
    cache_path = os.path.join(ctx.out, 'select', 'cv_cache.jsonl')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec['config_hash'] != ctx.chash:
                    raise SystemExit(f'{cache_path} has a foreign config_hash')
                cache[tuple(rec['probes'])] = rec['medae']
    _cv_init(truth_sel, folds)
    pool = multiprocessing.Pool(args.workers, _cv_init, (truth_sel, folds)) if args.workers > 1 else None

    def scores(keys):
        todo = [k for k in keys if k not in cache]
        if todo:
            t0 = time.time()
            vals = pool.map(cv_medae, todo) if pool else [cv_medae(k) for k in todo]
            with open(cache_path, 'a') as f:
                for k, v in zip(todo, vals):
                    cache[k] = float(v)
                    f.write(json.dumps({'probes': list(k), 'medae': float(v),
                                        'config_hash': ctx.chash}) + '\n')
            print(f'  {len(todo)} CV evaluations in {time.time() - t0:.0f}s', flush=True)
        return [cache[k] for k in keys]

    for budget in budgets:
        path = os.path.join(ctx.out, 'select', f'B{budget}.json')
        if read_unit(path, ctx.chash) is not None:
            continue
        probes, trace = [], []
        while True:
            remaining = budget - int(sizes[probes].sum())
            cands = [j for j in range(len(sizes)) if j not in probes and sizes[j] <= remaining]
            if not cands:
                break
            keys = [tuple(sorted(probes + [j])) for j in cands]
            vals = scores(keys)
            best_val, best_j = min(zip(vals, cands))
            probes.append(best_j)
            trace.append({'column': best_j, 'cv_medae': best_val,
                          'questions_used': int(sizes[probes].sum())})
            print(f'B={budget}: + {ctx.data.column_ids[best_j]} '
                  f'cv_medae={best_val:.3f} used={trace[-1]["questions_used"]}', flush=True)
        write_json_atomic(path, ctx.unit({
            'budget': budget, 'probes': probes, 'trace': trace,
            'questions_used': int(sizes[probes].sum())}))
    if pool:
        pool.close()


# ─── subtasks: held-out predictions with selected subtasks ───────────────────

_HO = {}


def _subtasks_one(task):
    target, probes = task
    truth = _HO['truth']
    matrix = np.vstack([truth[_HO['sel']], np.full(truth.shape[1], np.nan)])
    matrix[-1, probes] = truth[target, probes]
    return predict(matrix)[-1].tolist()


def cmd_subtasks(args):
    ctx = Context(args)
    budgets = ctx.subset(args.budgets, 'budgets')
    _HO.update(truth=ctx.data.truth, sel=ctx.sel)
    pool = multiprocessing.Pool(args.workers) if args.workers > 1 else None
    for budget in budgets:
        path = os.path.join(ctx.out, 'subtasks', f'B{budget}.json')
        if read_unit(path, ctx.chash) is not None:
            continue
        sel = read_unit(os.path.join(ctx.out, 'select', f'B{budget}.json'), ctx.chash)
        if sel is None:
            raise SystemExit(f'missing selection for B={budget}; run `select`')
        tasks = [(t, sel['probes']) for t in ctx.held]
        t0 = time.time()
        preds = pool.map(_subtasks_one, tasks) if pool else [_subtasks_one(t) for t in tasks]
        write_json_atomic(path, ctx.unit({
            'budget': budget, 'probes': sel['probes'],
            'questions_used': sel['questions_used'],
            'targets': ctx.held, 'pred': preds}))
        print(f'subtasks B={budget}: {len(tasks)} targets in {time.time() - t0:.0f}s', flush=True)
    if pool:
        pool.close()


# ─── shared coresets ─────────────────────────────────────────────────────────

def random_coresets(ctx, seed, budget):
    """Per-subtask random coresets (local question indices) shared by all random methods."""
    from benchpred import RandomSampling
    alloc = allocate(budget, ctx.data.sizes)
    coresets = []
    for j, q in enumerate(ctx.data.col_questions):
        source = ctx.data.answers[np.ix_(ctx.sel, q)].astype(np.float64)
        coresets.append(np.asarray(
            RandomSampling().fit(source, int(alloc[j]), seed=1000 * seed + j).get_coreset()))
    return alloc, coresets


# ─── questions: BenchPress on the model-by-question matrix ───────────────────

def cmd_questions(args):
    ctx = Context(args)
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    targets = ctx.subset(args.targets, 'heldout_models')
    data = ctx.data
    base = 100.0 * data.answers[ctx.sel].astype(np.float64)
    for seed in seeds:
        for budget in budgets:
            _, coresets = random_coresets(ctx, seed, budget)
            revealed = np.concatenate([q[c] for q, c in zip(data.col_questions, coresets)])
            for target in targets:
                path = os.path.join(ctx.out, 'questions', f's{seed}', f'B{budget}',
                                    f'm{target:03d}.json')
                if read_unit(path, ctx.chash) is not None:
                    continue
                t0 = time.time()
                row = np.full(data.answers.shape[1], np.nan)
                row[revealed] = 100.0 * data.answers[target, revealed]
                pred_row = predict(np.vstack([base, row]))[-1]
                if not np.array_equal(pred_row[revealed], row[revealed]):
                    raise AssertionError('predictor changed observed answers')
                pred = [float(pred_row[q].mean()) for q in data.col_questions]
                write_json_atomic(path, ctx.unit({
                    'seed': seed, 'budget': budget, 'target': target,
                    'questions_used': int(len(revealed)),
                    'revealed': sorted(int(i) for i in revealed), 'pred': pred}))
                print(f'questions s={seed} B={budget} m={target}: '
                      f'{time.time() - t0:.0f}s', flush=True)


# ─── estimators: Zhang, Dorner and Hardt (2025) ──────────────────────────────

def cmd_estimators(args):
    import benchpred
    commit = benchpred_commit()
    if commit != BENCHPRED_COMMIT:
        raise SystemExit(f'benchpred installed at {commit}, need {BENCHPRED_COMMIT}')
    ctx = Context(args)
    methods = args.methods or list(ESTIMATORS)
    bad = sorted(set(methods) - set(ESTIMATORS))
    if bad:
        raise SystemExit(f'unknown methods {bad}')
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    columns = args.columns if args.columns is not None else list(range(len(ctx.data.sizes)))
    data = ctx.data
    for seed in seeds:
        for budget in budgets:
            alloc, coresets = random_coresets(ctx, seed, budget)
            for j in columns:
                q = data.col_questions[j]
                source = data.answers[np.ix_(ctx.sel, q)].astype(np.float64)
                target = data.answers[np.ix_(ctx.held, q)].astype(np.float64)
                for name in methods:
                    path = os.path.join(ctx.out, 'estimators', name, f's{seed}',
                                        f'B{budget}', f'c{j:02d}.json')
                    if read_unit(path, ctx.chash) is not None:
                        continue
                    t0 = time.time()
                    est = getattr(benchpred, ESTIMATORS[name])()
                    est.fit(source, int(alloc[j]), seed=1000 * seed + j)
                    coreset = np.asarray(est.get_coreset())
                    if name != 'gp_irt' and not np.array_equal(coreset, coresets[j]):
                        raise AssertionError(f'{name} coreset differs from RandomSampling')
                    pred = 100.0 * np.asarray(est.predict(target[:, coreset]), dtype=float).ravel()
                    if pred.shape != (len(ctx.held),) or not np.isfinite(pred).all():
                        raise AssertionError(f'{name} returned shape {pred.shape}')
                    write_json_atomic(path, ctx.unit({
                        'method': name, 'seed': seed, 'budget': budget, 'column': j,
                        'allocation': int(alloc[j]),
                        'questions_used': int(len(np.unique(coreset))),
                        'coreset': coreset.tolist(), 'targets': ctx.held,
                        'pred': pred.tolist(), 'seconds': time.time() - t0}))
                    print(f'{name} s={seed} B={budget} c={j}: {time.time() - t0:.0f}s', flush=True)


# ─── summarize ───────────────────────────────────────────────────────────────

def load_predictions(ctx, method, seed, budget):
    """Return (pred[held, col], revealed[held, col], questions_used per target)."""
    n_held, n_cols = len(ctx.held), len(ctx.data.sizes)
    revealed = np.zeros((n_held, n_cols), dtype=bool)

    def need(path):
        unit = read_unit(path, ctx.chash)
        if unit is None:
            raise SystemExit(f'missing unit {path}')
        return unit

    if method == 'subtasks':
        unit = need(os.path.join(ctx.out, 'subtasks', f'B{budget}.json'))
        revealed[:, unit['probes']] = True
        return np.array(unit['pred']), revealed, [unit['questions_used']] * n_held
    if method == 'questions':
        units = [need(os.path.join(ctx.out, 'questions', f's{seed}', f'B{budget}',
                                   f'm{t:03d}.json')) for t in ctx.held]
        return (np.array([u['pred'] for u in units]), revealed,
                [u['questions_used'] for u in units])
    units = [need(os.path.join(ctx.out, 'estimators', method, f's{seed}', f'B{budget}',
                               f'c{j:02d}.json')) for j in range(n_cols)]
    used = sum(u['questions_used'] for u in units)
    return np.array([u['pred'] for u in units]).T, revealed, [used] * n_held


def cmd_summarize(args):
    ctx = Context(args, check_commit=False)
    truth = ctx.data.truth[ctx.held]
    rows = []
    for method in METHODS:
        for budget in ctx.config['budgets']:
            seeds = [None] if method == 'subtasks' else ctx.config['seeds']
            for seed in seeds:
                pred, revealed, used = load_predictions(ctx, method, seed, budget)
                pred = np.where(revealed, truth, pred)
                all_cells = compute_prediction_error(truth.ravel(), pred.ravel(),
                                                     aggregation='pool')
                hidden = ~revealed
                unrevealed = compute_prediction_error(truth[hidden], pred[hidden],
                                                      aggregation='pool')
                rows.append({
                    'method': method, 'seed': seed, 'budget': budget,
                    'medae_all': all_cells['medae'], 'n_all': all_cells['n'],
                    'medae_unrevealed': unrevealed['medae'], 'n_unrevealed': unrevealed['n'],
                    'questions_used_max': int(max(used)),
                    'questions_used_min': int(min(used)),
                })
    table = {}
    for method in METHODS:
        for budget in ctx.config['budgets']:
            sub = [r for r in rows if r['method'] == method and r['budget'] == budget]
            entry = {'n_seeds': len(sub)}
            for key in ('medae_all', 'medae_unrevealed'):
                vals = np.array([r[key] for r in sub])
                entry[f'{key}_mean'] = float(vals.mean())
                entry[f'{key}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            table.setdefault(method, {})[str(budget)] = entry
    summary = {'config_hash': ctx.chash, 'code_commit': ctx.config['code_commit'],
               'summarize_commit': args.code_commit,
               'n_heldout': len(ctx.held), 'n_columns': len(ctx.data.sizes),
               'per_seed': rows, 'table': table}
    path = os.path.join(ctx.out, 'summary.json')
    write_json_atomic(path, summary)
    for method in METHODS:
        cells = '  '.join(f'{b}:{table[method][str(b)]["medae_all_mean"]:.2f}'
                          for b in ctx.config['budgets'])
        print(f'{method:24s} {cells}')
    print(f'wrote {path}')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def int_list(text):
    return [int(x) for x in text.split(',')] if text else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    commands = {'init': cmd_init, 'select': cmd_select, 'subtasks': cmd_subtasks,
                'questions': cmd_questions, 'estimators': cmd_estimators,
                'summarize': cmd_summarize}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument('--data', default=DEFAULT_DATA)
        p.add_argument('--out-dir', default=DEFAULT_OUT)
        p.add_argument('--code-commit', required=True,
                       help='git commit of this repository that the run uses')
        if name in ('select', 'subtasks'):
            p.add_argument('--workers', type=int, default=1)
        if name in ('select', 'subtasks', 'questions', 'estimators'):
            p.add_argument('--budgets', type=int_list, default=None)
        if name in ('questions', 'estimators'):
            p.add_argument('--seeds', type=int_list, default=None)
        if name == 'questions':
            p.add_argument('--targets', type=int_list, default=None,
                           help='held-out model indices (rows of the data matrix)')
        if name == 'estimators':
            p.add_argument('--methods', type=lambda s: s.split(','), default=None)
            p.add_argument('--columns', type=int_list, default=None)
    args = ap.parse_args()
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
