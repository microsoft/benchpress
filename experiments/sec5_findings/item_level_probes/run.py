#!/usr/bin/env python
"""Question-level budgets on the Open LLM Leaderboard v2 (Appendix question_level_budgets).

Each held-out model is evaluated on B questions and every method predicts its
accuracy (percent) on the questions it was not evaluated on, per subtask.

Selections (which B questions are evaluated; shared by all predictors):
  random     per subtask, a uniform random subset
  anchors    per subtask, the GP-IRT anchor questions of tinyBenchmarks
B is split equally across the 37 subtasks, each capped at half its questions;
leftover budget goes to uncapped subtasks.

Predictors (all see the same evaluated questions, all graded on the same
unevaluated questions):
  questions  BenchPress (Logit Bias ALS) on the model-by-question matrix
             (answers 0/100); prediction = mean of the predicted unevaluated
             answers. Rank and lambda are chosen per selection and B by
             validation on training models only (see `tune`).
  random_sampling, random_sampling_learn, aipw, gp_irt
             estimators of Zhang, Dorner and Hardt (2025) (package `benchpred`).
             They estimate the full subtask mean f; the implied estimate on the
             unevaluated questions is (N f - sum of observed answers) / (N - n).

Subcommands (run in this order):
  init        write <out>/manifest.json (config + config_hash)
  estimators  coresets and benchpred predictions (unit per seed/B/column)
  tune        BenchPress on validation models drawn from the training models,
              one fit per grid point (unit per selection/B/rank/lambda/model)
  select      rank and lambda with the lowest validation MedAE per selection/B
  questions   BenchPress predictions (unit per selection/seed/B/target)
  summarize   MedAE over the 127 x 37 held-out cells per selection

Every unit is written atomically as JSON carrying the config_hash; existing
units are skipped, units from another config fail fast.
"""

import argparse
import hashlib
import json
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
from benchpress.methods.predictors import predict_logit_bias_als_scores  # noqa: E402

PCT_SPEC = {'type': 'pct', 'range': [0, 100]}
BUDGETS = [200, 400, 800, 1600, 3200, 6400]
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
RANKS = [2, 4, 8, 16, 32]
LAMS = [0.1, 1.0, 10.0]
TUNE_SEED = 0
N_TUNE_MODELS = 24
BENCHPRED_REPO = 'https://github.com/socialfoundations/benchmark-prediction'
BENCHPRED_COMMIT = '5b7a23e32ee738c69b4c408cb9746429a7fd0043'
ESTIMATORS = {
    'random_sampling': 'RandomSampling',
    'random_sampling_learn': 'RandomSamplingAndLearn',
    'aipw': 'AIPWPred',
    'gp_irt': 'GPIRTPred',
}
METHODS = ['questions'] + list(ESTIMATORS)
SELECTIONS = ['random', 'anchors']
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


def tune_models(train_models):
    """Validation models for tuning, drawn from the training models only."""
    perm = np.random.RandomState(SPLIT_SEED + 1).permutation(len(train_models))
    return sorted(int(train_models[i]) for i in perm[:N_TUNE_MODELS])


def allocate(budget, sizes):
    """Equal split of budget across subtasks, each capped at half its questions."""
    caps = np.asarray(sizes) // 2
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


def predict(matrix, rank, lam):
    ids = [f'c{j}' for j in range(matrix.shape[1])]
    pred = predict_logit_bias_als_scores(
        matrix, rank=rank, lam=lam, metric={i: PCT_SPEC for i in ids}, benchmark_ids=ids)
    if not np.isfinite(pred).all():
        raise AssertionError('BenchPress returned non-finite predictions')
    return pred


def predict_hidden(data, base, target, observed, hidden, rank, lam):
    """BenchPress mean prediction on each subtask's hidden questions for one target.

    base: (n, Q) answers of the other models (0/100); observed, hidden: per-subtask
    global question indices. Only the target's observed answers are revealed.
    """
    revealed = np.concatenate(observed)
    row = np.full(data.answers.shape[1], np.nan)
    row[revealed] = 100.0 * data.answers[target, revealed]
    pred_row = predict(np.vstack([base, row]), rank, lam)[-1]
    if not np.array_equal(pred_row[revealed], row[revealed]):
        raise AssertionError('predictor changed observed answers')
    return [float(pred_row[h].mean()) for h in hidden], int(len(revealed))


# ─── Context shared by subcommands ───────────────────────────────────────────

class Context:
    def __init__(self, args, need_manifest=True, check_commit=True):
        self.out = os.path.abspath(args.out_dir)
        self.data_path = os.path.abspath(args.data)
        self.data = Data(self.data_path)
        self.sel, self.held = split_models(len(self.data.model_ids))
        self.tune = tune_models(self.sel)
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
        if self.tune != self.config['tune_models']:
            raise SystemExit('tune models differ from manifest')

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
        'coreset_seed': '1000 * seed + column',
        'allocation': 'equal split, each subtask capped at floor(size / 2)',
        'target': 'accuracy on the unevaluated questions of each subtask',
        'predictor': 'benchpress.methods.predictors.predict_logit_bias_als_scores',
        'tune_ranks': RANKS,
        'tune_lams': LAMS,
        'tune_seed': TUNE_SEED,
        'tune_rule': 'lowest MedAE on the tune models per selection and budget; '
                     'ties to the smaller rank, then the smaller lambda',
        'tune_models': ctx.tune,
        'selections': SELECTIONS,
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


# ─── estimators: coresets and Zhang, Dorner and Hardt (2025) predictors ──────

def hidden_estimate(full_mean, observed, n_total):
    """Estimate on the unevaluated questions implied by a full-subtask estimate.

    full_mean: (n_targets,) estimate of the mean over all n_total questions.
    observed: (n_targets, n) answers on the n evaluated questions (n < n_total).
    """
    return (n_total * full_mean - observed.sum(1)) / (n_total - observed.shape[1])


def estimator_unit(source, target, k, seed):
    """Coresets of both selections and hidden-question predictions of every estimator."""
    import benchpred
    from sklearn.linear_model import RidgeCV

    gp = benchpred.GPIRTPred().fit(source, k, seed=seed)
    anchor_points = np.asarray(gp.anchor_points)
    coresets = {'random': np.asarray(benchpred.RandomSampling().fit(source, k, seed=seed).get_coreset()),
                'anchors': np.unique(anchor_points)}
    pred = {s: {} for s in SELECTIONS}
    for selection in SELECTIONS:
        c = coresets[selection]
        observed = target[:, c]
        for name in ESTIMATORS:
            if name == 'gp_irt':
                if selection == 'random':
                    # GP-IRT prediction with the random questions as equally weighted anchors.
                    own = gp.anchor_points, gp.anchor_weights
                    gp.anchor_points, gp.anchor_weights = c, np.full(len(c), 1.0 / len(c))
                    full = gp.predict(observed)
                    gp.anchor_points, gp.anchor_weights = own
                else:
                    full = gp.predict(target[:, anchor_points])
            else:
                est = getattr(benchpred, ESTIMATORS[name])().fit(source, k, seed=seed)
                if not np.array_equal(est.get_coreset(), coresets['random']):
                    raise AssertionError(f'{name} coreset differs from RandomSampling')
                if selection == 'anchors' and name != 'random_sampling':
                    est.compressed_data_indices = c
                    if name == 'random_sampling_learn':
                        # Refit as RandomSamplingAndLearn.fit does, on the given coreset.
                        est.rgs = RidgeCV().fit(source[:, c], source.mean(1))
                full = est.predict(observed)
            full = np.asarray(full, dtype=float).ravel()
            h = 100.0 * hidden_estimate(full, observed, source.shape[1])
            if h.shape != (target.shape[0],) or not np.isfinite(h).all():
                raise AssertionError(f'{name}/{selection} returned shape {h.shape}')
            pred[selection][name] = h.tolist()
    return coresets, anchor_points, pred


def estimator_path(ctx, seed, budget, j):
    return os.path.join(ctx.out, 'estimators', f's{seed}', f'B{budget}', f'c{j:02d}.json')


def cmd_estimators(args):
    commit = benchpred_commit()
    if commit != BENCHPRED_COMMIT:
        raise SystemExit(f'benchpred installed at {commit}, need {BENCHPRED_COMMIT}')
    ctx = Context(args)
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    columns = args.columns if args.columns is not None else list(range(len(ctx.data.sizes)))
    data = ctx.data
    for seed in seeds:
        for budget in budgets:
            alloc = allocate(budget, data.sizes)
            for j in columns:
                path = estimator_path(ctx, seed, budget, j)
                if read_unit(path, ctx.chash) is not None:
                    continue
                t0 = time.time()
                q = data.col_questions[j]
                source = data.answers[np.ix_(ctx.sel, q)].astype(np.float64)
                target = data.answers[np.ix_(ctx.held, q)].astype(np.float64)
                coresets, anchor_points, pred = estimator_unit(
                    source, target, int(alloc[j]), 1000 * seed + j)
                write_json_atomic(path, ctx.unit({
                    'seed': seed, 'budget': budget, 'column': j,
                    'allocation': int(alloc[j]),
                    'coresets': {s: c.tolist() for s, c in coresets.items()},
                    'gp_irt_anchor_points': anchor_points.tolist(),
                    'targets': ctx.held, 'pred': pred, 'seconds': time.time() - t0}))
                print(f'estimators s={seed} B={budget} c={j}: {time.time() - t0:.0f}s', flush=True)


def load_coresets(ctx, seed, budget):
    """{selection: [local question indices per column]} from the estimator units."""
    units = []
    for j in range(len(ctx.data.sizes)):
        unit = read_unit(estimator_path(ctx, seed, budget, j), ctx.chash)
        if unit is None:
            raise SystemExit(f'missing {estimator_path(ctx, seed, budget, j)}; run `estimators`')
        units.append(unit)
    return {s: [np.asarray(u['coresets'][s], dtype=int) for u in units] for s in SELECTIONS}, units


# ─── BenchPress on the model-by-question matrix ──────────────────────────────

def split_questions(data, coresets):
    """Per-subtask global indices of the evaluated and unevaluated questions."""
    observed = [q[c] for q, c in zip(data.col_questions, coresets)]
    hidden = [np.setdiff1d(q, o) for q, o in zip(data.col_questions, observed)]
    return observed, hidden


def tune_path(ctx, selection, budget, rank, lam, model):
    return os.path.join(ctx.out, 'tune', selection, f'B{budget}', f'r{rank}_l{lam:g}',
                        f'm{model:03d}.json')


def cmd_tune(args):
    ctx = Context(args)
    selections = ctx.subset(args.selections, 'selections')
    budgets = ctx.subset(args.budgets, 'budgets')
    ranks = ctx.subset(args.ranks, 'tune_ranks')
    lams = ctx.subset(args.lams, 'tune_lams')
    models = ctx.subset(args.targets, 'tune_models')
    data = ctx.data
    for budget in budgets:
        coresets, _ = load_coresets(ctx, ctx.config['tune_seed'], budget)
        for selection in selections:
            observed, hidden = split_questions(data, coresets[selection])
            for model in models:
                others = [m for m in ctx.sel if m != model]
                base = 100.0 * data.answers[others].astype(np.float64)
                for rank in ranks:
                    for lam in lams:
                        path = tune_path(ctx, selection, budget, rank, lam, model)
                        if read_unit(path, ctx.chash) is not None:
                            continue
                        t0 = time.time()
                        pred, used = predict_hidden(data, base, model, observed, hidden, rank, lam)
                        write_json_atomic(path, ctx.unit({
                            'selection': selection, 'budget': budget, 'rank': rank,
                            'lam': lam, 'target': model, 'questions_used': used,
                            'pred': pred}))
                        print(f'tune {selection} B={budget} r={rank} l={lam:g} m={model}: '
                              f'{time.time() - t0:.0f}s', flush=True)


def selection_path(ctx):
    return os.path.join(ctx.out, 'tune_selection.json')


def cmd_select(args):
    ctx = Context(args)
    data = ctx.data
    cfg = ctx.config
    chosen, grid = {}, []
    for budget in cfg['budgets']:
        coresets, _ = load_coresets(ctx, cfg['tune_seed'], budget)
        for selection in SELECTIONS:
            observed, hidden = split_questions(data, coresets[selection])
            truth = np.array([[100.0 * data.answers[m, h].mean() for h in hidden]
                              for m in cfg['tune_models']])
            best = None
            for rank in cfg['tune_ranks']:
                for lam in cfg['tune_lams']:
                    pred = []
                    for m in cfg['tune_models']:
                        unit = read_unit(tune_path(ctx, selection, budget, rank, lam, m), ctx.chash)
                        if unit is None:
                            raise SystemExit(f'missing {tune_path(ctx, selection, budget, rank, lam, m)}')
                        pred.append(unit['pred'])
                    medae = compute_prediction_error(
                        truth.ravel(), np.array(pred).ravel(), aggregation='pool')['medae']
                    grid.append({'selection': selection, 'budget': budget, 'rank': rank,
                                 'lam': lam, 'medae': medae})
                    if best is None or medae < best['medae']:
                        best = {'rank': rank, 'lam': lam, 'medae': medae}
            chosen.setdefault(selection, {})[str(budget)] = best
            print(f'{selection:8s} B={budget}: rank {best["rank"]}, lambda {best["lam"]:g}, '
                  f'tune MedAE {best["medae"]:.2f}')
    write_json_atomic(selection_path(ctx), ctx.unit({'chosen': chosen, 'grid': grid}))
    print(f'wrote {selection_path(ctx)}')


def load_selected(ctx):
    sel = read_unit(selection_path(ctx), ctx.chash)
    if sel is None:
        raise SystemExit(f'missing {selection_path(ctx)}; run `tune` and `select`')
    return sel['chosen']


def cmd_questions(args):
    ctx = Context(args)
    selections = ctx.subset(args.selections, 'selections')
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    targets = ctx.subset(args.targets, 'heldout_models')
    chosen = load_selected(ctx)
    data = ctx.data
    base = 100.0 * data.answers[ctx.sel].astype(np.float64)
    for seed in seeds:
        for budget in budgets:
            coresets, _ = load_coresets(ctx, seed, budget)
            for selection in selections:
                observed, hidden = split_questions(data, coresets[selection])
                rank, lam = (chosen[selection][str(budget)][k] for k in ('rank', 'lam'))
                for target in targets:
                    path = os.path.join(ctx.out, 'questions', selection, f's{seed}',
                                        f'B{budget}', f'm{target:03d}.json')
                    if read_unit(path, ctx.chash) is not None:
                        continue
                    t0 = time.time()
                    pred, used = predict_hidden(data, base, target, observed, hidden, rank, lam)
                    write_json_atomic(path, ctx.unit({
                        'selection': selection, 'seed': seed, 'budget': budget,
                        'target': target, 'rank': rank, 'lam': lam,
                        'questions_used': used, 'pred': pred}))
                    print(f'questions {selection} s={seed} B={budget} m={target}: '
                          f'{time.time() - t0:.0f}s', flush=True)


# ─── summarize ───────────────────────────────────────────────────────────────

def cmd_summarize(args):
    ctx = Context(args, check_commit=False)
    data = ctx.data
    rows = []
    for seed in ctx.config['seeds']:
        for budget in ctx.config['budgets']:
            coresets, units = load_coresets(ctx, seed, budget)
            for selection in SELECTIONS:
                hidden = [np.setdiff1d(np.arange(len(q)), c)
                          for q, c in zip(data.col_questions, coresets[selection])]
                truth = np.stack([100.0 * data.answers[np.ix_(ctx.held, q[h])].mean(1)
                                  for q, h in zip(data.col_questions, hidden)], axis=1)
                used = int(sum(len(c) for c in coresets[selection]))
                for method in METHODS:
                    if method == 'questions':
                        qunits = []
                        for t in ctx.held:
                            path = os.path.join(ctx.out, 'questions', selection, f's{seed}',
                                                f'B{budget}', f'm{t:03d}.json')
                            unit = read_unit(path, ctx.chash)
                            if unit is None:
                                raise SystemExit(f'missing unit {path}')
                            qunits.append(unit)
                        pred = np.array([u['pred'] for u in qunits])
                    else:
                        pred = np.array([u['pred'][selection][method] for u in units]).T
                    err = compute_prediction_error(truth.ravel(), pred.ravel(), aggregation='pool')
                    rows.append({'selection': selection, 'method': method, 'seed': seed,
                                 'budget': budget, 'medae': err['medae'], 'n': err['n'],
                                 'questions_used': used})
    table = {}
    for selection in SELECTIONS:
        for method in METHODS:
            for budget in ctx.config['budgets']:
                sub = [r for r in rows if (r['selection'], r['method'], r['budget'])
                       == (selection, method, budget)]
                vals = np.array([r['medae'] for r in sub])
                table.setdefault(selection, {}).setdefault(method, {})[str(budget)] = {
                    'n_seeds': len(sub), 'medae_mean': float(vals.mean()),
                    'medae_std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    'n_cells': sub[0]['n'],
                    'questions_used_mean': float(np.mean([r['questions_used'] for r in sub]))}
    summary = {'config_hash': ctx.chash, 'code_commit': ctx.config['code_commit'],
               'summarize_commit': args.code_commit,
               'n_heldout': len(ctx.held), 'n_columns': len(data.sizes),
               'benchpress_selected': load_selected(ctx),
               'per_seed': rows, 'table': table}
    path = os.path.join(ctx.out, 'summary.json')
    write_json_atomic(path, summary)
    for selection in SELECTIONS:
        for method in METHODS:
            cells = '  '.join(f'{b}:{table[selection][method][str(b)]["medae_mean"]:.2f}'
                              for b in ctx.config['budgets'])
            print(f'{selection:8s} {method:24s} {cells}')
    print(f'wrote {path}')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def int_list(text):
    return [int(x) for x in text.split(',')] if text else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    commands = {'init': cmd_init, 'estimators': cmd_estimators, 'tune': cmd_tune,
                'select': cmd_select, 'questions': cmd_questions, 'summarize': cmd_summarize}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument('--data', default=DEFAULT_DATA)
        p.add_argument('--out-dir', default=DEFAULT_OUT)
        p.add_argument('--code-commit', required=True,
                       help='git commit of this repository that the run uses')
        if name in ('questions', 'estimators'):
            p.add_argument('--seeds', type=int_list, default=None)
        if name in ('questions', 'estimators', 'tune'):
            p.add_argument('--budgets', type=int_list, default=None)
        if name in ('questions', 'tune'):
            p.add_argument('--selections', type=lambda s: s.split(','), default=None)
            p.add_argument('--targets', type=int_list, default=None,
                           help='model indices (rows of the data matrix): held-out '
                                'models for questions, tune models for tune')
        if name == 'tune':
            p.add_argument('--ranks', type=int_list, default=None)
            p.add_argument('--lams', type=lambda s: [float(x) for x in s.split(',')],
                           default=None)
        if name == 'estimators':
            p.add_argument('--columns', type=int_list, default=None)
    args = ap.parse_args()
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
