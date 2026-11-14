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
  benchpress BenchPress (Logit Bias ALS, ranks 2, 4, 8, lambda 0.1) on the
             model-by-question matrix (answers 0/100): one fit per selection,
             seed, B and rank on all training models plus all held-out models,
             each held-out model revealing only its evaluated questions;
             prediction = mean of the predicted unevaluated answers
  random_sampling, random_sampling_learn, aipw, gp_irt
             estimators of Zhang, Dorner and Hardt (2025) (package `benchpred`).
             They estimate the full subtask mean f; the implied estimate on the
             unevaluated questions is (N f - sum of observed answers) / (N - n).

Subcommands (run in this order):
  init        write <out>/manifest.json (config + config_hash)
  estimators  coresets and benchpred predictions (unit per seed/B/column)
  benchpress  BenchPress predictions (unit per selection/rank/seed/B); writes
              <out>/benchpress/manifest.json, which records the base config_hash
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
BENCHPRED_REPO = 'https://github.com/socialfoundations/benchmark-prediction'
BENCHPRED_COMMIT = '5b7a23e32ee738c69b4c408cb9746429a7fd0043'
ESTIMATORS = {
    'random_sampling': 'RandomSampling',
    'random_sampling_learn': 'RandomSamplingAndLearn',
    'aipw': 'AIPWPred',
    'gp_irt': 'GPIRTPred',
}
METHODS = ['questions'] + list(ESTIMATORS)
BP_RANKS = [2, 4, 8]
BP_LAM = 0.1
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
        'coreset_seed': '1000 * seed + column',
        'allocation': 'equal split, each subtask capped at floor(size / 2)',
        'target': 'accuracy on the unevaluated questions of each subtask',
        'predictor': 'benchpress.methods.predictors.predict_benchpress_scores',
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


def load_question_predictions(ctx, path, chash, observed):
    """Read one checked question-prediction array and its unit metadata."""
    unit = read_unit(path, chash)
    mask_hash = config_hash({'targets': ctx.held,
                             'observed': [o.tolist() for o in observed]})
    if unit is None or unit['targets'] != ctx.held or unit['mask_hash'] != mask_hash:
        raise ValueError(f'missing or mismatched predictions: {path}')
    if sha256_file(path[:-5] + '.npz') != unit['question_pred_sha256']:
        raise ValueError(f'corrupt question predictions: {path}')
    revealed = np.concatenate(observed)
    with np.load(path[:-5] + '.npz') as raw:
        np.testing.assert_array_equal(raw['targets'], ctx.held)
        np.testing.assert_array_equal(raw['revealed'], revealed)
        pred = raw['pred']
    if pred.shape != (len(ctx.held), ctx.data.answers.shape[1]) or not np.isfinite(pred).all():
        raise ValueError(f'invalid question prediction shape/values: {path}')
    np.testing.assert_array_equal(
        pred[:, revealed], 100. * ctx.data.answers[np.ix_(ctx.held, revealed)])
    hidden = [np.setdiff1d(q, o) for q, o in zip(ctx.data.col_questions, observed)]
    means = np.stack([pred[:, h].mean(1) for h in hidden], axis=1)
    np.testing.assert_allclose(means, unit['pred'], rtol=0, atol=1e-12)
    return pred, unit


# ─── benchpress: one pooled fit on the model-by-question matrix ─────────────

def benchpress_manifest_path(ctx):
    return os.path.join(ctx.out, 'benchpress', 'manifest.json')


def load_benchpress_manifest(ctx):
    path = benchpress_manifest_path(ctx)
    if not os.path.exists(path):
        raise SystemExit(f'missing {path}; run `benchpress`')
    with open(path) as f:
        manifest = json.load(f)
    if config_hash(manifest['config']) != manifest['config_hash']:
        raise SystemExit(f'{path} config_hash does not match its config')
    if manifest['config']['base_config_hash'] != ctx.chash:
        raise SystemExit(f'{path} was built on another base config')
    return manifest


def benchpress_path(ctx, selection, rank, seed, budget):
    return os.path.join(ctx.out, 'benchpress', selection, f'r{rank}', f's{seed}',
                        f'B{budget}.json')


def cmd_benchpress(args):
    ctx = Context(args, check_commit=False)
    config = {
        'base_config_hash': ctx.chash,
        'code_commit': args.code_commit,
        'predictor': 'benchpress.methods.predictors.predict_logit_bias_als_scores',
        'ranks': BP_RANKS,
        'lam': BP_LAM,
        'fit': 'one fit per selection, seed, B and rank on all training models '
               '(all answers) and all held-out models (evaluated questions only)',
    }
    manifest = {'config': config, 'config_hash': config_hash(config)}
    if os.path.exists(benchpress_manifest_path(ctx)):
        if load_benchpress_manifest(ctx)['config_hash'] != manifest['config_hash']:
            raise SystemExit(f'{benchpress_manifest_path(ctx)} exists with another config')
    else:
        write_json_atomic(benchpress_manifest_path(ctx), manifest)
    bp_hash = manifest['config_hash']
    selections = ctx.subset(args.selections, 'selections')
    seeds = ctx.subset(args.seeds, 'seeds')
    budgets = ctx.subset(args.budgets, 'budgets')
    ranks = BP_RANKS if args.ranks is None else args.ranks
    if set(ranks) - set(BP_RANKS):
        raise SystemExit(f'ranks {ranks} not in {BP_RANKS}')
    data = ctx.data
    train = 100.0 * data.answers[ctx.sel].astype(np.float64)
    n_train = len(ctx.sel)
    for seed in seeds:
        for budget in budgets:
            coresets, _ = load_coresets(ctx, seed, budget)
            for selection in selections:
                observed = [q[c] for q, c in zip(data.col_questions, coresets[selection])]
                hidden = [np.setdiff1d(q, o) for q, o in zip(data.col_questions, observed)]
                revealed = np.concatenate(observed)
                held = np.full((len(ctx.held), data.answers.shape[1]), np.nan)
                held[:, revealed] = 100.0 * data.answers[np.ix_(ctx.held, revealed)]
                matrix = np.vstack([train, held])
                for rank in ranks:
                    path = benchpress_path(ctx, selection, rank, seed, budget)
                    if read_unit(path, bp_hash) is not None:
                        continue
                    t0 = time.time()
                    pred = predict(matrix, rank, BP_LAM)[n_train:]
                    if not np.array_equal(pred[:, revealed], held[:, revealed]):
                        raise AssertionError('predictor changed observed answers')
                    write_json_atomic(path, {
                        'selection': selection, 'seed': seed, 'budget': budget,
                        'rank': rank, 'lam': BP_LAM, 'targets': ctx.held,
                        'questions_used': int(len(revealed)),
                        'pred': [[float(p[h].mean()) for h in hidden] for p in pred],
                        'seconds': time.time() - t0,
                        'config_hash': bp_hash, 'code_commit': args.code_commit})
                    print(f'benchpress {selection} r={rank} s={seed} B={budget}: '
                          f'{time.time() - t0:.0f}s', flush=True)


# ─── summarize ───────────────────────────────────────────────────────────────

def cmd_summarize(args):
    ctx = Context(args, check_commit=False)
    bp = load_benchpress_manifest(ctx)
    methods = [f'benchpress_r{r}' for r in bp['config']['ranks']] + list(ESTIMATORS)
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
                for method in methods:
                    if method.startswith('benchpress_r'):
                        path = benchpress_path(ctx, selection, int(method[12:]), seed, budget)
                        unit = read_unit(path, bp['config_hash'])
                        if unit is None:
                            raise SystemExit(f'missing unit {path}')
                        if unit['targets'] != ctx.held:
                            raise AssertionError(f'{path} targets differ from held-out models')
                        pred = np.array(unit['pred'])
                    else:
                        pred = np.array([u['pred'][selection][method] for u in units]).T
                    err = compute_prediction_error(truth.ravel(), pred.ravel(), aggregation='pool')
                    rows.append({'selection': selection, 'method': method, 'seed': seed,
                                 'budget': budget, 'medae': err['medae'], 'n': err['n'],
                                 'questions_used': used})
    table = {}
    for selection in SELECTIONS:
        for method in methods:
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
               'benchpress_config': bp['config'], 'benchpress_config_hash': bp['config_hash'],
               'summarize_commit': args.code_commit,
               'n_heldout': len(ctx.held), 'n_columns': len(data.sizes),
               'per_seed': rows, 'table': table}
    path = os.path.join(ctx.out, 'summary.json')
    write_json_atomic(path, summary)
    for selection in SELECTIONS:
        for method in methods:
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
    commands = {'init': cmd_init, 'estimators': cmd_estimators,
                'benchpress': cmd_benchpress, 'summarize': cmd_summarize}
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument('--data', default=DEFAULT_DATA)
        p.add_argument('--out-dir', default=DEFAULT_OUT)
        p.add_argument('--code-commit', required=True,
                       help='git commit of this repository that the run uses')
        if name in ('benchpress', 'estimators'):
            p.add_argument('--seeds', type=int_list, default=None)
            p.add_argument('--budgets', type=int_list, default=None)
        if name == 'benchpress':
            p.add_argument('--selections', type=lambda s: s.split(','), default=None)
            p.add_argument('--ranks', type=int_list, default=None)
        if name == 'estimators':
            p.add_argument('--columns', type=int_list, default=None)
    args = ap.parse_args()
    commands[args.cmd](args)


if __name__ == '__main__':
    main()
