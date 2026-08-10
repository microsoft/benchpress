"""BenchPress and observational scaling laws on OSL's own setup (nxZ2 Q1).

Reviewer nxZ2 asks whether the paper's contribution is already covered by
Observational Scaling Laws (OSL), pointing at its Sec. 4.2, and separately asks
how our prediction recipe compares to it. The cleanest answer runs both
predictors on OSL's released data, splits, and targets, so nothing about the
comparison depends on how we set up our own matrix.

OSL's prediction task is a score matrix with a structured hole: rows are models,
the base benchmark block is observed for everyone, and one target column is
observed only for the training models. Our predictor takes that matrix as is,
with no change to the method, which is the point of the comparison.

Three tasks are reproduced, one per OSL experiment, with the per target settings
transcribed from the notebooks in their repository rather than guessed: the
metric range passed to their `minmax_norm` (the random-guess floor and the
perfect-prediction ceiling, which is 0.25 for Persian QA and 100 for BLEU), the
exclusion of GSM8K from the predictor block whenever it would predict itself,
and the lower FLOPs cutoff they use for arithmetic.

- `emergent` (their Sec. 4.1): BigBench targets, split at a compute cutoff, so
  the models being predicted are all stronger than every model fitted on.
- `agentic` (their Sec. 4.2): the AgentBench and AgentBoard aggregate scores the
  reviewer points at, holding out the top decile.
- `post_training` (their Sec. 4.3): GSM8K under several prompting strategies,
  split at the same compute cutoff.

Targets the notebooks report are marked `headline`; targets that sit in the same
notebooks but are commented out as additional tasks are still run, and pooled
separately, so the choice of target set is visible rather than silent.

Three predictors are compared on identical held-out cells:

- `osl`: the reimplementation in `observational_scaling.py`.
- `benchpress`: the paper's fixed point predictor, rank-2 logit Bias ALS with
  lambda 0.1, run on the `[base block | target]` matrix with the test rows of
  the target column hidden.
- `compute`: OSL's own baseline, the same sigmoid regression fitted on
  log10 training FLOPs instead of the capability measures. It is included as a
  correctness gate: OSL reports that its capability measures beat this baseline,
  so a run where they do not would indicate the reimplementation is wrong.
- `median`: the no-information baseline, each held-out cell predicted by the
  median of the training scores in its column. Under a weak-to-strong split
  every held-out model is stronger than every training model, so this baseline
  is guaranteed to underpredict and marks the floor of the comparison.

`benchpress_joint` is also reported. It hides every target column sharing a
predictor block at once and completes them in a single fit, which is not
expressible in OSL's recipe because each target there needs its own regression.

Usage:
    python obs_scaling_setup.py [--smoke] [--out obs_scaling_setup.json]
"""
import argparse
import hashlib
import json
import os
import urllib.request

import numpy as np
import pandas as pd

from benchpress.evaluation_harness import compute_prediction_error
from benchpress.methods.predictors import (
    predict_benchmark_median_scores, predict_benchpress_scores,
)

from observational_scaling import ObservationalScalingPredictor, SigmoidCapabilityRegression

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'osl_data')
DATA_URL = 'https://raw.githubusercontent.com/ryoungj/ObsScaling/main/eval_results/{}.csv'

# OSL's base benchmark block; XWinograd and Winograd differ between the two files.
BASE_FEATURES = ['MMLU', 'ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA',
                 'GSM8K', 'XWinograd', 'HumanEval']
# OSL drops GSM8K from the predictor block whenever the target is arithmetic or
# is itself GSM8K, so that GSM8K does not predict GSM8K.
NONGSM_BASE_FEATURES = [f for f in BASE_FEATURES if f != 'GSM8K']
# `USED_INSTRUCT_LLM_METRICS` in their agent notebook; Arena-Elo and MTBench are
# present in the file but deliberately excluded.
INSTRUCT_FEATURES = ['MMLU', 'ARC-C', 'HellaSwag', 'Winogrande', 'TruthfulQA',
                     'GSM8K', 'HumanEval']

# Llama-2-7B training compute, OSL's default weak/strong cutoff, in units of 1E21
# FLOPs. Arithmetic targets use a lower cutoff, which OSL sets to make the
# extrapolation harder.
FLOPS_CUTOFF = 84.0
ARITHMETIC_FLOPS_CUTOFF = 22.0
AGENTIC_TEST_FRACTION = 0.1

# The completion pipeline expects columns on a [0, 100] scale, and OSL's sigmoid
# needs the metric mapped to [0, 1]. Both are served by rescaling each column
# with the same `metric_range` OSL passes to its own `minmax_norm`: the metric
# value of random guessing and the metric value of a perfect prediction, in the
# raw units of the released CSV. Errors are reported back in raw units.
FEATURE_RANGE = (0.0, 1.0)


def _target(name, metric_range, features, cutoff=FLOPS_CUTOFF, headline=True):
    return {'name': name, 'metric_range': metric_range, 'features': features,
            'cutoff': cutoff, 'headline': headline}


# Transcribed from OSL's released notebooks. `headline` marks the targets they
# report; the remainder appear in the same notebooks but are commented out as
# "additional tasks", so they are pooled separately rather than dropped.
TASKS = {
    'emergent': {
        'osl_section': '4.1',
        'feature_file': 'base_llm_benchmark_eval',
        'target_file': 'base_llm_emergent_capability_eval',
        'split': 'compute_cutoff',
        'targets': [
            _target('arithmetic_3ds_2_acc', (0.0, 1.0), NONGSM_BASE_FEATURES,
                    ARITHMETIC_FLOPS_CUTOFF),
            _target('arithmetic_2dm_2_acc', (0.0, 1.0), NONGSM_BASE_FEATURES,
                    ARITHMETIC_FLOPS_CUTOFF),
            _target('word_unscrambling_2_exact_match', (0.0, 1.0), BASE_FEATURES),
            # Persian multiple choice; OSL sets the floor at the random-guess rate.
            _target('parsinlu_qa_2_acc', (0.25, 1.0), BASE_FEATURES),
            _target('arithmetic_3da_2_acc', (0.0, 1.0), NONGSM_BASE_FEATURES,
                    ARITHMETIC_FLOPS_CUTOFF, headline=False),
            _target('arithmetic_2da_2_acc', (0.0, 1.0), NONGSM_BASE_FEATURES,
                    ARITHMETIC_FLOPS_CUTOFF, headline=False),
            # BLEU is released on a 0 to 100 scale, unlike every accuracy column.
            _target('ipa_transliterate_2_bleu', (0.0, 100.0), BASE_FEATURES,
                    headline=False),
        ],
    },
    'agentic': {
        'osl_section': '4.2',
        'feature_file': 'instruct_llm_benchmark_eval',
        'target_file': 'instruct_llm_agent_eval',
        'split': 'top_performers',
        'targets': [
            # AgentBench "Overall Score", a weighted average OSL normalises by 10.
            _target('ABench-OA', (0.0, 10.0), INSTRUCT_FEATURES),
            # AgentBoard "Average Success Rate", released as a percentage.
            _target('ABoard-Avg_SR', (0.0, 100.0), INSTRUCT_FEATURES),
        ],
    },
    'post_training': {
        'osl_section': '4.3',
        'feature_file': 'base_llm_benchmark_eval',
        'target_file': 'base_llm_post_training_eval',
        'split': 'compute_cutoff',
        'targets': [
            _target('gsm8k_5_exact_match,flexible-extract', (0.0, 1.0),
                    NONGSM_BASE_FEATURES),
            _target('gsm8k_cot_8_exact_match,flexible-extract', (0.0, 1.0),
                    NONGSM_BASE_FEATURES),
            _target('gsm8k_cot_self_consistency_8_exact_match,maj@5-flexible-extract',
                    (0.0, 1.0), NONGSM_BASE_FEATURES),
            _target('gsm8k_cot_zeroshot_0_exact_match,flexible-extract', (0.0, 1.0),
                    NONGSM_BASE_FEATURES, headline=False),
        ],
    },
}

MIN_TRAIN_MODELS = 8
# OSL's agentic split holds out the top decile, which is two models on
# AgentBench and one on AgentBoard, so a floor above one would silently drop
# the very experiment under discussion. The held-out count is reported per
# target instead, since it is what makes those results fragile.
MIN_TEST_MODELS = 1


def load_csv(name):
    """Download OSL's released CSV once, then read it from the local cache."""
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    path = os.path.join(DATA_DIR, name + '.csv')
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL.format(name), path)
    with open(path, 'rb') as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return pd.read_csv(path), digest


def build_task(spec):
    """Join the feature block with the target file and drop unusable rows."""
    features, feature_digest = load_csv(spec['feature_file'])
    targets, target_digest = load_csv(spec['target_file'])
    available = [t for t in spec['targets'] if t['name'] in targets.columns]
    if not available:
        raise ValueError('none of the configured target columns exist')

    names = [t['name'] for t in available]
    used = sorted({f for t in available for f in t['features']})
    merged = features.merge(targets[['Model'] + names], on='Model', how='inner')
    # OSL requires a complete base block per model; rows missing it are excluded
    # from its analysis, so every predictor sees the same model set.
    merged = merged[merged[used].notna().all(axis=1)]
    if spec['split'] == 'compute_cutoff':
        merged = merged[merged['FLOPs (1E21)'].notna()]
    return merged.reset_index(drop=True), available, {
        'feature_file_sha256': feature_digest,
        'target_file_sha256': target_digest,
    }


def split_models(frame, spec, target):
    """Reproduce OSL's held-out split for one target column.

    The compute-cutoff tasks train on models at or below the target's own FLOPs
    cutoff and predict the rest. The agentic task has no compute figures for
    proprietary models, so OSL instead holds out the top decile by target score;
    this reproduces that, which recovers GPT-4 and Claude-2 on AgentBench and
    GPT-4 on AgentBoard, the held-out models the paper names.
    """
    observed = frame[target['name']].notna().values
    if spec['split'] == 'compute_cutoff':
        weak = frame['FLOPs (1E21)'].values <= target['cutoff']
        train = observed & weak
        test = observed & ~weak
    else:
        scores = frame[target['name']].values
        n_test = max(1, int(round(AGENTIC_TEST_FRACTION * observed.sum())))
        ranked = np.argsort(np.where(observed, -scores, np.inf), kind='stable')
        test = np.zeros(len(frame), dtype=bool)
        test[ranked[:n_test]] = True
        train = observed & ~test
    return train, test


def to_unit_scale(values, metric_range):
    """Map raw metric values onto the [0, 100] scale the predictors expect."""
    lo, hi = metric_range
    return (np.asarray(values, dtype=float) - lo) / (hi - lo) * 100.0


def to_raw_scale(values, metric_range):
    """Invert `to_unit_scale`, so errors are reported in the released units."""
    lo, hi = metric_range
    return np.asarray(values, dtype=float) / 100.0 * (hi - lo) + lo


def complete_targets(base_block, target_columns, hidden, predictor):
    """Complete `[base block | targets]` with a whole-matrix score predictor.

    Args:
        base_block: (n_models, n_features) array on a [0, 100] scale.
        target_columns: (n_models, n_targets) array on a [0, 100] scale.
        hidden: boolean mask of the same shape as `target_columns`; True cells
            are removed before completion and predicted afterwards.
        predictor: callable mapping a training matrix to a completed matrix.
    """
    matrix = np.column_stack([base_block, target_columns])
    n_features = base_block.shape[1]
    observed = matrix.copy()
    observed[:, n_features:][hidden] = np.nan
    return predictor(observed)[:, n_features:]


def compute_baseline_predict(flops, train, test, unit_values):
    """OSL's compute scaling law baseline: sigmoid regression on log10 FLOPs."""
    x = np.log10(np.asarray(flops, dtype=float) * 1e21).reshape(-1, 1)
    model = SigmoidCapabilityRegression().fit(x[train], unit_values[train] / 100.0)
    return model.predict(x[test]) * 100.0


def run_task(name, spec, smoke=False):
    frame, targets, digests = build_task(spec)
    if smoke:
        targets = targets[:1]

    n_models = len(frame)
    # Every predictor column lives on a [0, 100] scale: feature accuracies are
    # released as fractions, and each target is mapped through its own metric
    # range. Predictions are mapped back before any error is computed.
    unit_targets = np.column_stack(
        [to_unit_scale(frame[t['name']].values, t['metric_range']) for t in targets]
    ) if targets else np.zeros((n_models, 0))
    hidden = np.zeros(unit_targets.shape, dtype=bool)
    usable, per_target, raw = [], [], {}

    for k, target in enumerate(targets):
        name_k = target['name']
        train, test = split_models(frame, spec, target)
        if train.sum() < MIN_TRAIN_MODELS or test.sum() < MIN_TEST_MODELS:
            per_target.append({
                'target': name_k, 'skipped': 'too few train or test models',
                'n_train': int(train.sum()), 'n_test': int(test.sum()),
            })
            continue
        usable.append(k)
        hidden[test, k] = True
        block = to_unit_scale(frame[target['features']].values, FEATURE_RANGE)
        truth = frame[name_k].values[test]

        def raw_pred(unit_pred):
            return to_raw_scale(unit_pred, target['metric_range'])

        osl_pred = raw_pred(ObservationalScalingPredictor(
            metric_range=(0.0, 100.0)).fit_predict(
                block[train], unit_targets[train, k], block[test],
        ))
        single = raw_pred(complete_targets(
            block, unit_targets[:, [k]], hidden[:, [k]], predict_benchpress_scores,
        )[test, 0])
        median = raw_pred(complete_targets(
            block, unit_targets[:, [k]], hidden[:, [k]],
            predict_benchmark_median_scores,
        )[test, 0])

        methods = {
            'osl': compute_prediction_error(truth, osl_pred),
            'benchpress': compute_prediction_error(truth, single),
            'median': compute_prediction_error(truth, median),
        }
        raw['truth__' + name_k] = truth
        raw['osl__' + name_k] = osl_pred
        raw['benchpress__' + name_k] = single
        raw['median__' + name_k] = median
        raw['test_models__' + name_k] = frame['Model'].values[test]

        # OSL notes that training compute is unpublished for the proprietary
        # models, so its own compute baseline does not apply to every task.
        flops = frame['FLOPs (1E21)'].values
        if np.isfinite(flops[train]).all() and np.isfinite(flops[test]).all():
            compute_pred = raw_pred(
                compute_baseline_predict(flops, train, test, unit_targets[:, k])
            )
            methods['compute'] = compute_prediction_error(truth, compute_pred)
            raw['compute__' + name_k] = compute_pred

        per_target.append({
            'target': name_k,
            'headline': bool(target['headline']),
            'metric_range': list(target['metric_range']),
            'n_features': len(target['features']),
            'excludes_gsm8k': 'GSM8K' not in target['features'],
            'flops_cutoff_1e21': (target['cutoff']
                                  if spec['split'] == 'compute_cutoff' else None),
            'n_train': int(train.sum()),
            'n_test': int(test.sum()),
            'test_models': list(frame['Model'].values[test]),
            'target_range_train': [float(frame[name_k].values[train].min()),
                                   float(frame[name_k].values[train].max())],
            'target_range_test': [float(truth.min()), float(truth.max())],
            'methods': methods,
        })

    joint = {}
    if usable:
        # Targets in one task can carry different feature blocks, since OSL drops
        # GSM8K whenever it would predict itself. The joint fit is therefore run
        # once per distinct block, which keeps that exclusion intact.
        truth_all, pred_all = [], []
        groups = {}
        for k in usable:
            groups.setdefault(tuple(targets[k]['features']), []).append(k)
        for block_features, keep in groups.items():
            block = to_unit_scale(frame[list(block_features)].values, FEATURE_RANGE)
            joint_pred = complete_targets(
                block, unit_targets[:, keep], hidden[:, keep], predict_benchpress_scores,
            )
            for column, k in enumerate(keep):
                mask = hidden[:, k]
                name_k = targets[k]['name']
                pred = to_raw_scale(joint_pred[mask, column], targets[k]['metric_range'])
                truth_all.append(frame[name_k].values[mask])
                pred_all.append(pred)
                raw['benchpress_joint__' + name_k] = pred
        joint = compute_prediction_error(np.concatenate(truth_all),
                                         np.concatenate(pred_all))

    def pool(keys):
        out = {}
        for method in ('osl', 'benchpress', 'benchpress_joint', 'compute', 'median'):
            truth_all, pred_all = [], []
            for name_k in keys:
                if method + '__' + name_k not in raw:
                    continue
                truth_all.append(raw['truth__' + name_k])
                pred_all.append(raw[method + '__' + name_k])
            if truth_all:
                out[method] = compute_prediction_error(
                    np.concatenate(truth_all), np.concatenate(pred_all),
                )
        return out

    names = [targets[k]['name'] for k in usable]
    headline = [targets[k]['name'] for k in usable if targets[k]['headline']]

    return {
        'osl_section': spec['osl_section'],
        'split': spec['split'],
        'n_models': int(len(frame)),
        'targets_evaluated': names,
        'headline_targets': headline,
        'data_sha256': digests,
        # OSL reports the headline targets; the rest sit commented out in the
        # same notebooks, so both poolings are given and neither is chosen here.
        'pooled': pool(headline),
        'pooled_all_targets': pool(names),
        'per_target': per_target,
    }, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='one target per task, for a fast sanity run')
    parser.add_argument('--tasks', nargs='*', default=sorted(TASKS),
                        choices=sorted(TASKS))
    parser.add_argument('--out', default=os.path.join(HERE, 'obs_scaling_setup.json'))
    args = parser.parse_args()

    results = {
        'description': 'BenchPress vs observational scaling laws on OSL released data',
        'source': 'https://github.com/ryoungj/ObsScaling',
        'flops_cutoff_1e21': {'default': FLOPS_CUTOFF,
                              'arithmetic': ARITHMETIC_FLOPS_CUTOFF},
        'config_source': 'transcribed from the notebooks in the OSL repository',
        'benchpress_predictor': 'logit Bias ALS, rank 2, lambda 0.1',
        'smoke': bool(args.smoke),
        'tasks': {},
    }
    raw_all = {}
    for name in args.tasks:
        summary, raw = run_task(name, TASKS[name], smoke=args.smoke)
        results['tasks'][name] = summary
        for key, value in raw.items():
            raw_all[name + '/' + key] = value

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    np.savez_compressed(args.out.replace('.json', '_raw.npz'), **raw_all)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
