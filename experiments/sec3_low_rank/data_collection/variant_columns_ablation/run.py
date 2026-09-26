#!/usr/bin/env python
"""Variant-column audit and ablation for the May 2026 BenchPress matrix."""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np


SEED = 42
N_SEEDS = 10
N_FOLDS = 3
MIN_SCORES = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
DEFAULT_RESULTS_PATH = os.path.join(SCRIPT_DIR, 'results.json')
DEFAULT_AUDIT_PATH = os.path.join(SCRIPT_DIR, 'variant_groups.json')


VARIANT_GROUPS = [
    {
        'id': 'aider_polyglot',
        'members': ['aider_polyglot_diff', 'aider_polyglot_whole'],
        'description': 'Same Aider Polyglot benchmark under diff-mode versus whole-file edit mode.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'same task family and score scale; only the edit harness differs.',
    },
    {
        'id': 'aime',
        'members': ['aime_2024', 'aime_2025', 'aime_2026', 'mt_aime_2024'],
        'description': 'AIME annual editions plus the multi-turn AIME 2024 protocol.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'the item set or interaction protocol differs by edition.',
        'pair_overrides': {
            'aime_2024|mt_aime_2024': [
                'ambiguous',
                'same AIME 2024 items, but multi-turn interaction changes the evaluation protocol',
            ],
        },
    },
    {
        'id': 'arc_agi',
        'members': ['arc_agi_1', 'arc_agi_2'],
        'description': 'ARC-AGI version 1 and version 2.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'new benchmark version with a different task distribution.',
    },
    {
        'id': 'bfcl',
        'members': ['bfcl', 'bfcl_v3'],
        'description': 'Berkeley Function Calling Leaderboard versions.',
        'default_rule_status': 'ambiguous',
        'default_rule_reason': 'versioned function-calling variants share a family but differ in benchmark release.',
    },
    {
        'id': 'browsecomp',
        'members': ['browsecomp', 'browsecomp_long_context_128k', 'browsecomp_zh'],
        'description': 'BrowseComp base, long-context, and Chinese variants.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'context length or language changes the evaluation setting.',
    },
    {
        'id': 'charxiv',
        'members': ['charxiv_descriptive', 'charxiv_reasoning'],
        'description': 'CharXiv descriptive and reasoning subtasks.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'subtasks require different visual reasoning behavior.',
    },
    {
        'id': 'frontiermath',
        'members': ['frontiermath', 'frontiermath_tier4'],
        'description': 'FrontierMath overall versus Tier 4 hard subset.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'Tier 4 is a subset of the same benchmark on the same percentage scale.',
    },
    {
        'id': 'gpqa',
        'members': ['gpqa_diamond', 'gpqa_main'],
        'description': 'GPQA Diamond subset and GPQA main/full set.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'same benchmark family and score scale; Diamond is a curated subset.',
    },
    {
        'id': 'graphwalks',
        'members': ['graphwalks_bfs_0k_128k', 'graphwalks_parents_0k_128k'],
        'description': 'GraphWalks BFS and parent-retrieval tasks at the same context range.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'same dataset family but different graph query task.',
    },
    {
        'id': 'hle',
        'members': ['hle', 'hle_text', 'hle_tools'],
        'description': 'Humanity\'s Last Exam full, text-only, and tool-access variants.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'same benchmark family; HLE Text is a subset-like same-scale view.',
        'pair_overrides': {
            'hle|hle_tools': [
                'allowed_protocol_variant',
                'tool access changes the evaluation protocol',
            ],
            'hle_text|hle_tools': [
                'allowed_protocol_variant',
                'tool access changes the evaluation protocol',
            ],
        },
    },
    {
        'id': 'hmmt',
        'members': ['hmmt_feb_2025', 'hmmt_feb_2026', 'hmmt_nov_2025'],
        'description': 'HMMT contest editions.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'different contest editions use different item sets.',
    },
    {
        'id': 'longfact',
        'members': ['longfact_concepts', 'longfact_objects'],
        'description': 'LongFact concepts and objects hallucination rates.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'parallel subsets of the same benchmark with the same lower-is-better metric.',
    },
    {
        'id': 'math',
        'members': ['math', 'math_500'],
        'description': 'MATH full benchmark and MATH-500 subset.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'MATH-500 is a subset-style same-scale view of MATH.',
    },
    {
        'id': 'mmlu',
        'members': ['mmlu_pro', 'mmmlu'],
        'description': 'MMLU-derived pro and multilingual variants retained in the May matrix.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'MMLU-Pro and MMMLU change difficulty construction or language coverage.',
    },
    {
        'id': 'mmmu',
        'members': ['mmmu', 'mmmu_pro'],
        'description': 'MMMU and MMMU-Pro.',
        'default_rule_status': 'ambiguous',
        'default_rule_reason': 'same benchmark family, but MMMU-Pro changes answer-option and vision-only construction.',
    },
    {
        'id': 'mrcr',
        'members': ['mrcr_v1', 'mrcr_v2', 'mrcr_v2_2needle_128k', 'mrcr_v2_8needle'],
        'description': 'MRCR benchmark versions and needle/context settings.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'version, number of needles, or context length changes the long-context protocol.',
    },
    {
        'id': 'multichallenge',
        'members': ['multichallenge', 'multichallenge_o3mini_grader'],
        'description': 'MultiChallenge with two grader choices.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'grader changes are protocol changes under the current rule text.',
    },
    {
        'id': 'omnidocbench',
        'members': ['omnidocbench', 'omnidocbench_1.5'],
        'description': 'OmniDocBench original normalized-edit-distance metric and OmniDocBench 1.5 percentage metric.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'same family is retained under different metric direction and scale.',
    },
    {
        'id': 'simpleqa',
        'members': ['simpleqa', 'simpleqa_verified', 'chinese_simpleqa'],
        'description': 'SimpleQA base, verified subset, and Chinese variant.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'SimpleQA and SimpleQA-Verified are same-scale subset variants.',
        'pair_overrides': {
            'simpleqa|chinese_simpleqa': [
                'allowed_protocol_variant',
                'language changes the evaluation setting',
            ],
            'simpleqa_verified|chinese_simpleqa': [
                'allowed_protocol_variant',
                'language changes the evaluation setting',
            ],
        },
    },
    {
        'id': 'swe_bench',
        'members': ['swe_bench_verified', 'swe_bench_pro', 'swe_bench_multilingual', 'multi_swe_bench'],
        'description': 'SWE-bench verified, pro, multilingual, and Multi-SWE-bench variants.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'the retained columns change instance source, language mix, or benchmark construction.',
    },
    {
        'id': 'swelancer',
        'members': ['swelancer', 'swelancer_freelance_dollars'],
        'description': 'SWE-Lancer percentage score and freelance-dollar score.',
        'default_rule_status': 'violation',
        'default_rule_reason': 'same family retained under different units, directly contrary to the different-scale exclusion sentence.',
    },
    {
        'id': 'tau_bench',
        'members': [
            'tau_bench_airline', 'tau_bench_retail',
            'tau2_bench_airline', 'tau2_bench_retail', 'tau2_bench_telecom',
            'tau3_bench',
        ],
        'description': 'τ-bench, τ²-bench, and τ³-bench domains/versions.',
        'default_rule_status': 'allowed_protocol_variant',
        'default_rule_reason': 'domain and version changes define separate agentic tasks.',
    },
    {
        'id': 'terminal_bench',
        'members': ['terminal_bench', 'terminal_bench_1', 'terminal_bench_hard'],
        'description': 'Terminal-Bench 2.0, Terminal-Bench 1.0, and hard subset.',
        'default_rule_status': 'ambiguous',
        'default_rule_reason': 'version and hard-subset variants share the same family but change task set/difficulty.',
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-json', default=None,
                        help='Score-matrix JSON. Sets BENCHPRESS_DATA before loading the harness.')
    parser.add_argument('--variant-config', choices=['reviewer_groups'], default='reviewer_groups',
                        help='Named variant-group configuration to deduplicate.')
    parser.add_argument('--drop-benchmarks', default=None,
                        help='Comma-separated benchmark ids to drop instead of deriving drops from the named config.')
    parser.add_argument('--output', default=DEFAULT_RESULTS_PATH)
    parser.add_argument('--audit-output', default=DEFAULT_AUDIT_PATH)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--lambda-reg', type=float, default=0.1)
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS)
    parser.add_argument('--n-folds', type=int, default=N_FOLDS)
    parser.add_argument('--base-seed', type=int, default=SEED)
    return parser.parse_args()


def load_harness(data_json):
    if data_json is not None:
        os.environ['BENCHPRESS_DATA'] = data_json
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from benchpress import evaluation_harness as eh
    from benchpress.methods.completers import complete_bias_als
    from benchpress.io_utils import write_json
    return eh, complete_bias_als, write_json


def current_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', ROOT, 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finite_float(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def pair_key(a, b):
    return '|'.join(sorted([a, b]))


def pair_rule_judgment(group, a, b):
    overrides = group.get('pair_overrides') or {}
    override = overrides.get(pair_key(a, b))
    if override is not None:
        return {'status': override[0], 'reason': override[1]}
    return {
        'status': group['default_rule_status'],
        'reason': group['default_rule_reason'],
    }


def metric_higher_is_better(metric_spec):
    if isinstance(metric_spec, dict) and metric_spec.get('higher_is_better') is False:
        return False
    return True


def aligned_values(values, higher_is_better):
    if higher_is_better:
        return values
    return -values


def pearson(x, y):
    if x.size < 2 or y.size < 2:
        return None
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def group_audit(eh):
    bench_idx = eh.BENCH_IDX
    observed_counts = {bid: int(eh.OBSERVED[:, j].sum()) for bid, j in bench_idx.items()}
    out_groups = []
    all_drop_ids = []

    for group in VARIANT_GROUPS:
        members = [bid for bid in group['members'] if bid in bench_idx]
        if len(members) < 2:
            continue
        member_rows = []
        for bid in members:
            j = bench_idx[bid]
            member_rows.append({
                'benchmark_id': bid,
                'name': eh.BENCH_NAMES[bid],
                'category': str(eh.BENCH_CATS[j]),
                'n_observed': observed_counts[bid],
                'metric': eh.BENCH_METRICS.get(bid),
            })
        keep_id = max(members, key=lambda bid: (observed_counts[bid], -members.index(bid)))
        drop_ids = [bid for bid in members if bid != keep_id]
        all_drop_ids.extend(drop_ids)
        pair_rows = []
        for a, b in itertools.combinations(members, 2):
            ja, jb = bench_idx[a], bench_idx[b]
            mask = np.isfinite(eh.M_FULL[:, ja]) & np.isfinite(eh.M_FULL[:, jb])
            xa = eh.M_FULL[mask, ja]
            xb = eh.M_FULL[mask, jb]
            ha = metric_higher_is_better(eh.BENCH_METRICS.get(a))
            hb = metric_higher_is_better(eh.BENCH_METRICS.get(b))
            union = int((np.isfinite(eh.M_FULL[:, ja]) | np.isfinite(eh.M_FULL[:, jb])).sum())
            smaller = min(observed_counts[a], observed_counts[b])
            pair_rows.append({
                'benchmark_a': a,
                'benchmark_b': b,
                'n_overlap': int(mask.sum()),
                'overlap_jaccard': float(mask.sum() / union) if union else None,
                'overlap_of_smaller': float(mask.sum() / smaller) if smaller else None,
                'pearson_raw': finite_float(pearson(xa, xb)),
                'pearson_direction_aligned': finite_float(pearson(
                    aligned_values(xa, ha), aligned_values(xb, hb))),
                'rule_judgment': pair_rule_judgment(group, a, b),
            })
        out_groups.append({
            'group_id': group['id'],
            'description': group['description'],
            'members': member_rows,
            'keep_rule': 'keep the member with the most observed cells; ties use the order in VARIANT_GROUPS',
            'kept_benchmark_id': keep_id,
            'dropped_benchmark_ids': drop_ids,
            'pairwise': pair_rows,
        })

    return {
        'variant_config': 'reviewer_groups',
        'n_groups': len(out_groups),
        'n_unique_benchmarks_in_groups': len(set(
            bid for group in out_groups for bid in [m['benchmark_id'] for m in group['members']])),
        'n_drop_benchmarks_for_dedup': len(set(all_drop_ids)),
        'drop_benchmark_ids_for_dedup': sorted(set(all_drop_ids), key=all_drop_ids.index),
        'groups': out_groups,
    }


def comma_ids(text):
    if text is None or not text.strip():
        return []
    return [part.strip() for part in text.split(',') if part.strip()]


def predictor_fn(eh, complete_bias_als, benchmark_ids, rank, lambda_reg):
    metric = {bid: eh.BENCH_METRICS[bid] for bid in benchmark_ids}
    return eh.make_score_predictor(
        lambda M, **kw: complete_bias_als(M, normalize=False, **kw),
        'logit', metric=metric, benchmark_ids=benchmark_ids,
        rank=rank, lam=lambda_reg)


def vector_metrics(eh, actual, predicted):
    return eh.compute_prediction_error(
        np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float),
        aggregation='pool')


def summarize_condition(eh, per_fold_rows, actual, predicted, n_requested):
    pooled = vector_metrics(eh, actual, predicted)
    medae_values = [row['medae'] for row in per_fold_rows if row['medae'] is not None]
    medape_values = [row['medape'] for row in per_fold_rows if row['medape'] is not None]
    n_covered = sum(row['n'] for row in per_fold_rows)
    return {
        'per_fold_median': {
            'n_folds': len(per_fold_rows),
            'n_requested': int(n_requested),
            'n_covered': int(n_covered),
            'coverage': float(n_covered / n_requested) if n_requested else 0.0,
            'medae': float(np.median(medae_values)) if medae_values else None,
            'medape': float(np.median(medape_values)) if medape_values else None,
        },
        'pooled': {
            'n': int(pooled['n']),
            'n_requested': int(n_requested),
            'coverage': float(pooled['n'] / n_requested) if n_requested else 0.0,
            'medae': finite_float(pooled['medae']),
            'medape': finite_float(pooled['medape']),
        },
        'per_fold': per_fold_rows,
    }


def run_ablation(eh, complete_bias_als, audit, explicit_drop_ids, rank, lambda_reg,
                 n_seeds, n_folds, base_seed):
    bench_ids = list(eh.BENCH_IDS)
    if explicit_drop_ids:
        drop_ids = explicit_drop_ids
        drop_source = 'explicit --drop-benchmarks'
    else:
        drop_ids = list(audit['drop_benchmark_ids_for_dedup'])
        drop_source = 'reviewer_groups keep-most-observed rule'
    unknown = [bid for bid in drop_ids if bid not in eh.BENCH_IDX]
    if unknown:
        raise ValueError(f'Unknown benchmark ids in drop list: {unknown}')
    drop_set = set(drop_ids)
    keep_indices = [idx for idx, bid in enumerate(bench_ids) if bid not in drop_set]
    keep_ids = [bench_ids[idx] for idx in keep_indices]
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    folds = eh.load_folds(
        n_seeds=n_seeds, n_folds=n_folds,
        base_seed=base_seed, min_scores=MIN_SCORES)
    full_predict = predictor_fn(eh, complete_bias_als, bench_ids, rank, lambda_reg)
    dedup_predict = predictor_fn(eh, complete_bias_als, keep_ids, rank, lambda_reg)

    full_actual, full_predicted = [], []
    dedup_actual, dedup_predicted = [], []
    full_per_fold, dedup_per_fold = [], []
    raw_predictions = []
    n_requested = 0
    start = time.time()

    for fold_idx, (m_train, test_set) in enumerate(folds):
        seed_idx = fold_idx // n_folds
        fold_in_seed = fold_idx % n_folds
        test_keep = [(i, j) for i, j in test_set if j in old_to_new]
        n_requested += len(test_keep)

        full_pred = full_predict(m_train)
        m_train_dedup = m_train[:, keep_indices]
        dedup_pred = dedup_predict(m_train_dedup)

        fold_full_actual, fold_full_predicted = [], []
        fold_dedup_actual, fold_dedup_predicted = [], []
        for i, j in test_keep:
            true_value = float(eh.M_FULL[i, j])
            full_value = float(full_pred[i, j]) if np.isfinite(full_pred[i, j]) else float('nan')
            dedup_j = old_to_new[j]
            dedup_value = (
                float(dedup_pred[i, dedup_j])
                if np.isfinite(dedup_pred[i, dedup_j]) else float('nan')
            )
            fold_full_actual.append(true_value)
            fold_full_predicted.append(full_value)
            fold_dedup_actual.append(true_value)
            fold_dedup_predicted.append(dedup_value)
            raw_predictions.append({
                'seed': int(seed_idx),
                'fold': int(fold_in_seed),
                'model_idx': int(i),
                'model_id': eh.MODEL_IDS[i],
                'benchmark_idx': int(j),
                'benchmark_id': eh.BENCH_IDS[j],
                'true': true_value,
                'full_pred': full_value if math.isfinite(full_value) else None,
                'deduplicated_pred': dedup_value if math.isfinite(dedup_value) else None,
            })
        full_m = vector_metrics(eh, fold_full_actual, fold_full_predicted)
        dedup_m = vector_metrics(eh, fold_dedup_actual, fold_dedup_predicted)
        full_per_fold.append({
            'fold_index': int(fold_idx),
            'seed': int(seed_idx),
            'fold': int(fold_in_seed),
            'n': int(full_m['n']),
            'n_requested': len(test_keep),
            'medae': finite_float(full_m['medae']),
            'medape': finite_float(full_m['medape']),
        })
        dedup_per_fold.append({
            'fold_index': int(fold_idx),
            'seed': int(seed_idx),
            'fold': int(fold_in_seed),
            'n': int(dedup_m['n']),
            'n_requested': len(test_keep),
            'medae': finite_float(dedup_m['medae']),
            'medape': finite_float(dedup_m['medape']),
        })
        full_actual.extend(fold_full_actual)
        full_predicted.extend(fold_full_predicted)
        dedup_actual.extend(fold_dedup_actual)
        dedup_predicted.extend(fold_dedup_predicted)
        print(
            f"fold {fold_idx + 1:02d}/{len(folds)} n={len(test_keep)} "
            f"full MedAE={full_m['medae']:.3f} dedup MedAE={dedup_m['medae']:.3f}",
            flush=True,
        )

    return {
        'protocol': {
            'matrix': 'May 2026 BenchPress matrix',
            'n_models': int(eh.N_MODELS),
            'n_benchmarks_full': int(eh.N_BENCH),
            'n_observed_full': int(np.isfinite(eh.M_FULL).sum()),
            'n_benchmarks_deduplicated': len(keep_ids),
            'n_observed_deduplicated': int(np.isfinite(eh.M_FULL[:, keep_indices]).sum()),
            'drop_source': drop_source,
            'dropped_benchmark_ids': drop_ids,
            'kept_benchmark_ids': keep_ids,
            'evaluation_universe': 'canonical 10 seed x 3 per-model folds filtered to kept benchmark columns',
            'n_requested_predictions_per_condition': int(n_requested),
            'predictor': {
                'transform': 'logit',
                'method': 'Bias ALS',
                'rank': int(rank),
                'lambda': float(lambda_reg),
            },
            'folds': {
                'n_seeds': int(n_seeds),
                'n_folds': int(n_folds),
                'base_seed': int(base_seed),
                'min_scores': int(MIN_SCORES),
            },
            'matrix_identity_sha256': eh.matrix_identity_sha256(eh.M_FULL),
            'benchmark_metric_identity_sha256': eh.benchmark_metric_identity_sha256(
                eh.BENCH_METRICS, eh.BENCH_IDS),
            'git_commit': current_commit(),
            'elapsed_sec': time.time() - start,
        },
        'summary': {
            'full_matrix_evaluated_on_kept_columns': summarize_condition(
                eh, full_per_fold, full_actual, full_predicted, n_requested),
            'deduplicated_matrix': summarize_condition(
                eh, dedup_per_fold, dedup_actual, dedup_predicted, n_requested),
        },
        'raw_predictions': raw_predictions,
    }


def main():
    args = parse_args()
    eh, complete_bias_als, write_json = load_harness(args.data_json)
    audit = group_audit(eh)
    write_json(args.audit_output, audit, indent=2, sort_keys=True)
    print(f"WROTE audit -> {args.audit_output}")
    if args.audit_only:
        return
    results = run_ablation(
        eh, complete_bias_als, audit, comma_ids(args.drop_benchmarks),
        args.rank, args.lambda_reg, args.n_seeds, args.n_folds, args.base_seed)
    write_json(args.output, results, indent=2, sort_keys=True)
    print(f"WROTE results -> {args.output}")


if __name__ == '__main__':
    main()
