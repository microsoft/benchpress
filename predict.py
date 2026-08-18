#!/usr/bin/env python3
"""
LLM Benchmark Score Predictor
==============================

Predict missing benchmark scores for LLM models using BenchPress
(Logit Bias ALS with lambda=0.1 and rank=2).

Usage:
    # Predict all missing scores (output CSV)
    python predict.py

    # Predict scores for a specific model
    python predict.py --model gpt-5.2

    # Predict scores on a specific benchmark
    python predict.py --benchmark aime_2025

    # Add calibrated 90% intervals
    python predict.py --model gpt-5.2 --confidence

    # Predict a single cell
    python predict.py --model gpt-5.2 --benchmark gpqa_diamond

    # Add a new model's known scores and predict the rest
    python predict.py --add-model my-model --scores "mmlu=85.2,gpqa_diamond=72.0,aime_2025=60.0"

    # Output as JSON instead of CSV
    python predict.py --model gpt-5.2 --format json

    # List all models or benchmarks
    python predict.py --list-models
    python predict.py --list-benchmarks
"""

import argparse
import csv
import io
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

# ── Setup paths ──
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Suppress the matrix print from evaluation_harness
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
import benchpress.evaluation_harness as benchpress_data  # noqa: E402
from benchpress.all_methods import predict_benchpress_scores  # noqa: E402
from benchpress.data.score_matrix import ScoreMatrix  # noqa: E402
sys.stdout = _old_stdout


def load_benchpress_matrix():
    return SimpleNamespace(
        values=benchpress_data.M_FULL,
        observed=benchpress_data.OBSERVED,
        model_ids=benchpress_data.MODEL_IDS,
        benchmark_ids=benchpress_data.BENCH_IDS,
        model_names=benchpress_data.MODEL_NAMES,
        benchmark_names=benchpress_data.BENCH_NAMES,
        model_providers=benchpress_data.MODEL_PROVIDERS,
        model_reasoning=benchpress_data.MODEL_REASONING,
        benchmark_categories=benchpress_data.BENCH_CATS,
    )


def render_prediction_rows(rows, fmt):
    """Render prediction rows as CSV or JSON."""
    if fmt == 'json':
        return json.dumps(rows, indent=2)
    if not rows:
        return "No predictions to show."
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _metric_range_label(metric):
    score_range = metric.get('range')
    if score_range is None:
        return ''
    lo, hi = score_range
    suffix = '%' if metric.get('type') == 'pct' else ''
    return f"{lo:g}..{hi:g}{suffix}"


def _metric_better_label(metric):
    return 'lower' if metric.get('higher_is_better') is False else 'higher'


def _format_metric_value(score, metric):
    if score is None or not np.isfinite(score):
        return 'n/a'
    value = f"{float(score):.1f}"
    return f"{value}%" if metric.get('type') == 'pct' else value


def _same_benchmark_token(left, right):
    def normalize(text):
        return text.lower().replace('-', '_').replace('.', '_').replace(' ', '_')
    return normalize(left) == normalize(right)


def _strip_benchmark_prefix(text, group):
    prefixes = [group, group.replace('-', '_'), group.replace('_', '-')]
    for prefix in prefixes:
        for sep in ['.', '_', '-']:
            marker = prefix + sep
            if text.startswith(marker):
                return text[len(marker):]
    return text


def _display_benchmark_parts(benchmark_id):
    pieces = [piece for piece in benchmark_id.split('/') if piece]
    if not pieces:
        return 'benchmarks', benchmark_id
    group = pieces[0]
    metric_parts = []
    for piece in pieces[1:] or pieces:
        name = _strip_benchmark_prefix(piece, group)
        if _same_benchmark_token(name, group):
            continue
        if metric_parts and _same_benchmark_token(name, metric_parts[-1]):
            continue
        metric_parts.append(name)
    return group, ' / '.join(metric_parts) if metric_parts else group


def _clip_to_metric(score, metric):
    score_range = metric.get('range')
    if score_range is None or not np.isfinite(score):
        return score
    lo, hi = score_range
    return float(np.clip(score, lo, hi))


def _format_interval(confidence):
    if confidence is None:
        return ''
    return f"[{confidence['lower']:.1f}, {confidence['upper']:.1f}]"


def _format_probability(probability):
    if probability is None or not np.isfinite(probability):
        return ''
    return f"{100 * float(probability):.0f}%"


def _median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _medape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (np.abs(actual) > 1e-12)
    if not valid.any():
        return None
    ape = np.abs((predicted[valid] - actual[valid]) / actual[valid]) * 100
    return float(np.median(ape))


def _summary_value(value, suffix=''):
    if value is None:
        return 'n/a'
    return f"{value:.2f}{suffix}"


def score_matrix_missing_cells(matrix, model_filter=None, bench_filter=None):
    observed = np.isfinite(matrix.values)
    cells = []
    for i, model_id in enumerate(matrix.model_ids):
        if model_filter and model_id != model_filter:
            continue
        for j, benchmark_id in enumerate(matrix.benchmark_ids):
            if bench_filter and benchmark_id != bench_filter:
                continue
            if not observed[i, j]:
                cells.append((i, j))
    return cells


def score_matrix_confidence_lookup(matrix, predictions, cells, artifact_path=None,
                                   method='combined_risk_model'):
    """Run the standard BenchPress confidence model on a custom score matrix."""
    from benchpress.methods.confidence import predict_confidence_intervals
    result = predict_confidence_intervals(
        matrix.values,
        M_pred=predictions,
        artifact_path=artifact_path,
        method=method,
        cells=cells,
        metric=matrix.metric,
        benchmark_ids=matrix.benchmark_ids,
    )
    confidence_level = float(result['confidence_level'])
    confidence = {
        '__metadata__': {
            'confidence_level': confidence_level,
            'confidence_method': result['method'],
            'trust_threshold': result.get('trust_threshold'),
        }
    }
    for index, (i, j) in enumerate(result['cells']):
        benchmark_id = matrix.benchmark_ids[j]
        metric = matrix.metric[benchmark_id]
        predicted = float(result['predicted'][index])
        lower = float(result['lower'][index])
        upper = float(result['upper'][index])
        trust_probability = None
        if 'trust_probability' in result:
            probability = result['trust_probability'][index]
            if np.isfinite(probability):
                trust_probability = float(probability)
        confidence[(int(i), int(j))] = {
            'confidence_method': result['method'],
            'predicted': predicted,
            'uncertainty': float(result['uncertainty'][index]),
            'lower': _clip_to_metric(lower, metric),
            'upper': _clip_to_metric(upper, metric),
            'trust_probability': trust_probability,
            'trust_threshold': result.get('trust_threshold'),
        }
    return confidence


def format_predictions(predictions, matrix, model_filter=None, bench_filter=None,
                       only_missing=True, fmt='csv', confidence=None):
    """Format built-in BenchPress matrix predictions as CSV or JSON rows."""
    rows = []
    for i in range(predictions.shape[0]):
        mid = matrix.model_ids[i]
        if model_filter and mid != model_filter:
            continue
        for j in range(predictions.shape[1]):
            bid = matrix.benchmark_ids[j]
            if bench_filter and bid != bench_filter:
                continue
            if only_missing and matrix.observed[i, j]:
                continue
            pred = predictions[i, j]
            actual = matrix.values[i, j] if matrix.observed[i, j] else None
            row = {
                'model': mid,
                'model_name': matrix.model_names[mid],
                'benchmark': bid,
                'benchmark_name': matrix.benchmark_names[bid],
                'predicted': round(float(pred), 1) if np.isfinite(pred) else None,
                'actual': round(float(actual), 1) if actual is not None else None,
                'is_observed': bool(matrix.observed[i, j]),
            }
            if confidence is not None and (i, j) in confidence:
                conf = confidence[(i, j)]
                row.update({
                    'confidence_method': conf['method'],
                    'uncertainty': round(float(conf['uncertainty']), 2),
                    'lower_90': round(float(conf['lower']), 1),
                    'upper_90': round(float(conf['upper']), 1),
                })
            rows.append(row)
    return render_prediction_rows(rows, fmt)


def format_score_matrix_predictions(predictions, matrix, model_filter=None,
                                    bench_filter=None, only_missing=True,
                                    fmt='csv', confidence=None):
    """Format predictions for a user-provided ScoreMatrix."""
    rows = []
    observed = np.isfinite(matrix.values)
    for i, model_id in enumerate(matrix.model_ids):
        if model_filter and model_id != model_filter:
            continue
        for j, benchmark_id in enumerate(matrix.benchmark_ids):
            if bench_filter and benchmark_id != bench_filter:
                continue
            if only_missing and observed[i, j]:
                continue
            pred = predictions[i, j]
            actual = matrix.values[i, j] if observed[i, j] else None
            row = {
                'model': model_id,
                'model_name': model_id,
                'benchmark': benchmark_id,
                'benchmark_name': benchmark_id,
                'predicted': round(float(pred), 1) if np.isfinite(pred) else None,
                'actual': round(float(actual), 1) if actual is not None else None,
                'is_observed': bool(observed[i, j]),
            }
            if confidence is not None and not observed[i, j]:
                conf = confidence.get((i, j))
                row.update({
                    'confidence_method': None if conf is None else conf['confidence_method'],
                    'uncertainty': None if conf is None else round(float(conf['uncertainty']), 2),
                    'lower_90': None if conf is None else round(float(conf['lower']), 1),
                    'upper_90': None if conf is None else round(float(conf['upper']), 1),
                    'trust_probability': (
                        None if conf is None or conf.get('trust_probability') is None
                        else round(float(conf['trust_probability']), 3)
                    ),
                    'trust_threshold': (
                        None if conf is None or conf.get('trust_threshold') is None
                        else round(float(conf['trust_threshold']), 1)
                    ),
                })
            rows.append(row)
    return render_prediction_rows(rows, fmt)


def format_score_matrix_report(predictions, matrix, matrix_path,
                               model_filter=None, bench_filter=None,
                               only_missing=True, confidence=None):
    """Format user-provided ScoreMatrix predictions for terminal reading."""
    rows = []
    observed = np.isfinite(matrix.values)
    for i, model_id in enumerate(matrix.model_ids):
        if model_filter and model_id != model_filter:
            continue
        for j, benchmark_id in enumerate(matrix.benchmark_ids):
            is_observed = observed[i, j]
            if bench_filter and benchmark_id != bench_filter:
                continue
            if only_missing and is_observed:
                continue
            score = matrix.values[i, j] if is_observed else predictions[i, j]
            metric = matrix.metric[benchmark_id]
            conf = None if is_observed or confidence is None else confidence.get((i, j))
            group, metric_name = _display_benchmark_parts(benchmark_id)
            rows.append({
                'model': model_id,
                'group': group,
                'metric': metric_name,
                'value': _format_metric_value(score, metric),
                'interval_90': _format_interval(conf),
                'trust_probability': _format_probability(None if conf is None else conf.get('trust_probability')),
                'source': 'matrix observed' if is_observed else 'BenchPress estimate',
                'model evidence': f"{int(observed[i].sum())} observed scores",
                'benchmark evidence': f"{int(observed[:, j].sum())} observed models",
                'type': metric.get('type', 'continuous'),
                'scale': _metric_range_label(metric),
                'better': _metric_better_label(metric),
            })

    title = "BenchPress estimates"
    if model_filter:
        title += f" for {model_filter}"
    lines = [title, f"Matrix: {matrix_path}"]
    if only_missing:
        lines.append("Estimated values for scores missing from this matrix. Use --all to include observed values.")
    else:
        lines.append("Showing observed matrix values and BenchPress estimates.")
    if model_filter and rows:
        lines.append(f"Model evidence: {rows[0]['model evidence']} for this model.")
    if confidence is not None:
        metadata = confidence.get('__metadata__', {})
        lines.extend([
            '',
            'Confidence:',
            f"  method: {metadata.get('confidence_method', 'combined_risk_model')}",
            f"  interval: calibrated {100 * metadata.get('confidence_level', 0.90):.0f}% conformal interval",
            '  trust probability: estimated P(abs error <= 10 score points) from hybrid uncertainty risk',
        ])
    lines.append('')

    if not rows:
        lines.append("No predictions to show.")
        return "\n".join(lines)

    show_model = model_filter is None
    headers = ['metric', 'estimated value', 'benchmark evidence', 'type', 'scale', 'better']
    if not only_missing:
        headers = ['metric', 'value', 'source', 'benchmark evidence', 'type', 'scale', 'better']
    if confidence is not None:
        headers.insert(2, '90% interval')
        headers.insert(3, 'trust probability')
    if show_model:
        headers.insert(0, 'model')
        headers.insert(1, 'model evidence')
    field_for_header = {
        'estimated value': 'value',
        '90% interval': 'interval_90',
        'trust probability': 'trust_probability',
    }
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(row)
    for group, group_rows in groups.items():
        lines.extend(['', f"=== {group} ==="])
        widths = {}
        for header in headers:
            field = field_for_header.get(header, header)
            widths[header] = max(len(header), max(len(str(row[field])) for row in group_rows))
        lines.append("  ".join(header.ljust(widths[header]) for header in headers))
        lines.append("  ".join('-' * widths[header] for header in headers))
        for row in group_rows:
            values = []
            for header in headers:
                field = field_for_header.get(header, header)
                values.append(str(row[field]).ljust(widths[header]))
            lines.append("  ".join(values))
    return "\n".join(lines)


def add_model_scores(scores_str, benchmark_ids, benchmark_index, M_input):
    """Add a new model row to the matrix with known scores, return augmented matrix."""
    scores = {}
    for pair in scores_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            print(f"Warning: skipping malformed score '{pair}' (expected bench=score)")
            continue
        bench, val = pair.split('=', 1)
        bench = bench.strip()
        val = float(val.strip())
        if bench not in benchmark_index:
            print(f"Warning: benchmark '{bench}' not found. Available: {', '.join(benchmark_ids[:10])}...")
            continue
        scores[bench] = val

    if not scores:
        print("Error: no valid scores provided.")
        sys.exit(1)

    new_row = np.full((1, len(benchmark_ids)), np.nan)
    for bench, val in scores.items():
        new_row[0, benchmark_index[bench]] = val
    return np.vstack([M_input, new_row]), scores


def predict_for_new_model(M_aug):
    """Run BenchPress on the augmented matrix (with the new model row appended)."""
    return predict_benchpress_scores(M_aug)


def confidence_lookup(conf_result):
    """Convert confidence arrays into a cell-index lookup."""
    lookup = {}
    for idx, cell in enumerate(conf_result['cells']):
        record = {
            'method': conf_result['method'],
            'uncertainty': conf_result['uncertainty'][idx],
            'lower': conf_result['lower'][idx],
            'upper': conf_result['upper'][idx],
        }
        if 'trust_probability' in conf_result:
            record['trust_probability'] = conf_result['trust_probability'][idx]
            record['trust_threshold'] = conf_result.get('trust_threshold')
        lookup[tuple(cell)] = record
    return lookup


def resolve_repo_path(path):
    """Resolve absolute paths or paths relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_matrix(path):
    """Load a custom score matrix from the CLI matrix arguments."""
    return ScoreMatrix.from_file(resolve_repo_path(path))


def run_matrix_mode(args):
    """Run prediction or export against a user-provided score matrix."""
    if args.add_model:
        print("Error: --add-model is only supported for the built-in BenchPress matrix.")
        sys.exit(1)

    matrix = load_matrix(args.matrix)
    observed = np.isfinite(matrix.values)

    try:
        matrix.validate()
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.list_models:
        print(f"{'Model ID':<30s} {'#Scores'}")
        print('-' * 40)
        for i, mid in enumerate(matrix.model_ids):
            print(f"{mid:<30s} {int(observed[i].sum())}")
        return

    if args.list_benchmarks:
        print(f"{'Benchmark ID':<30s} {'#Models'}")
        print('-' * 40)
        for j, bid in enumerate(matrix.benchmark_ids):
            print(f"{bid:<30s} {int(observed[:, j].sum())}")
        return

    model_index = {model_id: i for i, model_id in enumerate(matrix.model_ids)}
    benchmark_index = {
        benchmark_id: j for j, benchmark_id in enumerate(matrix.benchmark_ids)
    }
    if args.model and args.model not in model_index:
        print(f"Error: model '{args.model}' not found in --matrix.")
        print("Use --matrix ... --list-models to see available models.")
        sys.exit(1)
    if args.benchmark and args.benchmark not in benchmark_index:
        print(f"Error: benchmark '{args.benchmark}' not found in --matrix.")
        print("Use --matrix ... --list-benchmarks to see available benchmarks.")
        sys.exit(1)

    specific_observed_cell = False
    if args.model and args.benchmark:
        i = model_index[args.model]
        j = benchmark_index[args.benchmark]
        specific_observed_cell = bool(observed[i, j])
    only_missing = False if specific_observed_cell else not args.all

    predictions = predict_benchpress_scores(
        matrix.values,
        metric=matrix.metric,
        benchmark_ids=matrix.benchmark_ids,
    )

    confidence = None
    if args.confidence:
        confidence_cells = score_matrix_missing_cells(
            matrix,
            model_filter=args.model,
            bench_filter=args.benchmark,
        )
        confidence = score_matrix_confidence_lookup(
            matrix,
            predictions,
            confidence_cells,
            artifact_path=args.confidence_artifact,
        )
    if args.format:
        result = format_score_matrix_predictions(
            predictions,
            matrix,
            model_filter=args.model,
            bench_filter=args.benchmark,
            only_missing=only_missing,
            fmt=args.format,
            confidence=confidence,
        )
    else:
        result = format_score_matrix_report(
            predictions,
            matrix,
            args.matrix,
            model_filter=args.model,
            bench_filter=args.benchmark,
            only_missing=only_missing,
            confidence=confidence,
        )

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        if args.format == 'csv':
            print(f"Wrote {result.count(chr(10)) - 1} predictions to {args.output}")
        elif args.format == 'json':
            print(f"Wrote {result.count(chr(34) + 'model' + chr(34))} predictions to {args.output}")
        else:
            print(f"Wrote prediction report to {args.output}")
    else:
        print(result)


def main():
    parser = argparse.ArgumentParser(
        description='Predict missing LLM benchmark scores using BenchPress',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Filter predictions to a specific model ID')
    parser.add_argument('--benchmark', '-b', type=str, default=None,
                        help='Filter predictions to a specific benchmark ID')
    parser.add_argument('--format', '-f', type=str, default=None, choices=['csv', 'json'],
                        help='Output format (default: csv; text for --matrix)')
    parser.add_argument('--all', action='store_true',
                        help='Show all predictions (not just missing cells)')
    parser.add_argument('--list-models', action='store_true',
                        help='List all model IDs')
    parser.add_argument('--list-benchmarks', action='store_true',
                        help='List all benchmark IDs')
    parser.add_argument('--add-model', type=str, default=None,
                        help='Name of new model to add')
    parser.add_argument('--scores', '-s', type=str, default=None,
                        help='Known scores as "bench1=val1,bench2=val2" (used with --add-model)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: stdout)')
    parser.add_argument('--confidence', action='store_true',
                        help='Include calibrated 90%% intervals and trust probabilities')
    parser.add_argument('--confidence-artifact', type=str, default=None,
                        help='Path to confidence artifact (default: package artifact path)')
    parser.add_argument('--matrix', type=str, default=None,
                        help='Input score matrix file path')

    args = parser.parse_args()

    if args.matrix:
        run_matrix_mode(args)
        return

    matrix = load_benchpress_matrix()

    # ── List modes ──
    if args.list_models:
        print(f"{'Model ID':<30s} {'Display Name':<35s} {'Provider':<15s} {'Reasoning':<10s} {'#Scores'}")
        print('-' * 100)
        for i, mid in enumerate(matrix.model_ids):
            n = int(matrix.observed[i].sum())
            print(f"{mid:<30s} {matrix.model_names[mid]:<35s} "
                  f"{matrix.model_providers[i]:<15s} "
                  f"{'Y' if matrix.model_reasoning[i] else 'N':<10s} {n}")
        return

    if args.list_benchmarks:
        print(f"{'Benchmark ID':<30s} {'Display Name':<35s} {'Category':<20s} {'#Models'}")
        print('-' * 95)
        for j, bid in enumerate(matrix.benchmark_ids):
            n = int(matrix.observed[:, j].sum())
            print(f"{bid:<30s} {matrix.benchmark_names[bid]:<35s} "
                  f"{matrix.benchmark_categories[j]:<20s} {n}")
        return

    # ── Add-model mode ──
    if args.add_model:
        if not args.scores:
            print("Error: --add-model requires --scores")
            sys.exit(1)
        benchmark_index = {
            benchmark_id: j for j, benchmark_id in enumerate(matrix.benchmark_ids)
        }
        M_aug, known = add_model_scores(
            args.scores,
            matrix.benchmark_ids,
            benchmark_index,
            matrix.values.copy(),
        )

        M_pred_aug = predict_for_new_model(M_aug)
        new_predictions = M_pred_aug[-1]
        conf_by_cell = None
        new_model_index = len(matrix.model_ids)
        if args.confidence:
            from benchpress.methods.confidence import predict_confidence_intervals
            cells = [(new_model_index, j) for j in range(len(matrix.benchmark_ids))
                     if not np.isfinite(M_aug[new_model_index, j])]
            conf_by_cell = confidence_lookup(predict_confidence_intervals(
                M_aug,
                M_pred=M_pred_aug,
                artifact_path=args.confidence_artifact,
                cells=cells,
            ))

        print(f"\nPredictions for new model: {args.add_model}")
        print(f"Known scores provided: {len(known)}")
        if args.confidence:
            print(f"{'Benchmark':<35s} {'Predicted':>10s}  {'90% interval':>23s}  {'Known':>10s}")
            print('-' * 86)
        else:
            print(f"{'Benchmark':<35s} {'Predicted':>10s}  {'Known':>10s}")
            print('-' * 60)
        for j, bid in enumerate(matrix.benchmark_ids):
            is_known = bid in known
            pred_val = new_predictions[j]
            pred_str = f"{pred_val:>10.1f}" if np.isfinite(pred_val) else f"{'n/a':>10s}"
            known_str = f"({known[bid]:>5.1f})" if is_known else ""
            if args.confidence and conf_by_cell is not None and (new_model_index, j) in conf_by_cell:
                conf = conf_by_cell[(new_model_index, j)]
                interval = f"[{conf['lower']:.1f}, {conf['upper']:.1f}]"
                print(f"  {matrix.benchmark_names[bid]:<33s} {pred_str}  {interval:>23s}  {known_str:>10s}")
            else:
                print(f"  {matrix.benchmark_names[bid]:<33s} {pred_str}  {known_str:>10s}")
        return

    # ── Standard prediction mode ──
    model_index = {
        model_id: i for i, model_id in enumerate(matrix.model_ids)
    }
    benchmark_index = {
        benchmark_id: j for j, benchmark_id in enumerate(matrix.benchmark_ids)
    }
    if args.model and args.model not in model_index:
        print(f"Error: model '{args.model}' not found.")
        print(f"Use --list-models to see available models.")
        sys.exit(1)
    if args.benchmark and args.benchmark not in benchmark_index:
        print(f"Error: benchmark '{args.benchmark}' not found.")
        print(f"Use --list-benchmarks to see available benchmarks.")
        sys.exit(1)

    # Single-cell prediction
    if args.model and args.benchmark:
        i = model_index[args.model]
        j = benchmark_index[args.benchmark]
        if matrix.observed[i, j]:
            print(f"{matrix.model_names[args.model]} on "
                  f"{matrix.benchmark_names[args.benchmark]}: "
                  f"{matrix.values[i, j]:.1f} (observed)")
        else:
            M_pred = predict_benchpress_scores(matrix.values.copy())
            pred = M_pred[i, j]
            if args.confidence:
                from benchpress.methods.confidence import predict_confidence_intervals
                conf = predict_confidence_intervals(
                    matrix.values,
                    M_pred=M_pred,
                    artifact_path=args.confidence_artifact,
                    cells=[(i, j)],
                )
                print(f"{matrix.model_names[args.model]} on "
                      f"{matrix.benchmark_names[args.benchmark]}: "
                      f"{pred:.1f} (predicted), 90% interval "
                      f"[{conf['lower'][0]:.1f}, {conf['upper'][0]:.1f}]")
            else:
                print(f"{matrix.model_names[args.model]} on "
                      f"{matrix.benchmark_names[args.benchmark]}: "
                      f"{pred:.1f} (predicted)")
        return

    # Predict and output
    M_pred = predict_benchpress_scores(matrix.values.copy())
    conf_by_cell = None
    if args.confidence:
        from benchpress.methods.confidence import predict_confidence_intervals
        cells = []
        for i, mid in enumerate(matrix.model_ids):
            if args.model and mid != args.model:
                continue
            for j, bid in enumerate(matrix.benchmark_ids):
                if args.benchmark and bid != args.benchmark:
                    continue
                if not args.all and matrix.observed[i, j]:
                    continue
                cells.append((i, j))
        conf_by_cell = confidence_lookup(predict_confidence_intervals(
            matrix.values,
            M_pred=M_pred,
            artifact_path=args.confidence_artifact,
            cells=cells,
        ))
    result = format_predictions(M_pred,
                                matrix,
                                model_filter=args.model,
                                bench_filter=args.benchmark,
                                only_missing=not args.all,
                                fmt=args.format or 'csv',
                                confidence=conf_by_cell)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        output_format = args.format or 'csv'
        n = result.count('\n') - 1 if output_format == 'csv' else result.count('"model"')
        print(f"Wrote {n} predictions to {args.output}")
    else:
        print(result)


if __name__ == '__main__':
    main()
