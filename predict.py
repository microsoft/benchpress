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
import json
import sys
import os
import io
import numpy as np

# ── Setup paths ──
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Suppress the matrix print from evaluation_harness
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
from benchpress.evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH,
    MODEL_IDS, BENCH_IDS, MODEL_NAMES, BENCH_NAMES,
    MODEL_IDX, BENCH_IDX, MODEL_REASONING, MODEL_PROVIDERS,
)
from benchpress.all_methods import predict_benchpress_scores
sys.stdout = _old_stdout

def predict_all(M_input=None):
    """Run BenchPress on the matrix and return predictions."""
    if M_input is None:
        M_input = M_FULL.copy()
    return predict_benchpress_scores(M_input)

def format_predictions(predictions, model_filter=None, bench_filter=None,
                       only_missing=True, fmt='csv', confidence=None):
    """Format predictions as CSV or JSON rows."""
    rows = []
    obs = ~np.isnan(M_FULL) if only_missing else np.ones_like(OBSERVED)

    for i in range(predictions.shape[0]):
        mid = MODEL_IDS[i]
        if model_filter and mid != model_filter:
            continue
        for j in range(predictions.shape[1]):
            bid = BENCH_IDS[j]
            if bench_filter and bid != bench_filter:
                continue
            if only_missing and OBSERVED[i, j]:
                continue
            pred = predictions[i, j]
            actual = M_FULL[i, j] if OBSERVED[i, j] else None
            row = {
                'model': mid,
                'model_name': MODEL_NAMES[mid],
                'benchmark': bid,
                'benchmark_name': BENCH_NAMES[bid],
                'predicted': round(float(pred), 1) if np.isfinite(pred) else None,
                'actual': round(float(actual), 1) if actual is not None else None,
                'is_observed': bool(OBSERVED[i, j]),
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

    if fmt == 'json':
        return json.dumps(rows, indent=2)
    else:
        if not rows:
            return "No predictions to show."
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()

def add_model_scores(model_name, scores_str, M_input):
    """Add a new model row to the matrix with known scores, return augmented matrix."""
    # Parse scores
    scores = {}
    for pair in scores_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            print(f"Warning: skipping malformed score '{pair}' (expected bench=score)")
            continue
        bench, val = pair.split('=', 1)
        bench = bench.strip()
        val = float(val.strip())
        if bench not in BENCH_IDX:
            print(f"Warning: benchmark '{bench}' not found. Available: {', '.join(BENCH_IDS[:10])}...")
            continue
        scores[bench] = val

    if not scores:
        print("Error: no valid scores provided.")
        sys.exit(1)

    # Add new row
    new_row = np.full((1, N_BENCH), np.nan)
    for bench, val in scores.items():
        new_row[0, BENCH_IDX[bench]] = val

    M_aug = np.vstack([M_input, new_row])
    return M_aug, scores

def predict_for_new_model(M_aug):
    """Run BenchPress on the augmented matrix (with the new model row appended)."""
    return predict_benchpress_scores(M_aug)

def confidence_lookup(conf_result):
    """Convert confidence arrays into a cell-index lookup."""
    return {
        tuple(cell): {
            'method': conf_result['method'],
            'uncertainty': conf_result['uncertainty'][idx],
            'lower': conf_result['lower'][idx],
            'upper': conf_result['upper'][idx],
        }
        for idx, cell in enumerate(conf_result['cells'])
    }

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
    parser.add_argument('--format', '-f', type=str, default='csv', choices=['csv', 'json'],
                        help='Output format (default: csv)')
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
                        help='Include calibrated 90%% confidence intervals')
    parser.add_argument('--confidence-artifact', type=str, default=None,
                        help='Path to confidence artifact (default: package artifact path)')

    args = parser.parse_args()

    # ── List modes ──
    if args.list_models:
        print(f"{'Model ID':<30s} {'Display Name':<35s} {'Provider':<15s} {'Reasoning':<10s} {'#Scores'}")
        print('-' * 100)
        for i, mid in enumerate(MODEL_IDS):
            n = int(OBSERVED[i].sum())
            print(f"{mid:<30s} {MODEL_NAMES[mid]:<35s} {MODEL_PROVIDERS[i]:<15s} "
                  f"{'Y' if MODEL_REASONING[i] else 'N':<10s} {n}")
        return

    if args.list_benchmarks:
        from benchpress.evaluation_harness import BENCH_CATS
        print(f"{'Benchmark ID':<30s} {'Display Name':<35s} {'Category':<20s} {'#Models'}")
        print('-' * 95)
        for j, bid in enumerate(BENCH_IDS):
            n = int(OBSERVED[:, j].sum())
            print(f"{bid:<30s} {BENCH_NAMES[bid]:<35s} {BENCH_CATS[j]:<20s} {n}")
        return

    # ── Add-model mode ──
    if args.add_model:
        if not args.scores:
            print("Error: --add-model requires --scores")
            sys.exit(1)
        M_input = M_FULL.copy()
        M_aug, known = add_model_scores(args.add_model, args.scores, M_input)

        # Run real BenchPress on the augmented matrix.
        M_pred_aug = predict_for_new_model(M_aug)
        new_predictions = M_pred_aug[-1]
        conf_by_cell = None
        if args.confidence:
            from benchpress.methods.confidence import predict_confidence_intervals
            cells = [(N_MODELS, j) for j in range(N_BENCH)
                     if not np.isfinite(M_aug[N_MODELS, j])]
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
        for j in range(N_BENCH):
            bid = BENCH_IDS[j]
            is_known = bid in known
            pred_val = new_predictions[j]
            pred_str = f"{pred_val:>10.1f}" if np.isfinite(pred_val) else f"{'n/a':>10s}"
            known_str = f"({known[bid]:>5.1f})" if is_known else ""
            if args.confidence and conf_by_cell is not None and (N_MODELS, j) in conf_by_cell:
                conf = conf_by_cell[(N_MODELS, j)]
                interval = f"[{conf['lower']:.1f}, {conf['upper']:.1f}]"
                print(f"  {BENCH_NAMES[bid]:<33s} {pred_str}  {interval:>23s}  {known_str:>10s}")
            else:
                print(f"  {BENCH_NAMES[bid]:<33s} {pred_str}  {known_str:>10s}")
        return

    # ── Standard prediction mode ──
    if args.model and args.model not in MODEL_IDX:
        print(f"Error: model '{args.model}' not found.")
        print(f"Use --list-models to see available models.")
        sys.exit(1)
    if args.benchmark and args.benchmark not in BENCH_IDX:
        print(f"Error: benchmark '{args.benchmark}' not found.")
        print(f"Use --list-benchmarks to see available benchmarks.")
        sys.exit(1)

    # Single-cell prediction
    if args.model and args.benchmark:
        i = MODEL_IDX[args.model]
        j = BENCH_IDX[args.benchmark]
        if OBSERVED[i, j]:
            print(f"{MODEL_NAMES[args.model]} on {BENCH_NAMES[args.benchmark]}: "
                  f"{M_FULL[i, j]:.1f} (observed)")
        else:
            M_pred = predict_all()
            pred = M_pred[i, j]
            if args.confidence:
                from benchpress.methods.confidence import predict_confidence_intervals
                conf = predict_confidence_intervals(
                    M_FULL,
                    M_pred=M_pred,
                    artifact_path=args.confidence_artifact,
                    cells=[(i, j)],
                )
                print(f"{MODEL_NAMES[args.model]} on {BENCH_NAMES[args.benchmark]}: "
                      f"{pred:.1f} (predicted), 90% interval "
                      f"[{conf['lower'][0]:.1f}, {conf['upper'][0]:.1f}]")
            else:
                print(f"{MODEL_NAMES[args.model]} on {BENCH_NAMES[args.benchmark]}: "
                      f"{pred:.1f} (predicted)")
        return

    # Predict and output
    M_pred = predict_all()
    conf_by_cell = None
    if args.confidence:
        from benchpress.methods.confidence import predict_confidence_intervals
        cells = []
        for i, mid in enumerate(MODEL_IDS):
            if args.model and mid != args.model:
                continue
            for j, bid in enumerate(BENCH_IDS):
                if args.benchmark and bid != args.benchmark:
                    continue
                if not args.all and OBSERVED[i, j]:
                    continue
                cells.append((i, j))
        conf_by_cell = confidence_lookup(predict_confidence_intervals(
            M_FULL,
            M_pred=M_pred,
            artifact_path=args.confidence_artifact,
            cells=cells,
        ))
    result = format_predictions(M_pred,
                                model_filter=args.model,
                                bench_filter=args.benchmark,
                                only_missing=not args.all,
                                fmt=args.format,
                                confidence=conf_by_cell)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        n = result.count('\n') - 1 if args.format == 'csv' else result.count('"model"')
        print(f"Wrote {n} predictions to {args.output}")
    else:
        print(result)

if __name__ == '__main__':
    main()
