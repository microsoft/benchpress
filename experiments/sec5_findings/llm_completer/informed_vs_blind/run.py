#!/usr/bin/env python3
"""GPT-5.5 informed-vs-blind prediction on the §4 method-comparison folds."""

import sys, os, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from benchpress.call_model import get_client
from shared import (
    LLM_MODELS, PROTOCOL,
    load_method_comparison_folds, fold_metadata,
    run_llm_predictions, run_benchpress_baseline,
    evaluate_predictions, save_results,
    fold_has_valid_result, drop_fold_results, load_protocol_results,
)


def load_results():
    return load_protocol_results(
        RESULTS_PATH,
        {"protocol": PROTOCOL, "informed": {}, "blind": {}, "bp": []},
    )


def save_all_results(data):
    save_results(data, RESULTS_PATH)


def run(models=None, fold_ids=None, dry_run=False, tensor_parallel=1,
        batch_size=10, max_tokens=16384, force=False):
    models = models or LLM_MODELS
    all_data = load_results()
    all_data["protocol"] = PROTOCOL
    for condition in ["informed", "blind"]:
        all_data.setdefault(condition, {})
    all_data.setdefault("bp", [])
    folds = load_method_comparison_folds()
    if fold_ids is None:
        fold_ids = list(range(len(folds)))
    else:
        fold_ids = [int(f) for f in fold_ids]

    # BenchPress baseline
    bp_data = all_data["bp"]
    for fold_id in fold_ids:
        if fold_has_valid_result(bp_data, fold_id) and not force:
            continue
        if any(r.get("fold_id") == fold_id for r in bp_data):
            bp_data = drop_fold_results(bp_data, fold_id)
            all_data["bp"] = bp_data
        M_train, test_cells = folds[fold_id]
        meta = fold_metadata(fold_id)
        bp_metrics = run_benchpress_baseline(M_train, test_cells, fold_id)
        bp_data.append({
            **meta,
            "medape": bp_metrics["medape"],
            "medae": float(bp_metrics.get("medae", np.nan)),
            "n": bp_metrics["n"],
            "n_total": bp_metrics["n_total"],
            "coverage": bp_metrics["coverage"],
            "raw_predictions": bp_metrics.get("raw_predictions", []),
        })
        print(f"  BenchPress fold={fold_id}: MedAPE={bp_metrics['medape']:.1f}%")
        save_all_results(all_data)

    # Each LLM
    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")
        model_seen = any(model in all_data[condition] for condition in ["informed", "blind"])
        if model_seen:
            print(f"  Loaded existing results for {model}")

        for condition in ["informed", "blind"]:
            condition_data = all_data[condition]
            if model not in condition_data:
                condition_data[model] = []

            for fold_id in fold_ids:
                done = fold_has_valid_result(condition_data[model], fold_id)
                if done and not force:
                    print(f"  {condition} fold={fold_id}: already done")
                    continue
                if any(r.get("fold_id") == fold_id for r in condition_data[model]):
                    print(f"  {condition} fold={fold_id}: rerunning prior result")
                    condition_data[model] = drop_fold_results(condition_data[model], fold_id)

                M_train, test_cells = folds[fold_id]
                meta = fold_metadata(fold_id)
                is_blind = condition == "blind"
                print(f"\n  --- {condition}, fold={fold_id} (seed={meta['seed']}, fold={meta['fold']}) ---")

                if dry_run:
                    rng_seed = fold_id + sum(ord(c) for c in model + condition)
                    rng = np.random.RandomState(rng_seed)
                    bp_m = run_benchpress_baseline(M_train, test_cells, fold_id)
                    noise = rng.uniform(0.8, 1.2)
                    penalty = 1.3 if is_blind else 1.0
                    condition_data[model].append({
                        **meta,
                        "medape": bp_m["medape"] * noise * penalty,
                        "medae": float(bp_m.get("medae", np.nan)) * noise * penalty,
                        "n": bp_m["n"],
                        "n_total": bp_m["n_total"],
                        "coverage": 100.0,
                        "raw_predictions": bp_m["raw_predictions"],
                    })
                    save_all_results(all_data)
                    continue

                try:
                    client = get_client(model, tensor_parallel_size=tensor_parallel)
                    art_dir = os.path.join(SCRIPT_DIR, "predictions",
                                           f"{condition}_fold{fold_id:03d}_{model}")
                    preds, usage = run_llm_predictions(
                        client, model, M_train, test_cells, blind=is_blind,
                        batch_size=batch_size, max_tokens=max_tokens,
                        save_dir=art_dir,
                    )
                    llm_metrics = evaluate_predictions(preds, test_cells, fold_id)
                    condition_data[model].append({
                        **meta,
                        "medape": llm_metrics["medape"],
                        "medae": float(llm_metrics.get("medae", np.nan)),
                        "n": llm_metrics["n"],
                        "n_total": llm_metrics.get("n_total", len(test_cells)),
                        "coverage": llm_metrics.get("coverage", 0.0),
                        "tokens": usage,
                        "raw_predictions": llm_metrics.get("raw_predictions", []),
                    })
                    print(f"    MedAPE={llm_metrics['medape']:.1f}%, "
                          f"cov={llm_metrics.get('coverage', 0):.0f}%")
                except Exception as e:
                    print(f"    FAILED: {e}")

                save_all_results(all_data)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--results-path", default=RESULTS_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RESULTS_PATH = args.results_path
    fold_ids = args.fold_ids
    if fold_ids is None and args.fold_limit is not None:
        fold_ids = list(range(args.fold_limit))
    run(models=args.models, fold_ids=fold_ids, dry_run=args.dry_run,
        tensor_parallel=args.tensor_parallel, batch_size=args.batch_size,
        max_tokens=args.max_tokens, force=args.force)
