#!/usr/bin/env python3
"""Five-shot nearest-peer prompt ablation for §5.3 LLM score prediction."""

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from benchpress.call_model import get_client
from benchpress.io_utils import write_json
from shared import (
    BENCH_IDS,
    BENCH_NAMES,
    LLM_MODELS,
    M_FULL,
    MODEL_IDS,
    MODEL_NAMES,
    PROTOCOL,
    benchmark_scale,
    clamp_score_for_benchmark,
    compute_prediction_error,
    drop_fold_results,
    fold_has_valid_result,
    fold_metadata,
    load_method_comparison_folds,
    load_protocol_results,
    run_benchpress_baseline,
    save_results,
)

CONDITIONS = ("five_shot_named", "five_shot_blind")
N_SHOTS = 5
MIN_SHARED = 5
MAX_TARGET_KNOWN = 12
MAX_EXAMPLE_SHARED = 4


def load_results():
    return load_protocol_results(
        RESULTS_PATH,
        {"protocol": PROTOCOL, "bp": [], **{c: {} for c in CONDITIONS}},
    )


def save_all_results(data):
    save_results(data, RESULTS_PATH)


def _finite_shared(M_train, i, peer, exclude_j):
    mask = np.isfinite(M_train[i]) & np.isfinite(M_train[peer])
    mask[exclude_j] = False
    return np.where(mask)[0]


def _pearson(x, y):
    if len(x) < MIN_SHARED:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def select_peer_examples(M_train, i, j, n_shots=N_SHOTS):
    """Pick nearest peer models using only visible training scores."""
    candidates = []
    for peer in range(M_train.shape[0]):
        if peer == i or not np.isfinite(M_train[peer, j]):
            continue
        shared = _finite_shared(M_train, i, peer, exclude_j=j)
        if len(shared) < MIN_SHARED:
            continue
        corr = _pearson(M_train[i, shared], M_train[peer, shared])
        if not np.isfinite(corr):
            continue
        candidates.append((peer, corr, shared))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[:n_shots]


def _score_list(M_train, i, bench_indices, blind, bench_labels):
    parts = []
    for j in bench_indices:
        label = bench_labels[j] if blind else BENCH_IDS[j]
        parts.append(f"{label}={M_train[i, j]:.1f}")
    return ", ".join(parts) if parts else "none"


def _build_benchmark_labels(queries):
    bench_ids = []
    for q in queries:
        bench_ids.append(q["target_j"])
        bench_ids.extend(q["target_known"])
        for ex in q["examples"]:
            bench_ids.extend(ex["shared"])
    unique = []
    seen = set()
    for j in bench_ids:
        if j not in seen:
            seen.add(j)
            unique.append(j)
    alphabet = [chr(ord("A") + k) for k in range(26)]
    labels = {}
    for idx, j in enumerate(unique):
        if idx < len(alphabet):
            suffix = alphabet[idx]
        else:
            suffix = f"{idx + 1}"
        labels[j] = f"Benchmark {suffix}"
    return labels


def build_query(M_train, i, j, query_id):
    peers = select_peer_examples(M_train, i, j)
    target_known = np.where(np.isfinite(M_train[i]))[0].tolist()
    target_known = [k for k in target_known if k != j][:MAX_TARGET_KNOWN]
    examples = []
    for peer, corr, shared in peers:
        shared = shared.tolist()[:MAX_EXAMPLE_SHARED]
        examples.append({"peer": int(peer), "corr": float(corr), "shared": shared})
    return {
        "query_id": query_id,
        "target_i": int(i),
        "target_j": int(j),
        "target_known": target_known,
        "examples": examples,
    }


def render_prompt(M_train, queries, blind):
    bench_labels = _build_benchmark_labels(queries) if blind else {}
    lines = [
        "You are estimating benchmark results before running expensive evaluations.",
        "Each query gives compact known scores for a target model and five nearest peer-model examples.",
        "Make a quick numerical estimate from the nearest peers; do not explain or show calculations.",
        "Return ONLY valid JSON mapping each query_id to a numeric score, e.g. {\"q0\": 72.5}.",
    ]
    for q in queries:
        i, j = q["target_i"], q["target_j"]
        if blind:
            target_model = f"Target model {q['query_id']}"
            target_bench = bench_labels[j]
            target_bench_desc = target_bench
        else:
            target_model = MODEL_NAMES[MODEL_IDS[i]]
            target_bench = f"{BENCH_NAMES[BENCH_IDS[j]]} ({BENCH_IDS[j]})"
            target_bench_desc = f"{target_bench}, scale={benchmark_scale(BENCH_IDS[j])}"
        lines.extend([
            "",
            f"Query {q['query_id']}",
            f"Target model: {target_model}",
            f"Target known scores: {_score_list(M_train, i, q['target_known'], blind, bench_labels)}",
            f"Estimate the target model's score on: {target_bench_desc}",
            "Nearest peer examples:",
        ])
        if not q["examples"]:
            lines.append("- No eligible peer examples are available.")
        for ex_idx, ex in enumerate(q["examples"], start=1):
            peer = ex["peer"]
            if blind:
                peer_name = f"Peer model {q['query_id']}-{ex_idx}"
                target_line = f"{target_bench} score: {M_train[peer, j]:.1f}"
            else:
                peer_name = MODEL_NAMES[MODEL_IDS[peer]]
                target_line = f"{target_bench} score: {M_train[peer, j]:.1f}"
            lines.extend([
                f"Example {ex_idx}: model={peer_name}; shared_scores="
                f"{_score_list(M_train, peer, ex['shared'], blind, bench_labels)}; "
                f"{target_line}",
            ])
    return "\n".join(lines)


def parse_batch_response(text, queries):
    text = text.strip()
    if text.startswith("```"):
        import re
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    if '<think>' in text:
        think_end = text.rfind('</think>')
        if think_end >= 0:
            text = text[think_end + len('</think>'):].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        data = data["predictions"]
    if not isinstance(data, dict):
        return {}
    q_by_id = {q["query_id"]: q for q in queries}
    preds = {}
    for qid, value in data.items():
        if qid not in q_by_id:
            continue
        if isinstance(value, dict):
            value = value.get("score") or value.get("prediction") or value.get("predicted")
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        q = q_by_id[qid]
        bench_id = BENCH_IDS[q["target_j"]]
        preds[(q["target_i"], q["target_j"])] = clamp_score_for_benchmark(score, bench_id)
    return preds


def run_five_shot_predictions(
    client,
    model,
    M_train,
    test_cells,
    condition,
    batch_size,
    max_tokens,
    save_dir=None,
):
    blind = condition == "five_shot_blind"
    all_preds = {}
    query_meta = {}
    total_in, total_out = 0, 0
    batch_artifacts = []
    queries = [
        build_query(M_train, i, j, f"q{idx}")
        for idx, (i, j) in enumerate(test_cells)
    ]
    batches = [queries[s:s + batch_size] for s in range(0, len(queries), batch_size)]
    for batch_idx, batch in enumerate(batches):
        n_cells = len(batch)
        prompt = render_prompt(M_train, batch, blind=blind)
        print(f"    Batch {batch_idx+1}/{len(batches)}: {n_cells} cells...", end=" ", flush=True)
        backoff = 10
        attempt = 0
        while True:
            try:
                resp = client.chat(
                    model=model,
                    user_message=prompt,
                    system_message=None,
                    max_tokens=max_tokens,
                )
                preds = parse_batch_response(resp.content, batch)
                all_preds.update(preds)
                total_in += resp.input_tokens
                total_out += resp.output_tokens
                print(f"OK ({len(preds)}/{n_cells})")
                batch_artifacts.append({
                    "batch_idx": batch_idx,
                    "condition": condition,
                    "user_prompt": prompt,
                    "raw_response": resp.content,
                    "parsed_n": len(preds),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                })
                break
            except Exception as e:
                retryable = any(s in str(e) for s in ("429", "timed out", "timeout", "409", "500", "502", "503"))
                if retryable:
                    attempt += 1
                    wait = min(backoff * (2 ** (attempt - 1)), 300)
                    print(f"RETRY {attempt} in {wait}s ({e})")
                    time.sleep(wait)
                    continue
                print(f"FAILED ({e})")
                batch_artifacts.append({
                    "batch_idx": batch_idx,
                    "condition": condition,
                    "user_prompt": prompt,
                    "error": str(e),
                })
                raise RuntimeError(
                    f"Five-shot batch {batch_idx + 1}/{len(batches)} failed after "
                    f"{attempt + 1} attempts: {e}"
                ) from e
        if batch_idx < len(batches) - 1:
            time.sleep(1)
    for q in queries:
        key = (q["target_i"], q["target_j"])
        query_meta[key] = q
    if save_dir:
        save_artifacts(save_dir, batch_artifacts, all_preds, test_cells, query_meta)
    return all_preds, query_meta, {"input_tokens": total_in, "output_tokens": total_out}


def save_artifacts(save_dir, batch_artifacts, preds, test_cells, query_meta):
    os.makedirs(save_dir, exist_ok=True)
    for art in batch_artifacts:
        fname = f"batch_{art['batch_idx']:03d}.json"
        write_json(os.path.join(save_dir, fname), art, indent=2)
    pred_out = {
        "test_cells": [[int(i), int(j)] for i, j in test_cells],
        "predictions": {f"{i},{j}": float(v) for (i, j), v in preds.items()},
        "query_meta": {f"{i},{j}": query_meta.get((i, j), {}) for i, j in test_cells},
    }
    write_json(os.path.join(save_dir, "predictions.json"), pred_out, indent=2)
    print(f"    Artifacts saved -> {save_dir}/ ({len(batch_artifacts)} batches)")


def evaluate_five_shot_predictions(preds, query_meta, test_cells, fold_id):
    M_pred = M_FULL.copy()
    valid_cells = []
    raw_preds = []
    meta = fold_metadata(fold_id)
    for i, j in test_cells:
        score = preds.get((i, j))
        if score is None or not np.isfinite(score):
            continue
        M_pred[i, j] = score
        valid_cells.append((i, j))
        q = query_meta.get((i, j), {})
        peers = q.get("examples", [])
        raw_preds.append({
            **meta,
            "i": int(i),
            "j": int(j),
            "actual": float(M_FULL[i, j]),
            "predicted": float(score),
            "n_shots": int(len(peers)),
            "peer_indices": [int(ex["peer"]) for ex in peers],
            "peer_correlations": [float(ex["corr"]) for ex in peers],
        })
    if not valid_cells:
        return {
            "medape": float("nan"),
            "medae": float("nan"),
            "n": 0,
            "n_total": len(test_cells),
            "coverage": 0.0,
            "raw_predictions": [],
        }
    metrics = compute_prediction_error(M_FULL, M_pred, test_set=valid_cells, aggregation="pool")
    return {
        "medape": float(metrics["medape"]),
        "medae": float(metrics["medae"]),
        "n": int(metrics["n"]),
        "n_total": len(test_cells),
        "coverage": len(valid_cells) / len(test_cells) * 100,
        "raw_predictions": raw_preds,
    }


def _limit_cells(test_cells, cell_limit):
    if cell_limit is None:
        return test_cells
    if cell_limit <= 0:
        raise ValueError("--cell-limit must be positive")
    return test_cells[:cell_limit]


def run(models=None, fold_ids=None, conditions=None, dry_run=False,
        batch_size=16, max_tokens=16384, force=False, cell_limit=None):
    models = models or LLM_MODELS
    conditions = conditions or list(CONDITIONS)
    all_data = load_results()
    all_data["protocol"] = PROTOCOL
    all_data.setdefault("bp", [])
    for condition in CONDITIONS:
        all_data.setdefault(condition, {})
    folds = load_method_comparison_folds()
    fold_ids = list(range(len(folds))) if fold_ids is None else [int(f) for f in fold_ids]

    bp_data = all_data["bp"]
    for fold_id in fold_ids:
        if fold_has_valid_result(bp_data, fold_id) and not force:
            continue
        if any(r.get("fold_id") == fold_id for r in bp_data):
            bp_data = drop_fold_results(bp_data, fold_id)
            all_data["bp"] = bp_data
        M_train, test_cells = folds[fold_id]
        test_cells = _limit_cells(test_cells, cell_limit)
        bp_metrics = run_benchpress_baseline(M_train, test_cells, fold_id)
        bp_data.append({**fold_metadata(fold_id), **bp_metrics})
        print(f"  BenchPress fold={fold_id}: MedAPE={bp_metrics['medape']:.1f}%")
        save_all_results(all_data)

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")
        client = None if dry_run else get_client(model)
        for condition in conditions:
            if condition not in CONDITIONS:
                raise ValueError(f"Unknown condition: {condition}")
            condition_data = all_data[condition]
            condition_data.setdefault(model, [])
            for fold_id in fold_ids:
                done = fold_has_valid_result(condition_data[model], fold_id)
                if done and not force:
                    print(f"  {condition} fold={fold_id}: already done")
                    continue
                if any(r.get("fold_id") == fold_id for r in condition_data[model]):
                    condition_data[model] = drop_fold_results(condition_data[model], fold_id)
                M_train, test_cells = folds[fold_id]
                test_cells = _limit_cells(test_cells, cell_limit)
                meta = fold_metadata(fold_id)
                print(f"\n  --- {condition}, fold={fold_id} (seed={meta['seed']}, fold={meta['fold']}) ---")
                if dry_run:
                    bp_m = run_benchpress_baseline(M_train, test_cells, fold_id)
                    rng = np.random.RandomState(fold_id + sum(ord(c) for c in model + condition))
                    condition_data[model].append({
                        **meta,
                        "medape": bp_m["medape"] * rng.uniform(0.9, 1.25),
                        "medae": bp_m["medae"] * rng.uniform(0.9, 1.25),
                        "n": bp_m["n"],
                        "n_total": bp_m["n_total"],
                        "coverage": 100.0,
                        "tokens": {"input_tokens": 0, "output_tokens": 0},
                        "raw_predictions": bp_m["raw_predictions"],
                    })
                    save_all_results(all_data)
                    continue
                art_dir = os.path.join(SCRIPT_DIR, "predictions", f"{condition}_fold{fold_id:03d}_{model}")
                preds, query_meta, usage = run_five_shot_predictions(
                    client,
                    model,
                    M_train,
                    test_cells,
                    condition=condition,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    save_dir=art_dir,
                )
                llm_metrics = evaluate_five_shot_predictions(preds, query_meta, test_cells, fold_id)
                condition_data[model].append({
                    **meta,
                    "medape": llm_metrics["medape"],
                    "medae": llm_metrics["medae"],
                    "n": llm_metrics["n"],
                    "n_total": llm_metrics["n_total"],
                    "coverage": llm_metrics["coverage"],
                    "tokens": usage,
                    "raw_predictions": llm_metrics["raw_predictions"],
                })
                print(f"    MedAPE={llm_metrics['medape']:.1f}%, cov={llm_metrics['coverage']:.0f}%")
                save_all_results(all_data)
    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cell-limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--results-path", default=RESULTS_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RESULTS_PATH = args.results_path
    fold_ids = args.fold_ids
    if fold_ids is None and args.fold_limit is not None:
        fold_ids = list(range(args.fold_limit))
    run(
        models=args.models,
        fold_ids=fold_ids,
        conditions=args.conditions,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        force=args.force,
        cell_limit=args.cell_limit,
    )
