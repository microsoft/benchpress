#!/usr/bin/env python3
"""Shared utilities for the §5.3 LLM score-prediction diagnostics."""

import json
import os
import re
import time
from copy import deepcopy
from collections import defaultdict

import numpy as np

from benchpress.evaluation_harness import (
    M_FULL, OBSERVED, N_MODELS, N_BENCH, MODEL_IDS, BENCH_IDS,
    MODEL_NAMES, BENCH_NAMES, MODEL_REASONING, BENCH_CATS,
    benchmark_mean_median_metric,
    compute_prediction_error,
    load_folds,
)
from benchpress.artifact_utils import ensure_default_predictions
from benchpress.io_utils import load_json, write_json
from benchpress.methods.transforms import benchmark_scale, clamp_score_for_benchmark
from benchpress.stats import median_metric

DEFAULT_PREDICTIONS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "benchpress",
    "evaluation",
    "default_predictions",
    "benchpress_default",
    "predictions.npz",
))

# ── Models to evaluate ──
LLM_MODELS = [
    "gpt-5.5",
]

PROTOCOL = {
    "name": "method_comparison_s10_f3_bs42_ms1",
    "n_seeds": 10,
    "n_folds": 3,
    "base_seed": 42,
    "min_scores": 1,
    "fold_source": "benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json",
}


def protocol_matches(protocol):
    return protocol == PROTOCOL


def load_protocol_results(path, default_payload):
    """Load a protocol-matched results cache or return a fresh payload."""
    if os.path.exists(path):
        data = load_json(path)
        if protocol_matches(data.get("protocol", {})):
            return data
        print("Existing results.json uses an old protocol; starting a fresh cache.")
    return deepcopy(default_payload)


def load_method_comparison_folds():
    """Load the exact folds used by §4.2 method comparison."""
    return load_folds(
        n_seeds=PROTOCOL["n_seeds"],
        n_folds=PROTOCOL["n_folds"],
        base_seed=PROTOCOL["base_seed"],
        min_scores=PROTOCOL["min_scores"],
    )


def fold_metadata(fold_id):
    """Map global fold_id to the persisted seed/fold metadata."""
    return {
        "fold_id": int(fold_id),
        "seed": int(PROTOCOL["base_seed"] + fold_id // PROTOCOL["n_folds"]),
        "fold": int(fold_id % PROTOCOL["n_folds"]),
    }


# ══════════════════════════════════════════════════════════════
#  PROMPT BUILDING
# ══════════════════════════════════════════════════════════════

def format_matrix_csv(M, obs_mask, include_models=None, blind=False):
    """Format observed matrix as CSV for the system prompt.
    
    If blind=True, replace model/benchmark names with anonymous labels.
    """
    if blind:
        bench_headers = [f"B{j}" for j in range(N_BENCH)]
    else:
        bench_headers = list(BENCH_IDS)
    
    lines = ["model,reasoning," + ",".join(bench_headers)]
    model_indices = range(N_MODELS) if include_models is None else include_models
    
    for idx, i in enumerate(model_indices):
        if blind:
            name = f"M{idx}"
        else:
            name = MODEL_NAMES[MODEL_IDS[i]]
        reasoning = "Y" if MODEL_REASONING[i] else "N"
        vals = []
        for j in range(N_BENCH):
            if obs_mask[i, j]:
                v = M[i, j]
                vals.append(str(int(v)) if v == int(v) else f"{v:.1f}")
            else:
                vals.append("?")
        lines.append(f"{name},{reasoning},{','.join(vals)}")
    return "\n".join(lines)


def build_system_prompt(matrix_csv, blind=False):
    """Build system prompt with matrix context."""
    if blind:
        bench_info_str = "(Benchmark identifiers and metadata are anonymized.)"
    else:
        bench_info = []
        for j, bid in enumerate(BENCH_IDS):
            cat = BENCH_CATS[j]
            scale = benchmark_scale(bid)
            bench_info.append(f"  {bid}: {BENCH_NAMES[bid]} [{cat}] scale={scale}")
        bench_info_str = "Benchmark definitions:\n" + "\n".join(bench_info)

    return f"""You are a matrix completion system for LLM benchmark scores.

PURPOSE: We have a model × benchmark matrix of evaluation scores with some cells filled. Your job is to predict the missing scores as accurately as possible.

KEY STRUCTURAL FACTS:
- The matrix is approximately rank-2: two latent factors explain most of the variance.
- Most benchmarks use 0-100% accuracy. Exceptions: chatbot_arena_elo (Elo ~1000-1500), codeforces_rating (~800-2200).
- reasoning=Y models tend to score higher on hard math/reasoning benchmarks.
- Some benchmarks are near-saturated (most models score 90%+).

{bench_info_str}

Observed matrix ('?' = missing):
{matrix_csv}

RESPONSE FORMAT: Return ONLY valid JSON, no commentary, no markdown fences. Format:
{{"model_name": {{"benchmark_id": score, ...}}, ...}}

Do NOT wrap the answer in a top-level "predictions" list or any other schema.
Use the column headers as benchmark keys and the exact model names from the matrix."""


def build_prediction_request(
    batch_models,
    obs_mask,
    blind=False,
    model_index_map=None,
    target_benches_by_model=None,
):
    """Build user prompt for a batch of models to predict."""
    requests = []
    for i in batch_models:
        if blind and model_index_map is not None:
            name = f"M{model_index_map[i]}"
            bench_label = lambda j: f"B{j}"
        else:
            name = MODEL_NAMES[MODEL_IDS[i]]
            bench_label = lambda j: BENCH_IDS[j]
        
        target_benches = target_benches_by_model.get(i) if target_benches_by_model else None
        if target_benches is None:
            target_benches = [j for j in range(N_BENCH) if not obs_mask[i, j]]
        missing_benches = [bench_label(j) for j in target_benches]
        if missing_benches:
            known = []
            for j in range(N_BENCH):
                if obs_mask[i, j]:
                    v = M_FULL[i, j]
                    known.append(f"{bench_label(j)}={v:.1f}")
            requests.append(
                f"\n{name} (reasoning={'Y' if MODEL_REASONING[i] else 'N'}):\n"
                f"  Known: {', '.join(known[:15])}"
                + (f" + {len(known)-15} more" if len(known) > 15 else "")
                + f"\n  Predict: {', '.join(missing_benches)}"
            )
    return (
        "Predict the missing benchmark scores for these models. "
        "Return valid JSON only as {\"model_name\": {\"benchmark_id\": score, ...}, ...}. "
        "Do not use a top-level \"predictions\" key.\n"
        + "\n".join(requests)
    )


def _normalize_prediction_payload(data):
    """Coerce supported JSON schemas into {model_name: {benchmark_id: score}}."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("predictions"), list):
        items = data["predictions"]
    elif isinstance(data, dict):
        return data
    else:
        return {}

    normalized = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        model_name = item.get("model") or item.get("model_name")
        bench_id = item.get("benchmark") or item.get("bench_name") or item.get("bench")
        score = item.get("score")

        if not isinstance(model_name, str) or not isinstance(bench_id, str):
            continue

        normalized.setdefault(model_name, {})[bench_id] = score

    return normalized


# ══════════════════════════════════════════════════════════════
#  RESPONSE PARSING
# ══════════════════════════════════════════════════════════════

def parse_predictions(response_text, batch_models, blind=False, model_index_map=None):
    """Parse LLM's JSON response into {(i, j): score} dict."""
    text = response_text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    # Strip <think> blocks from reasoning models
    if '<think>' in text:
        think_end = text.rfind('</think>')
        if think_end >= 0:
            text = text[think_end + len('</think>'):].strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    data = _normalize_prediction_payload(data)
    if not isinstance(data, dict):
        return {}

    # Build name→index mapping
    if blind and model_index_map is not None:
        name_to_idx = {f"M{v}": k for k, v in model_index_map.items()}
    else:
        name_to_idx = {}
        for i in batch_models:
            name = MODEL_NAMES[MODEL_IDS[i]]
            name_to_idx[name] = i
            name_to_idx[name.strip()] = i

    # Build bench label→index mapping
    if blind:
        bench_to_idx = {f"B{j}": j for j in range(N_BENCH)}
    else:
        bench_to_idx = {bid: j for j, bid in enumerate(BENCH_IDS)}
        # Also try without underscores
        for j, bid in enumerate(BENCH_IDS):
            bench_to_idx[bid.replace('_', '')] = j

    predictions = {}
    for model_name, bench_scores in data.items():
        idx = name_to_idx.get(model_name)
        if idx is None:
            # Fuzzy match
            for known_name, known_idx in name_to_idx.items():
                if known_name.lower() in model_name.lower() or model_name.lower() in known_name.lower():
                    idx = known_idx
                    break
        if idx is None or not isinstance(bench_scores, dict):
            continue
        
        for bench_id, score in bench_scores.items():
            j = bench_to_idx.get(bench_id, bench_to_idx.get(bench_id.replace('_', ''), -1))
            if j < 0:
                continue
            try:
                score = float(score)
            except (ValueError, TypeError):
                continue
            # Clamp
            bid = BENCH_IDS[j]
            predictions[(idx, j)] = clamp_score_for_benchmark(score, bid)
    
    return predictions


# ══════════════════════════════════════════════════════════════
#  LLM PREDICTION RUNNER
# ══════════════════════════════════════════════════════════════

def run_llm_predictions(
    client,
    model: str,
    M_train: np.ndarray,
    test_cells: list,
    include_models=None,
    blind=False,
    batch_size=10,
    max_tokens=16384,
    save_dir=None,
):
    """Run LLM predictions on test_cells.
    
    Args:
        client: object with a chat(...) method
        model: Short model name (e.g., "gpt-5.5")
        M_train: Training matrix (NaN for hidden cells)
        test_cells: List of (i, j) tuples to predict
        include_models: If set, only show these model indices in the prompt
        blind: If True, anonymize all names
        batch_size: Number of models per API call
        max_tokens: Max output tokens
        save_dir: If set, save system prompt, per-batch I/O, and predictions
    
    Returns:
        predictions: {(i, j): score}
        usage: {'input_tokens': int, 'output_tokens': int}
    """
    obs_mask = ~np.isnan(M_train)
    
    # Build model index map for blind mode
    if include_models is not None:
        model_index_map = {i: idx for idx, i in enumerate(include_models)}
    else:
        model_index_map = {i: i for i in range(N_MODELS)}
    
    matrix_csv = format_matrix_csv(M_train, obs_mask, include_models, blind=blind)
    system_prompt = build_system_prompt(matrix_csv, blind=blind)
    
    # Group test cells by model
    model_to_cells = defaultdict(list)
    for i, j in test_cells:
        model_to_cells[i].append(j)
    model_indices = sorted(model_to_cells.keys())
    
    batches = [model_indices[s:s+batch_size] for s in range(0, len(model_indices), batch_size)]
    
    all_preds, batch_artifacts, total_in, total_out = _load_saved_artifacts(save_dir)
    
    for batch_idx, batch in enumerate(batches):
        n_cells = sum(len(model_to_cells[i]) for i in batch)
        target_pairs = {(i, j) for i in batch for j in model_to_cells[i]}
        if target_pairs and target_pairs.issubset(all_preds):
            print(
                f"    Batch {batch_idx+1}/{len(batches)}: {len(batch)} models, "
                f"{n_cells} cells... SKIP cached"
            )
            continue
        print(f"    Batch {batch_idx+1}/{len(batches)}: {len(batch)} models, {n_cells} cells...", end=" ", flush=True)
        
        user_prompt = build_prediction_request(
            batch,
            obs_mask,
            blind=blind,
            model_index_map=model_index_map,
            target_benches_by_model=model_to_cells,
        )
        
        max_retries = 4
        backoff = 10  # seconds
        for attempt in range(max_retries):
            try:
                resp = client.chat(
                    model=model,
                    user_message=user_prompt,
                    system_message=system_prompt,
                    max_tokens=max_tokens,
                )
                preds = parse_predictions(resp.content, batch, blind=blind, model_index_map=model_index_map)
                all_preds.update(preds)
                total_in += resp.input_tokens
                total_out += resp.output_tokens
                print(f"OK ({len(preds)}/{n_cells})")
                batch_artifacts[batch_idx] = {
                    "batch_idx": batch_idx,
                    "user_prompt": user_prompt,
                    "raw_response": resp.content,
                    "parsed_n": len(preds),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                }
                if save_dir:
                    _save_artifacts(save_dir, system_prompt, batch_artifacts,
                                    all_preds, test_cells)
                break
            except Exception as e:
                retryable = any(s in str(e) for s in ("429", "timed out", "timeout", "409", "500", "502", "503"))
                if retryable and attempt < max_retries - 1:
                    wait = backoff * (2 ** attempt)
                    print(f"RETRY {attempt+1}/{max_retries-1} in {wait}s ({e})")
                    time.sleep(wait)
                    continue
                print(f"FAILED ({e})")
                batch_artifacts[batch_idx] = {
                    "batch_idx": batch_idx,
                    "user_prompt": user_prompt,
                    "error": str(e),
                }
                if save_dir:
                    _save_artifacts(save_dir, system_prompt, batch_artifacts,
                                    all_preds, test_cells)
                raise RuntimeError(
                    f"LLM batch {batch_idx + 1}/{len(batches)} failed after "
                    f"{attempt + 1} attempts: {e}"
                ) from e
        
        if batch_idx < len(batches) - 1:
            time.sleep(3)
    
    # Save all artifacts if requested
    if save_dir:
        _save_artifacts(save_dir, system_prompt, batch_artifacts,
                        all_preds, test_cells)
    
    return all_preds, {'input_tokens': total_in, 'output_tokens': total_out}


def _load_saved_artifacts(save_dir):
    """Load per-batch cached predictions so interrupted LLM folds resume."""
    if not save_dir or not os.path.isdir(save_dir):
        return {}, {}, 0, 0

    preds = {}
    pred_path = os.path.join(save_dir, "predictions.json")
    if os.path.exists(pred_path):
        data = load_json(pred_path)
        for key, value in data.get("predictions", {}).items():
            try:
                i, j = key.split(",", 1)
                preds[(int(i), int(j))] = float(value)
            except (TypeError, ValueError):
                continue

    batch_artifacts = {}
    total_in, total_out = 0, 0
    for fname in sorted(os.listdir(save_dir)):
        if not (fname.startswith("batch_") and fname.endswith(".json")):
            continue
        art = load_json(os.path.join(save_dir, fname))
        batch_idx = art.get("batch_idx")
        if not isinstance(batch_idx, int):
            continue
        batch_artifacts[batch_idx] = art
        if "error" not in art:
            total_in += int(art.get("input_tokens", 0) or 0)
            total_out += int(art.get("output_tokens", 0) or 0)

    return preds, batch_artifacts, total_in, total_out


def _save_artifacts(save_dir, system_prompt, batch_artifacts, preds, test_cells):
    """Save system prompt, per-batch I/O, and parsed predictions."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "system_prompt.txt"), 'w') as f:
        f.write(system_prompt)
    for art in batch_artifacts.values():
        fname = f"batch_{art['batch_idx']:03d}.json"
        write_json(os.path.join(save_dir, fname), art, indent=2)
    pred_out = {
        "test_cells": [[int(i), int(j)] for i, j in test_cells],
        "predictions": {f"{i},{j}": float(v) for (i, j), v in preds.items()},
        "n_predicted": len(preds),
        "n_total": len(test_cells),
    }
    write_json(os.path.join(save_dir, "predictions.json"), pred_out, indent=2)
    print(f"    Artifacts saved → {save_dir}/ ({len(batch_artifacts)} batches, {len(preds)} preds)")


# ══════════════════════════════════════════════════════════════
#  BENCHPRESS BASELINE
# ══════════════════════════════════════════════════════════════

def _record_fold_predictions(M_pred, test_cells, fold_id):
    """Return raw per-cell predictions for one persisted §4 fold."""
    meta = fold_metadata(fold_id)
    raw_preds = []
    for i, j in test_cells:
        t, p = M_FULL[i, j], M_pred[i, j]
        if np.isfinite(t) and np.isfinite(p):
            raw_preds.append({
                **meta,
                'i': int(i), 'j': int(j),
                'actual': float(t), 'predicted': float(p),
            })
    return raw_preds


def _fold_metrics_from_matrix(M_pred, test_cells):
    if not test_cells:
        return {'medape': float('nan'), 'medae': float('nan'), 'n': 0}
    m = compute_prediction_error(M_FULL, M_pred, test_set=test_cells, aggregation='pool')
    return {
        'medape': float(m['medape']),
        'medae': float(m['medae']),
        'n': int(m['n']),
    }


def run_benchpress_baseline(M_train, test_cells, fold_id):
    """Read the canonical BenchPress default predictions for one §4 fold."""
    del M_train  # Baseline predictions are canonical cached artifacts.
    if not test_cells:
        return {'medape': float('nan'), 'medae': float('nan'),
                'n': 0, 'n_total': 0, 'coverage': 0.0, 'raw_predictions': []}
    if not os.path.exists(DEFAULT_PREDICTIONS_PATH):
        ensure_default_predictions()

    cache = np.load(DEFAULT_PREDICTIONS_PATH, allow_pickle=False)
    fold_mask = cache['fold_id'] == fold_id
    cached = {
        (int(i), int(j)): (float(actual), float(predicted))
        for i, j, actual, predicted in zip(
            cache['test_i'][fold_mask],
            cache['test_j'][fold_mask],
            cache['actual'][fold_mask],
            cache['predicted'][fold_mask],
        )
    }
    requested = [(int(i), int(j)) for i, j in test_cells]
    missing = [cell for cell in requested if cell not in cached]
    if missing:
        raise ValueError(
            f"Canonical BenchPress cache lacks {len(missing)} requested cells "
            f"for fold_id={fold_id}; first missing={missing[:5]}"
        )

    M_bp = M_FULL.copy()
    raw_preds = []
    meta = fold_metadata(fold_id)
    for i, j in requested:
        actual, predicted = cached[(i, j)]
        M_bp[i, j] = predicted
        raw_preds.append({
            **meta,
            'i': i,
            'j': j,
            'actual': actual,
            'predicted': predicted,
        })
    metrics = _fold_metrics_from_matrix(M_bp, requested)
    return {
        **metrics,
        'n_total': len(test_cells),
        'coverage': 100.0,
        'raw_predictions': raw_preds,
    }


def evaluate_predictions(preds, test_cells, fold_id):
    """Compute fold metrics from an LLM predictions dict."""
    # Build full M_pred: start from M_FULL, overwrite test cells with predictions
    M_pred = M_FULL.copy()
    valid_cells = []
    raw_preds = []
    for i, j in test_cells:
        score = preds.get((i, j))
        if score is not None and np.isfinite(score):
            M_pred[i, j] = score
            valid_cells.append((i, j))
            t = M_FULL[i, j]
            if np.isfinite(t):
                raw_preds.append({'i': int(i), 'j': int(j),
                                  'actual': float(t), 'predicted': float(score)})

    if not valid_cells:
        return {'medape': float('nan'), 'medae': float('nan'),
                'n': 0, 'coverage': 0.0,
                'n_total': len(test_cells),
                'raw_predictions': []}

    m = _fold_metrics_from_matrix(M_pred, valid_cells)
    meta = fold_metadata(fold_id)
    return {
        'medape': m['medape'],
        'medae': m['medae'],
        'n': m['n'],
        'n_total': len(test_cells),
        'coverage': len(valid_cells) / len(test_cells) * 100,
        'raw_predictions': [{**meta, **r} for r in raw_preds],
    }


def is_valid_result_entry(entry):
    """Return True only for entries with finite score-error metrics and nonzero coverage."""
    medape = entry.get("medape")
    if medape is None or not np.isfinite(medape):
        return False
    if "n" in entry and entry.get("n", 0) <= 0:
        return False
    if "coverage" in entry and entry.get("coverage", 0.0) <= 0:
        return False
    return True


def fold_has_valid_result(entries, fold_id):
    expected_meta = fold_metadata(fold_id)
    return any(
        all(r.get(k) == v for k, v in expected_meta.items())
        and is_valid_result_entry(r)
        for r in entries
    )


def drop_fold_results(entries, fold_id):
    return [r for r in entries if r.get("fold_id") != fold_id]


def save_results(results, path):
    """Save results to JSON (converting numpy types)."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    write_json(path, convert(results), indent=2)
    print(f"  Saved: {path}")
