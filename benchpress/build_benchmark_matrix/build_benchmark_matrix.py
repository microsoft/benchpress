"""
BenchPress score matrix.

Single source of truth for loading the (model × benchmark) score matrix from
`benchpress/data/llm_benchmark_data.json`. Used by all downstream experiments.

Quick start:
    from benchpress.build_benchmark_matrix import load_score_matrix
    df = load_score_matrix()                                      # deduped, thresholded paper matrix
    df, info = load_score_matrix(return_info=True)
    raw = load_score_matrix(m_threshold=0, b_threshold=0)         # original audit pool
    dedup = load_score_matrix(m_threshold=0, b_threshold=0, deduplicate=True)
                                                                    # original pool after canonicalization

Backward-compatible exports for legacy code:
    MODELS, BENCHMARKS, DATA  — tuple-shaped (read-only), pre-filtered to (15, 8).

CLI:
    python -m benchpress.build_benchmark_matrix                              # default (15, 8)
    python -m benchpress.build_benchmark_matrix --m-threshold 0 --b-threshold 0  # unfiltered
    python -m benchpress.build_benchmark_matrix --excel scores.xlsx
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ───────────────────────── Constants ─────────────────────────

#: Paper-canonical asymmetric thresholds (§3.1): keep models with ≥15
#: observations and benchmarks with ≥8 observations, iterated to fixed point.
DEFAULT_M_THRESHOLD: int = 15
DEFAULT_B_THRESHOLD: int = 8

# JSON path resolved relative to this package (works installed or in-place)
_DEFAULT_JSON = Path(__file__).resolve().parents[1] / "data" / "llm_benchmark_data.json"


# ───────────────────────── Data classes ─────────────────────────

@dataclass(frozen=True)
class CanonicalRule:
    """One canonicalization decision applied before threshold filtering."""

    drop_id: str
    keep_id: str
    family: str
    reason: str


@dataclass(frozen=True)
class FilterInfo:
    """Metadata returned by `load_score_matrix(..., return_info=True)`."""

    m_threshold: int
    b_threshold: int
    iterations: int
    n_models: int
    n_benchmarks: int
    n_observations: int
    fill_rate: float  # in [0, 1]
    dropped_models: tuple[str, ...]
    dropped_benchmarks: tuple[str, ...]
    excluded_models: tuple[str, ...] = ()
    excluded_benchmarks: tuple[str, ...] = ()
    model_canonical_rules: tuple[CanonicalRule, ...] = ()
    benchmark_canonical_rules: tuple[CanonicalRule, ...] = ()


MODEL_DROP_RULES: tuple[CanonicalRule, ...] = (
    CanonicalRule(
        drop_id="deepseek-v3.2-speciale",
        keep_id="deepseek-v3.2",
        family="DeepSeek-V3.2",
        reason="same base model; keep the default-effort row because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="kimi-k2-thinking",
        keep_id="kimi-k2",
        family="Kimi K2",
        reason="same base model with a different mode; keep the broader-coverage default row",
    ),
    CanonicalRule(
        drop_id="lfm2.5-1.2b-thinking",
        keep_id="lfm2.5-1.2b-instruct",
        family="LFM2.5-1.2B",
        reason="same base model with a different mode; keep the instruct/non-thinking row",
    ),
)

MODEL_FILL_RULES: tuple[CanonicalRule, ...] = (
    CanonicalRule(
        drop_id="gemma-3-1b",
        keep_id="gemma-3-1b-it",
        family="Gemma 3 1B",
        reason="same base model; many sources do not disambiguate base vs instruct, so use the base-only cells to fill missing instruct cells (small differences within evaluation noise)",
    ),
    CanonicalRule(
        drop_id="granite-3.3-8b-base",
        keep_id="granite-3.3-8b-instruct",
        family="Granite 3.3 8B",
        reason="same base model; many sources do not disambiguate base vs instruct, so use the base-only cells to fill missing instruct cells (small differences within evaluation noise)",
    ),
    CanonicalRule(
        drop_id="llama-3.2-1b",
        keep_id="llama-3.2-1b-instruct",
        family="Llama 3.2 1B",
        reason="same base model; many sources do not disambiguate base vs instruct, so use the base-only cells to fill missing instruct cells (small differences within evaluation noise)",
    ),
    CanonicalRule(
        drop_id="qwen-2.5-14b",
        keep_id="qwen-2.5-14b-instruct",
        family="Qwen2.5 14B",
        reason="same base model; many sources do not disambiguate base vs instruct, so use the base-only cells to fill missing instruct cells (small differences within evaluation noise)",
    ),
)

MODEL_CANONICAL_RULES: tuple[CanonicalRule, ...] = (
    *MODEL_DROP_RULES,
    *MODEL_FILL_RULES,
)

BENCHMARK_DROP_RULES: tuple[CanonicalRule, ...] = (
    CanonicalRule(
        drop_id="codeforces_avg8",
        keep_id="codeforces_rating",
        family="Codeforces",
        reason="same task family; keep Codeforces Rating because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="codeforces_pass8",
        keep_id="codeforces_rating",
        family="Codeforces",
        reason="same task family; keep Codeforces Rating because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="livecodebench_pro",
        keep_id="livecodebench",
        family="LiveCodeBench",
        reason="same benchmark family; keep the standard LiveCodeBench because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="tau1_bench_avg",
        keep_id="tau_bench_airline/tau_bench_retail",
        family="τ-bench",
        reason="composite average of existing domain benchmarks; keep per-domain benchmarks",
    ),
    CanonicalRule(
        drop_id="tau2_bench_avg",
        keep_id="tau2_bench_airline/tau2_bench_retail/tau2_bench_telecom",
        family="τ²-bench",
        reason="composite average of existing domain benchmarks; keep per-domain benchmarks",
    ),
    CanonicalRule(
        drop_id="apex_shortlist",
        keep_id="matharena_apex_2025",
        family="MathArena Apex",
        reason="subset/shortlist variant; keep MathArena Apex 2025 because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="healthbench_hard",
        keep_id="healthbench",
        family="HealthBench",
        reason="hard-subset variant; keep HealthBench because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="mmlu",
        keep_id="mmlu_pro",
        family="MMLU",
        reason="same benchmark family; keep MMLU-Pro because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="mmlu_redux",
        keep_id="mmlu_pro",
        family="MMLU",
        reason="same benchmark family; keep MMLU-Pro because it has broader coverage",
    ),
    CanonicalRule(
        drop_id="global_mmlu_lite",
        keep_id="mmlu_pro",
        family="MMLU",
        reason="same benchmark family; keep MMLU-Pro because it has broader coverage",
    ),
)

BENCHMARK_FILL_RULES: tuple[CanonicalRule, ...] = (
    CanonicalRule(
        drop_id="livecodebench_v5",
        keep_id="livecodebench",
        family="LiveCodeBench",
        reason="same benchmark family; use v5 only to fill missing standard LiveCodeBench cells, marking them non-canonical because the version differs",
    ),
    CanonicalRule(
        drop_id="livecodebench_v6",
        keep_id="livecodebench",
        family="LiveCodeBench",
        reason="same benchmark family; use v6 only to fill missing standard LiveCodeBench cells, marking them non-canonical because the version differs",
    ),
)

BENCHMARK_CANONICAL_RULES: tuple[CanonicalRule, ...] = (
    *BENCHMARK_DROP_RULES,
    *BENCHMARK_FILL_RULES,
)

DEFAULT_EXCLUDED_MODELS: tuple[str, ...] = tuple(r.drop_id for r in MODEL_CANONICAL_RULES)
DEFAULT_EXCLUDED_BENCHMARKS: tuple[str, ...] = tuple(r.drop_id for r in BENCHMARK_CANONICAL_RULES)


# ───────────────────────── JSON load ─────────────────────────

def _load_raw(json_path: str | Path | None = None) -> dict:
    """Read the canonical JSON. Cached at module level for legacy globals."""
    path = Path(json_path) if json_path else Path(
        os.environ.get("BENCHPRESS_DATA", _DEFAULT_JSON)
    )
    if not path.exists():
        raise FileNotFoundError(
            f"score matrix JSON not found at {path}. "
            "Run `python -m benchpress.download_data`, pass `json_path=`, "
            "or set `BENCHPRESS_DATA` env var."
        )
    with open(path) as f:
        return json.load(f)


# ───────────────────────── Helpers ─────────────────────────

def _filter_status(scores: list[dict], status_filter: str | Sequence[str] | None) -> list[dict]:
    if status_filter is None or status_filter == "all":
        return list(scores)
    if isinstance(status_filter, str):
        status_filter = [status_filter]
    keep = set(status_filter)
    return [s for s in scores if (s.get("audit_status") or "pending") in keep]


def _apply_canonical_rules(
    scores: list[dict],
    *,
    exclude_models: set[str],
    exclude_benchmarks: set[str],
) -> list[dict]:
    """Apply canonical representative rules before threshold filtering.

    This does not mutate or average raw scores. It removes rows/columns whose
    canonical representative is another model/benchmark in the original audit
    pool. The raw matrix remains available with canonicalization disabled.
    """
    if not exclude_models and not exclude_benchmarks:
        return scores
    return [
        s for s in scores
        if s["model_id"] not in exclude_models
        and s["benchmark_id"] not in exclude_benchmarks
    ]


def _apply_benchmark_fill_rules(
    scores: list[dict],
    *,
    fill_rules: Sequence[CanonicalRule],
    exclude_models: set[str],
) -> list[dict]:
    """Use a non-canonical benchmark variant only when the canonical cell is missing."""
    if not fill_rules:
        return scores
    out = list(scores)
    existing = {
        (s["model_id"], s["benchmark_id"])
        for s in scores
        if s.get("score") is not None and s["model_id"] not in exclude_models
    }
    for rule in fill_rules:
        for s in scores:
            if (
                s["benchmark_id"] != rule.drop_id
                or s["model_id"] in exclude_models
                or s.get("score") is None
            ):
                continue
            target = (s["model_id"], rule.keep_id)
            if target in existing:
                continue
            filled = dict(s)
            filled["benchmark_id"] = rule.keep_id
            filled["matches_canonical"] = False
            filled["_canonical_fill_from"] = rule.drop_id
            note = (
                f"Canonical-fill from {rule.drop_id}: same benchmark family but a "
                "different version, so the value may differ from standard "
                f"{rule.keep_id}; used only because the canonical cell is missing."
            )
            filled["notes"] = f"{s.get('notes', '').rstrip()} {note}".strip()
            out.append(filled)
            existing.add(target)
    return out


def _apply_model_fill_rules(
    scores: list[dict],
    *,
    fill_rules: Sequence[CanonicalRule],
    exclude_benchmarks: set[str],
) -> list[dict]:
    """Use a non-canonical model variant only when the canonical cell is missing.

    Mirror of ``_apply_benchmark_fill_rules`` but for model-side merges
    (e.g. ``gemma-3-1b`` -> ``gemma-3-1b-it`` when sources do not
    disambiguate base vs instruct).
    """
    if not fill_rules:
        return scores
    out = list(scores)
    existing = {
        (s["model_id"], s["benchmark_id"])
        for s in scores
        if s.get("score") is not None and s["benchmark_id"] not in exclude_benchmarks
    }
    for rule in fill_rules:
        for s in scores:
            if (
                s["model_id"] != rule.drop_id
                or s["benchmark_id"] in exclude_benchmarks
                or s.get("score") is None
            ):
                continue
            target = (rule.keep_id, s["benchmark_id"])
            if target in existing:
                continue
            filled = dict(s)
            filled["model_id"] = rule.keep_id
            filled["matches_canonical"] = False
            filled["_canonical_fill_from"] = rule.drop_id
            note = (
                f"Canonical-fill from {rule.drop_id}: same base model but a "
                "different release variant (base vs instruct), so the value "
                f"may differ slightly from {rule.keep_id}; used only because "
                "the canonical cell is missing."
            )
            filled["notes"] = f"{s.get('notes', '').rstrip()} {note}".strip()
            out.append(filled)
            existing.add(target)
    return out


def _iterate_threshold(
    obs: list[tuple[str, str]],
    m_threshold: int,
    b_threshold: int,
    max_iter: int = 50,
) -> tuple[set[str], set[str], int]:
    """Iteratively drop rows below *m_threshold* and cols below *b_threshold*."""
    if m_threshold <= 0 and b_threshold <= 0:
        return set(m for m, _ in obs), set(b for _, b in obs), 0
    keep_m: set[str] = set()
    keep_b: set[str] = set()
    prev = -1
    for itr in range(1, max_iter + 1):
        if len(obs) == prev:
            return keep_m, keep_b, itr - 1
        prev = len(obs)
        mc: dict[str, int] = {}
        bc: dict[str, int] = {}
        for m, b in obs:
            mc[m] = mc.get(m, 0) + 1
            bc[b] = bc.get(b, 0) + 1
        keep_m = {m for m, c in mc.items() if c >= m_threshold} if m_threshold > 0 else set(mc)
        keep_b = {b for b, c in bc.items() if c >= b_threshold} if b_threshold > 0 else set(bc)
        obs = [(m, b) for m, b in obs if m in keep_m and b in keep_b]
    raise RuntimeError(f"threshold filter did not converge after {max_iter} iterations")


# ───────────────────────── Main API ─────────────────────────

def load_score_matrix(
    *,
    m_threshold: int = DEFAULT_M_THRESHOLD,
    b_threshold: int = DEFAULT_B_THRESHOLD,
    status_filter: str | Sequence[str] = ("verified", "verified_third_party"),
    use_candidates: bool = False,
    iterate: bool = True,
    json_path: str | Path | None = None,
    deduplicate: bool | None = None,
    exclude_models: Sequence[str] | None = None,
    exclude_benchmarks: Sequence[str] | None = None,
    return_info: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple:
    """Build the (model × benchmark) score matrix as a pandas DataFrame.

    Args:
        m_threshold: minimum observations per model row (default 15).
        b_threshold: minimum observations per benchmark column (default 8).
            Set both to 0 to disable filtering.
        status_filter: which ``audit_status`` rows to include.  Default
            ``("verified", "verified_third_party")``.
        use_candidates: if True, fall back to ``candidates[]`` values when
            primary cell is missing or filtered out.
        iterate: iterate threshold filter to fixed point (default True).
        json_path: override default JSON location.
        deduplicate: apply paper-canonical representative-selection rules.
            By default this is enabled for thresholded matrices and disabled
            when both thresholds are zero, so ``m_threshold=0, b_threshold=0``
            remains the full raw audit matrix.
        exclude_models: additional model ids to exclude before thresholding.
        exclude_benchmarks: additional benchmark ids to exclude before
            thresholding.
        return_info: also return a :class:`FilterInfo` dataclass.
        return_metadata: also return ``(models_df, benchmarks_df)`` DataFrames.

    Returns:
        DataFrame indexed by ``model_id``, columns = ``benchmark_id``,
        values = score (NaN where missing).  If ``return_info`` /
        ``return_metadata`` are set, returns a tuple.
    """
    raw = _load_raw(json_path)
    # Pipeline:
    #   original audit pool
    #   → audit-status filter
    #   → canonical representative selection (optional)
    #   → iterative threshold filter
    #   → final score matrix
    scores = _filter_status(raw["scores"], status_filter)
    if deduplicate is None:
        deduplicate = m_threshold > 0 or b_threshold > 0
    excluded_models = set(DEFAULT_EXCLUDED_MODELS) if deduplicate else set()
    excluded_benchmarks = set(DEFAULT_EXCLUDED_BENCHMARKS) if deduplicate else set()
    if exclude_models:
        excluded_models.update(exclude_models)
    if exclude_benchmarks:
        excluded_benchmarks.update(exclude_benchmarks)
    if deduplicate:
        scores = _apply_benchmark_fill_rules(
            scores,
            fill_rules=BENCHMARK_FILL_RULES,
            exclude_models=excluded_models,
        )
        scores = _apply_model_fill_rules(
            scores,
            fill_rules=MODEL_FILL_RULES,
            exclude_benchmarks=excluded_benchmarks,
        )
    scores = _apply_canonical_rules(
        scores,
        exclude_models=excluded_models,
        exclude_benchmarks=excluded_benchmarks,
    )

    if use_candidates:
        primary_keys = {(s["model_id"], s["benchmark_id"]) for s in scores}
        for s in raw["scores"]:
            if s["model_id"] in excluded_models or s["benchmark_id"] in excluded_benchmarks:
                continue
            for c in s.get("candidates") or []:
                key = (s["model_id"], s["benchmark_id"])
                if key in primary_keys:
                    continue
                scores.append({
                    "model_id": s["model_id"],
                    "benchmark_id": s["benchmark_id"],
                    "score": c.get("score"),
                    "audit_status": "verified",  # promoted from candidate
                    "source_type": c.get("source_type"),
                    "reference_url": c.get("reference_url"),
                    "_from_candidate": True,
                })

    obs = [(s["model_id"], s["benchmark_id"]) for s in scores if s.get("score") is not None]

    keep_m: set[str] = set(m for m, _ in obs)
    keep_b: set[str] = set(b for _, b in obs)
    iterations = 0
    if m_threshold > 0 or b_threshold > 0:
        if iterate:
            keep_m, keep_b, iterations = _iterate_threshold(obs, m_threshold, b_threshold)
        else:
            mc: dict[str, int] = {}
            bc: dict[str, int] = {}
            for m, b in obs:
                mc[m] = mc.get(m, 0) + 1
                bc[b] = bc.get(b, 0) + 1
            keep_m = {m for m, c in mc.items() if c >= m_threshold} if m_threshold > 0 else set(mc)
            keep_b = {b for b, c in bc.items() if c >= b_threshold} if b_threshold > 0 else set(bc)
            iterations = 1

    keep_m_list = sorted(keep_m)
    keep_b_list = sorted(keep_b)
    df = pd.DataFrame(np.nan, index=keep_m_list, columns=keep_b_list, dtype=float)
    df.index.name = "model_id"
    df.columns.name = "benchmark_id"
    n_obs_kept = 0
    for s in scores:
        m = s["model_id"]
        b = s["benchmark_id"]
        sc = s.get("score")
        if sc is None or m not in keep_m or b not in keep_b:
            continue
        try:
            sc = float(sc)
        except (TypeError, ValueError):
            continue
        if pd.isna(df.at[m, b]):
            df.at[m, b] = sc
            n_obs_kept += 1

    out: list = [df]
    if return_info:
        all_input_models = set(m for m, _ in obs)
        all_input_bench = set(b for _, b in obs)
        info = FilterInfo(
            m_threshold=m_threshold,
            b_threshold=b_threshold,
            iterations=iterations,
            n_models=len(keep_m),
            n_benchmarks=len(keep_b),
            n_observations=n_obs_kept,
            fill_rate=n_obs_kept / (len(keep_m) * len(keep_b)) if keep_m and keep_b else 0.0,
            dropped_models=tuple(sorted(all_input_models - keep_m)),
            dropped_benchmarks=tuple(sorted(all_input_bench - keep_b)),
            excluded_models=tuple(sorted(excluded_models)),
            excluded_benchmarks=tuple(sorted(excluded_benchmarks)),
            model_canonical_rules=MODEL_CANONICAL_RULES if deduplicate else (),
            benchmark_canonical_rules=BENCHMARK_CANONICAL_RULES if deduplicate else (),
        )
        out.append(info)
    if return_metadata:
        models_df = pd.DataFrame(raw["models"]).set_index("id").loc[keep_m_list]
        models_df.index.name = "model_id"
        bench_df = pd.DataFrame(raw["benchmarks"]).set_index("id").loc[keep_b_list]
        bench_df.index.name = "benchmark_id"
        out.extend([models_df, bench_df])
    return out[0] if len(out) == 1 else tuple(out)


def load_long_format(
    *,
    status_filter: str | Sequence[str] = "verified",
    json_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return all (model, benchmark, score, ...) rows in long format with metadata.

    Note: this does NOT apply the threshold filter; use `load_score_matrix()`
    for the dense subset.
    """
    raw = _load_raw(json_path)
    scores = _filter_status(raw["scores"], status_filter)
    rows = [{
        "model_id": s["model_id"],
        "benchmark_id": s["benchmark_id"],
        "score": s.get("score"),
        "reference_url": s.get("reference_url"),
        "source_type": s.get("source_type"),
        "audit_status": s.get("audit_status"),
        "matches_canonical": s.get("matches_canonical"),
        "n_candidates": len(s.get("candidates") or []),
    } for s in scores]
    return pd.DataFrame(rows)


# ───────────────────────── Legacy globals (backward compat) ─────────────────────────
# Existing experiment scripts import MODELS / BENCHMARKS / DATA as tuple lists.
# These are derived from the canonical JSON on import — DO NOT edit them in place;
# edit `benchpress/data/llm_benchmark_data.json` instead.
#
# MODELS / BENCHMARKS / DATA are filtered to the paper-canonical (M≥15, B≥8)
# subset.  To get the unfiltered audit pool, use
# ``load_score_matrix(m_threshold=0, b_threshold=0)``.

_RAW_CACHE = _load_raw()
_LEGACY_KEEP = {"verified", "verified_third_party"}

_LEGACY_EXCLUDED_MODELS = set(DEFAULT_EXCLUDED_MODELS)
_LEGACY_EXCLUDED_BENCHMARKS = set(DEFAULT_EXCLUDED_BENCHMARKS)

_LEGACY_SCORES = [
    s for s in _RAW_CACHE["scores"]
    if s.get("audit_status") in _LEGACY_KEEP and s.get("score") is not None
]
_LEGACY_SCORES = _apply_benchmark_fill_rules(
    _LEGACY_SCORES,
    fill_rules=BENCHMARK_FILL_RULES,
    exclude_models=_LEGACY_EXCLUDED_MODELS,
)
_LEGACY_SCORES = _apply_model_fill_rules(
    _LEGACY_SCORES,
    fill_rules=MODEL_FILL_RULES,
    exclude_benchmarks=_LEGACY_EXCLUDED_BENCHMARKS,
)
_LEGACY_SCORES = _apply_canonical_rules(
    _LEGACY_SCORES,
    exclude_models=_LEGACY_EXCLUDED_MODELS,
    exclude_benchmarks=_LEGACY_EXCLUDED_BENCHMARKS,
)
_LEGACY_OBS = [(s["model_id"], s["benchmark_id"]) for s in _LEGACY_SCORES]
_KEEP_M, _KEEP_B, _ = _iterate_threshold(
    list(_LEGACY_OBS), DEFAULT_M_THRESHOLD, DEFAULT_B_THRESHOLD
)

MODELS: list[tuple] = [
    (m["id"], m.get("name"), m.get("provider"), m.get("release_date"),
     m.get("params_total_M"), m.get("params_active_M"), m.get("architecture"),
     m.get("is_reasoning"), m.get("open_weights"))
    for m in _RAW_CACHE["models"]
    if m["id"] in _KEEP_M
]

BENCHMARKS: list[tuple] = [
    (b["id"], b.get("name"), b.get("category"), b.get("metric"),
     b.get("num_problems"), b.get("source_url"))
    for b in _RAW_CACHE["benchmarks"]
    if b["id"] in _KEEP_B
]

DATA: list[tuple] = [
    (s["model_id"], s["benchmark_id"], s["score"], s.get("reference_url"))
    for s in _LEGACY_SCORES
    if s["model_id"] in _KEEP_M and s["benchmark_id"] in _KEEP_B
]


# ───────────────────────── Excel export ─────────────────────────

def build_excel(output: str | Path = "llm_benchmark_matrix.xlsx") -> Path:
    """Generate a multi-sheet Excel workbook (Scores / References / Metadata / Long)."""
    raw = _load_raw()
    df = load_score_matrix(m_threshold=0, b_threshold=0)  # full unfiltered
    long_df = load_long_format()
    models_df = pd.DataFrame(raw["models"]).set_index("id")
    bench_df = pd.DataFrame(raw["benchmarks"]).set_index("id")
    # URL matrix matching df shape
    url_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for s in raw["scores"]:
        if s.get("audit_status") != "verified":
            continue
        m, b = s["model_id"], s["benchmark_id"]
        if m in url_df.index and b in url_df.columns:
            url_df.at[m, b] = s.get("reference_url")
    out = Path(output)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Scores")
        url_df.to_excel(w, sheet_name="References")
        models_df.to_excel(w, sheet_name="Models")
        bench_df.to_excel(w, sheet_name="Benchmarks")
        long_df.to_excel(w, sheet_name="Flat", index=False)
    return out


# ───────────────────────── CLI ─────────────────────────

def _main() -> None:
    p = argparse.ArgumentParser(description="BenchPress score matrix loader")
    p.add_argument("--m-threshold", "-Tm", type=int, default=DEFAULT_M_THRESHOLD,
                   help=f"min observations per model row (default {DEFAULT_M_THRESHOLD}; 0=no filter)")
    p.add_argument("--b-threshold", "-Tb", type=int, default=DEFAULT_B_THRESHOLD,
                   help=f"min observations per benchmark col (default {DEFAULT_B_THRESHOLD}; 0=no filter)")
    p.add_argument("--status", default="verified,verified_third_party",
                   help="audit_status filter (comma-sep or 'all').")
    p.add_argument("--use-candidates", action="store_true",
                   help="fall back to candidates[] when primary missing")
    dedup_group = p.add_mutually_exclusive_group()
    dedup_group.add_argument("--deduplicate", action="store_true",
                             help="apply canonical rules even when thresholds are zero")
    dedup_group.add_argument("--no-deduplicate", action="store_true",
                             help="include near-duplicate benchmarks and setting variants")
    p.add_argument("--list-canonical-rules", action="store_true",
                   help="print canonical representative rules and exit")
    p.add_argument("--no-iterate", action="store_true",
                   help="disable iterated threshold filter (single pass)")
    p.add_argument("--output", "-o", default=None, help="write CSV to this path")
    p.add_argument("--excel", default=None, help="write multi-sheet Excel to this path")
    p.add_argument("--long", action="store_true",
                   help="output long format instead of wide matrix")
    p.add_argument("--json-path", default=None, help="override JSON path")
    args = p.parse_args()

    status = "all" if args.status == "all" else [s.strip() for s in args.status.split(",")]

    if args.list_canonical_rules:
        print("Model canonicalization rules:")
        for r in MODEL_CANONICAL_RULES:
            print(f"  drop {r.drop_id} -> keep {r.keep_id} ({r.family}): {r.reason}")
        print("Benchmark canonicalization rules:")
        for r in BENCHMARK_CANONICAL_RULES:
            print(f"  drop {r.drop_id} -> keep {r.keep_id} ({r.family}): {r.reason}")
        return

    if args.excel:
        out = build_excel(args.excel)
        print(f"Wrote {out}")
        return

    if args.long:
        df = load_long_format(status_filter=status, json_path=args.json_path)
        print(f"Loaded {len(df)} rows")
    else:
        deduplicate = None
        if args.deduplicate:
            deduplicate = True
        elif args.no_deduplicate:
            deduplicate = False
        df, info = load_score_matrix(
            m_threshold=args.m_threshold,
            b_threshold=args.b_threshold,
            status_filter=status,
            use_candidates=args.use_candidates,
            iterate=not args.no_iterate,
            json_path=args.json_path,
            deduplicate=deduplicate,
            return_info=True,
        )
        print(f"Score matrix: {info.n_models} models × {info.n_benchmarks} benchmarks")
        print(f"  Observations: {info.n_observations:,} / {info.n_models * info.n_benchmarks:,} possible")
        print(f"  Fill rate: {100 * info.fill_rate:.1f}%")
        if info.m_threshold > 0 or info.b_threshold > 0:
            print(f"  Threshold (M≥{info.m_threshold}, B≥{info.b_threshold}), converged in {info.iterations} iterations")
            print(f"  Dropped {len(info.dropped_models)} models, {len(info.dropped_benchmarks)} benchmarks")
        if info.excluded_models or info.excluded_benchmarks:
            print(f"  Dedup excluded {len(info.excluded_models)} models, {len(info.excluded_benchmarks)} benchmarks")

    if args.output:
        df.to_csv(args.output)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    _main()
