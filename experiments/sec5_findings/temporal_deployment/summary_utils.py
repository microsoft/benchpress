"""Shared temporal-deployment summaries from merged JSON rows."""

from __future__ import annotations

import numpy as np

from benchpress.evaluation_harness import compute_prediction_error

DISPLAY_K = [1, 5, 10]


def ordered_landmarks(payload: dict) -> list[dict]:
    return sorted(
        payload["landmarks"],
        key=lambda row: (row["cutoff_date"], row["family_name"]),
    )


def _metric_from_rows(rows: list[dict]) -> dict:
    actual = np.array([row["actual"] for row in rows], dtype=float)
    pred = np.array([row["pred"] for row in rows], dtype=float)
    metrics = compute_prediction_error(actual, pred, aggregation="pool")
    return {
        "n": int(metrics["n"]),
        "medape": float(metrics["medape"]) if np.isfinite(metrics["medape"]) else None,
        "medae": float(metrics["medae"]) if np.isfinite(metrics["medae"]) else None,
    }


def temporal_family_k_seed_metric(payload: dict, family_key: str, k: int,
                                  seed: int, hidden_only: bool) -> dict:
    rows = [
        row for row in payload["raw_predictions"]
        if row["family_key"] == family_key
        and int(row["k"]) == int(k)
        and int(row["seed"]) == int(seed)
    ]
    if hidden_only:
        rows = [
            row for row in rows
            if not row.get("is_revealed")
            and row.get("pred") is not None
            and np.isfinite(float(row["pred"]))
        ]
    else:
        rows = [row for row in rows if row.get("is_metric_cell")]
    return _metric_from_rows(rows)


def temporal_family_k_summary(payload: dict, hidden_only: bool,
                              k_values: list[int] | None = None) -> dict:
    k_values = DISPLAY_K if k_values is None else k_values
    seed_values = sorted({
        int(row["seed"])
        for row in payload["raw_predictions"]
    })
    summaries = {}
    for landmark in ordered_landmarks(payload):
        family_key = landmark["family_key"]
        family_summary = payload["summary_by_family"][family_key]
        observed_counts = family_summary.get("target_observed_counts", {})
        summaries[family_key] = {
            "family_key": family_key,
            "family_name": family_summary["family_name"],
            "cutoff_date": landmark["cutoff_date"],
            "observed": max(observed_counts.values()) if observed_counts else None,
            "n_train_models": family_summary["n_train_models"],
            "by_k": {},
        }
        for k in k_values:
            per_seed = [
                temporal_family_k_seed_metric(
                    payload, family_key, int(k), int(seed), hidden_only,
                )
                for seed in seed_values
            ]
            medape_values = [row["medape"] for row in per_seed if row["medape"] is not None]
            medae_values = [row["medae"] for row in per_seed if row["medae"] is not None]
            summaries[family_key]["by_k"][str(k)] = {
                "n_seeds": len(per_seed),
                "medape": float(np.median(medape_values)) if medape_values else None,
                "medae": float(np.median(medae_values)) if medae_values else None,
                "per_seed": per_seed,
            }
    return summaries


def temporal_values_by_k(payload: dict, metric: str, hidden_only: bool,
                         k_values: list[int] | None = None) -> list[list[float]]:
    summaries = temporal_family_k_summary(payload, hidden_only, k_values=k_values)
    k_values = DISPLAY_K if k_values is None else k_values
    values = []
    for k in k_values:
        current = []
        for landmark in ordered_landmarks(payload):
            value = summaries[landmark["family_key"]]["by_k"][str(k)][metric]
            if value is not None and np.isfinite(float(value)):
                current.append(float(value))
        values.append(current)
    return values


def temporal_overall_medians(payload: dict, hidden_only: bool,
                             k_values: list[int] | None = None) -> dict:
    return {
        metric: {
            int(k): float(np.median(vals)) if vals else None
            for k, vals in zip(
                DISPLAY_K if k_values is None else k_values,
                temporal_values_by_k(payload, metric, hidden_only, k_values=k_values),
            )
        }
        for metric in ("medae", "medape")
    }
