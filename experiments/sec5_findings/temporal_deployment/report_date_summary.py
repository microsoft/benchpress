#!/usr/bin/env python
"""Compare temporal-deployment results across --report-date-filter variants.

Reads ``<results-dir>/<filter>/results.json`` for every filter written by
``run.py --mode merge --report-date-filter <filter> --results-root
<results-dir>/<filter>`` and writes ``<results-dir>/summary.json``. All filters
must score the same target cells with the same revealed probes. Metrics are the
hidden-only median over targets of per-target MedAE / MedAPE
(``summary_utils``), for BenchPress (``pred``) and the logit model mean
(``model_mean_pred``), on each filter's own finite cells and on the hidden cells
that every filter and both predictors predict.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchpress.io_utils import load_json, write_json
from summary_utils import DISPLAY_K, temporal_family_k_summary, temporal_overall_medians

FILTERS = ["none", "verified_later", "strict"]
PREDICTORS = {"benchpress": "pred", "model_mean": "model_mean_pred"}


def cell_key(row: dict) -> tuple:
    return (row["family_key"], int(row["k"]), int(row["seed"]), row["model_id"], row["benchmark_id"])


def finite(value) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def hidden_payload(payload: dict, prediction_field: str, keep_keys: set | None) -> dict:
    """Payload whose rows are hidden cells scored by ``prediction_field``."""
    rows = [
        {**row, "pred": row[prediction_field]}
        for row in payload["raw_predictions"]
        if not row["is_revealed"] and (keep_keys is None or cell_key(row) in keep_keys)
    ]
    return {
        "landmarks": payload["landmarks"],
        "summary_by_family": payload["summary_by_family"],
        "raw_predictions": rows,
    }


def scored_summary(payload: dict, prediction_field: str, keep_keys: set | None) -> dict:
    scored = hidden_payload(payload, prediction_field, keep_keys)
    per_target = temporal_family_k_summary(scored, hidden_only=True)
    return {
        "n_hidden_cells_scored": sum(1 for row in scored["raw_predictions"] if finite(row["pred"])),
        "median_over_targets": {
            metric: {str(k): value for k, value in by_k.items()}
            for metric, by_k in temporal_overall_medians(scored, hidden_only=True).items()
        },
        "per_target": {
            family_key: {
                str(k): {"medae": row["by_k"][str(k)]["medae"], "medape": row["by_k"][str(k)]["medape"]}
                for k in DISPLAY_K
            }
            for family_key, row in per_target.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(HERE, "report_date_filter"),
                        help="Directory containing <filter>/results.json for every filter.")
    args = parser.parse_args()

    payloads = {
        name: load_json(os.path.join(args.results_dir, name, "results.json"))
        for name in FILTERS
    }
    for name, payload in payloads.items():
        if payload["config"]["report_date_filter"] != name:
            raise ValueError(f"{name}: results.json was written with filter {payload['config']['report_date_filter']}")

    reference_cells = {
        cell_key(row): bool(row["is_revealed"]) for row in payloads["none"]["raw_predictions"]
    }
    for name, payload in payloads.items():
        cells = {cell_key(row): bool(row["is_revealed"]) for row in payload["raw_predictions"]}
        if cells != reference_cells:
            raise ValueError(f"{name}: evaluated or revealed cells differ from the unfiltered run")

    common_keys = {
        key for key, is_revealed in reference_cells.items() if not is_revealed
    }
    for payload in payloads.values():
        common_keys &= {
            cell_key(row) for row in payload["raw_predictions"]
            if not row["is_revealed"] and all(finite(row[field]) for field in PREDICTORS.values())
        }

    variants = {}
    for name, payload in payloads.items():
        landmarks = payload["landmarks"]
        removed = [int(row["n_report_date_removed"]) for row in landmarks]
        variants[name] = {
            "manifest": payload["manifest"],
            "n_eval_cells": len(payload["raw_predictions"]),
            "n_hidden_cells": sum(1 for row in payload["raw_predictions"] if not row["is_revealed"]),
            "training_scores_removed": {
                "total_over_targets": int(sum(removed)),
                "median_per_target": float(np.median(removed)),
                "median_fraction_per_target": float(np.median([
                    int(row["n_report_date_removed"]) / int(row["n_train_scores"]) for row in landmarks
                ])),
                "per_target": {row["family_key"]: int(row["n_report_date_removed"]) for row in landmarks},
            },
            "own_cells": {
                predictor: scored_summary(payload, field, None)
                for predictor, field in PREDICTORS.items()
            },
            "common_cells": {
                predictor: scored_summary(payload, field, common_keys)
                for predictor, field in PREDICTORS.items()
            },
        }

    comparisons = {}
    reference = variants["none"]["common_cells"]["benchpress"]["per_target"]
    for name in FILTERS:
        common = variants[name]["common_cells"]
        comparisons[name] = {}
        for k in DISPLAY_K:
            k = str(k)
            bp = {key: row[k]["medae"] for key, row in common["benchpress"]["per_target"].items()}
            mm = {key: row[k]["medae"] for key, row in common["model_mean"]["per_target"].items()}
            delta = [bp[key] - reference[key][k]["medae"] for key in bp]
            comparisons[name][k] = {
                "median_target_medae_change_vs_none": float(np.median(delta)),
                "max_abs_target_medae_change_vs_none": float(np.max(np.abs(delta))),
                "n_targets_benchpress_below_model_mean": int(sum(bp[key] < mm[key] for key in bp)),
                "n_targets": len(bp),
            }

    summary = {
        "description": (
            "Hidden-only median over targets of per-target MedAE / MedAPE (median over seeds of the "
            "pooled per-seed metric) for each --report-date-filter. own_cells scores each filter on its "
            "finite hidden predictions; common_cells restricts every filter and predictor to hidden "
            "cells finite in all of them."
        ),
        "filters": FILTERS,
        "k_values": DISPLAY_K,
        "identical_eval_and_revealed_cells": True,
        "n_eval_cells": len(reference_cells),
        "n_hidden_cells": sum(1 for is_revealed in reference_cells.values() if not is_revealed),
        "n_common_hidden_cells": len(common_keys),
        "variants": variants,
        "comparison_common_cells": comparisons,
    }
    out_path = os.path.join(args.results_dir, "summary.json")
    write_json(out_path, summary, indent=2, trailing_newline=True)
    print(f"Wrote {out_path}")
    for name in FILTERS:
        common = variants[name]["common_cells"]
        print(
            f"{name:15s} removed median/target={variants[name]['training_scores_removed']['median_per_target']:.0f} "
            f"BenchPress MedAE {common['benchpress']['median_over_targets']['medae']} "
            f"model mean MedAE {common['model_mean']['median_over_targets']['medae']}"
        )


if __name__ == "__main__":
    main()
