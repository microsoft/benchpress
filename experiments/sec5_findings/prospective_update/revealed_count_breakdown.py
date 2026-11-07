"""Break the new-model comparison of run.py down by revealed probe count.

A new model reveals only the first-k probes it reports, so a k-probe budget
often reveals fewer than k scores. This post-hoc analysis reads the cached
raw_predictions.json.gz of run.py and pools, for each greedy ordering, the
unrevealed cells of all (target, k) units by the number of probe scores the
target actually revealed. BenchPress and the logit-space model mean are scored
on the same cells with the canonical harness metric.
"""
import argparse
import gzip
import json
import os

import numpy as np

from benchpress.evaluation_harness import compute_prediction_error

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(HERE, "results", "may_to_aug", "raw_predictions.json.gz")
METHODS = {
    "benchpress": "benchpress_fixed_order",
    "logit_model_mean": "logit_model_mean_fixed_order",
}
ORDERING_GROUPS = {
    "medae_any": ("medae_any",),
    "medae_low_cost": ("medae_low_cost",),
    "medae_both": ("medae_any", "medae_low_cost"),
}
REVEALED_BINS = (("1", 1, 1), ("2", 2, 2), ("3+", 3, None))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "may_to_aug", "revealed_count_breakdown.json"))
    return parser.parse_args()


def unit_key(row):
    return (row["ordering"], row["k"], row["model_id"], row["benchmark_id"])


def in_bin(count, low, high):
    return count >= low and (high is None or count <= high)


def score(rows_by_method, keys):
    out = {}
    for name, rows in rows_by_method.items():
        actual = np.array([rows[key]["actual"] for key in keys], dtype=float)
        pred = np.array([rows[key]["pred"] for key in keys], dtype=float)
        metrics = compute_prediction_error(actual, pred, aggregation="pool")
        out[name] = {"medae": float(metrics["medae"]), "medape": float(metrics["medape"]),
                     "n": int(metrics["n"])}
    return out


def main():
    args = parse_args()
    with gzip.open(args.raw, "rt") as handle:
        raw = json.load(handle)
    rows_by_method = {
        name: {unit_key(row): row for row in raw
               if row["method"] == method and not row["is_probe"]}
        for name, method in METHODS.items()
    }
    shared = set.intersection(*(set(rows) for rows in rows_by_method.values()))
    bp_rows = rows_by_method["benchpress"]
    result = {"raw": os.path.relpath(args.raw, HERE), "groups": {}}
    for group, orderings in ORDERING_GROUPS.items():
        result["groups"][group] = []
        for label, low, high in REVEALED_BINS:
            keys = sorted(key for key in shared if key[0] in orderings
                          and in_bin(bp_rows[key]["n_observed_probe"], low, high))
            units = {key[:3] for key in keys}
            entry = {"revealed": label, "cells": len(keys), "target_budget_units": len(units),
                     "targets": len({unit[2] for unit in units})}
            entry.update(score(rows_by_method, keys))
            result["groups"][group].append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
