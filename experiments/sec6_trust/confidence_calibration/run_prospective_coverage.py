#!/usr/bin/env python3
"""Coverage of the May-trained hybrid reliability intervals on later August scores.

The hybrid reliability estimator and its conformal scales are fit once on the
May matrix (the Section 6.2 procedure) and then applied unchanged to the
prospective May-to-August cells of `sec5_findings/prospective_update`.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROSPECTIVE_RUN = os.path.join(
    REPO_ROOT, "experiments", "sec5_findings", "prospective_update", "run.py")
DEFAULT_OUT_DIR = os.path.join(HERE, "results", "prospective_coverage")
METHOD = "combined_risk_model"
FOLD_SPEC = {"n_seeds": 10, "n_folds": 3, "base_seed": 42, "min_scores": 1}
TERCILES = ("low_risk", "medium_risk", "high_risk")

_spec = importlib.util.spec_from_file_location("prospective_update_run", PROSPECTIVE_RUN)
prospective = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prospective)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--may-json", required=True, help="May 2026 score JSON (training).")
    parser.add_argument("--aug-json", required=True, help="August 26, 2026 score JSON.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--orderings", nargs="+", default=["medae_any", "medae_low_cost"],
                        choices=list(prospective.FIXED_ORDERING_KEYS))
    parser.add_argument("--probe-order-json", default=prospective.DEFAULT_ORDER_PATH)
    parser.add_argument("--seed", type=int, default=42,
                        help="Risk-model seed (Section 6.2 uses 42).")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def fold_crossfit(task):
    from benchpress.methods.confidence import leave_fold_mlp_error_calibrator
    actual, predicted, fold_id, cell_id, features, fold, seed = task
    uncertainty, _, selected = leave_fold_mlp_error_calibrator(
        actual, predicted, fold_id, cell_id, features,
        folds_to_run=[fold], label=METHOD, seed=seed)
    return fold, uncertainty, selected


def frozen_unit_risk(task):
    """BenchPress prediction matrix and frozen-calibrator risk for `cells` of `train`."""
    from benchpress.methods.confidence import predict_confidence_intervals
    from benchpress.methods.predictors import predict_benchpress_scores
    train, cells, metric, benchmark_ids, artifact = task
    pred = predict_benchpress_scores(train, metric=metric, benchmark_ids=benchmark_ids)
    out = predict_confidence_intervals(
        train, M_pred=pred, artifact=artifact, method=METHOD,
        train_if_missing=False, cells=cells, metric=metric,
        benchmark_ids=benchmark_ids)
    return pred, np.asarray(out["uncertainty"], dtype=float)


def fit_frozen_calibrator(out_dir, seed, executor):
    """Fit (or reload) the May hybrid calibrator and its leave-fold-out risks."""
    from benchpress.evaluation_harness import (
        BENCH_IDS, BENCH_METRICS, M_FULL, load_folds, matrix_identity_sha256)
    from benchpress.io_utils import write_npz_compressed_atomic
    from benchpress.methods.confidence import (
        confidence_training_records, fit_default_confidence_calibrator)

    calibrator_path = os.path.join(out_dir, "frozen_calibrator.pkl")
    crossfit_path = os.path.join(out_dir, "may_crossfit.npz")
    if os.path.exists(calibrator_path) and os.path.exists(crossfit_path):
        with open(calibrator_path, "rb") as f:
            artifact = pickle.load(f)
        if (artifact["matrix_identity_sha256"] != matrix_identity_sha256(M_FULL)
                or int(artifact["seed"]) != int(seed)
                or artifact["methods"] != [METHOD]):
            raise RuntimeError(
                f"Existing {calibrator_path} does not match the May matrix/seed; "
                "remove it to refit.")
        with np.load(crossfit_path, allow_pickle=False) as data:
            crossfit = {key: data[key] for key in data.files}
        print(f"Reusing frozen calibrator {calibrator_path}", flush=True)
        return artifact, crossfit, calibrator_path

    folds = load_folds(**FOLD_SPEC)
    records = confidence_training_records(
        M_FULL, folds, metric=BENCH_METRICS, benchmark_ids=BENCH_IDS,
        map_fn=executor.map)
    cell_id = (records["test_i"].astype(np.int64) * M_FULL.shape[1]
               + records["test_j"].astype(np.int64))
    features = records["feature_sets"][METHOD]
    tasks = [
        (records["actual"], records["predicted"], records["fold_id"], cell_id,
         features, int(fold), seed)
        for fold in np.unique(records["fold_id"])
    ]
    uncertainty = np.full(len(cell_id), np.nan, dtype=float)
    selected = {}
    for fold, fold_uncertainty, fold_selected in executor.map(fold_crossfit, tasks):
        mask = records["fold_id"] == fold
        uncertainty[mask] = fold_uncertainty[mask]
        selected.update(fold_selected)

    crossfit = {
        "fold_id": records["fold_id"].astype(np.int16),
        "test_i": records["test_i"].astype(np.int16),
        "test_j": records["test_j"].astype(np.int16),
        "actual": records["actual"].astype(float),
        "predicted": records["predicted"].astype(float),
        "uncertainty": uncertainty,
    }
    os.makedirs(out_dir, exist_ok=True)
    write_npz_compressed_atomic(crossfit_path, **crossfit)
    artifact = fit_default_confidence_calibrator(
        records, M_FULL, methods=[METHOD], artifact_path=calibrator_path,
        seed=seed, metric=BENCH_METRICS, benchmark_ids=BENCH_IDS,
        crossfit_uncertainty={METHOD: uncertainty},
        crossfit_selected={METHOD: selected})
    return artifact, crossfit, calibrator_path


def check_may_matches_harness(may):
    """The calibrator is trained on harness M_FULL; it must be the May snapshot."""
    from benchpress.evaluation_harness import (
        BENCH_IDS, BENCH_METRICS, M_FULL, MODEL_IDS,
        benchmark_metric_identity_sha256, matrix_identity_sha256)
    reindexed = may["df"].reindex(index=MODEL_IDS, columns=BENCH_IDS).to_numpy(dtype=float)
    if matrix_identity_sha256(reindexed) != matrix_identity_sha256(M_FULL):
        raise RuntimeError("May snapshot differs from the harness matrix M_FULL")
    if (benchmark_metric_identity_sha256(may["benchmark_metrics"], BENCH_IDS)
            != benchmark_metric_identity_sha256(BENCH_METRICS, BENCH_IDS)):
        raise RuntimeError("May benchmark metrics differ from the harness metrics")


def level_tag(level_key):
    return str(int(round(float(level_key) * 100)))


def risk_tercile(risk, cuts):
    if risk <= cuts[0]:
        return TERCILES[0]
    if risk <= cuts[1]:
        return TERCILES[1]
    return TERCILES[2]


def interval_metrics(actual, lower, upper):
    above = actual > upper
    below = actual < lower
    n_miss = int(above.sum() + below.sum())
    return {
        "n": int(len(actual)),
        "coverage": float(np.mean(~above & ~below)) if len(actual) else None,
        "median_width": float(np.median(upper - lower)) if len(actual) else None,
        "miss_rate_above": float(np.mean(above)) if len(actual) else None,
        "miss_rate_below": float(np.mean(below)) if len(actual) else None,
        "n_miss": n_miss,
        "share_of_misses_above": float(above.sum() / n_miss) if n_miss else None,
    }


def group_summary(records, level_keys):
    """Coverage, width, and miss direction overall and by May risk tercile."""
    rows = [r for r in records if r["pred"] is not None and r["risk"] is not None]
    actual = np.asarray([r["actual"] for r in rows], dtype=float)
    pred = np.asarray([r["pred"] for r in rows], dtype=float)
    tercile = np.asarray([r["risk_tercile"] for r in rows])

    def block(mask):
        out = {
            "n": int(mask.sum()),
            "n_models": int(len({rows[i]["model_id"] for i in np.flatnonzero(mask)})),
            "medae": float(np.median(np.abs(pred[mask] - actual[mask]))) if mask.any() else None,
            "fraction_over_predicted": float(np.mean(pred[mask] > actual[mask])) if mask.any() else None,
            "median_risk": float(np.median([rows[i]["risk"] for i in np.flatnonzero(mask)])) if mask.any() else None,
            "levels": {},
        }
        for key in level_keys:
            tag = level_tag(key)
            lower = np.asarray([rows[i][f"lower_{tag}"] for i in np.flatnonzero(mask)], dtype=float)
            upper = np.asarray([rows[i][f"upper_{tag}"] for i in np.flatnonzero(mask)], dtype=float)
            out["levels"][key] = interval_metrics(actual[mask], lower, upper)
        return out

    summary = block(np.ones(len(rows), dtype=bool))
    summary["n_total_cells"] = int(len(records))
    summary["by_risk_tercile"] = {name: block(tercile == name) for name in TERCILES}
    return summary


def may_crossfit_reproduction(crossfit, level_keys, cuts):
    """Section 6.2 leave-fold-out conformal coverage on May (paper: 90% hybrid 0.899)."""
    from benchpress.evaluation_harness import M_FULL
    from benchpress.methods.confidence import conformal_interval
    actual = crossfit["actual"]
    predicted = crossfit["predicted"]
    risk = crossfit["uncertainty"]
    cell_id = (crossfit["test_i"].astype(np.int64) * M_FULL.shape[1]
               + crossfit["test_j"].astype(np.int64))
    tercile = np.asarray([
        risk_tercile(r, cuts) if np.isfinite(r) else "missing_risk" for r in risk
    ])
    out = {"n": int(len(actual)), "levels": {}, "by_risk_tercile": {}}
    bounds = {}
    for key in level_keys:
        lower, upper, _ = conformal_interval(
            actual, predicted, risk, crossfit["fold_id"], cell_id, ci=float(key))
        valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
        bounds[key] = (lower, upper, valid)
        out["levels"][key] = interval_metrics(actual[valid], lower[valid], upper[valid])
    for name in TERCILES:
        out["by_risk_tercile"][name] = {}
        for key, (lower, upper, valid) in bounds.items():
            mask = valid & (tercile == name)
            out["by_risk_tercile"][name][key] = interval_metrics(
                actual[mask], lower[mask], upper[mask])
    return out


def interval_fields(pred, risk, actual, scales):
    fields = {}
    for key, q in scales.items():
        tag = level_tag(key)
        if pred is None or risk is None:
            fields.update({f"lower_{tag}": None, f"upper_{tag}": None, f"covered_{tag}": None})
            continue
        lower = pred - q * risk
        upper = pred + q * risk
        fields.update({
            f"lower_{tag}": float(lower),
            f"upper_{tag}": float(upper),
            f"covered_{tag}": bool(lower <= actual <= upper),
        })
    return fields


def main():
    args = parse_args()
    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    bp = prospective.bootstrap_benchpress(args.may_json)
    cfg = prospective.load_json_file(prospective.DEFAULT_CONFIG_PATH)
    may = prospective.load_snapshot(args.may_json, bp["load_score_matrix"])
    aug = prospective.load_snapshot(args.aug_json, bp["load_score_matrix"])
    if (list(may["matrix"].shape) != cfg["expected_train_shape"]
            or int(np.isfinite(may["matrix"]).sum()) != int(cfg["expected_train_observed"])):
        raise RuntimeError("May matrix identity check failed")
    check_may_matches_harness(may)
    orderings = prospective.load_json_file(args.probe_order_json)["orderings"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        artifact, crossfit, calibrator_path = fit_frozen_calibrator(
            args.out_dir, args.seed, executor)
        calibrator = artifact["calibrators"][METHOD]
        scales = calibrator["conformal_scales"]
        if not np.isclose(scales["0.90"], calibrator["conformal_scale"]):
            raise RuntimeError("Calibrator 0.90 scale mismatch")
        risk_valid = np.isfinite(crossfit["uncertainty"])
        cuts = [float(c) for c in np.quantile(crossfit["uncertainty"][risk_valid], [1 / 3, 2 / 3])]

        metric = may["benchmark_metrics"]
        bids = may["benchmark_ids"]
        model_idx = {mid: i for i, mid in enumerate(may["model_ids"])}
        bench_idx = {bid: j for j, bid in enumerate(bids)}

        units = []
        part_i_rows = [r for r in prospective.part_i_new_cells(may, aug, bp, {})
                       if r["method"] == "benchpress"]
        part_i_cells = [(model_idx[r["model_id"]], bench_idx[r["benchmark_id"]])
                        for r in part_i_rows]
        units.append({"group": "new_cells_existing_models", "ordering": None, "k": None,
                      "model_id": None, "prefix": None, "rows": part_i_rows,
                      "train": may["matrix"], "cells": part_i_cells})

        new_models = [mid for mid in aug["model_ids"] if mid not in model_idx]
        for ordering in args.orderings:
            order_ids = orderings[ordering]["benchmark_ids"]
            for k in cfg["k_values"]:
                prefix = order_ids[:k]
                for mid in new_models:
                    observed = prospective.observed_aug_values_for_may_benchmarks(aug, mid, bids)
                    if not prospective.probe_unit_included(observed, prefix):
                        continue
                    train = prospective.append_target_row(may, observed, prefix)
                    target = train.shape[0] - 1
                    cells = [(int(target), int(j))
                             for j in np.flatnonzero(~np.isfinite(train[target]))]
                    units.append({"group": "new_models", "ordering": ordering, "k": int(k),
                                  "model_id": mid, "prefix": prefix, "rows": None,
                                  "train": train, "cells": cells})

        print(f"Applying frozen calibrator to {len(units)} prediction units", flush=True)
        outputs = executor.map(frozen_unit_risk, [
            (u["train"], u["cells"], metric, bids, artifact) for u in units
        ])
        records = []
        for unit, (pred, risk) in zip(units, outputs):
            risk_by_cell = {cell: float(r) for cell, r in zip(unit["cells"], risk)}
            if unit["group"] == "new_cells_existing_models":
                rows = unit["rows"]
                for row, cell in zip(rows, unit["cells"]):
                    if row["pred"] is not None and not np.isclose(pred[cell], row["pred"], atol=1e-9):
                        raise RuntimeError("Part (i) prediction differs from prospective_update")
            else:
                rows = []
                target = unit["train"].shape[0] - 1
                prospective.add_part_ii_rows(
                    rows, may, aug, unit["model_id"], "benchpress_fixed_order", unit["k"],
                    unit["ordering"], None, unit["prefix"], pred[target, :])
            for row in rows:
                if unit["group"] == "new_models":
                    row_index = unit["train"].shape[0] - 1
                else:
                    row_index = model_idx[row["model_id"]]
                cell = (row_index, bench_idx[row["benchmark_id"]])
                risk_value = risk_by_cell.get(cell)
                if risk_value is not None and not np.isfinite(risk_value):
                    risk_value = None
                records.append({
                    "group": unit["group"],
                    "ordering": unit["ordering"],
                    "k": unit["k"],
                    "model_id": row["model_id"],
                    "model_name": row["model_name"],
                    "benchmark_id": row["benchmark_id"],
                    "benchmark_category": row["benchmark_category"],
                    "actual": row["actual"],
                    "pred": row["pred"],
                    "risk": risk_value,
                    "risk_tercile": None if risk_value is None else risk_tercile(risk_value, cuts),
                    **interval_fields(row["pred"], risk_value, row["actual"], scales),
                })

        reproduction = may_crossfit_reproduction(crossfit, list(scales), cuts)

    level_keys = list(scales)
    summary = {
        "frozen_calibrator": {
            "method": METHOD,
            "risk_model_config": calibrator["risk_model_config"],
            "conformal_scales": scales,
            "may_risk_tercile_cuts": cuts,
            "n_training_rows": int(len(crossfit["actual"])),
        },
        "may_leave_fold_out_reproduction": reproduction,
        "new_cells_existing_models": group_summary(
            [r for r in records if r["group"] == "new_cells_existing_models"], level_keys),
        "new_models": {
            ordering: {
                "pooled_k": group_summary(
                    [r for r in records if r["group"] == "new_models"
                     and r["ordering"] == ordering], level_keys),
                "by_k": {
                    str(k): group_summary(
                        [r for r in records if r["group"] == "new_models"
                         and r["ordering"] == ordering and r["k"] == k], level_keys)
                    for k in cfg["k_values"]
                },
            }
            for ordering in args.orderings
        },
    }

    from benchpress.evaluation_harness import (
        BENCH_IDS, BENCH_METRICS, M_FULL, benchmark_metric_identity_sha256)
    with open(calibrator_path, "rb") as f:
        calibrator_sha256 = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "git_commit": prospective.git_commit(),
        "may_json": os.path.abspath(os.path.expanduser(args.may_json)),
        "aug_json": os.path.abspath(os.path.expanduser(args.aug_json)),
        "may_matrix_identity_sha256": bp["matrix_identity_sha256"](may["matrix"]),
        "harness_matrix_identity_sha256": bp["matrix_identity_sha256"](M_FULL),
        "aug_matrix_identity_sha256": bp["matrix_identity_sha256"](aug["matrix"]),
        "benchmark_metric_identity_sha256": benchmark_metric_identity_sha256(
            BENCH_METRICS, BENCH_IDS),
        "may_shape": [int(x) for x in may["matrix"].shape],
        "aug_shape": [int(x) for x in aug["matrix"].shape],
        "frozen_calibrator_path": os.path.relpath(calibrator_path, REPO_ROOT),
        "frozen_calibrator_sha256": calibrator_sha256,
        "crossfit_selected_risk_model_by_fold":
            artifact["crossfit_selected_risk_model_by_fold"][METHOD],
        "config": {
            "method": METHOD,
            "fold_spec": FOLD_SPEC,
            "seed": int(args.seed),
            "final_risk_model_seed": int(args.seed) + 2000,
            "conformal_levels": level_keys,
            "orderings": list(args.orderings),
            "k_values": cfg["k_values"],
            "probe_order_json": os.path.relpath(
                os.path.abspath(os.path.expanduser(args.probe_order_json)), REPO_ROOT),
            "workers": int(args.workers),
            "point_predictor": "predict_benchpress_scores (Logit Bias ALS, rank=2, lambda=0.1)",
            "inclusion_rule_new_models": "prospective_update.probe_unit_included",
        },
        "n_records": int(len(records)),
    }

    write_json_atomic = bp["write_json_atomic"]
    write_json_atomic(os.path.join(args.out_dir, "raw_records.json.gz"), records)
    write_json_atomic(os.path.join(args.out_dir, "manifest.json"), manifest, indent=2)
    write_json_atomic(os.path.join(args.out_dir, "summary.json"), summary, indent=2)
    print(f"Wrote prospective coverage results under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
