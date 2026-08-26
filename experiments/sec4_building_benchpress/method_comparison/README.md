# Section 4.2 Method Comparison

## Paper mapping

- Main text: `\Cref{sec:method_comparison}`, `\Cref{tab:top15}`
- Appendix: `\Cref{fig:transform_method_grid}` and `\Cref{tab:full_grid}`
- Appendix script: `experiments/appendix_c_sec4_methods/method_comparison/gen_full_table.py` reads this directory's `results.json`

## Purpose

Evaluate the full transform-by-method grid under the shared per-model holdout folds. The grid covers seven feature transforms and twelve prediction methods: mean baselines, KNN baselines, BenchReg, ModelReg, Soft-Impute, Bias ALS, NMF, PMF, Nuclear Norm, and MLP. This experiment is prediction-first: every shard saves full prediction matrices before any metric aggregation, so score-error metrics and coverage can be recomputed without rerunning predictors.

## How to run

This is a 329-shard sweep over `(transform, method, hyperparameter)`. Each shard runs all 30 shared folds and writes one `.npz` file under `predictions/`. Run sequentially (slow), in a local process pool, or distribute over a CPU cluster.

```bash
# Confirm the current matrix and ordered benchmark-metric semantics before
# reusing or generating prediction shards.
python - <<'PY'
from benchpress.evaluation_harness import (
    BENCH_IDS,
    BENCH_METRICS,
    M_FULL,
    benchmark_metric_identity_sha256,
    matrix_identity_sha256,
)

matrix_identity = matrix_identity_sha256(M_FULL)
metric_identity = benchmark_metric_identity_sha256(BENCH_METRICS, BENCH_IDS)
assert M_FULL.shape == (129, 253)
assert matrix_identity == (
    "4d882cce44ff1d20c9fbada545a21a6dadf4bb8e3eb1a989f9d93bfed38bef25"
)
assert metric_identity == (
    "96713a65a02bb93f7ee6668b0421cff189e88de5264a8ca6c99e7c05c2a74566"
)
print("matrix_identity_sha256:", matrix_identity)
print("benchmark_metric_identity_sha256:", metric_identity)
PY

# List all shards (one JSON line per shard)
python experiments/sec4_building_benchpress/method_comparison/run.py --list-shards

# Run a single shard (resumable: existing .npz is skipped unless --force)
python experiments/sec4_building_benchpress/method_comparison/run.py --shard-index 0
# ... shard-index 1 .. 328 (parallelize across cores or pods as your infra allows)

# Run the full sweep with bounded CPU parallelism
experiments/sec4_building_benchpress/method_comparison/run.sh \
  --workers 48 --merge --table-out /tmp/sec4_top15.tex

# Merge: recompute metrics + figures from predictions/*.npz
python experiments/sec4_building_benchpress/method_comparison/run.py --merge
python experiments/sec4_building_benchpress/method_comparison/gen_table.py > /tmp/sec4_top15.tex
# Requires the nested-selection inner pass described below:
python experiments/sec4_building_benchpress/method_comparison/plot.py
```

## Nested hyperparameter selection

`results.json` picks each pair's hyperparameter by the lowest MedAPE on the outer
test cells, so selection and reporting share cells. The nested pass re-selects on
inner validation cells carved out of each outer fold's *training* cells
(`holdout_inner_per_model`), leaving the outer test cells untouched, and reports
the selected configuration on those same outer test cells.

The outer sweep is not rerun: `--inner-shard-index` only adds validation scores,
and `--merge-nested` reads them together with the existing `predictions/*.npz`.

```bash
# Score one shard on the inner validation folds (resumable, same 0..328 indices)
python experiments/sec4_building_benchpress/method_comparison/run.py --inner-shard-index 0

# Re-select hyperparameters on inner validation → results_nested.json
python experiments/sec4_building_benchpress/method_comparison/run.py --merge-nested
```

Consumed by `experiments/appendix_c_sec4_methods/method_comparison/` for the
Appendix C.2 model-selection table. Because the selected hyperparameter may
differ across outer folds, each row reports the most frequently selected one
plus `modal_hp_share` and `n_distinct_hp_selected`.

## Parallel execution

Unit of work: one `(transform, method, hyperparameter)` shard. Each shard is fully independent and resumable.

Recommended layout: one CPU core per shard (set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) — running many small jobs is faster than multiprocessing inside one large job.

Expected prediction shard count is 329. This is larger than the 84 transform-method cells because each hyperparameter setting is cached separately:

- 7 transforms
- 12 methods
- hyperparameter grids: 1, 1, 4, 4, 9, 9, 1, 3, 4, 4, 4, 3 per method

## Inputs

- `benchpress.evaluation_harness.M_FULL`
- `benchpress/evaluation/folds/folds_s10_f3_bs42_ms1.json`, loaded through `load_folds()`
- Completion method implementations in `benchpress/methods/completers.py`; `benchpress/all_methods.py` remains a compatibility re-export for older scripts. The MLP method needs the optional `mlp` dependency group: install the environment with `pip install -e .[mlp]`, otherwise its 21 shards raise `ImportError` instead of contributing rows
- Transform pipeline in `benchpress/methods/transforms.py`

## Outputs

- `predictions/*.npz`: source-of-truth prediction cache, one file per `(transform, method, HP)` shard
- `manifest.json`: derived inventory of completed/missing shards and per-HP metrics; `gen_table.py` reads this directly so each metric can choose its own top HPs
- `results.json`: derived MedAPE-best-HP summary used by `plot.py` and appendix table generation
- `figures/bp_transform_method_grid*.{pdf,png}`
- Top-15 LaTeX table printed by `gen_table.py`
- `inner_scores/*.npz`: one file per shard, one row per `(outer fold, inner fold)` with the configuration's inner-validation MedAPE, MedAE, coverage, and cell count. Prediction matrices are not persisted here because nested selection consumes only these scores; the reported test error still comes from `predictions/*.npz`
- `results_nested.json`: per-pair leaderboard whose hyperparameters were selected on inner validation

Paper-facing score-error metrics use per-fold MedAPE/MedAE followed by the median over the 10 seeds x 3 folds. `predictions/*.npz` remains the source of truth, so changing this aggregation only requires rerunning `run.py --merge`; predictor shards do not need to be rerun.

Each `predictions/*.npz` contains:

| Key | Shape / type | Meaning |
|-----|--------------|---------|
| `M_pred_by_fold` | `(30, N, D)` float | Full prediction matrix for every fold |
| `fold_id` | `(n_test,)` int | Fold index for each held-out cell |
| `test_i`, `test_j` | `(n_test,)` int | Held-out model and benchmark indices |
| `actual` | `(n_test,)` float | True held-out scores |
| `predicted` | `(n_test,)` float | Predicted held-out scores |
| `metadata_json` | scalar string | Transform, method, HP, matrix shape, fold settings, matrix identity, and ordered benchmark-metric identity |

`results.json` is intentionally not the cache. If only the aggregation of the
same stored `actual` and `predicted` vectors changes, rerun only:

```bash
python experiments/sec4_building_benchpress/method_comparison/run.py --merge
```

If benchmark metric semantics, transforms, prediction methods, matrix values, or
observed cells change, rerun the affected prediction shards. The runner rejects
stale shards whose matrix or ordered benchmark-metric identity differs.

## Resume / rerun

Resume is file-based. `run.py --shard-index K` skips only when the expected
`predictions/*.npz` file contains the required arrays and its matrix identity,
ordered benchmark-metric identity, fold protocol, method, transform, and
hyperparameters all match the current run. A shard from the same numeric matrix
but different metric semantics is stale and is recomputed. To rerun one bad
shard explicitly:

```bash
python experiments/sec4_building_benchpress/method_comparison/run.py --shard-index K --force
```

To list shard indices and completion status:

```bash
python experiments/sec4_building_benchpress/method_comparison/run.py --list-shards
```

To submit only a range or a few missing shards:

```bash
experiments/sec4_building_benchpress/method_comparison/run.sh --start 80 --end 120
experiments/sec4_building_benchpress/method_comparison/run.sh --limit 20
experiments/sec4_building_benchpress/method_comparison/run.sh --workers 48 --force
```

Do not delete `predictions/` unless intentionally invalidating the whole experiment. Delete `results.json` or `manifest.json` freely; they are derived from `predictions/*.npz`.

## Hyperparameters

| Method | Grid |
|--------|------|
| Benchmark Mean, Model Mean | none |
| Bench-KNN, Model-KNN | `k ∈ {3, 5, 7, 10}` |
| BenchReg, ModelReg | `top_k ∈ {3, 5, 7}`, `R²_min ∈ {0.1, 0.2, 0.3}` |
| Soft-Impute | none (rank fixed at 2 per §3 rank-2 evidence) |
| Bias ALS | `λ ∈ {0.01, 0.1, 1.0}` (rank fixed at 2) |
| NMF, PMF | `rank ∈ {1, 2, 3, 5}` |
| Nuclear Norm | `λ ∈ {0.1, 0.5, 1.0, 5.0}` |
| MLP | `lr ∈ {1e-4, 1e-3, 1e-2}` |

BenchReg and ModelReg also require at least 5 shared observations for a pairwise regression; this fixed guard is part of the method definition and is documented in Appendix C.1.

## Last valid result

Current metric-aware result:

- code commit: `f4319afa41c47fac470a8a5b5ca0dacbd95ee3c8`
- matrix: 129 models x 253 benchmarks, 4,905 observed cells
- matrix identity: `4d882cce44ff1d20c9fbada545a21a6dadf4bb8e3eb1a989f9d93bfed38bef25`
- ordered benchmark-metric identity: `96713a65a02bb93f7ee6668b0421cff189e88de5264a8ca6c99e7c05c2a74566`
- completed shards: 329/329
- selected full-coverage predictor: Logit Bias ALS, rank 2, lambda 0.1
- held-out MedAPE: 7.740388161995237
- held-out MedAE: 4.434680035745227
- coverage: 100%
