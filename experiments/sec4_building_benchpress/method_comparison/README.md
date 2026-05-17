# Section 4.2 Method Comparison

## Paper mapping

- Main text: `\Cref{sec:method_comparison}`, `\Cref{tab:top15}`
- Appendix: `\Cref{fig:transform_method_grid}` and `\Cref{tab:full_grid}`
- Appendix script: `experiments/appendix_b_sec4_methods/method_comparison/gen_full_table.py` reads this directory's `results.json`

## Purpose

Evaluate the full transform-by-method grid under the shared per-model holdout folds. The grid covers seven feature transforms and twelve prediction methods: mean baselines, KNN baselines, BenchReg, ModelReg, Soft-Impute, Bias ALS, NMF, PMF, Nuclear Norm, and MLP. This experiment is prediction-first: every shard saves full prediction matrices before any metric aggregation, so score-error metrics and coverage can be recomputed without rerunning predictors.

## How to run

This is a 329-shard sweep over `(transform, method, hyperparameter)`. Each shard runs all 30 shared folds and writes one `.npz` file under `predictions/`. Run sequentially (slow), in a local process pool, or distribute over a CPU cluster.

```bash
# List all shards (one JSON line per shard)
python experiments/sec4_building_benchpress/method_comparison/run.py --list-shards

# Run a single shard (resumable: existing .npz is skipped unless --force)
python experiments/sec4_building_benchpress/method_comparison/run.py --shard-index 0
# ... shard-index 1 .. 328 (parallelize across cores or pods as your infra allows)

# Merge: recompute metrics + figures from predictions/*.npz
python experiments/sec4_building_benchpress/method_comparison/run.py --merge
python experiments/sec4_building_benchpress/method_comparison/gen_table.py > /tmp/sec4_top15.tex
python experiments/sec4_building_benchpress/method_comparison/plot.py
```

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
- Completion method implementations in `benchpress/methods/completers.py`; `benchpress/all_methods.py` remains a compatibility re-export for older scripts
- Transform pipeline in `benchpress/methods/transforms.py`

## Outputs

- `predictions/*.npz`: source-of-truth prediction cache, one file per `(transform, method, HP)` shard
- `manifest.json`: derived inventory of completed/missing shards and per-HP metrics; `gen_table.py` reads this directly so each metric can choose its own top HPs
- `results.json`: derived MedAPE-best-HP summary used by `plot.py` and appendix table generation
- `figures/bp_transform_method_grid*.{pdf,png}`
- Top-15 LaTeX table printed by `gen_table.py`

Paper-facing score-error metrics use per-fold MedAPE/MedAE followed by the median over the 10 seeds x 3 folds. `predictions/*.npz` remains the source of truth, so changing this aggregation only requires rerunning `run.py --merge`; predictor shards do not need to be rerun.

Each `predictions/*.npz` contains:

| Key | Shape / type | Meaning |
|-----|--------------|---------|
| `M_pred_by_fold` | `(30, N, D)` float | Full prediction matrix for every fold |
| `fold_id` | `(n_test,)` int | Fold index for each held-out cell |
| `test_i`, `test_j` | `(n_test,)` int | Held-out model and benchmark indices |
| `actual` | `(n_test,)` float | True held-out scores |
| `predicted` | `(n_test,)` float | Predicted held-out scores |
| `metadata_json` | scalar string | Transform, method, HP, matrix shape, fold settings |

`results.json` is intentionally not the cache. If metric definitions change, rerun only:

```bash
python experiments/sec4_building_benchpress/method_comparison/run.py --merge
```

## Resume / rerun

Resume is file-based. `run.py --shard-index K` skips if the expected `predictions/*.npz` file exists and contains the required keys. To rerun one bad shard, delete only that `.npz` file or run:

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

BenchReg and ModelReg also require at least 5 shared observations for a pairwise regression; this fixed guard is part of the method definition and is documented in Appendix B.1.

## Last valid result

Previous valid result before adding prediction-first sharded runs: `results.json` from the old serial runner on the current 84 x 133 matrix. The current valid result has 329 prediction shard files, `manifest.json` with `n_completed_shards = 329`, and `results.json` derived from those shard files.
