# H6 Strong-Neighbor Support

## Paper mapping
- Section: `sec:reliability_analysis`, benchmark-side Table `tab:predictability_factors` and Figure `fig:predictability_factors_51`.
- Parent runner: `../run.sh`.

## Purpose
H6 tests whether weakening the target benchmark's best score-matrix neighbor hurts prediction. For each target benchmark, it drops nested fractions of the best neighbor's cells that overlap with the target benchmark's observed rows and compares each treatment to the paired drop-rate 0 baseline.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- `N_SEEDS = 5`, `SEED = 42`, drop rates `[0.25, 0.50, 0.75]` compared against the same drop-rate 0 baseline.
- For each `(benchmark, seed)`, the hide-half split is fixed across all drop rates and the 25%, 50%, and 75% neighbor masks are nested prefixes of one fixed permutation, so the intervention is a strict dose-response in removed neighbor evidence.

## Outputs
- `ablation_records_nested.jsonl`: resumable per-record cache, one JSON object per `(benchmark, seed, drop_rate)`.
- `ablation_results.json`:
  - `records`: one record per `(benchmark, seed, drop_rate)` for nonzero drop rates.
  - each record stores base/treatment MedAPE and MedAE, deltas, nested mask metadata, dropped model indices, and paired `raw_base` / `raw_treat` predictions.
  - `wilcoxon`: paired benchmark-level summary at drop rate 0.75.

## Run
This ablation is embarrassingly parallel across target benchmarks. A full serial run takes too long for an interactive session, so use a single-benchmark smoke test first and then run benchmark shards in a clean remote worktree.

```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H6_strong_neighbor_support/ablation.py
```

Smoke test:
```bash
python H6_strong_neighbor_support/ablation.py \
  --max-benchmarks 1 --max-seeds 1 --force \
  --cache-path /tmp/h6_smoke_records.jsonl \
  --output-path /tmp/h6_smoke_results.json
```

The script is deterministic and resumable from `ablation_records_nested.jsonl`; use `--force` only when intentionally rerunning from scratch.

Sharded run (across multiple processes/machines):
```bash
cd experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python H6_strong_neighbor_support/ablation.py \
  --cache-path H6_strong_neighbor_support/shards/records_<shard>.jsonl \
  --output-path H6_strong_neighbor_support/shards/results_<shard>.json \
  --num-shards <N> \
  --shard-id <ID>
```

Shard caches are append-only JSONL files; existing records are skipped, so rerunning the same shard resumes from the cache. After all shards finish, merge shard caches into the local `ablation_records_nested.jsonl` / `ablation_results.json`.
