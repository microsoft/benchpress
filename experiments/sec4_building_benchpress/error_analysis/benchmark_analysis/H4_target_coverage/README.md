# H4 Target Coverage Ablation

H4 measures whether benchmark prediction error increases when the target benchmark has fewer observed training scores. For each benchmark and seed, the script hides half of the benchmark's observed scores as test cells, keeps progressively smaller fractions of the remaining training cells, predicts with the fixed BenchPress predictor, and records MedAPE/MedAE deltas against the paired drop-rate 0 no-intervention baseline.

H4 is an ablation only. H1-H3 read official BenchPress default errors from `benchpress/evaluation/default_predictions/benchpress_default/by_benchmark.json`, not from H4 outputs.

## Outputs

- `shards/bench_<benchmark_id>_seed_<seed>.json`: one resumable unit per `(benchmark, seed)`, storing raw base/treatment predictions and metrics.
- `ablation_results.json`: merged output with the original downstream schema:
  - `records`: one record per `(benchmark, seed, drop_rate)` for nonzero drop rates.
  - each record stores `base_medape`, `treat_medape`, `delta_medape`, `base_medae`, `treat_medae`, `delta_medae`, `raw_base`, and `raw_treat`.
  - `wilcoxon`: paired benchmark-level Wilcoxon summary at `drop_rate=0.75`.

## How to run

H4 has about `N_BENCH * N_SEEDS` independent `(benchmark, seed)` units. The script is fully resumable: if a unit's shard JSON already exists, it is skipped.

### Single process (slow but simple)

```bash
cd experiments/sec4_building_benchpress/error_analysis/benchmark_analysis
python H4_target_coverage/ablation.py
```

### Sharded run (recommended)

Split the work into 4 shards across 4 machines or pods (each with 48 worker processes):

```bash
# Run on shard 0..3 in parallel (different machines or processes)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BENCHPRESS_H4_WORKERS=48

python H4_target_coverage/ablation.py --num-shards 4 --shard-id 0 --workers 48
python H4_target_coverage/ablation.py --num-shards 4 --shard-id 1 --workers 48
python H4_target_coverage/ablation.py --num-shards 4 --shard-id 2 --workers 48
python H4_target_coverage/ablation.py --num-shards 4 --shard-id 3 --workers 48
```

After all shards finish, merge once:

```bash
python H4_target_coverage/ablation.py --merge-only
```

## Resume and full rerun

The script is resumable at the `(benchmark, seed)` unit level. If a pod fails, rerun the same `--num-shards/--shard-id`; existing shard JSON files are skipped.

For a true full rerun, delete both the merged file and the unit shards first:

```bash
rm -f H4_target_coverage/ablation_results.json
rm -rf H4_target_coverage/shards
```

## Smoke test

Smoke tests must not overwrite the real `ablation_results.json`. Use temporary shard and output paths:

```bash
rm -rf /tmp/benchpress_h4_smoke_shards /tmp/benchpress_h4_smoke.json
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python H4_target_coverage/ablation.py \
  --workers 2 \
  --limit-units 2 \
  --shards-dir /tmp/benchpress_h4_smoke_shards \
  --output-path /tmp/benchpress_h4_smoke.json
```

Expected smoke-test result: two unit shard files under `/tmp/benchpress_h4_smoke_shards` and a readable `/tmp/benchpress_h4_smoke.json` with nonempty `records`.
