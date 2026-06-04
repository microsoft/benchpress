# Brute-force optimal probe search

## Paper mapping

- Section: `\Cref{sec:probe_selection}` / Appendix probe-selection diagnostics.
- This directory is an exhaustive-search counterpart to
  `../all_known/`: it uses the same all-known-cell evaluation protocol, but
  evaluates every `k`-benchmark probe subset instead of greedily selecting one
  benchmark at a time.

## Purpose

Find the optimal low-cost probe set for the current all-known scorecard
recovery objective. For each target model, the selected probe columns are
visible, probe cells count as exact predictions, and all other observed cells
are predicted by BenchPress. The primary comparison is greedy versus exhaustive
optimal under the same candidate universe and metric.

## How to run

Smoke-test one small shard on Bonete or another remote CPU environment:

```bash
cd experiments/sec5_findings/optimal_probe/brute_force
./run.sh run-shard \
  --candidate-allowlist ../candidate_allowlists/user_cheap_20260505.json \
  --k 3 --candidate-limit 8 \
  --num-waves 2 --wave-index 0 \
  --num-shards 2 --shard-index 0 \
  --workers 4 --chunk-size 2 \
  --max-subsets 4
```

Full low-cost choose-5 run, split into 10 waves and 8 shards per wave:

```bash
python submit_bonete.py \
  --candidate-allowlist ../candidate_allowlists/user_cheap_20260505.json \
  --k 5 --metric medae \
  --num-waves 10 --num-shards 8 \
  --waves 0 \
  --workers 128 --cpus 128 --memory 256 \
  --setup benchpress-cpu \
  --branch main
```

Submit one wave at a time by changing `--waves 0` to `--waves 1`, ..., or use
`--waves 0-9` only when intentionally launching the full 80-job sweep.

Merge after all shards finish:

```bash
./run.sh merge \
  --out-dir results/exhaustive_medae_k5_candidates-user_cheap_20260505 \
  --top-n 100
```

## Inputs

- Score matrix and observed mask from `benchpress.evaluation_harness`.
- Predictor: `predict_benchpress_scores`.
- Candidate allowlists from `../candidate_allowlists/`.
- Default low-cost candidate set:
  `../candidate_allowlists/user_cheap_20260505.json` (25 current-matrix
  benchmarks, so `C(25, 5) = 53,130` subsets).

## Outputs

Local runs default to `results/<run_id>/`. Bonete submissions default to the
shared PVC directory `/data/benchpress/runs/benchpress/probe_bruteforce_results/<run_id>/`
so separately scheduled shards can be merged.

```text
results/exhaustive_<metric>_k<k>_candidates-<source>/
├── config.json
├── shards/
│   └── wave_XX/shard_YYY/chunk_ZZZZZZ.json.gz
└── merged_summary.json.gz
```

Each chunk stores raw per-cell predictions for every evaluated subset. The
merged summary stores per-subset metrics and the best subset without duplicating
raw predictions.

## Parallel execution

Subset evaluations are independent. The runner partitions global combination
indices by:

```text
combo_index % (num_waves * num_shards) == wave_index + num_waves * shard_index
```

This makes every wave/shard disjoint and deterministic. Use large CPU pods with
many internal workers rather than one pod per subset.

## Resume / rerun

- Done unit: one chunk file.
- Re-running a shard validates existing chunks against the expected config and
  combo indices, then skips valid chunks.
- If a chunk is stale or protocol-incompatible, delete that chunk only and rerun
  the same wave/shard.
- `merge` fails fast unless all expected chunks are present, unless
  `--allow-incomplete` is explicitly provided for diagnostics.

## Last valid result

Not yet run for the exhaustive low-cost choose-5 setting. The intended first
full result is:

- Candidate set: `user_cheap_20260505`
- `k=5`
- Metric: `medae`
- Protocol: `all_known_probe_bruteforce_v1`
