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

Smoke-test one small shard on any CPU machine:

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

Full low-cost choose-5 run, split into 10 waves and 8 shards per wave. Each
`(wave-index, shard-index)` pair is an independent CPU job; submit them with
whatever scheduler is available:

```bash
for wave in $(seq 0 9); do
  for shard in $(seq 0 7); do
    ./run.sh run-shard \
      --candidate-allowlist ../candidate_allowlists/user_cheap_20260505.json \
      --k 5 --metric medae \
      --num-waves 10 --wave-index "$wave" \
      --num-shards 8 --shard-index "$shard" \
      --workers 128
  done
done
```

Full greedy-rank top-30 choose-5 run, split into 20 waves with one shard each:

```bash
for wave in $(seq 0 19); do
  ./run.sh run-shard \
    --candidate-allowlist ../candidate_allowlists/full_rank_top30_count_by_greedy_20260605.json \
    --k 5 --metric medae \
    --num-waves 20 --wave-index "$wave" \
    --num-shards 1 --shard-index 0 \
    --workers 24 \
    --out-dir results/exhaustive_medae_k5_candidates-full_rank_top30_count_by_greedy_20260605
done
```

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
- Greedy-rank top-30 diagnostic candidate set:
  `../candidate_allowlists/full_rank_top30_count_by_greedy_20260605.json`
  (30 full-matrix benchmarks, so `C(30, 5) = 142,506` subsets).

## Outputs

Runs default to `results/<run_id>/`; pass `--out-dir` to write shards to a
shared filesystem instead
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

## Last valid results

Both results use `k=5`, metric `medae`, and protocol
`all_known_probe_bruteforce_v1`.

### Greedy-rank top-30 candidates

- Candidate set: `full_rank_top30_count_by_greedy_20260605`
- Local merged summary:
  `results/top30_bruteforce/merged_summary.json.gz`
- Remote PVC source:
  `results/exhaustive_medae_k5_candidates-full_rank_top30_count_by_greedy_20260605/merged_summary.json.gz`
- Completeness: `142,506 / 142,506` subsets, `missing_chunks=0`
- Best probe set: `gpqa_diamond`, `hle`, `mmlu_pro`, `arc_agi_1`,
  `codeforces_rating`
- Best score: MedAE `3.9264436813102748`; MedAPE
  `6.588142223218943`; `n=2604`

### Low-cost candidates

- Candidate set: `user_cheap_20260505`
- Local merged summary:
  `results/lowcost_bruteforce/merged_summary.json.gz`
- Remote PVC source:
  `results/exhaustive_medae_k5_candidates-user_cheap_20260505/merged_summary.json.gz`
- Completeness: `53,130 / 53,130` subsets, `missing_combo_indices=0`
- Best probe set: `tau2_bench_telecom`, `matharena_apex_2025`,
  `gpqa_diamond`, `aider_polyglot_diff`, `mmlu_pro`
- Best score: MedAE `4.464725682233784`; MedAPE
  `7.411531542389898`; `n=2604`
