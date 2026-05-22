# Section 5.2 Ranking-Preservation Greedy Probe Set

## Paper mapping

- Main text: `\Cref{sec:ranking_preservation}` / Section 5.2.
- Purpose: choose a top-10 probe set whose objective is margin-5 pairwise ranking accuracy. The default run is cost-unaware; the cost-aware run constrains greedy candidates to the curated benchmark-cost allowlist.

## Purpose

This experiment asks which probes best preserve benchmark leaderboards when the downstream decision metric is margin-aware pairwise ranking accuracy. It uses the same all-known-cell probe-set setting as Section 5.1: for a target model, selected probe benchmarks are revealed exactly, and every other observed target-model cell is predicted by BenchPress.

The greedy objective is Section 5.2's pairwise ranking accuracy at margin 5. For each candidate probe set, the script groups the completed predictions by benchmark and calls `benchpress.evaluation_harness.compute_ranking_accuracy(..., margin=5, aggregation="per_group_median")`. Probe cells are exact predictions and remain in the fixed all-known-cell denominator, so adding a probe improves the completed leaderboard rather than removing that benchmark from evaluation.

`run_model_split_validation.py` is the held-out check. It uses the same 70/30 model split and isolated held-out model protocol as Section 5.1's probe validation: select probes on training model rows, then validate the fixed prefix on held-out model rows where BenchPress sees only training rows plus that held-out target model's probe scores.

## How to run

This is a slow CPU sweep. Recommended on a remote machine with many cores.

```bash
cd experiments/sec5_findings/ranking_preservation/greedy_probe_set
WORKERS=48 ./run.sh
```

Smoke test:

```bash
MAX_STEPS=1 CANDIDATE_LIMIT=2 WORKERS=2 OUT=smoke_pairwise_margin5.json.gz ./run.sh
```

Cost-aware candidate-constrained run:

```bash
CANDIDATE_ALLOWLIST=../../optimal_probe/candidate_allowlists/user_cheap_20260505.json \
  OUT=greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz \
  WORKERS=48 ./run.sh
```

Model-split validation smoke test:

```bash
MAX_STEPS=1 CANDIDATE_LIMIT=2 MODEL_LIMIT=8 WORKERS=2 \
  OUT=smoke_model_split_pairwise_margin5.json.gz \
  ./run_model_split_validation.sh
```

Model-split validation full runs:

```bash
WORKERS=48 ./run_model_split_validation.sh

CANDIDATE_ALLOWLIST=../../optimal_probe/candidate_allowlists/user_cheap_20260505.json \
  OUT=model_split_validation_pairwise_margin5_train70_usercheap.json.gz \
  WORKERS=48 ./run_model_split_validation.sh
```

## Inputs

- Score matrix and benchmark IDs from `benchpress.evaluation_harness`.
- Probe-set prediction primitive from `benchpress.evaluation_harness.evaluate_probe_set`.
- Ranking metric from `benchpress.evaluation_harness.compute_ranking_accuracy`.

## Outputs

Results are written under `results/`:

- `greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz`
- `greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz`
- `model_split_validation_pairwise_margin5_train70_all.json.gz`
- `model_split_validation_pairwise_margin5_train70_usercheap.json.gz`

The result file contains:

| Key | Meaning |
|-----|---------|
| `config` | objective, margin, fixed-universe protocol, matrix size, candidate count, seed, workers |
| `trajectory` | one entry per greedy step |
| `trajectory[*].candidate_results` | raw candidate evaluations for that step |
| `trajectory[*].candidate_results[*].predictions` | raw per-cell prediction lists (`i`, `j`, `true`, `pred`) |
| `trajectory[*].pairwise_accuracy_margin5` | selected prefix's benchmark-median margin-5 pairwise ranking accuracy |
| `trajectory[*].candidate_results[*].per_benchmark_ranking` | per-benchmark ranking accuracy/count diagnostics |

The model-split validation result files additionally contain:

| Key | Meaning |
|-----|---------|
| `split` | train and validation model IDs |
| `trajectory[*].train` | selected prefix's train-split ranking accuracy and diagnostics |
| `trajectory[*].validation_non_probe` | held-out ranking accuracy excluding already measured probe cells |
| `trajectory[*].validation_with_probe_zero` | held-out ranking accuracy including observed probe cells as exact predictions |

Raw per-cell predictions are the bottleneck output and are saved for every candidate at every step.

## Resume / rerun

Re-running the same `OUT=...` resumes from the completed trajectory and candidate cache when the objective, margin, protocol, candidate allowlist, candidate limit, and candidate count match. Candidate caches are keyed by step, candidate benchmark, and the probe prefix before the candidate, so a changed greedy prefix refuses to reuse stale shards.

The model-split validation script has the same resume behavior, with the model split and protocol included in the cache key and output config.

## Last valid result

Model-split validation run at commit `6109f74`.

| Candidate pool | k | Probe prefix | Train margin-5 pairwise accuracy | Held-out non-probe accuracy | Held-out with-probe-zero accuracy |
|---|---:|---|---:|---:|---:|
| Any benchmark | 5 | `gpqa_diamond`, `hle`, `bullshit_pushback`, `mmlu_pro`, `erqa` | 0.8462 | 0.9310 | 0.9370 |
| Any benchmark | 10 | `gpqa_diamond`, `hle`, `bullshit_pushback`, `mmlu_pro`, `erqa`, `c_eval`, `aime_2024`, `arena_hard`, `babyvision`, `popqa` | 0.8800 | 0.8868 | 0.9412 |
| Low-cost benchmarks | 5 | `gpqa_diamond`, `mmlu_pro`, `arena_hard`, `alpacaeval_2`, `hmmt_feb_2026` | 0.8208 | 0.9091 | 0.9214 |
| Low-cost benchmarks | 10 | `gpqa_diamond`, `mmlu_pro`, `arena_hard`, `alpacaeval_2`, `hmmt_feb_2026`, `math_500`, `aime_2025`, `vibe_eval`, `tau2_bench_airline`, `aider_polyglot_whole` | 0.8512 | 0.9091 | 0.9304 |

Cost-unaware legacy run at commit `3d6ac50`, using 48 workers. The selected top-10 prefix ends with `alpacaeval_2` and reaches margin-5 pairwise ranking accuracy `0.888543823326432` over `27245` comparable pairs.

Cost-aware candidate-constrained legacy run at commit `2ccb061`, using 48 workers. The selected top-10 prefix is `gpqa_diamond`, `mmlu_pro`, `aime_2025`, `bullshit_pushback`, `hmmt_feb_2026`, `math_500`, `alpacaeval_2`, `hmmt_feb_2025`, `aider_polyglot_whole`, `arena_hard`, and reaches margin-5 pairwise ranking accuracy `0.8619555353901995` over `27245` comparable pairs.
