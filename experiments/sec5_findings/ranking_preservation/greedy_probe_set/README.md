# Section 5.2 Ranking-Preservation Greedy Probe Set

## Paper mapping

- Main text: `\Cref{sec:ranking_preservation}` / Section 5.2.
- Purpose: choose a top-10 probe set whose objective is margin-5 pairwise ranking accuracy. The default run is cost-unaware; the cost-aware run constrains greedy candidates to the curated benchmark-cost allowlist.

## Purpose

This experiment asks which probes best preserve benchmark leaderboards when the downstream decision metric is margin-aware pairwise ranking accuracy. It uses the same all-known-cell probe-set setting as Section 5.1: for a target model, selected probe benchmarks are revealed exactly, and every other observed target-model cell is predicted by BenchPress.

The greedy objective is Section 5.2's pairwise ranking accuracy at margin 5. For each candidate probe set, the script groups the completed predictions by benchmark and calls `benchpress.evaluation_harness.compute_ranking_accuracy(..., margin=5, aggregation="per_group_median")`. Probe cells are exact predictions and remain in the fixed all-known-cell denominator, so adding a probe improves the completed leaderboard rather than removing that benchmark from evaluation.

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

## Inputs

- Score matrix and benchmark IDs from `benchpress.evaluation_harness`.
- Probe-set prediction primitive from `benchpress.evaluation_harness.evaluate_probe_set`.
- Ranking metric from `benchpress.evaluation_harness.compute_ranking_accuracy`.

## Outputs

Results are written under `results/`:

- `greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz`
- `greedy_pairwise_margin5_top10_targets_usercheap_candidates_usercheap.json.gz`

The result file contains:

| Key | Meaning |
|-----|---------|
| `config` | objective, margin, fixed-universe protocol, matrix size, candidate count, seed, workers |
| `trajectory` | one entry per greedy step |
| `trajectory[*].candidate_results` | raw candidate evaluations for that step |
| `trajectory[*].candidate_results[*].predictions` | raw per-cell prediction lists (`i`, `j`, `true`, `pred`) |
| `trajectory[*].pairwise_accuracy_margin5` | selected prefix's benchmark-median margin-5 pairwise ranking accuracy |
| `trajectory[*].candidate_results[*].per_benchmark_ranking` | per-benchmark ranking accuracy/count diagnostics |

Raw per-cell predictions are the bottleneck output and are saved for every candidate at every step.

## Resume / rerun

Re-running the same `OUT=...` resumes from the completed trajectory and candidate cache when the objective, margin, protocol, candidate allowlist, candidate limit, and candidate count match. Candidate caches are keyed by step, candidate benchmark, and the probe prefix before the candidate, so a changed greedy prefix refuses to reuse stale shards.

## Last valid result

Cost-unaware run on branch `yz`, commit `3d6ac50`, using 48 workers. The selected top-10 prefix ends with `alpacaeval_2` and reaches margin-5 pairwise ranking accuracy `0.888543823326432` over `27245` comparable pairs.

Cost-aware candidate-constrained run on branch `yz`, commit `2ccb061`, using 48 workers. The selected top-10 prefix is `gpqa_diamond`, `mmlu_pro`, `aime_2025`, `bullshit_pushback`, `hmmt_feb_2026`, `math_500`, `alpacaeval_2`, `hmmt_feb_2025`, `aider_polyglot_whole`, `arena_hard`, and reaches margin-5 pairwise ranking accuracy `0.8619555353901995` over `27245` comparable pairs.
