# Section 5.2 Ranking Preservation

## Paper mapping

- Main text: `\Cref{sec:ranking_preservation}` reports pairwise ranking accuracy.
- Appendix: `\Cref{app:ranking_preservation}` reports auxiliary top-fraction shortlist recovery.
- Purpose: evaluate whether BenchPress preserves decision-relevant leaderboard structure without relying on Kendall tau.

## Purpose

The operational question is whether BenchPress preserves which model is better when two models differ meaningfully on the same benchmark. Exact leaderboard positions are unstable when many models are separated by tiny score gaps: a small prediction error can swap two nearly tied models, so exact-rank agreement is too sensitive for this decision question. This experiment evaluates ranking preservation at coarser, decision-facing resolutions:

1. **Margin-aware pairwise ranking accuracy**: for each fold and benchmark, the completed leaderboard uses true scores for seen cells and BenchPress predictions for held-out cells. Among all same-benchmark model pairs, pairs where both cells were seen are discarded; every pair with at least one held-out cell is scored if its true score gap is at least the margin. The metric is computed by `benchpress.evaluation_harness.compute_ranking_accuracy`: number of comparable pairs whose completed ordering matches the true ordering divided by the number of comparable pairs. Margin 0 includes every non-tied pair and is therefore most sensitive to near-ties; larger margins focus on clearer score gaps.
2. **Top-fraction recovery**: for each benchmark leaderboard, seen cells keep their true scores and held-out cells use BenchPress predictions; how well does the completed top fraction recover the true top fraction among all observed models on that benchmark?

The `greedy_probe_set/` child experiment uses the same pairwise ranking metric at margin 5 as a probe-selection objective. It belongs under Section 5.2 because it asks which probes optimize ranking preservation, while reusing Section 5.1's all-known probe-set prediction primitive.

## How to run

This experiment is lightweight post-processing of the Section 4.2 prediction cache. Do not rerun BenchPress predictors unless the source NPZ is missing or stale.

```bash
cd "$(git rev-parse --show-toplevel)"
bash experiments/sec5_findings/ranking_preservation/run.sh
bash experiments/sec5_findings/ranking_preservation/run.sh --predictor logit_model_mean
bash experiments/sec5_findings/ranking_preservation/run.sh --predictor logit_benchmark_mean

# Include the stricter reviewer-facing subset where both compared cells are held out:
bash experiments/sec5_findings/ranking_preservation/run.sh --include-both-hidden
bash experiments/sec5_findings/ranking_preservation/run.sh --predictor logit_model_mean --include-both-hidden
bash experiments/sec5_findings/ranking_preservation/run.sh --predictor logit_benchmark_mean --include-both-hidden
```

## Inputs

- Source prediction cache:
  `experiments/sec4_building_benchpress/method_comparison/predictions/0124__logit__bias_als__hp01_b16f05a66b.npz`
- This is the Logit Bias ALS BenchPress default used in the paper:
  `lambda=0.1`, `rank=2`, 10 seeds x 3 folds, `base_seed=42`.
- Same-cell baseline caches:
  `experiments/sec4_building_benchpress/method_comparison/predictions/0095__logit__model_mean__hp00_bf21a9e8fb.npz`
  and
  `experiments/sec4_building_benchpress/method_comparison/predictions/0094__logit__benchmark_mean__hp00_bf21a9e8fb.npz`.

## Outputs

- `results.json`: raw per-benchmark/per-fold metric rows plus benchmark-median summaries.
- `results_logit_model_mean.json`: same ranking metrics for the logit-space model mean baseline.
- `results_logit_benchmark_mean.json`: same ranking metrics for the logit-space benchmark mean baseline.
- `greedy_probe_set/all_known/results/greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz`: top-10 cost-unaware greedy probe set selected for margin-5 pairwise ranking accuracy.

The file contains:

| Key | Meaning |
|-----|---------|
| `metadata` | source cache, fold setting, margins, top fractions |
| `pairwise_rows` | one row per `(fold, benchmark, margin)` with correct/total pair counts and accuracy |
| `top_rows` | one row per `(fold, benchmark, top_fraction)` with full-observed-leaderboard top-k overlap metrics |
| `pairwise_both_hidden_rows` | when `--include-both-hidden` is used, pairwise rows restricted to pairs where both cells were held out in the same fold and benchmark |
| `top_hidden_rows` | when `--include-both-hidden` is used, top-fraction recovery over the held-out-only leaderboard for the same fold and benchmark |
| `summary` | benchmark-level median summaries for each margin and top fraction; with `--include-both-hidden`, also includes `pairwise_both_hidden_by_margin` and `top_hidden_by_fraction` |

## Resume / rerun

The script is deterministic and cheap. Re-run `run.sh` to overwrite `results.json` atomically.

## Last valid result

Latest regeneration: CHTC job `6307820`, tag `rb_rank_bh`, code commit `1ee4fa68b5954fb3f338480a3391101b11ecf7c9`, May 2026 matrix `84 x 133`, 10 seeds x 3 folds.
The job regenerated the three source prediction shards (`benchpress`, `logit_model_mean`, `logit_benchmark_mean`) and ran `run.py --include-both-hidden` for each predictor.

Pairwise ranking accuracy, median across benchmarks:

| Predictor | Pair subset | Margin 0 | Margin 1 | Margin 2 | Margin 5 | Pair counts at margins 0 / 1 / 2 / 5 |
|-----------|-------------|----------|----------|----------|----------|---------------------------------------|
| \benchpress{} | all scored pairs | 84.1% | 86.5% | 87.8% | 92.2% | 589,830 / 561,283 / 531,498 / 454,090 |
| logit-space model mean | all scored pairs | 78.3% | 80.2% | 82.1% | 86.7% | 589,830 / 561,283 / 531,498 / 454,090 |
| logit-space benchmark mean | all scored pairs | 59.2% | 60.0% | 61.2% | 64.4% | 589,830 / 561,283 / 531,498 / 454,090 |
| \benchpress{} | both held out | 81.3% | 82.8% | 84.4% | 88.1% | 117,990 / 112,117 / 106,202 / 90,810 |
| logit-space model mean | both held out | 76.7% | 78.0% | 79.2% | 83.3% | 117,990 / 112,117 / 106,202 / 90,810 |
| logit-space benchmark mean | both held out | 0.0% | 0.0% | 0.0% | 0.0% | 117,990 / 112,117 / 106,202 / 90,810 |

The both-held-out subset restricts each fold-benchmark leaderboard to held-out cells before scoring pairs, so every scored pair compares two predicted scores. The logit-space benchmark mean produces identical predictions within a benchmark, so all both-held-out same-benchmark pairs are predicted ties and count as incorrect under the pairwise metric.

Top-fraction overlap, median across benchmarks:

| Predictor | Leaderboard universe | Top 10% | Top 20% | Top 30% | Slots at top 10% / 20% / 30% |
|-----------|----------------------|---------|---------|---------|-------------------------------|
| \benchpress{} | all observed cells | 72.4% | 79.3% | 84.3% | 9,515 / 17,154 / 25,091 |
| logit-space model mean | all observed cells | 66.7% | 72.5% | 79.6% | 9,515 / 17,154 / 25,091 |
| logit-space benchmark mean | all observed cells | 66.7% | 66.7% | 67.6% | 9,515 / 17,154 / 25,091 |
| \benchpress{} | held-out cells only | 58.6% | 65.9% | 75.0% | 4,497 / 6,660 / 9,501 |
| logit-space model mean | held-out cells only | 51.7% | 59.8% | 68.1% | 4,497 / 6,660 / 9,501 |
| logit-space benchmark mean | held-out cells only | 20.7% | 29.0% | 43.8% | 4,497 / 6,660 / 9,501 |

Raw rows and full summaries are in `results.json`, `results_logit_model_mean.json`, and `results_logit_benchmark_mean.json`. The current ICLR main table displays the earlier BenchPress all-pairs row as `83.8%`, `86.3%`, `88.0%`, and `92.1%`; the same-pair baseline columns above match the paper-current baseline numbers.
