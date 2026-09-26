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

Regenerated the same-cell baselines as lightweight post-processing using the cached logit-space model mean and logit-space benchmark mean prediction shards.
Latest baseline regeneration: CHTC job `6307703`, tag `rk_rank_bases`, commit `c54ce04d07cd82c7b4bfc098ec33a9e5ac52da1e`, May 2026 matrix `84 x 133`, 10 seeds x 3 folds.
The BenchPress column below reports the paper-current numbers from `results.json` / `tab:ranking_preservation`.

Key aggregate results:

| Metric | Setting | logit-space benchmark mean | logit-space model mean | BenchPress | Paper location |
|--------|---------|----------------------------|------------------------|------------|----------------|
| Pairwise ranking accuracy | margin 0 | 59.2% | 78.3% | 83.8% | Main |
| Pairwise ranking accuracy | margin 1 | 60.0% | 80.2% | 86.3% | Main |
| Pairwise ranking accuracy | margin 2 | 61.2% | 82.1% | 88.0% | Main |
| Pairwise ranking accuracy | margin 5 | 64.4% | 86.7% | 92.1% | Main |
| Top-fraction overlap | top 10% | 66.7% | 66.7% | 72.4% | Appendix |
| Top-fraction overlap | top 20% | 66.7% | 72.5% | 79.3% | Appendix |
| Top-fraction overlap | top 30% | 67.6% | 79.6% | 83.9% | Appendix |

Raw rows and full summaries are in `results.json`, `results_logit_model_mean.json`, and `results_logit_benchmark_mean.json`.
