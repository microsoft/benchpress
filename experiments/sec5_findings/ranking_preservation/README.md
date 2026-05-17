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
```

## Inputs

- Source prediction cache:
  `experiments/sec4_building_benchpress/method_comparison/predictions/0124__logit__bias_als__hp01_b16f05a66b.npz`
- This is the Logit Bias ALS BenchPress default used in the paper:
  `lambda=0.1`, `rank=2`, 10 seeds x 3 folds, `base_seed=42`.

## Outputs

- `results.json`: raw per-benchmark/per-fold metric rows plus benchmark-median summaries.
- `greedy_probe_set/results/greedy_pairwise_margin5_top10_targets_all_candidates_all.json.gz`: top-10 cost-unaware greedy probe set selected for margin-5 pairwise ranking accuracy.

The file contains:

| Key | Meaning |
|-----|---------|
| `metadata` | source cache, fold setting, margins, top fractions |
| `pairwise_rows` | one row per `(fold, benchmark, margin)` with correct/total pair counts and accuracy |
| `top_rows` | one row per `(fold, benchmark, top_fraction)` with full-observed-leaderboard top-k overlap metrics |
| `summary` | benchmark-level median summaries for each margin and top fraction |

## Resume / rerun

The script is deterministic and cheap. Re-run `run.sh` to overwrite `results.json` atomically.

## Last valid result

Regenerated as lightweight post-processing using the cached Logit Bias ALS prediction shard.

Key aggregate results:

| Metric | Setting | Accuracy / recovery | Paper location |
|--------|---------|---------------------|----------------|
| Pairwise ranking accuracy | margin 0 | 83.8% | Main |
| Pairwise ranking accuracy | margin 1 | 86.3% | Main |
| Pairwise ranking accuracy | margin 2 | 88.0% | Main |
| Pairwise ranking accuracy | margin 5 | 92.1% | Main |
| Top-fraction overlap | top 10% | 72.4% | Appendix |
| Top-fraction overlap | top 20% | 79.3% | Appendix |
| Top-fraction overlap | top 30% | 83.9% | Appendix |

Raw rows and full summaries are in `results.json`.
