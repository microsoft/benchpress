# Benchmark Correlation

## Paper mapping

- Appendix C.1, `\Cref{app:probe_selection}`
- Table: `tab:pairwise_ols`
- Paragraph: "Widespread redundancy across benchmarks."

## Purpose

This appendix-only analysis quantifies benchmark redundancy before the probe-set selection results. For every ordered benchmark pair with at least five shared model scores, `run_pairwise_ols.py` fits a univariate regression in logit-plus-z-score space and reports the best neighbor for each target benchmark.

## How to run

```bash
cd experiments/appendix_c_sec5_findings/benchmark_correlation
python run_pairwise_ols.py
```

`plot_mds.py` is an exploratory visualization of the same benchmark-correlation structure and selected greedy probe benchmarks.

## Inputs

- Current score matrix from `benchpress.all_methods`
- Shared normalization and metric helpers from `benchpress.evaluation_harness`
- Greedy probe outputs under `experiments/sec5_findings/optimal_probe/results/` for the optional MDS overlays

## Outputs

- `pairwise_ols_stats.json`: best-neighbor table source for Appendix C.1
- `figures/`: exploratory MDS figures

## Resume / rerun

The pairwise OLS script is deterministic and overwrites `pairwise_ols_stats.json`. Re-run it after matrix updates or changes to the minimum shared-model threshold.

## Last valid result

Current Appendix C.1 text reports 132 benchmarks with at least one valid neighbor pair, 127 with best-neighbor `|r| >= 0.85`, and median best-neighbor absolute correlation 0.97.
