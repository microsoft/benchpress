# Prospective May-to-August snapshot test

## Paper mapping

Section 5.3 / Appendix D.3 candidate diagnostic for prospective updates.

## Purpose

Test whether the May 2026 BenchPress matrix predicts later observations in the
August 26, 2026 snapshot. The main metrics are MedAE and MedAPE from
`benchpress.evaluation_harness.compute_prediction_error`.

## How to run

```bash
cd experiments/sec5_findings/prospective_update
python run.py \
  --train-json "$DATA/may.json" \
  --eval-json "$DATA/aug.json" \
  --out-dir results/may_to_aug \
  --n-random-seeds 10 \
  --base-seed 42
```

## Parallel execution

The script is single-process and idempotently rewrites the output directory. It
performs one BenchPress fit for Part (i), then one fit per included
`(target model, k, ordering/seed)` unit for Part (ii). Run it on a CHTC CPU node;
do not run it on the local Mac.

## Inputs

- May JSON: expected to build an `84 x 133` matrix with `2,604` observed cells.
- August JSON: expected to build a `129 x 253` matrix with `4,905` observed cells.
- Probe orderings: `../optimal_probe/all_known/probe_orderings.json` (`medae_any`, `medae_low_cost`, and the most-reported-first baselines `coverage_any`, `coverage_low_cost`). Each fixed-order unit also records `logit_model_mean_fixed_order`, the logit-space model mean fit on the same appended matrix and scored on the same hidden cells.

## Outputs

Under `--out-dir`:

- `config.json`: normalized run configuration.
- `manifest.json`: git commit, matrix identities, shapes, and observed counts.
- `raw_predictions.json.gz`: raw per-cell rows for both parts and baselines.
- `summary.json`: pooled, per-seed, per-target, and category summaries.
- `table_summary.json`: compact table-ready summary.

## Resume / rerun

No shards are used. Rerunning the same command overwrites the same files with
the same deterministic seeds and matrix identities.

## Last valid result

CHTC CPU job 6307633 (`b3_mm`), github commit `9ac9963`, 2 CPUs, `python:3.11` container. It reproduces every BenchPress number of the earlier job 6307243 (commit `2f1ca06`) and adds the logit-space model mean and the most-reported-first orderings. May matrix 84 x 133 / 2,604 observed (same content as `evaluation_harness.M_FULL`; the manifest hash differs only because `load_score_matrix` orders models differently); August matrix 129 x 253 / 4,905 observed. The 2,604 shared cells are 99.65% identical.

- Part (i), 178 August cells of May models on May benchmarks that were missing in May: BenchPress MedAE 4.44 (MedAPE 6.39%); logit model mean 4.86; logit benchmark mean 9.55. Signed error (pred minus actual): BenchPress median +2.48, over-predicts 65.2%; logit model mean +1.26, 58.4%; logit benchmark mean -4.76, 28.7%.
- Part (ii), 45 August-only models, pooled MedAE on unrevealed cells, BenchPress / logit model mean / May benchmark median on the same cells (targets / cells):

| Ordering | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| `medae_any` | 8.41 / 6.58 / 10.20 (29 / 483) | 6.81 / 6.19 / 10.30 (33 / 497) | 6.88 / 6.95 / 10.15 (40 / 541) | 6.38 / 6.13 / 9.30 (41 / 502) |
| `medae_low_cost` | 8.41 / 6.58 / 10.20 (29 / 483) | 8.18 / 7.12 / 10.10 (36 / 529) | 8.08 / 7.70 / 10.00 (37 / 517) | 7.00 / 7.64 / 9.61 (37 / 485) |
| `coverage_any` | 8.41 / 6.58 / 10.20 (29 / 483) | 8.58 / 6.90 / 9.95 (30 / 477) | 8.22 / 6.89 / 10.25 (39 / 520) | 6.69 / 5.72 / 9.61 (42 / 499) |
| `coverage_low_cost` | 8.41 / 6.58 / 10.20 (29 / 483) | 8.20 / 7.52 / 9.87 (37 / 534) | 8.08 / 7.40 / 9.80 (37 / 533) | 7.71 / 7.08 / 9.61 (37 / 498) |

  Random orderings, BenchPress pooled over 10 seeds: k=1 7.87 (1,227 cells), k=3 8.70 (2,522), k=5 8.50 (2,935), k=10 7.29 (4,033); benchmark medians 10.50 / 10.80 / 10.80 / 10.20.
  Per target (median per-target MedAE), BenchPress is lower than the model mean on 42% to 59% of targets in each greedy row, and the target-bootstrap 95% interval (2,000 resamples) of the model-mean minus BenchPress difference contains zero in every greedy row.
- Paper: ICLR `tab:prospective_new_models` and prospective paragraph in `app:temporal_deployment`; main-body prospective sentence in `sec:temporal_deployment`.
