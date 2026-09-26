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
- Probe orderings: `../optimal_probe/all_known/probe_orderings.json`.

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

CHTC CPU job 6307243, github commit `2f1ca06`, 2 CPUs, `python:3.11` container. May matrix 84 x 133 / 2,604 observed (same content as `evaluation_harness.M_FULL`; the manifest hash differs only because `load_score_matrix` orders models differently); August matrix 129 x 253 / 4,905 observed. The 2,604 shared cells are 99.65% identical.

- Part (i), 178 August cells of May models on May benchmarks that were missing in May: BenchPress MedAE 4.44 (MedAPE 6.39%); logit model mean 4.86; logit benchmark mean 9.55.
- Part (ii), 45 August-only models, pooled MedAE on unrevealed cells (targets / cells) with May benchmark medians on the same cells: `medae_any` k=5 6.88 (40 / 541) vs 10.15; k=10 6.38 (41 / 502) vs 9.30. `medae_low_cost` k=5 8.08 (37 / 517) vs 10.00; k=10 7.00 (37 / 485) vs 9.61. Random orderings pooled over 10 seeds: k=5 8.50 (2,935 cells) vs 10.80; k=10 7.29 (4,033 cells) vs 10.20.
- Paper: ICLR `tab:prospective_new_models` in `app:temporal_deployment`.
