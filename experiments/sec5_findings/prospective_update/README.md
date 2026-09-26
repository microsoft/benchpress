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

Not yet run. This directory was added for the ICLR 2027 final experiment batch.
