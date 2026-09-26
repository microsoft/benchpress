# Variant columns ablation

## Paper mapping

Appendix B.1 Data Collection: reviewer-facing audit of retained benchmark-family variants and the sensitivity test for duplicate or variant benchmark columns.

## Purpose

This experiment answers whether retained benchmark-family variants make the canonical held-out prediction task artificially easy. It uses the May 2026 84 x 133 matrix, audits each retained variant group, then compares the fixed BenchPress predictor on the same held-out cells with and without the non-kept variant columns.

## How to run

Local is allowed only for `--help` and `--audit-only`. The full 30-fold predictor run must run on CHTC.

```bash
BENCHPRESS_DATA=/path/to/may.json PYTHONPATH=$PWD \
python experiments/sec3_low_rank/data_collection/variant_columns_ablation/run.py --audit-only

BENCHPRESS_DATA=/path/to/may.json PYTHONPATH=$PWD \
python experiments/sec3_low_rank/data_collection/variant_columns_ablation/run.py
```

The CLI also accepts `--drop-benchmarks id1,id2,...` to test an explicit deduplication list without editing score data.

## Parallel execution

This is a single CPU job: 30 folds x two conditions with Logit Bias ALS rank 2, lambda 0.1. It should run as one CHTC job rather than many shards so both conditions share the exact same fold loop and output file.

## Inputs

- May 2026 matrix JSON via `BENCHPRESS_DATA`.
- `benchpress.evaluation_harness.load_folds()` for the canonical 10 seed x 3 per-model folds.
- `benchpress.methods.completers.complete_bias_als` through `make_score_predictor`.

## Outputs

- `variant_groups.json`: observed counts, overlap, pairwise raw and direction-aligned correlations, and rule judgment for each retained variant group.
- `results.json`: manifest, drop list, full-vs-deduplicated per-fold and pooled MedAE/MedAPE, and raw per-cell predictions for the shared evaluation cells.

## Resume / rerun

The script overwrites `variant_groups.json` and `results.json` atomically through `benchpress.io_utils.write_json`. Rerun the single CHTC command after code or group-definition changes. No score data file is edited.

## Last valid result

- CHTC job: `6307697` (`dd_varnoise2`)
- code commit: `b34cd43b913a988be591c6e3d2031c2d848ff12e`
- matrix: May 2026, 84 models x 133 benchmarks, 2,604 observed cells
- dedup rule: 23 variant groups, keep the member with the most observed cells, drop 38 columns, leaving 95 columns
- evaluation cells: 18,820 held-out predictions from the canonical 10 seed x 3 per-model folds after filtering to kept columns
- full matrix on kept-column cells: per-fold median MedAE 4.693, MedAPE 7.924%; pooled MedAE 4.704, MedAPE 7.816%
- deduplicated matrix: per-fold median MedAE 4.982, MedAPE 8.119%; pooled MedAE 4.947, MedAPE 8.148%
- artifacts: `results.json`, `variant_groups.json`; remote tarball `CHTC:~/bp_iclr/out_dd_varnoise2.tar.gz`
