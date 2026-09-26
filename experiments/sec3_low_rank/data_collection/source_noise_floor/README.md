# Source noise floor

## Paper mapping

Appendix B.1 Data Collection: quantifies inter-source variation from stored alternative score candidates in the May 2026 matrix JSON.

## Purpose

This aggregation estimates a public-record noise floor: when a primary cell has alternative candidate scores from other sources or settings, how far apart are those values? The reported quantities are median absolute differences between the chosen score and alternatives, and between alternatives within the same cell.

## How to run

This script only reads the JSON data file and performs local aggregation, so it may run locally.

```bash
BENCHPRESS_DATA=/path/to/may.json PYTHONPATH=$PWD \
python experiments/sec3_low_rank/data_collection/source_noise_floor/run.py
```

## Inputs

- May 2026 matrix JSON via `BENCHPRESS_DATA` or `--data-json`.
- Stored `scores[].candidates[]` values.

## Outputs

- `results.json`: overall summaries, metric-type and category breakdowns, and final-matrix cell records with at least two distinct numeric values across the primary score and alternatives.

## Resume / rerun

The script overwrites `results.json`. No score data file is edited.

## Last valid result

- CHTC job: `6307697` (`dd_varnoise2`); local aggregation gives the same values because the script only reads `may.json`
- code commit: `b34cd43b913a988be591c6e3d2031c2d848ff12e`
- matrix: May 2026, 84 models x 133 benchmarks, 2,604 observed cells
- final-matrix cells with at least two distinct numeric values across the chosen score and stored alternatives: 373 cells, 610 alternatives
- chosen vs alternative: median per-cell absolute difference 3.4 score points; pooled median 3.0 over 466 chosen-alternative pairs
- alternative vs alternative: median per-cell absolute difference 3.2 score points; pooled median 3.0 over 213 alternative-alternative pairs
- percentage-metric subset: chosen vs alternative median per-cell absolute difference 3.2 points over 361 cells
- artifact: `results.json`; remote tarball `CHTC:~/bp_iclr/out_dd_varnoise2.tar.gz`
