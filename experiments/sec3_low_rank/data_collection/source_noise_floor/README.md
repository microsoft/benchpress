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

Pending aggregation.
