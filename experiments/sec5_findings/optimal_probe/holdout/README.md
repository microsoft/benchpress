# Held-out scorecard-recovery probe validation

## Paper mapping

- Robustness artifacts for `\Cref{sec:probe_selection}`.
- Current paper-facing Hero Figure panel B uses the full-matrix `all_known/`
  curves; this directory is for held-out sensitivity checks and update notes.

## Purpose

Validate probe-set selection on held-out model rows. The split is deterministic:
70% train models and 30% validation models with seed 42. Greedy probes are
selected on train rows; validation isolates each held-out target model so
BenchPress sees train rows plus that target model's probe scores, but not other
held-out rows.

## How to run

```bash
cd experiments/sec5_findings/optimal_probe/holdout

METRIC=medae OUT=model_split_validation_medae_train70_all.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run_model_split_validation.sh

CANDIDATE_ALLOWLIST=../candidate_allowlists/user_cheap_20260505.json \
  METRIC=medae OUT=model_split_validation_medae_train70_usercheap.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run_model_split_validation.sh

python run_model_split_random.py --k-max 10 --n-seeds 10 --workers 48

# Baselines on the same held-out cells and the same probes: logit-space model
# mean and offsets only (Logit Bias ALS rank 0).
python score_heldout_baselines.py \
  results/model_split_validation_medae_train70_all.json.gz \
  results/model_split_validation_medae_train70_usercheap.json.gz \
  --out results/heldout_baselines_s42.json.gz

# Low-cost pool under the documented <= $30 public-cost rule (drops MMLU-Pro):
CANDIDATE_ALLOWLIST=../candidate_allowlists/public_cost_le_30usd.json \
  METRIC=medae OUT=model_split_validation_medae_train70_cost_le_30usd.json.gz \
  MAX_STEPS=10 WORKERS=24 ./run_model_split_validation.sh
```

CHTC-clean checkout commands with the May matrix:

```bash
cd "$REPO/experiments/sec5_findings/optimal_probe/holdout"

BENCHPRESS_DATA="$DATA/may.json" \
  METRIC=medae OUT=model_split_validation_medae_train70_all.json.gz \
  MAX_STEPS=10 WORKERS="$NCPU" ./run_model_split_validation.sh

BENCHPRESS_DATA="$DATA/may.json" \
  CANDIDATE_ALLOWLIST=../candidate_allowlists/user_cheap_20260505.json \
  METRIC=medae OUT=model_split_validation_medae_train70_usercheap.json.gz \
  MAX_STEPS=10 WORKERS="$NCPU" ./run_model_split_validation.sh

BENCHPRESS_DATA="$DATA/may.json" \
  python run_model_split_random.py --k-max 10 --n-seeds 10 --workers "$NCPU"
```

Smoke test:

```bash
MAX_STEPS=1 CANDIDATE_LIMIT=2 MODEL_LIMIT=8 WORKERS=2 \
  OUT=smoke_model_split_validation.json.gz \
  ./run_model_split_validation.sh
```

## Inputs

- Score matrix and model split from `benchpress.evaluation_harness`.
- Shared candidate allowlists from `../candidate_allowlists/`.
- Predictor: `predict_benchpress_scores`.

## Outputs

Results are under `results/`.

- `results/model_split_validation_medae_train70_all.json.gz`
- `results/model_split_validation_medae_train70_usercheap.json.gz`
- `results/model_split_random_medae_train70.json.gz`
- `results/heldout_baselines_s42.json.gz`: per-cell BenchPress, model-mean and
  offsets-only predictions on the stored `validation_non_probe` cells, with
  per-k pooled MedAE/MedAPE and per-target win fractions.
- `results/model_split_validation_medae_train70_cost_le_30usd.json.gz`

Each validation result contains `trajectory[*].validation_non_probe` as the
primary held-out metric and `trajectory[*].validation_with_probe_zero` for the
compatibility denominator.

## Resume / rerun

The validation script resumes only when metric, candidate universe, split,
train fraction, seed, model limit, and candidate count match. Training candidate
caches are under `results/.candidate_cache/`.

## Last valid result

Split seed 42 (59 selection models, 25 held-out models), github commit `2f1ca06`, CHTC CPU jobs 6307250 (`b2_any`, any-benchmark, 16 CPUs, 10,552 s) and the earlier `b2_low` / `b2_rand` jobs (low-cost greedy, random baseline). Held-out non-probe MedAE:

| k | Any-benchmark greedy | Low-cost greedy | Random (median of 10 seeds) |
|---|---|---|---|
| 1 | 6.59 | 6.59 | 10.37 |
| 5 | 5.32 | 5.23 | 7.22 |
| 10 | 4.26 | 5.54 | 5.91 |

Both greedy searches are below the random median at every k from 1 to 10. Any-benchmark probes on the selection split: GPQA Diamond, Terminal-Bench 2.0, AIME 2024, LiveCodeBench, ARC-AGI-1, MMLU-Pro, Aider Polyglot (diff), HLE Text, BrowseComp, Codeforces Rating. Paper: ICLR App D.1 held-out paragraph and main-body transfer sentence in `sec:probe_selection`.


Same-cell baselines (`score_heldout_baselines.py`, protocol `heldout_baselines_v2`, CHTC CPU job 7925132, github commit `7996392`; `results/heldout_baselines_s42.json.gz`). MedAE at k = 1 / 5 / 10, with the fraction of held-out models on which BenchPress has the lower per-target MedAE:

| Ordering | BenchPress | Offsets only (rank 0) | Logit model mean |
|---|---|---|---|
| Any-benchmark greedy | 6.59 / 5.32 / 4.26 | 5.96 / 5.64 / 4.95 (24 / 48 / 64%) | 8.11 / 8.43 / 8.80 (76 / 84 / 96%) |
| Low-cost greedy | 6.59 / 5.23 / 5.54 | 5.96 / 5.78 / 5.44 (24 / 40 / 52%) | 8.11 / 9.10 / 8.88 (76 / 80 / 84%) |

Offsets only beats BenchPress at k = 1 and 2; on the any-benchmark ordering BenchPress is lower at every k from 3 to 10.

Low-cost pool without MMLU-Pro (`candidate_allowlists/public_cost_le_30usd.json`, 24 benchmarks; CHTC job 7925234, tag `rb_e4h`, commit `7996392`; `results/model_split_validation_medae_train70_cost_le_30usd.json.gz`, `results/heldout_baselines_cost_le_30usd_s42.json.gz`). Same split; selection-split probes: GPQA Diamond, MATH-500, Aider Polyglot (diff), HMMT Feb 2026, Aider Polyglot (whole), AIME 2026, tau2-bench Retail, Arena-Hard Auto, SciCode, AIME 2025. Held-out non-probe MedAE at k = 1 / 5 / 10: 6.59 / 6.30 / 6.01 (with MMLU-Pro 6.59 / 5.23 / 5.54); below the random median at k = 1 to 9, above at k = 10 (6.01 vs 5.91). Offsets only on the same cells: 5.96 / 6.13 / 5.65; logit model mean 8.11 / 8.40 / 9.07.

On the May `84 x 133` matrix, predictor-fit counts are:

- Any-benchmark greedy validation: `76,315` fits (`1,285` candidate prefixes x
  `59` train models, plus `500` validation fits).
- Low-cost greedy validation: `12,595` fits (`205` candidate prefixes x `59`
  train models, plus `500` validation fits).
- Random baseline: `2,500` fits (`10` k values x `10` seeds x `25` validation
  models).

## Predictor comparison across requested probe budgets

`plot_baselines.py` draws two panels from the original
`results/heldout_baselines_s42.json.gz` summary: cost-unaware and cost-aware
greedy probes, each with BenchPress, offsets only, and logit-space model mean
at every requested budget from 1 to 10. It does not fit predictors or change
the appendix evaluation setup. The y-axis is pooled MedAE on unrevealed
observed cells; all 25 held-out models remain in each row. Test cells match
across predictors within a budget but change across budgets.

```bash
python plot_baselines.py
python plot_baselines.py --metric medape
python plot_baselines.py --metric both
python plot_baselines.py --metric both --layout row
```

Outputs: `figures/bp_heldout_probe_baselines.pdf` and `.png`, plus
`figures/bp_heldout_probe_baselines_caption.md` and
`figures/bp_heldout_probe_baselines_data.json`. The data companion records all
60 plotted values, test-cell counts, and the source artifact hash/manifest.
The figure uses the shared visual identity: policy colors for BenchPress,
violet dashed lines for offsets only, and gray dotted lines for model mean.
Both panels use the same axes. It is a standalone figure for the rebuttal
discussion, with no manuscript placement applied.

The `--metric medape` variant writes the same four files with stem
`bp_heldout_probe_baselines_medape`. It keeps the MedAE-selected probes and
changes only the plotted metric. MedAPE excludes zero true scores. All 60
plotted values are checked against the saved baseline summaries.

The `--metric both` variant writes one four-panel figure with stem
`bp_heldout_probe_baselines_both`: MedAE on the top row, MedAPE on the bottom,
cost-unaware probes on the left, and cost-aware probes on the right. All 120
plotted values match the original summaries. Its PNG is embedded in the local
ICLR rebuttal W1-2 / Q1 draft, with a short caption and the original 59/25
evaluation scope. OpenReview comments do not render images, so submission
requires an anonymous figure link or revised-paper appendix reference.

Adding `--layout row` writes `bp_heldout_probe_baselines_both_row` with all four
panels in one row. It identifies the 59/25 split in its title and uses the same
metric axis ranges as `all_known/figures/bp_main_probe_offsets_both_row`.
This single-row PNG is the current embedded version in the local rebuttal.
All 120 values are unchanged from the original held-out baseline summaries.

## Matched revealed-count comparison

`summarize_revealed_counts.py` re-analyzes `results/heldout_baselines_s42.json.gz`
without fitting any predictors; it is safe to run locally. It follows the W1
comparison at exactly 1, 2, and 3 revealed scores. Within each ordering, it
retains models with predictions at all three counts, chooses the earliest
requested-probe prefix reaching each count, and evaluates all three predictors
on the intersection of the hidden test cells. Each model-benchmark pair is
counted once per revealed count. Selection uses availability only.

```bash
python summarize_revealed_counts.py
```

Output: `results/heldout_revealed_counts_s42.json`, including the source SHA-256,
source manifest, retained model IDs, selected prefix at each count, common test
benchmark IDs, pooled metrics, and per-model MedAE. The split and source
predictions remain those of `heldout_baselines_v2` (CHTC job 7925132, commit
`7996392`). The re-analysis runs on cached predictions only.

The fixed cohorts contain 22 models / 608 cells for cost-unaware probes and
19 models / 564 cells for cost-aware probes. MedAE in original score units,
ordered BenchPress / offsets only / model mean:

| Revealed scores | Cost-unaware probes | Cost-aware probes |
|---|---|---|
| 1 | 6.01 / 5.67 / 8.24 | 6.61 / 6.10 / 7.88 |
| 2 | 4.92 / 5.59 / 8.47 | 6.43 / 6.53 / 8.21 |
| 3 | 5.11 / 5.51 / 8.65 | 6.18 / 6.06 / 8.56 |

Validation: the same earliest-prefix/common-cell procedure reproduces all six
BenchPress/model-mean pairs in the W1 table (11 models / 162 cells and 8 models /
134 cells). The canonical metric also reproduces all 60 pooled MedAE values in
the original held-out baseline artifact. These matched results support a gain
over model mean in this subset, with mixed gains over offsets only.

For exactly 1 through 5 revealed scores, use:

```bash
python summarize_revealed_counts.py --max-revealed 5
```

This writes `results/heldout_revealed_counts_s42_through5.json`. The default
three-count artifact is unchanged. Requiring all five counts retains 12 models /
398 fixed test cells under cost-unaware selection and 7 models / 273 fixed test
cells under cost-aware selection. All five rows use these same cohorts and
cells; rows 1 to 3 therefore differ from the wider-cohort table above.

| Revealed scores | Cost-unaware probes | Cost-aware probes |
|---|---|---|
| 1 | 5.89 / 5.74 / 6.59 | 7.64 / 6.62 / 9.14 |
| 2 | 4.74 / 4.75 / 5.92 | 7.17 / 7.10 / 9.36 |
| 3 | 4.68 / 4.82 / 6.56 | 6.50 / 6.75 / 9.64 |
| 4 | 4.72 / 5.00 / 6.14 | 6.09 / 6.84 / 9.58 |
| 5 | 4.57 / 4.92 / 6.02 | 6.16 / 6.51 / 9.92 |

The order remains BenchPress / offsets only / model mean. The five-count table
is a separate matched-subset diagnostic. Its small, availability-selected
cohorts limit generalization; the reported advantage is in pooled MedAE and
does not imply a majority of per-model wins. The parameterized summarizer
reproduces the original three-count JSON byte for byte.
