# All-known scorecard-recovery probe sets

## Paper mapping

- Main text: `\Cref{sec:probe_selection}` current-matrix construction.
- Figures: `figures/bp_probe_evaluation_*.pdf`.
- Hero Figure panel B uses the MedAE all-known curves as historical comparison inputs.

## Purpose

Select compact probe sets on the current observed matrix. For each target model,
selected probe columns stay visible, probe cells are exact (`pred=true`), and
all other observed target-model cells are predicted by BenchPress. The evaluated
cell universe is fixed across probe budgets.

## How to run

```bash
cd experiments/sec5_findings/optimal_probe/all_known

OUT=greedy_medape_targets_tall_candidates_tall.json.gz MAX_STEPS=10 WORKERS=48 ./run.sh

CANDIDATE_ALLOWLIST=../candidate_allowlists/user_cheap_20260505.json \
  OUT=greedy_medape_targets_tall_candidates_usercheap.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run.sh

METRIC=medae OUT=greedy_medae_targets_tall_candidates_tall.json.gz MAX_STEPS=10 WORKERS=48 ./run.sh

CANDIDATE_ALLOWLIST=../candidate_allowlists/user_cheap_20260505.json \
  METRIC=medae OUT=greedy_medae_targets_tall_candidates_usercheap.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run.sh

python run_random.py --k-max 30 --n-seeds 10 --workers 24

python evaluate_order.py \
  --fixed-order-json probe_orderings.json \
  --fixed-order medae_any \
  --k-max 10 \
  --workers 10 \
  --out fixed_order_medae_any_hidden_only.json.gz

python evaluate_order.py \
  --fixed-order-json probe_orderings.json \
  --fixed-order medae_low_cost \
  --k-max 10 \
  --workers 10 \
  --out fixed_order_medae_low_cost_hidden_only.json.gz

# Same command with --fixed-order medape_any / medape_low_cost and
# --out fixed_order_medape_{any,low_cost}_hidden_only.json.gz for the MedAPE orderings.

# Probe baselines on the same hidden cells:
# --fixed-order coverage_any / coverage_low_cost (most-reported benchmarks first);
# --fixed-order medae_any --rank 0 (offsets-only Bias ALS: model level from the probes,
# no interaction term); default output fixed_order_<key>[_rank<r>]_hidden_only.json.gz.
# Use --predictor logit_model_mean to score the same fixed-order hidden cells
# with the logit-space model mean baseline.

python run_coverage_matched_random.py \
  --k-values 1 3 5 10 \
  --n-subsets 10 \
  --tolerance-fraction 0.10 \
  --coverage-proposal-power 2.0 \
  --workers 8 \
  --out coverage_matched_random_hidden_only.json.gz
```

Plot from existing results:

```bash
python plot.py --compare \
  --random-in random_medape_hero_all_known.json.gz \
  --cheap-in greedy_medape_targets_tall_candidates_usercheap.json.gz \
  --out bp_probe_evaluation_cost_unaware

python plot.py --compare \
  --all-in greedy_medae_targets_tall_candidates_tall.json.gz \
  --cheap-in greedy_medae_targets_tall_candidates_usercheap.json.gz \
  --metric medae \
  --random-in random_medape_hero_all_known.json.gz \
  --out bp_probe_evaluation_medae_cost_unaware
```

## Inputs

- Score matrix and observed mask from `benchpress.evaluation_harness`.
- Shared candidate allowlists from `../candidate_allowlists/`.
- Predictor: `predict_benchpress_scores`.

## Outputs

Results are under `results/`; figures are under `figures/`.

- `results/greedy_medape_targets_tall_candidates_tall.json.gz`
- `results/greedy_medape_targets_tall_candidates_usercheap.json.gz`
- `results/greedy_medae_targets_tall_candidates_tall.json.gz`
- `results/greedy_medae_targets_tall_candidates_usercheap.json.gz`
- `results/random_medape_hero_all_known.json.gz`
- `results/fixed_order_medae_any_hidden_only.json.gz`
- `results/fixed_order_medae_low_cost_hidden_only.json.gz`
- `results/fixed_order_medape_any_hidden_only.json.gz`
- `results/fixed_order_medape_low_cost_hidden_only.json.gz`
- `results/coverage_matched_random_hidden_only.json.gz`

Raw per-cell predictions are saved in every greedy candidate result and random
baseline shard output. Fixed-order evaluation stores raw per `(target, k, cell)`
rows plus both with-probe-zero and hidden-only summaries. Random baseline output
also includes `summary_non_probe_by_k_seed` and `summary_non_probe_by_k`.

## Resume / rerun

`run.py` resumes only when metric, candidate universe, candidate count, target
cell count, and protocol match. Candidate caches are under `results/.candidate_cache/`.
`run_random.py` resumes by `(k, seed)` shard.

## Last valid result

Full-matrix MedAE construction:

- Any-benchmark k=5 MedAE: 3.93; k=10 MedAE: 3.07.
- Low-cost k=5 MedAE: 4.55; k=10 MedAE: 3.80.

Hidden-only evaluation (unrevealed cells only; CHTC jobs 6307236/6307237/6307249/6307410/6307411, commits `2f1ca06`/`d3f65e2`, May matrix identity `9603f902…`):

- MedAE orderings, k=5 / k=10 MedAE: any 4.745 / 4.345; low-cost 5.318 / 4.919.
- MedAPE orderings, k=5 / k=10 MedAPE: any 7.873% / 7.329%; low-cost 8.851% / 8.375%.
- Random prefixes, median over 10 seeds, k=5 / k=10 MedAE: 7.503 / 6.120.
- Probe-choice and rank baselines (CHTC jobs 6307629-6307632, commit `9ac9963`), hidden-only MedAE at k=1 / 3 / 5 / 10:
  most-reported-first any 6.27 / 6.18 / 5.73 / 5.17 and low-cost 6.27 / 5.94 / 5.50 / 4.99 (BenchPress rank 2);
  offsets only (`--rank 0`) on the MedAE orderings, any 5.92 / 5.53 / 5.41 / 5.18 and low-cost 5.92 / 5.82 / 5.81 / 5.73.
- The with-probe-zero MedAE of the any ordering is 3.955 at k=5, above the arXiv 3.93 because commit `f4319af` made default predictions metric-aware for non-percentage metrics; the low-cost value reproduces 4.55.

ICLR figure: `plot.py --compare --hidden-only --metric medape --all-in fixed_order_medape_any_hidden_only.json.gz --cheap-in fixed_order_medape_low_cost_hidden_only.json.gz --random-in random_medape_hero_all_known.json.gz --out bp_probe_evaluation_cost_aware` writes `overleaf/iclr2027/figures/bp_probe_evaluation_cost_aware.pdf`.

`probe_orderings.json` stores the four top-10 prefixes from
`tab:probe_sets` and two most-reported-first baselines (`coverage_any`,
`coverage_low_cost`; construction and counts recorded per ordering). Its unrestricted k=5 prefix matches the brute-force diagnostic
set `gpqa_diamond, hle, codeforces_rating, mmlu_pro, arc_agi_1` up to ordering.
