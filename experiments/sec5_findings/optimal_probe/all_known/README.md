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

Raw per-cell predictions are saved in every greedy candidate result and random
baseline shard output.

## Resume / rerun

`run.py` resumes only when metric, candidate universe, candidate count, target
cell count, and protocol match. Candidate caches are under `results/.candidate_cache/`.
`run_random.py` resumes by `(k, seed)` shard.

## Last valid result

Full-matrix MedAE construction:

- Any-benchmark k=5 MedAE: 3.93; k=10 MedAE: 3.07.
- Low-cost k=5 MedAE: 4.55; k=10 MedAE: 3.80.
