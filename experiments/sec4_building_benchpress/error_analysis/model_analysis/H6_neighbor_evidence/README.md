# H6 Strong-Peer Support

## Paper mapping
- Section: `sec:reliability_analysis`, model-side Table `tab:model_hypotheses`.
- Parent runner: `../run.sh`.

## Purpose
H6 tests whether weakening the target model's strongest score-matrix peer hurts prediction. It mirrors benchmark-side H7: for each target model, the script finds the strongest peer model with `|r| >= NEIGHBOR_THRESH` (`0.95`), restricts to benchmarks observed by both target and peer, and masks nested prefixes of that peer-overlap evidence.

## Setting
For each eligible `(target_model, seed)` unit:
1. Use the standard per-model hide-half split from `../_shared.py`.
2. Identify the strongest peer model by absolute Pearson correlation on shared observed scores; keep the unit only if the strongest peer has `|r| >= 0.95`.
3. Restrict the intervention to benchmarks observed by both the target and the strongest peer.
4. Compare the baseline with all strongest-peer overlap available against treatments that remove nested prefixes of 25%, 50%, and 75% of those overlapping peer cells.
5. Predict the target model's held-out cells and store the paired raw predictions for every drop rate.

## Inputs
- Canonical score matrix from `benchpress.all_methods.M_FULL`.
- Per-model hide-half baseline from `../_shared.py` with `N_SEEDS = 10` and `SEED = 42`.
- Nested peer-overlap drop rates `[0.25, 0.50, 0.75]`; the table uses the 75% condition.

## Outputs
- `results.json`:
  - `records`: one record per `(model, seed, drop_rate)` with base/treatment metrics, masked benchmark indices, and paired raw predictions.
  - `by_drop_rate`: paired model-level Wilcoxon summaries for every drop rate.
  - `tests`: paired model-level Wilcoxon summaries for MedAPE and MedAE.
- `shards/model_<model_id>_seed_<seed>.json`: resumable per-unit cache.

## Current result
`results.json` is the full rerun used by Section 4.3:

| Drop rate | MedAPE median delta | MedAPE p | MedAE median delta | MedAE p |
|-----------|---------------------|----------|--------------------|---------|
| 25% | `+0.008845` | `0.012` | `+0.000170` | `0.484` |
| 50% | `+0.017273` | `0.025` | `+0.002877` | `0.260` |
| 75% | `+0.010201` | `0.033` | `+0.002877` | `0.309` |

The paper reports the 75% condition as `+0.01` MedAPE with `p=0.033` and `+0.00` MedAE with `p=0.309`. The full run contains 83 eligible models, 10 seeds per model, 830 unit shards, and 2,490 `(model, seed, drop_rate)` records.

## Run
```bash
cd ~/Documents/submission/benchpress/github/experiments/sec4_building_benchpress/error_analysis/model_analysis
python H6_neighbor_evidence/ablation.py --workers 8
```

Smoke test without overwriting the real result:
```bash
python H6_neighbor_evidence/ablation.py \
  --max-models 1 --max-seeds 1 --force \
  --shards-dir /tmp/benchpress_h6_smoke_shards \
  --output-path /tmp/benchpress_h6_smoke.json
```

The script is deterministic and resumable at the `(model, seed)` level. Reuse existing `results.json` unless intentionally rerunning the ablation; use `--force` only when starting a clean rerun.

## Rerun provenance
- Code commit for the setting: `7f229b7` (`sec4: make model H6 nested peer support`).
- Result commit: `7325b47` (`sec4: rerun model H6 peer-support ablation`).
- Full rerun environment: remote CPU worker, conda env `benchpress`, clean per-run worktree.
- Full rerun command: `python H6_neighbor_evidence/ablation.py --workers 16 --force`.
- Notes: outputs were copied back to this directory after the remote run finished.
