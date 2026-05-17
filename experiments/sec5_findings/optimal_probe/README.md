# Optimal probe selection

## Paper mapping

- Section: `\Cref{sec:probe_selection}` in the main text.
- Main figure: `bp_probe_evaluation_cost_unaware.pdf` and `bp_probe_evaluation_medae_cost_unaware.pdf` as side-by-side MedAPE / MedAE panels.

## Purpose

This experiment asks which benchmark probes a practitioner should run on a new model so that BenchPress can recover the model's full known score profile. The active setting evaluates the fixed universe of all observed model--benchmark cells at every probe-set size. A revealed probe cell is already known and contributes zero error (`pred=true`); unrevealed known cells are predicted by BenchPress.

Two greedy probe selectors use the same all-known-cell evaluator and objective. The cost-unaware selector can choose any benchmark as a probe. The constrained selector uses an explicit user-provided cheap benchmark set as its complete candidate universe. At each step, both selectors add the candidate that gives the lowest pooled MedAPE over all observed cells.

Main-text results use pooled MedAPE as the greedy objective (`METRIC=medape`). Appendix sensitivity runs may switch only the objective to pooled MedAE (`METRIC=medae`); the target cells, predictor, masking protocol, raw-prediction logging, and parallel execution rules stay identical. The random baseline follows the same all-known-cell protocol: for each seed, it draws one global random benchmark ordering, applies the first `k` benchmarks in that ordering to every model, stores observed probe cells with zero error, and predicts unrevealed known cells.

If a held-out BenchPress reference is shown alongside probe-set curves, label it as a reference line rather than as another probe-set method. It does not share the all-known-cell zero-error contribution from revealed probes, so it should anchor the best-method prediction scale without replacing the fixed-universe comparison between random and greedy probe sets.

## Inputs

- Score matrix and observed mask from `benchpress.evaluation_harness`.
- Explicit candidate allowlists under `candidate_allowlists/` for user-provided candidate-set runs.
- Predictor: `predict_benchpress_scores` from `benchpress.all_methods` (Logit Bias ALS default).

## Run commands

This is a slow greedy sweep. Recommended on a remote CPU machine with many cores.
For CPU-only independent configurations, submit one job per configuration and run them in parallel whenever resources allow. Do not serialize independent objectives in one job: the pooled-MedAPE and pooled-MedAE greedy runs should be separate processes. Within each greedy step, all remaining candidate benchmarks are independent; set `WORKERS` close to the number of candidates, capped by the CPU available on one machine. The only serial dependency is between greedy steps, because step `k+1` depends on the probe selected at step `k`.

```bash
cd experiments/sec5_findings/optimal_probe

# Cost-unaware upper bound: all benchmarks are eligible probes.
OUT=greedy_medape_targets_tall_candidates_tall.json.gz MAX_STEPS=10 WORKERS=48 ./run.sh

# User-provided cheap probe set: the allowlist is the full candidate universe.
CANDIDATE_ALLOWLIST=candidate_allowlists/user_cheap_20260505.json \
  OUT=greedy_medape_targets_tall_candidates_usercheap.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run.sh

# Appendix sensitivity: same two policies, but optimize pooled MedAE.
METRIC=medae OUT=greedy_medae_targets_tall_candidates_tall.json.gz MAX_STEPS=10 WORKERS=48 ./run.sh
CANDIDATE_ALLOWLIST=candidate_allowlists/user_cheap_20260505.json \
  METRIC=medae OUT=greedy_medae_targets_tall_candidates_usercheap.json.gz \
  MAX_STEPS=10 WORKERS=48 ./run.sh

# Random probe-prefix baseline, reusable by Figure 1 and Figure 8.
python run_random.py --k-max 30 --n-seeds 10 --workers 24
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

Smoke test (small problem, finishes in seconds):

```bash
CANDIDATE_ALLOWLIST=candidate_allowlists/user_cheap_20260505.json \
  MAX_STEPS=1 CANDIDATE_LIMIT=2 WORKERS=2 OUT=smoke_usercheap.json.gz ./run.sh
```

## Outputs

Results are written under `results/`:

- `greedy_medape_targets_tall_candidates_tall.json.gz`
- `greedy_medae_targets_tall_candidates_tall.json.gz` (appendix sensitivity)
- `greedy_medape_targets_tall_candidates_usercheap.json.gz` (user-provided cheap candidate allowlist)
- `greedy_medae_targets_tall_candidates_usercheap.json.gz` (user-provided cheap candidate allowlist; MedAE objective)
- `random_medape_hero_all_known.json.gz` (random probe-prefix raw predictions)

The greedy result files contain:

- `config`: matrix size, candidate source, target-cell count, seed, workers.
- `trajectory`: one entry per greedy step.
- `trajectory[*].candidate_results`: raw candidate evaluations for that step.
- `trajectory[*].candidate_results[*].predictions`: raw per-cell prediction lists (`i`, `j`, `true`, `pred`).
- `trajectory[*].medape` and `trajectory[*].medae`: both are saved regardless of which objective was optimized.
- When `CANDIDATE_ALLOWLIST` is set, `config.candidate_allowlist_path`
  and `config.candidate_allowlist_ids` record the exact candidate universe.

The random probe-prefix result file contains:

- `config`: protocol, matrix size, benchmark IDs, `k_max`, `n_seeds`, and `base_seed`.
- `summary_by_k_seed`: MedAPE / MedAE summaries for each `(k, seed)` shard.
- `raw_predictions`: all observed cells with `seed`, `k`, model index, benchmark index, actual score, and prediction. Revealed cells have `pred=actual`; the revealed cells are exactly the observed cells that fall in the first `k` columns of that seed's global random benchmark ordering.

The raw predictions are the expensive output. MedAPE, MedAE, and per-benchmark summaries can be recomputed from them without rerunning BenchPress.

The random baseline stores the same all-observed evaluation universe as greedy, uses the same shared `benchpress.evaluation_harness.evaluate_probe_set` primitive, and differs only in how the probe set is chosen: nested random prefixes rather than greedy minimization. The MedAPE random output currently stores `k=1..30` with 10 seeds and 781,200 raw predictions over the 84-by-133 matrix; Section 5.1 plots display the first 10 points, while Figure 1 displays the random curve at `k=0,3,6,...,30`. `plot.py --compare --random-in random_medape_hero_all_known.json.gz --cheap-in greedy_medape_targets_tall_candidates_usercheap.json.gz` compares random, all-candidate greedy, and user-cheap greedy on the same denominator.

## Resume / rerun

The greedy-search script resumes from an existing output file only when `metric`,
`candidate_source`, `candidate_allowlist_ids`, `candidate_limit`,
`n_target_cells`, `n_candidates`, and `eval_protocol` match the requested run. Within an
unfinished greedy step, candidate evaluations are cached under
`results/.candidate_cache/` as soon as each candidate completes, including the raw
per-cell predictions. If a remote pod is preempted mid-step, re-running the same
`OUT=...` reuses the completed candidate shards and evaluates only the missing
candidates. The cache is keyed by output name, metric, candidate source, step,
candidate benchmark, and the probe set before that candidate, so a
changed greedy prefix refuses to reuse stale shards instead of silently mixing
objectives.

The random baseline uses one shard file per `(k, seed)` and resumes only when the
existing shard declares the nested probe-prefix protocol and matches the requested
`k`, `seed`, `model_limit`, matrix shape, and seed configuration. Merge is
fail-fast: the shard directory must contain exactly the requested grid and no
extra smoke-test or stale-protocol shards.
