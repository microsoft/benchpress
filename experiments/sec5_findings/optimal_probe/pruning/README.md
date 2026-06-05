# Greedy-elimination probe pruning

## Paper mapping

- Section: `\Cref{sec:probe_selection}` / Appendix probe-selection diagnostics.
- This is a candidate-set pruning diagnostic for the same all-known-cell protocol
  used by `../all_known/` and `../brute_force/`.

## Purpose

Use greedy search as an elimination tool, not as the final selector. The runner
fixes one or more anchor probes, follows the greedy path to create realistic
contexts, and records every remaining benchmark's conditional gain in each
context. A benchmark is marked removable only when it never provides meaningful
gain across those contexts and passes explicit coverage/category/protection
guards.

## How to run

Smoke-test on a remote CPU environment:

```bash
cd experiments/sec5_findings/optimal_probe/pruning
CANDIDATE_LIMIT=10 MAX_STEPS=3 WORKERS=4 ./run.sh
```

Full conservative scan over the current matrix candidates:

```bash
cd experiments/sec5_findings/optimal_probe/pruning
METRIC=medae \
FIXED_PROBES=gpqa_diamond \
PROTECTED_PROBES=gpqa_diamond,mmlu_pro \
MAX_STEPS=5 \
WORKERS=48 \
ALLOWLIST_OUT=../candidate_allowlists/full_pruned_by_gpqa_greedy_20260605.json \
./run.sh
```

If an existing all-known greedy result already contains the same greedy
contexts, reuse it instead of re-evaluating candidates:

```bash
METRIC=medae \
FIXED_PROBES=gpqa_diamond \
PROTECTED_PROBES=gpqa_diamond,mmlu_pro \
MAX_STEPS=5 \
SOURCE_GREEDY_RESULT=../all_known/results/greedy_medae_targets_tall_candidates_tall.json.gz \
OUT=results/greedy_elimination_medae_fixed-gpqa_diamond_from_existing_greedy.json.gz \
ALLOWLIST_OUT=../candidate_allowlists/full_pruned_by_gpqa_greedy_20260605.json \
./run.sh
```

## Inputs

- Score matrix and observed mask from `benchpress.evaluation_harness`.
- Predictor: `predict_benchpress_scores`.
- Optional candidate allowlists from `../candidate_allowlists/`; when omitted,
  all current matrix benchmarks are candidates.
- Optional source greedy result from `../all_known/results/`. This is the
  preferred path when the source trajectory starts with the requested fixed
  probes, because every step already stores raw candidate predictions.

## Outputs

Results are written under `results/`.

```text
results/greedy_elimination_<metric>_fixed-<anchors>_candidates-<source>.json.gz
results/.candidate_cache/<run-id>/step_XXX/<benchmark>.json.gz
```

When `SOURCE_GREEDY_RESULT` is set, no candidate cache is written; the result
copies the relevant source greedy contexts and raw predictions.

The result stores:

- fixed/protected probes and pruning thresholds;
- the greedy trajectory;
- raw per-cell predictions for every evaluated candidate in every context;
- per-candidate max conditional gain across contexts;
- keep/remove decisions and guard reasons;
- optional generated allowlist metadata.

## Resume / rerun

Done unit is one cached candidate evaluation for one greedy context. Re-running
with the same output/config validates cache metadata and skips completed
candidates. If fixed probes, metric, candidate universe, thresholds, or guards
change, use a different `OUT` or delete the incompatible cache.

## Last valid result

Not yet run for the full GPQA-D anchored pruning scan.
