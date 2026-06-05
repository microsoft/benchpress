# Greedy-elimination probe pruning

## Paper mapping

- Section: `\Cref{sec:probe_selection}` / Appendix probe-selection diagnostics.
- This is a candidate-set pruning diagnostic for the same all-known-cell protocol
  used by `../all_known/` and `../brute_force/`.

## Purpose

Use greedy search as an elimination tool, not as the final selector. The runner
derives pruning diagnostics from an existing all-known greedy result that
already stores every candidate's predictions in each greedy context. It does not
re-evaluate candidates.

## How to run

Derive the GPQA-D anchored pruning allowlist from the existing full all-known
greedy result:

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

- Source greedy result from `../all_known/results/`.
- Score-matrix metadata from `benchpress.evaluation_harness` for benchmark names,
  categories, and coverage guards.
- Optional candidate allowlist only when the source result does not record its
  candidate IDs.

## Outputs

Results are written under `results/`.

```text
results/greedy_elimination_<metric>_fixed-<anchors>_from_existing_greedy.json.gz
```

The result stores:

- fixed/protected probes and pruning thresholds;
- the greedy trajectory;
- raw per-cell predictions for every evaluated candidate in every context;
- per-candidate max conditional gain across contexts;
- keep/remove decisions and guard reasons;
- optional generated allowlist metadata.

## Last valid result

- Source: `../all_known/results/greedy_medae_targets_tall_candidates_tall.json.gz`
- Derived result: `results/greedy_elimination_medae_fixed-gpqa_diamond_from_existing_greedy.json.gz`
- Allowlist: `../candidate_allowlists/full_pruned_by_gpqa_greedy_20260605.json`
- Setting: fixed `gpqa_diamond`, protected `gpqa_diamond,mmlu_pro`, source
  greedy steps 2--6.
- Decision: remove 26 candidates and keep 107.
