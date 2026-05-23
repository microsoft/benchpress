# Optimal probe selection

## Directory split

Probe-set experiments are separated by protocol so code, results, and figures do
not share a directory across incompatible evaluation settings.

| Protocol | Directory | Owns |
|---|---|---|
| Current-matrix / all-known-cell construction | `all_known/` | `run.py`, `run_random.py`, `plot.py`, `results/`, `figures/` |
| Model-split held-out validation | `holdout/` | `run_model_split_validation.py`, `run_model_split_random.py`, Bonete launcher, `results/` |

Shared candidate-set definitions stay in `candidate_allowlists/` because both
protocols use the same curated low-cost benchmark universe.

## Paper mapping

- Main text: `\Cref{sec:probe_selection}`.
- Appendix: `\Cref{app:probe_selection}`.
- Current paper-facing Hero Figure panel B uses the `all_known/` curves. The
  `holdout/` outputs are robustness artifacts for sensitivity checks and update
  notes, not the active main recommendation.

## Protocol decision

Use `all_known/` when the question is the full-matrix construction result: all
historical rows are available, a target model reveals only the chosen probe
columns, and every observed target-model cell remains in the denominator with
probe cells counted as exact.

Use `holdout/` when the question is model-level validation: split model rows
70/30, select probes on training rows, and validate fixed prefixes on held-out
model rows under the isolated target-row protocol.
