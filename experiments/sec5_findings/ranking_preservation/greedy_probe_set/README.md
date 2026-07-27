# Ranking-preservation probe sets

## Directory split

Ranking probe-set experiments are separated by protocol so code and results for
current-matrix construction and held-out validation do not share paths.

| Protocol | Directory | Owns |
|---|---|---|
| Current-matrix / all-known-cell construction | `all_known/` | `run.py`, `results/` |
| Model-split held-out validation | `holdout/` | `run_model_split_validation.py`, `results/` |

## Paper mapping

- Main text: `\Cref{sec:ranking_preservation}`.
- Appendix: `\Cref{app:ranking_preservation}`, table `tab:ranking_greedy_probe_set`.
- Hero Figure ranking panel reads all-known ranking probe results.

## Protocol decision

Use `all_known/` for the original ranking-aware construction: every observed
cell remains in the fixed denominator, probe cells are exact, and candidate
prefixes are scored by margin-5 same-benchmark pairwise accuracy.

Use `holdout/` for model-level validation: split model rows 70/30, select probes
on training model rows, and validate fixed prefixes on held-out model rows.
