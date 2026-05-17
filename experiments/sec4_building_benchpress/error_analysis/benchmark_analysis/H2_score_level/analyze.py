#!/usr/bin/env python3
"""H2: Score level — median observed score (denominator / z-score effect)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import *

def main():
    np.random.seed(SEED)
    rows = build_rows()
    result = {
        "hypothesis": "H2", "feature": "med_score", "seed": SEED,
        "default_error_source": DEFAULT_PREDICTIONS_BY_BENCHMARK_REL,
        "raw_predictions_source": DEFAULT_PREDICTIONS_BY_BENCHMARK_REL,
        "targets": TARGETS, "n_benchmarks": len(rows),
        "benchmarks": rows,
        "univariate_spearman": {t: univariate_spearman(rows, t, ["med_score"]) for t in TARGETS},
    }
    write_json_next_to(__file__, result, indent=2)
    print(f"[saved] H2 results.json (n={len(rows)})")

if __name__ == "__main__":
    main()
