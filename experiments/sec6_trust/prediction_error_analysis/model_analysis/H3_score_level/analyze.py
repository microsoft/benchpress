#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _shared import *

def main():
    np.random.seed(SEED)
    print("[H3] Running BenchPress hide-half per-model...")
    loo, raw = benchpress_half_holdout_per_model()
    rows = build_rows(loo)
    spearman = {t: univariate_spearman(rows, t, ["med_true"]) for t in TARGETS}
    result = {"feature": "med_true", "models": rows, "univariate_spearman": spearman,
              "raw_predictions": raw}
    write_json_next_to(__file__, result, indent=2)
    print(f"[H3] Saved ({len(rows)} models)")

if __name__ == "__main__":
    main()
