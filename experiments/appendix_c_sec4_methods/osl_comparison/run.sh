#!/bin/bash
# Appendix C.4: Observational Scaling Laws (Ruan et al. 2024) vs BenchPress.
# obs_scaling_baseline.py runs OSL on our score matrix -> tab:osl_comparison.
# Run on GCR CPU; not for local Mac.
set -euo pipefail
cd "$(dirname "$0")"

python obs_scaling_baseline.py --workers 8
