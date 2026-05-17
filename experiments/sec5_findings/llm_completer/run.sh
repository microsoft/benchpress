#!/bin/bash
# §5.3 LLM prompt ablations: matrix completer and five-shot predictor.
set -euo pipefail
cd "$(dirname "$0")"

bash informed_vs_blind/run.sh
python informed_vs_blind/plot.py
bash five_shot_predictor/run.sh
python five_shot_predictor/plot.py
