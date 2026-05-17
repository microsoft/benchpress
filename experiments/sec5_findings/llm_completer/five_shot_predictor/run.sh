#!/bin/bash
# §5.3 LLM five-shot predictor prompt ablation.
set -euo pipefail
cd "$(dirname "$0")"

python run.py --models gpt-5.5 --conditions five_shot_named \
  --batch-size 64 --max-tokens 16384
python run.py --models gpt-5.5 --conditions five_shot_blind \
  --batch-size 16 --max-tokens 16384
