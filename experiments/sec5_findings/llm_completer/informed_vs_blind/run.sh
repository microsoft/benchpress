#!/bin/bash
# §5.3 LLM completer: informed vs blind prompting on the §4 folds.
set -euo pipefail
cd "$(dirname "$0")"

python run.py --models gpt-5.5 --batch-size 10
