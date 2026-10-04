#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
: "${CONCEPT:?set CONCEPT to a celebrity slug}"
case "$CONCEPT" in taylor_swift|elon_musk|adam_lambert) ;; *) exit 1 ;; esac
source scripts/gpu_env.sh
"$PYTHON_BIN" scripts/prepare/generate_dataset.py \
    --input-path "prompts/$CONCEPT.csv" --input-format csv \
    --output-path "$DATA_ROOT/$CONCEPT" --output-format dir \
    --output-overwrite "$OUTPUT_OVERWRITE" --model-name-or-path "$BASE_MODEL" \
    --device "$DEVICE" --sampling-steps 25 --guidance-source row
