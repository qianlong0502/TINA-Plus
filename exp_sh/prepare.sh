#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${CONCEPT:?set CONCEPT}"
case "$CONCEPT" in
    vangogh|nudity|church|garbage_truck|parachute|tench) ;;
    *) echo "use exp_sh/celebrity/00_prepare.sh for celebrity targets" >&2; exit 1 ;;
esac
source scripts/gpu_env.sh
TARGET_STEPS="${TARGET_STEPS:-25}"
FILTER_LONG_PROMPTS=0
[[ "$CONCEPT" == nudity ]] && FILTER_LONG_PROMPTS=1
"$PYTHON_BIN" scripts/prepare/generate_dataset.py \
    --input-path "prompts/$CONCEPT.csv" --input-format csv \
    --output-path "$DATA_ROOT/$CONCEPT" --output-format dir \
    --output-overwrite "$OUTPUT_OVERWRITE" \
    --model-name-or-path "$BASE_MODEL" --device "$DEVICE" \
    --filter-long-prompts "$FILTER_LONG_PROMPTS" --sampling-steps "$TARGET_STEPS" --guidance-source legacy --guidance 7.5
