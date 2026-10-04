#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${CONCEPT:?set CONCEPT}"
EXPECTED_COUNT="${EXPECTED_COUNT:-50}"
[[ "$CONCEPT" == nudity ]] && EXPECTED_COUNT="${NUDITY_EXPECTED_COUNT:-118}"
RUN_ROOT="$OUTPUT_ROOT/$CONCEPT/$DEFENSE"
case "$CONCEPT" in
    taylor_swift|elon_musk|adam_lambert)
        : "${GCD_PYTHON_BIN:?set GCD_PYTHON_BIN}"
        : "${GCD_ROOT:?set GCD_ROOT}"
        "$GCD_PYTHON_BIN" scripts/evaluate/gcd_asr.py \
            --input-path "$RUN_ROOT/tina_plus" --input-format dir \
            --output-path "$RUN_ROOT/evaluation" --output-format dir \
            --output-overwrite "$OUTPUT_OVERWRITE" --expected-count "$EXPECTED_COUNT" \
            --celebrity "$CONCEPT" --gcd-root "$GCD_ROOT" \
            --gcd-data-dir "${GCD_DATA_DIR:-}" --use-cuda 0 ;;
    *)
        "$PYTHON_BIN" scripts/evaluate/asr.py \
            --input-path "$RUN_ROOT/tina_plus" --input-format dir \
            --input-no-attack-path "$RUN_ROOT/no_attack" \
            --output-path "$RUN_ROOT/evaluation" --output-format dir \
            --output-overwrite "$OUTPUT_OVERWRITE" --expected-count "$EXPECTED_COUNT" ;;
esac
