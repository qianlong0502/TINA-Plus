#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
: "${CONCEPT:?set CONCEPT}"
: "${CKPT:?set CKPT to a concept-erased checkpoint}"
case "$CONCEPT" in
    vangogh|church|garbage_truck|parachute|tench|taylor_swift|elon_musk|adam_lambert) EXPECTED_COUNT=50 ;;
    nudity) EXPECTED_COUNT=118 ;;
    *) echo "unsupported concept: $CONCEPT" >&2; exit 1 ;;
esac
EXTRA_ARGS=()
if [[ "$CONCEPT" == vangogh ]]; then
    : "${STYLE_CLASSIFIER:?set STYLE_CLASSIFIER}"
    EXTRA_ARGS+=(--classifier-dir "$STYLE_CLASSIFIER")
elif [[ "$CONCEPT" == nudity ]]; then
    : "${NUDENET_ONNX_PATH:?set NUDENET_ONNX_PATH}"
    EXTRA_ARGS+=(--nudenet-onnx-path "$NUDENET_ONNX_PATH")
fi
source scripts/gpu_env.sh
for attacker in no_attack tina; do
    name=no_attack
    [[ "$attacker" == tina ]] && name=tina_plus
    "$PYTHON_BIN" tina_run.py \
        --input-path "$DATA_ROOT/$CONCEPT" --input-format dir \
        --output-path "$OUTPUT_ROOT/$CONCEPT/$DEFENSE/$name" --output-format dir \
        --output-overwrite "$OUTPUT_OVERWRITE" --resume "$RESUME" \
        --attacker "$attacker" --concept "$CONCEPT" \
        --base-model-name-or-path "$BASE_MODEL" \
        --input-checkpoint-path "$CKPT" --checkpoint-type "$CHECKPOINT_TYPE" \
        --device "$DEVICE" \
        --sampling-step-num 50 --tina-num-ddim-steps 50 --tina-opt-round 25 \
        --tina-lr 0.001 --tina-fmi-ratio 0.1 --tina-fmi-seed 0 \
        --tina-energy-weight 1.0 --tina-energy-tau 10 \
        --tina-energy-t-min 0.1 --tina-energy-t-max 1.0 --seed 0 \
        --attack-idx-start "${ATTACK_IDX_START:-0}" \
        --attack-idx-end "${ATTACK_IDX_END:-$EXPECTED_COUNT}" \
        "${EXTRA_ARGS[@]}"
done
