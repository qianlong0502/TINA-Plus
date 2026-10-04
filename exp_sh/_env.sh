#!/usr/bin/env bash
TINA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TINA_ROOT"
if [[ -f configs/paths.local.sh ]]; then
    source configs/paths.local.sh
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-CompVis/stable-diffusion-v1-4}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-unet}"
DEFENSE="${DEFENSE:-esd}"
RESUME="${RESUME:-1}"
OUTPUT_OVERWRITE="${OUTPUT_OVERWRITE:-0}"
