#!/usr/bin/env bash
# Sequentially run and evaluate a user-supplied checkpoint table.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
TABLE="${CHECKPOINT_TABLE:-configs/checkpoints.local.csv}"
[[ -f "$TABLE" ]] || { echo "missing checkpoint table: $TABLE" >&2; exit 1; }
while IFS=, read -r concept defense checkpoint_type checkpoint; do
    [[ "$concept" == concept || -z "$concept" ]] && continue
    [[ -n "${MATRIX_CONCEPT:-}" && "$concept" != "$MATRIX_CONCEPT" ]] && continue
    [[ -n "${MATRIX_DEFENSE:-}" && "$defense" != "$MATRIX_DEFENSE" ]] && continue
    [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
    CONCEPT="$concept" DEFENSE="$defense" CHECKPOINT_TYPE="$checkpoint_type" CKPT="$checkpoint" \
        bash exp_sh/run_attack.sh </dev/null
    CONCEPT="$concept" DEFENSE="$defense" bash exp_sh/evaluate.sh </dev/null
done < "$TABLE"
