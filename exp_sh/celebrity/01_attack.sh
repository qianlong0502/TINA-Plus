#!/usr/bin/env bash
set -euo pipefail
export CONCEPT="${CONCEPT:-taylor_swift}"
bash "$(dirname "${BASH_SOURCE[0]}")/../run_attack.sh"
