#!/usr/bin/env bash
set -euo pipefail
export CONCEPT="${CONCEPT:-tench}"
bash "$(dirname "${BASH_SOURCE[0]}")/../evaluate.sh"
