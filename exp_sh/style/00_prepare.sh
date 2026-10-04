#!/usr/bin/env bash
set -euo pipefail
export CONCEPT="${CONCEPT:-vangogh}"
bash "$(dirname "${BASH_SOURCE[0]}")/../prepare.sh"
