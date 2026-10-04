#!/usr/bin/env bash
# Source in the shell that executes the GPU workload.
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
if [ -d /lib/x86_64-linux-gnu ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *:/lib/x86_64-linux-gnu:*) ;;
        *) export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
"${PYTHON_BIN:-python}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable after runtime initialization")
print("CUDA device:", torch.cuda.get_device_name(0))
PY
