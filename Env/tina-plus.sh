#!/usr/bin/env bash
# Environment: tina-plus; target: SD1.4 attacks; mode: fresh
# Python: 3.10.18; torch: 2.8.0+cu126; torchvision: 0.23.0+cu126
# CUDA wheel: 12.6; hardware: CUDA GPU (paper uses A100)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n tina-plus python=3.10.18 -y
conda activate tina-plus
# pip/wheel versions observed in the source environment; setuptools is pinned
# to its Python 3.10-compatible release (fresh-install recipe).
python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1
# All installation stages share exact pins so subsequent installs cannot alter
# the selected Torch / Hugging Face / NumPy stack silently.
python -m pip install -c requirements/attack.txt \
    torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c requirements/attack.txt -r requirements/attack.txt
python -V
python -m pip check
export PYTHON_BIN=python
source scripts/gpu_env.sh
python - <<'PY'
import diffusers, transformers, torch, onnxruntime
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
print("onnxruntime", onnxruntime.__version__)
PY
