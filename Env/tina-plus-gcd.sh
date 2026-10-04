#!/usr/bin/env bash
# Environment: tina-plus-gcd; target: offline celebrity ASR; mode: fresh
# Python: 3.6.13; torch: 1.10.1; CPU evaluation, no CUDA required
# Execute the whole file from the repository root. No GCD checkout is installed
# by pip; evaluation loads the separately supplied GCD_ROOT directly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n tina-plus-gcd python=3.6.13 -y
conda activate tina-plus-gcd
# Python 3.6 compatible packaging tools.
python -m pip install pip==21.3.1 setuptools==59.6.0 wheel==0.37.1
python -m pip install -c requirements/gcd.txt -r requirements/gcd.txt
python -V
python -m pip check
python - <<'PY'
import torch, tensorflow, cv2, skimage
print("torch", torch.__version__, "tensorflow", tensorflow.__version__)
print("opencv", cv2.__version__, "skimage", skimage.__version__)
PY
