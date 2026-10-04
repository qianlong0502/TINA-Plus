# Celebrity identity attack

SD1.4 identity erasure for Taylor Swift, Elon Musk and Adam Lambert, using ESD
and STEREO; 50 retained targets per identity.

## Run

Set `GCD_PYTHON_BIN`, `GCD_ROOT` and `GCD_DATA_DIR` for evaluation.
Example for Taylor Swift × STEREO, from the repository root:

```bash
CONCEPT=taylor_swift bash exp_sh/celebrity/00_prepare.sh
CONCEPT=taylor_swift DEFENSE=stereo CKPT=/path/to/stereo_taylor_swift.pt bash exp_sh/celebrity/01_attack.sh
CONCEPT=taylor_swift DEFENSE=stereo bash exp_sh/celebrity/02_evaluate.sh
```

## Workflow

| Script | Role | Dependencies | Outputs |
|---|---|---|---|
| `00_prepare.sh` | Final 50 prompts/seeds → SD images | Attack environment + SD1.4 | Targets |
| `01_attack.sh` | No Attack + TINA+ | Targets, erased UNet checkpoint | Baseline + generated image for each target |
| `02_evaluate.sh` | CPU GCD exact identity ASR | GCD checkout/weights, complete attack | Per-sample labels/scores and summary |

## Settings

Set `CONCEPT` explicitly for every command, and `DEFENSE=esd|stereo` for attack
and evaluation. Supply the corresponding checkpoint through `CKPT`.
The preparation script requires `CONCEPT`; attack/evaluation default to Taylor
Swift. Preparation reads `prompts/<concept>.csv` directly and needs no GCD setup.
The original per-row seeds and guidance are retained; no further selection occurs.

GCD detects faces and evaluates the first returned face, with top-1 normalized
name matching. No detected face is failure. Every target undergoes TINA+
inversion; merged ASR evaluates baseline first and attack if needed.

GCD runs under `GCD_PYTHON_BIN`, isolated from modern Torch/Transformers.
Set `GCD_ROOT` and `GCD_DATA_DIR` to external GCD code/resources.
See [GCD setup](../../docs/resources.md) and [evaluation](../../docs/evaluation.md).

Erased-model weights are supplied by the user through `CKPT`.
