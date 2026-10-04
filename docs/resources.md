# Resources and datasets

Use SD1.4 and supply your concept-erased checkpoint through `CKPT`.
Erased-model weights and target images are not distributed in this repository.

Copy `configs/paths.example.sh` to `configs/paths.local.sh` and configure your
resource locations. For multiple defenses, fill in `configs/checkpoints.local.csv`
from the example table; its paths are placeholders.

## Evaluation resources

Nudity and style evaluation reuse resources from
[UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack):
set `NUDENET_ONNX_PATH` to its `best.onnx` and `STYLE_CLASSIFIER` to the artist
classifier directory (`checkpoint-2800`). Obtain these resources from upstream.

Object evaluation uses [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
and its accompanying image processor. The code downloads them automatically;
`TINA_RESNET50_MODEL_PATH` optionally selects a local copy.

Celebrity evaluation uses [GIPHY Celebrity Detector](https://github.com/Giphy/celeb-detection-oss).
Follow its model-download instructions and set `GCD_ROOT`, `GCD_DATA_DIR` and
`GCD_PYTHON_BIN`. Its resources include the face detector arrays, recognition
weights and labels. These resources retain their upstream licenses.

## Checkpoint formats

- `unet`: full Diffusers UNet state dict in `.pt`; `.safetensors` can contain a
  parameter subset, retaining other parameters from SD1.4.
- `text_encoder`: state dict for the historical AdvUnlearn `CustomTextEncoder`
  wrapper.

Other formats, including original LDM checkpoints and LoRA adapters, require
conversion with the corresponding defense's tools.

## Released prompt lists

| Concept | CSV rows | Attack targets |
|---|---:|---:|
| Van Gogh | 50 | 50 |
| Nudity | 142 | 118 after runtime token filtering |
| Church / Garbage Truck / Parachute / Tench | 50 each | 50 each |
| Taylor Swift / Elon Musk / Adam Lambert | 50 each | 50 each |

The three celebrity CSVs contain the final prompts and original generation
seeds. No candidate generation or identity-based selection is performed.
Nudity retains its original 142 rows and excludes >60-token prompts at runtime.
All other lists are loaded in full.

Preparation uses 25 LMS steps and 512×512 images. Standard tasks use generation
guidance 7.5 unless `sd_guidance_scale` is supplied; celebrity generation uses
`evaluation_guidance`. Baseline guidance always comes from `evaluation_guidance`
(7 for the released Nudity rows), independently of target generation guidance.

Output layout:

```text
data/<concept>/
  prompts.csv
  imgs/<case_number>_0.png
  ignore.json
  samples.csv
  generation.json
```

`ignore.json` records excluded Nudity CSV rows and is empty for other tasks.
`samples.csv` records seeds, guidance and image hashes. `generation.json` records
image-generation arguments. The generation seed is `evaluation_seed`; case IDs
are only used for naming images.

## GCD compatibility

Use the separate [environment recipe](../Env/tina-plus-gcd.sh). Evaluation imports
the GCD checkout directly, so editable installation is unnecessary.

The research integration uses two compatibility changes in the external GCD
checkout:

1. In `model_training/preprocessors/face_detection/network.py`, load the trusted
   upstream detector arrays with
   `np.load(data_path, encoding='latin1', allow_pickle=True).item()`.
2. If installing GCD through its old `setup.py`, use `x.requirement` instead of
   `x.req`. The wrappers here import the checkout directly, so editable installation
   is unnecessary.

Install OpenCV system libraries (`libsm6`, `libxext6`, `libxrender1`) if required.
GCD wrappers use CPU and a separate environment.
