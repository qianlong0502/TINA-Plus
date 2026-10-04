# Direct Python Commands

The [experiment scripts](../exp_sh/README.md) provide the standard workflow.
Use the commands below to run individual stages or select individual samples.
Run from the repository root with the attack environment active.

## Generate, Attack and Evaluate

Set `PYTHON_BIN` to the attack interpreter and initialize CUDA in the same shell
before running generation or attack commands:

```bash
export PYTHON_BIN=python
source scripts/gpu_env.sh
```

```bash
python scripts/prepare/generate_dataset.py \
  --input-path prompts/tench.csv --input-format csv \
  --output-path data/tench --output-format dir \
  --model-name-or-path CompVis/stable-diffusion-v1-4

python tina_run.py \
  --input-path data/tench --input-format dir \
  --output-path outputs/tench/esd/tina_plus --output-format dir \
  --concept tench --base-model-name-or-path CompVis/stable-diffusion-v1-4 \
  --input-checkpoint-path checkpoints/esd_tench.pt --checkpoint-type unet

python scripts/evaluate/asr.py \
  --input-path outputs/tench/esd/tina_plus --input-format dir \
  --output-path outputs/tench/esd/evaluation --output-format dir \
  --expected-count 50
```

## Dataset Inputs

Each dataset contains `prompts.csv` and one image per row at
`imgs/<case_number>_0.png`. `evaluation_seed` determines the generation seed;
`case_number` only determines the image filename.

For Nudity, add `--filter-long-prompts 1` to the generation command.
`ignore.json` stores the excluded CSV row indices, not case numbers. Generation
retains all 142 rows; the attack loader excludes the listed rows. Other tasks
use the default `--filter-long-prompts 0` and load their final lists in full.

Celebrity generation also requires `--guidance-source row` to use the supplied
`evaluation_guidance`. Standard tasks use the default `legacy` guidance source.
See [resources](resources.md) for the generation and baseline guidance settings.

## Attack Options

The default is full TINA+, with FMI ratio 0.1 and energy weight 1.0.
Use `--attacker no_attack` for the baseline with the same inputs, model and seeds.

`--attack-idx-start` and `--attack-idx-end` select a half-open range. Alternatively,
`--attack-indices 0,3,7` selects individual samples. Indices refer to the loaded
list, after filtering for Nudity. Every result records its source CSV row and
case number.

`--resume 1` skips completed samples with matching configurations.
`--output-overwrite 1 --resume 0` replaces selected sample directories.
Use a separate output directory when changing model or attack settings.

## Output Layout

```text
outputs/<concept>/<defense>/tina_plus/
  manifest.json
  attack_idx_<n>/
    config.json
    log.json
    complete.json
    images/
    noises/
```

| File | Meaning |
|---|---|
| `images/orig.png` | Prompt-conditioned erased-model baseline |
| `images/reconstructed.png` | Target VAE reconstruction, when inversion runs |
| `images/generated.png` | Null-text attack output, when inversion runs |
| `noises/optimized_noise_<n>.pt` | Optimized inversion noise, when inversion runs |
| `complete.json` | Marks a finished sample |

Baseline-success object and Nudity samples skip inversion. Style and celebrity
samples always undergo inversion. See [evaluation](evaluation.md) for ASR rules.

## Evaluation

Pass `--expected-count 50` for standard final lists or `118` for Nudity.
The evaluator requires every expected sample and rejects incomplete results.
Optionally pass `--input-no-attack-path` to compare an independently generated
baseline with the one stored in the attack results.

Celebrity ASR uses `scripts/evaluate/gcd_asr.py` in the separate GCD environment.
It takes the same typed input/output arguments, plus `--celebrity`, `--gcd-root`
and optionally `--gcd-data-dir`. It does not take `--input-no-attack-path`.
The [celebrity scripts](../exp_sh/celebrity/README.md) provide the full command.
