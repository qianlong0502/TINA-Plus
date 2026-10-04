<div align="center">

<h2>TINA+: Probing Residual Visual Knowledge in Unlearned Diffusion Models<br>via Diffusion-Consistent Text-Free Inversion</h2>

<p>
Qianlong Xiang<sup>1,2,3</sup>,
Miao Zhang<sup>1,✉</sup>,
Kun Wang<sup>5</sup>,
Haoyu Zhang<sup>1,4</sup>,
Junhui Hou<sup>2,✉</sup>,
Liqiang Nie<sup>1</sup>
</p>

<p>
<sup>1</sup>Harbin Institute of Technology (Shenzhen)&nbsp;&nbsp;
<sup>2</sup>City University of Hong Kong<br>
<sup>3</sup>Shenzhen Loop Area Institute&nbsp;&nbsp;
<sup>4</sup>Pengcheng Laboratory&nbsp;&nbsp;
<sup>5</sup>National University of Singapore
</p>

<p><sup>✉</sup> Corresponding authors</p>

<p>
<a href="https://arxiv.org/abs/2608.17747"><img src="https://img.shields.io/badge/Paper-arXiv-green" alt="TINA+ paper on arXiv"></a>
&nbsp;
<a href="https://qianlong0502.github.io/TINA-Plus-Homepage/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="TINA+ project page"></a>
&nbsp;
<a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Xiang_TINA_Text-Free_Inversion_Attack_for_Unlearned_Text-to-Image_Diffusion_Models_CVPR_2026_paper.pdf"><img src="https://img.shields.io/badge/TINA-CVPR%202026-orange" alt="TINA conference version at CVPR 2026"></a>
</p>

</div>

## Method Overview

TINA+ recovers erased concepts by optimizing a target image's inversion under
an **empty text condition**. It combines fixed-point inversion with **Forward
Marginal Initialization (FMI)** and **Marginal Energy Regularization**, then
generates an image from the optimized noise using the same concept-erased model.
No model parameters are trained by the attack.

This repository provides **Stable Diffusion v1.4 attack reproduction** for style,
nudity, object and celebrity identity erasure.

## Main Results

Selected TINA+ ASR (%) from the paper, using the evaluation
protocol in [docs/evaluation.md](docs/evaluation.md):

| Task / Concept | ESD | STEREO | Cases |
|---|---:|---:|---:|
| Style / Van Gogh | 60.00 | 46.00 | 50 |
| Nudity | 86.44 | 97.46 | 118 |
| Object / Church | 90.00 | 78.00 | 50 |
| Object / Garbage Truck | 74.00 | 66.00 | 50 |
| Object / Parachute | 84.00 | 80.00 | 50 |
| Object / Tench | 64.00 | 66.00 | 50 |
| Celebrity / Taylor Swift | 96.00 | 94.00 | 50 |
| Celebrity / Elon Musk | 98.00 | 46.00 | 50 |
| Celebrity / Adam Lambert | 94.00 | 64.00 | 50 |

## Environment

The attack environment uses Python 3.10, PyTorch 2.8.0/CUDA 12.6, Diffusers 0.35.1
and Transformers 4.50.0. Create a fresh environment using
[Env/tina-plus.sh](Env/tina-plus.sh). Celebrity detection uses a separate Python
3.6 environment: [Env/tina-plus-gcd.sh](Env/tina-plus-gcd.sh).

The NudeNet ONNX backend defaults to CPU; the diffusion attack requires CUDA.

## Resource Setup

Provide SD1.4, your concept-erased checkpoint and the relevant classifier as
described in [docs/resources.md](docs/resources.md). Configure local paths:

```bash
cp configs/paths.example.sh configs/paths.local.sh
# Edit configs/paths.local.sh for your model, classifiers and environments.
```

Use `CHECKPOINT_TYPE=unet` for UNet weights and `text_encoder` for AdvUnlearn
wrapper checkpoints. Checkpoint type is explicit; filenames are not used to
infer it. Checkpoints, datasets and generated results are excluded from Git.

## Quick Start: Tench × ESD

Run from the repository root in the attack environment:

```bash
# Generate target images from the pretrained SD1.4 model.
bash exp_sh/object/00_prepare.sh

# Run the prompt-conditioned baseline, then the full null-text TINA+ attack.
CKPT=/path/to/ESD-Tench-Diffusers-UNet-noxattn.pt \
  bash exp_sh/object/01_attack.sh

# Aggregate all 50 targets, checking the independently generated baseline.
bash exp_sh/object/02_evaluate.sh
```

Each GPU shell script initializes driver library paths and checks CUDA before
loading models. For direct Python commands, first run
`source scripts/gpu_env.sh` in the same shell.

## Other Tasks

Each group follows the same sequence: prepare targets, run No Attack and TINA+,
then evaluate ASR. The group pages provide task-specific commands and settings.

| Task | Released prompts | Attack targets | Instructions |
|---|---:|---:|---|
| Van Gogh | 50 | 50 | [Style](exp_sh/style/README.md) |
| Nudity | 142 | 118 after runtime filtering | [Nudity](exp_sh/nudity/README.md) |
| Church, Garbage Truck, Parachute, Tench | 50 per concept | 50 per concept | [Objects](exp_sh/object/README.md) |
| Taylor Swift, Elon Musk, Adam Lambert | 50 per identity | 50 per identity | [Celebrities](exp_sh/celebrity/README.md) |

Celebrity preparation uses the final prompt/seed lists directly. GCD is needed
only for offline evaluation. All datasets are generated from the CSVs in
[prompts/](prompts/); target images and erased-model weights are supplied or
generated locally.

## Attack Settings and Outputs

The default attack is full TINA+: 50 DDIM inversion steps, 25 AdamW updates per
step, learning rate 0.001, FMI ratio 0.1, energy weight 1.0, tolerance 10 and
regularization window [0.1, 1.0]. Final generation uses LMS. Dataset row guidance
is retained (7.5 when absent); the released I2P rows use their evaluation guidance.

Results are written to `outputs/<concept>/<defense>/{no_attack,tina_plus}/`.
Each sample stores its configuration, baseline, attack image and optimized noise.
Evaluation produces `samples.csv` and `summary.json` with baseline ASR, additional
successes and merged ASR. See [command contracts](docs/commands.md) for direct CLI,
sample indices and resume behavior; [exp_sh/README.md](exp_sh/README.md) indexes
the experiment scripts.

For multiple defenses, configure the checkpoint table and sequential matrix
entry described in [exp_sh/README.md](exp_sh/README.md).

## Acknowledgments

This work builds on [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack).
We thank its authors and the developers of Diffusers, Transformers, NudeNet and
GIPHY Celebrity Detector. See [third-party notices](THIRD_PARTY_NOTICES.md).
The code distribution uses [AGPL-3.0](LICENSE); external resources retain their
publisher's terms.

## Citation

If you use TINA+ in your research, please cite:

```bibtex
@article{xiang2026tina+,
  title={TINA+: Probing Residual Visual Knowledge in Unlearned Diffusion Models via Diffusion-Consistent Text-Free Inversion},
  author={Xiang, Qianlong and Zhang, Miao and Wang, Kun and Zhang, Haoyu and Hou, Junhui and Nie, Liqiang},
  journal={arXiv preprint arXiv:2608.17747},
  year={2026}
}
```

For the conference version, TINA:

```bibtex
@inproceedings{xiang2026tina,
  title={TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models},
  author={Xiang, Qianlong and Zhang, Miao and Zhang, Haoyu and Wang, Kun and Hou, Junhui and Nie, Liqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2026},
  pages={30076--30086}
}
```
