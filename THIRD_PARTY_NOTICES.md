# Third-party notices

This distribution uses AGPL-3.0 (see [LICENSE](LICENSE)) because it includes an
adapted NudeNet detector. External model weights and datasets are not covered by
the code license; obtain them from their original publishers under their terms.

| Component | Origin | Notice |
|---|---|---|
| SD sampling, inversion infrastructure and evaluation conventions | [UnlearnDiffAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack) and the authors' TINA implementation | Copyright (c) 2023 OPTML Group; [MIT notice](licenses/UnlearnDiffAtk-MIT.txt) retained |
| CLIP attention-mask helpers in `src/tasks/utils/text_encoder.py` | [Transformers](https://github.com/huggingface/transformers/tree/v4.50.0) | Apache-2.0; [license](licenses/Transformers-Apache-2.0.txt) retained |
| `src/tasks/utils/metrics/nudenet/detector.py` | [NudeNet](https://github.com/notAI-tech/NudeNet/tree/v3) | AGPL-3.0; adapted in the research code and for this export on 2026-10-04 (configurable model path/providers, CPU default, removed local demo) |
| GIPHY Celebrity Detector | [GCD](https://github.com/Giphy/celeb-detection-oss) | External dependency, MPL-2.0; its source and weights are not bundled |

The authors' dataset and GCD wrapper scripts are provided in `scripts/prepare/`
and `scripts/evaluate/`. GCD retains its own license in the user's checkout.
