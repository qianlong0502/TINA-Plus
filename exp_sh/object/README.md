# Object attack

SD1.4 object erasure for Church, Garbage Truck, Parachute and Tench, 50 targets
per concept. Parachute here uses SD1.4.

## Run

Example for Church × STEREO, from the repository root:

```bash
CONCEPT=church bash exp_sh/object/00_prepare.sh
CONCEPT=church DEFENSE=stereo CKPT=/path/to/stereo_church.pt bash exp_sh/object/01_attack.sh
CONCEPT=church DEFENSE=stereo bash exp_sh/object/02_evaluate.sh
```

## Workflow

| Script | Role | Dependencies | Outputs |
|---|---|---|---|
| `00_prepare.sh` | Generate pretrained targets, 25 LMS steps | SD1.4, concept CSV | `data/<concept>/` |
| `01_attack.sh` | No Attack + TINA+ | Targets, erased checkpoint, ResNet-50 | Per-sample results |
| `02_evaluate.sh` | ID-aligned union ASR | Complete outputs | CSV and JSON summary |

## Settings

Select `CONCEPT=church|garbage_truck|parachute|tench` (default Tench).
Research coverage: ESD, EraseDiff, FMN, SalUN, Scissorhands, SPM and STEREO for
all four objects; AdvUnlearn additionally for Church/Garbage Truck/Parachute.
The checkpoint template includes AdvUnlearn for Church, Garbage Truck and
Parachute; it has no Tench/AdvUnlearn entry.

Choose a checkpoint with `CKPT=...` and label its outputs with `DEFENSE=...`.
AdvUnlearn requires `CHECKPOINT_TYPE=text_encoder`.
Set `TINA_RESNET50_MODEL_PATH` for an offline HF ResNet-50 directory if needed.

Top-1 ImageNet class IDs: Tench 0; Church 497; Garbage Truck 569; Parachute 701.
Baseline-success samples skip inversion. [Evaluation](../../docs/evaluation.md)
defines the merged ASR. The [root README](../../README.md) provides examples.
