# Nudity attack

This group reproduces SD1.4 attacks on the I2P nudity prompt subset.

## Run

Set `NUDENET_ONNX_PATH` in your local paths file. Run from the repository root:

```bash
bash exp_sh/nudity/00_prepare.sh
CKPT=/path/to/esd_nudity.pt bash exp_sh/nudity/01_attack.sh
bash exp_sh/nudity/02_evaluate.sh
```

## Workflow

| Script | Role | Dependencies | Outputs |
|---|---|---|---|
| `00_prepare.sh` | Generate 142 targets; export token filter mapping | Base SD1.4 | `data/nudity/` |
| `01_attack.sh` | No Attack + TINA+ on 118 effective targets | Dataset, erased checkpoint, `best.onnx` | Per-sample results |
| `02_evaluate.sh` | Full-set ASR with ID-matched baseline | Complete outputs | CSV and JSON summary |

## Settings

Research coverage: ESD, FMN, UCE, MACE, RECE, AdvUnlearn, SalUN, STEREO and SPM.
The paper's main nudity table uses eight defenses, excluding SPM. Use `DEFENSE`
for output naming and supply matching `CKPT`; AdvUnlearn uses
`CHECKPOINT_TYPE=text_encoder`.

The `NUDENET_ONNX_PATH` weight is external. Detector classes, internal NMS/score
thresholds and outer `if_nude` threshold are documented in
[evaluation](../../docs/evaluation.md). Original CSV row IDs differ from filtered
attack IDs; never infer the retained set from `0..117` in the raw CSV.

Report `pre_asr_percent`, `additional_success` and `merged_asr_percent`.
The evaluator rejects incomplete results. See [root commands](../../README.md)
and [resources](../../docs/resources.md).
