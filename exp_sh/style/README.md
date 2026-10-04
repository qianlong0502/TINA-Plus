# Van Gogh style attack

This group reproduces SD1.4 style-erasure attacks, using 50 Van Gogh targets.

## Run

Set `STYLE_CLASSIFIER` in your local paths file. Run from the repository root:

```bash
bash exp_sh/style/00_prepare.sh
CKPT=/path/to/esd_vangogh.pt bash exp_sh/style/01_attack.sh
bash exp_sh/style/02_evaluate.sh
```

## Workflow

| Script | Role | Dependencies | Outputs |
|---|---|---|---|
| `00_prepare.sh` | Pretrained SD target generation, 25 LMS steps | Base SD1.4 | `data/vangogh/` |
| `01_attack.sh` | No Attack + TINA+ | Dataset, erased checkpoint, WikiArt classifier | Per-sample results |
| `02_evaluate.sh` | Top-1 baseline/attack union ASR | Complete outputs from attack | CSV and JSON summary |

## Settings

Defense coverage of the research scripts: ESD, FMN, AC (concept-ablation), SPM,
RECE, MACE, AdvUnlearn and STEREO. Select one using `DEFENSE=<slug>` and provide
its weights with `CKPT=...`; AdvUnlearn uses `CHECKPOINT_TYPE=text_encoder`.

The classifier must be the matching 129-class WikiArt checkpoint. Top-1
`vincent-van-gogh` is success. All targets undergo inversion, even if the baseline
is already successful. Report `merged_asr_percent` for the reference table.

See [resources](../../docs/resources.md) for model paths and
[evaluation](../../docs/evaluation.md) for ASR definitions.
