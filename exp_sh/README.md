# Attack reproduction scripts

Scripts define the SD1.4 attack/evaluation workflow. Code details and command
contracts are documented in [commands](../docs/commands.md).

| Group | Scope | Entry |
|---|---|---|
| style | Van Gogh | [README](style/README.md) |
| nudity | I2P nudity subset | [README](nudity/README.md) |
| object | Church, Garbage Truck, Parachute, Tench | [README](object/README.md) |
| celebrity | Taylor Swift, Elon Musk, Adam Lambert | [README](celebrity/README.md) |

Run `00_prepare.sh`, then `01_attack.sh`, then `02_evaluate.sh` in a group.
The attack script runs No Attack and TINA+ sequentially. Use the same `CONCEPT`
and `DEFENSE` for attack and evaluation. `DEFENSE` names the output subdirectory;
it does not select/download weights. `CKPT` and `CHECKPOINT_TYPE` select weights.

## Shared Settings

Common scripts `prepare.sh`, `run_attack.sh`, `evaluate.sh` implement shared
commands, avoiding copies of the same settings. `_env.sh` sources
`configs/paths.local.sh` and changes to the repository root.

`ATTACK_IDX_START` and `ATTACK_IDX_END` select a half-open range for attack.
Evaluation expects the complete task (50, or 118 for nudity); partial outputs
cannot be presented as full benchmark ASR. Repeated disjoint invocations with
identical settings extend the run manifest. Do not run overlapping writers in
the same output tree concurrently.

## Multiple Defenses

For a complete defense matrix, copy `configs/checkpoints.example.csv` to
`configs/checkpoints.local.csv` and replace its example paths with real files.
The template records the research script coverage (54 concept/defense cells),
including SPM nudity and excluding the undocumented Tench/AdvUnlearn cell.
Paths must not contain commas; quote-free CSV is expected by the shell reader.
Prepare all selected datasets first, then run `bash exp_sh/run_matrix.sh`.
Use `MATRIX_CONCEPT=tench` or `MATRIX_DEFENSE=esd` to select a subset. Runs and
evaluations are sequential; missing checkpoints and failed commands stop the
matrix immediately. Keep the optional local table out of Git.

## GPU Runtime

GPU call chain: group wrapper → shared script → `_env.sh` →
`scripts/gpu_env.sh` (exports + CUDA probe) → Python workload. Celebrity generation
uses the same runtime initialization; GCD evaluation is CPU only. There are no
cluster submission or machine-specific scheduling scripts.
