#!/usr/bin/env python3
"""Batch TINA+ attacks on SD1.4 concept-erased checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--input-format", required=True, choices=["dir"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-format", required=True, choices=["dir"])
    parser.add_argument("--output-overwrite", type=int, default=0, choices=[0, 1])
    parser.add_argument("--attacker", choices=["tina", "no_attack"], default="tina")
    parser.add_argument("--concept", required=True, choices=["vangogh", "nudity", "church", "garbage_truck",
                        "parachute", "tench", "taylor_swift", "elon_musk", "adam_lambert"])
    parser.add_argument("--base-model-name-or-path", required=True)
    parser.add_argument("--input-checkpoint-path", required=True)
    parser.add_argument("--checkpoint-type", required=True, choices=["unet", "text_encoder"])
    parser.add_argument("--classifier-dir")
    parser.add_argument("--nudenet-onnx-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-path", default=".cache")
    parser.add_argument("--sampling-step-num", type=int, default=50)
    parser.add_argument("--tina-lr", type=float, default=0.001)
    parser.add_argument("--tina-opt-round", type=int, default=25)
    parser.add_argument("--tina-num-ddim-steps", type=int, default=50)
    parser.add_argument("--tina-fmi-ratio", type=float, default=0.1)
    parser.add_argument("--tina-fmi-seed", type=int)
    parser.add_argument("--tina-energy-weight", type=float, default=1.0)
    parser.add_argument("--tina-energy-tau", type=float, default=10.0)
    parser.add_argument("--tina-energy-t-min", type=float, default=0.1)
    parser.add_argument("--tina-energy-t-max", type=float, default=1.0)
    parser.add_argument("--attack-idx-start", type=int)
    parser.add_argument("--attack-idx-end", type=int)
    parser.add_argument("--attack-indices", help="comma-separated filtered dataset indices")
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    return parser.parse_args()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    args = parse_args()
    source = Path(args.input_path).resolve()
    output = Path(args.output_path).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("input and output directories must not contain each other")
    for path in (source / "prompts.csv", Path(args.input_checkpoint_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.concept == "vangogh" and not args.classifier_dir:
        raise ValueError("Van Gogh requires --classifier-dir")
    if args.concept == "nudity" and not args.nudenet_onnx_path:
        raise ValueError("Nudity requires --nudenet-onnx-path")
    if not 0 <= args.tina_fmi_ratio < 1 or not 0 <= args.tina_energy_t_min < args.tina_energy_t_max <= 1:
        raise ValueError("invalid FMI ratio or energy window")
    if min(args.tina_energy_weight, args.tina_energy_tau) < 0 or args.tina_lr <= 0:
        raise ValueError("invalid energy parameters or learning rate")
    if args.tina_opt_round < 1 or not 1 <= args.tina_num_ddim_steps <= 1000 or args.sampling_step_num < 1:
        raise ValueError("step counts must be positive (inversion at most 1000)")
    if round(args.tina_fmi_ratio * args.tina_num_ddim_steps) >= args.tina_num_ddim_steps:
        raise ValueError("FMI must leave at least one inversion step")
    if args.tina_fmi_seed is None:
        args.tina_fmi_seed = args.seed

    import numpy as np
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required; no CPU fallback")
    torch.cuda.get_device_properties(torch.device(args.device))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from src.tasks.classifier_ import ClassifierTask
    from src.attackers.tina_ import TINA
    from src.attackers.no_attack_ import generate
    from src.loggers.json_ import JSONLogger, write_json
    task = ClassifierTask(args)
    size = len(task.dataset)
    if args.attack_indices:
        if args.attack_idx_start is not None or args.attack_idx_end is not None:
            raise ValueError("--attack-indices cannot be combined with start/end")
        indices = [int(value) for value in args.attack_indices.split(",")]
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate attack indices")
    else:
        start = args.attack_idx_start if args.attack_idx_start is not None else 0
        end = args.attack_idx_end if args.attack_idx_end is not None else size
        if not 0 <= start < end <= size:
            raise ValueError(f"invalid range [{start}, {end}) for {size} targets")
        indices = list(range(start, end))
    if not indices or any(index < 0 or index >= size for index in indices):
        raise ValueError(f"attack indices must be in [0, {size})")

    config = vars(args).copy()
    for name in ("output_overwrite", "resume", "attack_idx_start", "attack_idx_end", "attack_indices"):
        config.pop(name)
    config.update({"dataset_size": size, "prompts_sha256": digest(source / "prompts.csv"),
                   "checkpoint_sha256": digest(args.input_checkpoint_path),
                   "ignored_source_rows": task.dataset.ignored,
                   "checkpoint_report": task.checkpoint_report,
                   "runtime_versions": {name: importlib.metadata.version(name)
                       for name in ("torch", "torchvision", "diffusers", "transformers", "numpy", "pandas")}})
    if args.nudenet_onnx_path:
        config["detector_sha256"] = digest(args.nudenet_onnx_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["config"] != config:
            raise ValueError("output configuration differs; use a new output directory")
        intended = sorted(set(old["expected_indices"]) | set(indices))
    else:
        intended = sorted(indices)
    write_json(manifest_path, {"config": config, "expected_indices": intended})
    attacker = TINA(args, task) if args.attacker == "tina" else None
    for index in indices:
        root = output / f"attack_idx_{index}"
        sample_config = {**config, "attack_idx": index, **task.dataset.identity(index),
                         "target_sha256": digest(task.dataset.image_path(task.dataset.idxs[index]))}
        if root.exists():
            existing = root / "config.json"
            if existing.exists() and json.loads(existing.read_text()) != sample_config:
                raise ValueError(f"sample configuration differs: {root}")
            if args.resume and existing.is_file() and (root / "complete.json").is_file():
                print(f"skip completed {index}")
                continue
            if not args.resume and not args.output_overwrite:
                raise FileExistsError(root)
            shutil.rmtree(root)
        logger = JSONLogger(root, sample_config)
        image, prompt, seed, guidance = task.dataset[index]
        seed = args.eval_seed if seed is None else seed
        baseline_image = generate(task, prompt, seed, guidance)
        baseline_path = logger.image("orig", baseline_image)
        baseline = task.score(baseline_image, baseline_path)
        # Style is always inverted, matching the original offline style protocol.
        skip_attack = baseline["success"] is True and args.concept != "vangogh"
        if attacker is None or skip_attack:
            logger.finish(baseline)
        else:
            noise, reconstructed = attacker.invert(image)
            logger.image("reconstructed", reconstructed)
            generated = task.sample("", seed, guidance, initial_noise=noise,
                                    steps=args.tina_num_ddim_steps)
            generated_path = logger.image("generated", generated)
            attack = task.score(generated, generated_path)
            logger.finish(baseline, attack, noise, index)
        print(f"completed {index} ({args.concept})")


if __name__ == "__main__":
    main()
