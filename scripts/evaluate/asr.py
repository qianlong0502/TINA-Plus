#!/usr/bin/env python3
"""Strict, ID-aligned SD1.4 attack success aggregation (stdlib only)."""
import argparse
import csv
import json
from pathlib import Path


def load_run(root, expected_count):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    indices = manifest["expected_indices"]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("invalid expected indices in manifest")
    if expected_count != len(indices):
        raise ValueError(f"expected {expected_count} samples, manifest has {len(indices)}")
    samples = {}
    for index in indices:
        sample = root / f"attack_idx_{index}"
        if not (sample / "complete.json").is_file():
            raise ValueError(f"incomplete sample: {sample}")
        config = json.loads((sample / "config.json").read_text())
        if config["attack_idx"] != index:
            raise ValueError(f"sample ID mismatch: {sample}")
        for key, value in manifest["config"].items():
            if config.get(key) != value:
                raise ValueError(f"sample configuration differs from manifest: {sample}: {key}")
        logs = json.loads((sample / "log.json").read_text())
        if not logs or len(logs) > 2:
            raise ValueError(f"invalid logs: {sample}")
        if not (sample / "images" / "orig.png").is_file():
            raise FileNotFoundError(sample / "images" / "orig.png")
        if len(logs) == 2 and not (sample / "images" / "generated.png").is_file():
            raise FileNotFoundError(sample / "images" / "generated.png")
        samples[index] = (config, logs, sample)
    return samples


def write_summary(rows, output, overwrite):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    pre = sum(row["pre_success"] for row in rows)
    added = sum(row["attack_success"] and not row["pre_success"] for row in rows)
    summary = {"count": n, "pre_success": pre, "additional_success": added,
               "merged_success": pre + added, "pre_asr_percent": 100 * pre / n,
               "merged_asr_percent": 100 * (pre + added) / n,
               "attack_asr_on_baseline_failures_percent": 100 * added / (n - pre) if n > pre else None}
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--input-format", required=True, choices=["dir"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-format", required=True, choices=["dir"])
    parser.add_argument("--output-overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--input-no-attack-path", help="optional independently generated baseline")
    args = parser.parse_args()
    samples = load_run(args.input_path, args.expected_count)
    baselines = load_run(args.input_no_attack_path, args.expected_count) if args.input_no_attack_path else None
    if baselines is not None and samples.keys() != baselines.keys():
        raise ValueError("attack/baseline ID sets differ")
    rows = []
    for index, (config, logs, _) in sorted(samples.items()):
        if config["concept"] in ("taylor_swift", "elon_musk", "adam_lambert"):
            raise ValueError("use gcd_asr.py for celebrity identities")
        baseline = logs[0]["baseline"]["success"]
        attack = logs[1]["attack"]["success"] if len(logs) == 2 else False
        if type(baseline) is not bool or type(attack) is not bool:
            raise ValueError(f"missing boolean classifier decision at sample {index}")
        if config["attacker"] == "tina" and not baseline and len(logs) != 2:
            raise ValueError(f"missing attack result at sample {index}")
        if baselines is not None:
            base_config, base_logs, _ = baselines[index]
            for key in ("case_number", "source_row", "target_sha256", "prompts_sha256", "concept",
                        "base_model_name_or_path", "checkpoint_sha256", "checkpoint_type", "eval_seed", "sampling_step_num", "seed"):
                if config[key] != base_config[key]:
                    raise ValueError(f"baseline mismatch at {index}: {key}")
            for key in ("runtime_versions", "detector_sha256", "classifier_dir", "nudenet_onnx_path"):
                if config.get(key) != base_config.get(key):
                    raise ValueError(f"baseline evaluator mismatch at {index}: {key}")
            if base_config["attacker"] != "no_attack" or baseline != base_logs[0]["baseline"]["success"]:
                raise ValueError(f"baseline result differs at {index}")
        rows.append({"attack_idx": index, "source_row": config["source_row"],
                     "case_number": config["case_number"], "pre_success": int(baseline),
                     "attack_success": int(attack), "merged_success": int(baseline or attack)})
    write_summary(rows, args.output_path, args.output_overwrite)


if __name__ == "__main__":
    main()
