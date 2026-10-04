#!/usr/bin/env python3
"""Generate SD1.4 target images and a filtered-index manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--input-format", required=True, choices=["csv"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-format", required=True, choices=["dir"])
    parser.add_argument("--output-overwrite", type=int, default=0, choices=[0, 1])
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--guidance-source", choices=["legacy", "row"], default="legacy",
                        help="legacy: sd_guidance_scale or --guidance; row: evaluation_guidance")
    parser.add_argument("--filter-long-prompts", type=int, choices=[0, 1], default=0,
                        help="exclude >60-token rows from attack indices (Nudity protocol)")
    parser.add_argument("--cache-path", default=".cache")
    args = parser.parse_args()
    source, output = Path(args.input_path).resolve(), Path(args.output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output or output in source.parents:
        raise ValueError("output must not contain the input CSV")
    if output.exists() and any(output.iterdir()) and not args.output_overwrite:
        raise FileExistsError(output)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    if args.sampling_steps < 1:
        raise ValueError("sampling steps must be positive")
    table = pd.read_csv(source, keep_default_na=False)
    if table.empty or "prompt" not in table:
        raise ValueError("prompt CSV must contain non-empty prompt rows")
    if "case_number" not in table:
        table["case_number"] = range(len(table))
    table["case_number"] = table.case_number.astype(int)
    if table.case_number.duplicated().any():
        raise ValueError("duplicate case numbers")
    seeds = table["evaluation_seed"] if "evaluation_seed" in table else table.get("sd_seed", table.case_number)
    table["evaluation_seed"] = seeds.astype(int)
    if "evaluation_guidance" not in table:
        table["evaluation_guidance"] = args.guidance
    load = dict(cache_dir=args.cache_path)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name_or_path, subfolder="tokenizer", **load)
    vae = AutoencoderKL.from_pretrained(args.model_name_or_path, subfolder="vae", **load).to(args.device)
    encoder = CLIPTextModel.from_pretrained(args.model_name_or_path, subfolder="text_encoder", **load).to(args.device)
    unet = UNet2DConditionModel.from_pretrained(args.model_name_or_path, subfolder="unet", **load).to(args.device)
    for model in (vae, encoder, unet):
        model.eval().requires_grad_(False)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                     beta_schedule="scaled_linear", num_train_timesteps=1000)
    ignored = [i for i, prompt in enumerate(table.prompt) if args.filter_long_prompts and len(tokenizer(str(prompt))["input_ids"]) > 60]
    images = output / "imgs"
    images.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.no_grad():
        for row_index, row in table.iterrows():
            guidance = float(row.evaluation_guidance) if args.guidance_source == "row" else float(row.get("sd_guidance_scale", "") or args.guidance)
            prompt = str(row.prompt)
            text_ids = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(args.device)
            null_ids = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                                 return_tensors="pt").input_ids.to(args.device)
            context = torch.cat([encoder(null_ids)[0], encoder(text_ids)[0]])
            torch.manual_seed(int(row.evaluation_seed))
            latents = torch.randn((1, 4, 64, 64)).to(args.device)
            scheduler.set_timesteps(args.sampling_steps)
            latents *= scheduler.init_noise_sigma
            for timestep in scheduler.timesteps:
                model_input = scheduler.scale_model_input(torch.cat([latents] * 2), timestep=timestep)
                noise = unet(model_input, timestep, encoder_hidden_states=context).sample
                unconditional, conditional = noise.chunk(2)
                noise = unconditional + guidance * (conditional - unconditional)
                latents = scheduler.step(noise, timestep, latents).prev_sample
            decoded = (vae.decode(latents / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            array = (decoded.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).round().astype(np.uint8)
            image_path = images / f"{int(row.case_number)}_0.png"
            Image.fromarray(array).save(image_path)
            manifest.append({"source_row": row_index, "case_number": int(row.case_number),
                             "attack_idx": "" if row_index in ignored else row_index - sum(i < row_index for i in ignored),
                             "prompt": prompt, "evaluation_seed": int(row.evaluation_seed),
                             "evaluation_guidance": float(row.evaluation_guidance),
                             "generation_guidance": guidance,
                             "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
            print(f"generated case {int(row.case_number)}")
    table.to_csv(output / "prompts.csv", index=False)
    (output / "ignore.json").write_text(json.dumps(ignored) + "\n")
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output / "generation.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
