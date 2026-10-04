"""SD1.4 targets with stable mappings between attack indices and CSV rows."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer


class PNGImageDataset:
    def __init__(self, root_dir, model_name_or_path, cache_dir=None, filter_long_prompts=False):
        self.root = Path(root_dir)
        self.data = pd.read_csv(self.root / "prompts.csv", keep_default_na=False)
        if self.data.empty or "prompt" not in self.data:
            raise ValueError("prompts.csv must contain non-empty prompt rows")
        if "case_number" not in self.data:
            self.data["case_number"] = range(len(self.data))
        self.data["case_number"] = self.data["case_number"].astype(int)
        if self.data.case_number.duplicated().any():
            raise ValueError("one unique case_number per CSV row is required")
        ignore_path = self.root / "ignore.json"
        if not filter_long_prompts:
            ignored = []
        elif ignore_path.is_file():
            ignored = json.loads(ignore_path.read_text())
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_name_or_path, subfolder="tokenizer", cache_dir=cache_dir
            )
            ignored = [i for i, prompt in enumerate(self.data.prompt)
                       if len(tokenizer(str(prompt))["input_ids"]) > 60]
        if (not isinstance(ignored, list) or
                any(type(i) is not int or not 0 <= i < len(self.data) for i in ignored)):
            raise ValueError("ignore.json must be a list of valid CSV row indices")
        self.ignored = ignored
        self.idxs = [i for i in range(len(self.data)) if i not in set(ignored)]
        if not self.idxs:
            raise ValueError("no targets remain after prompt filtering")
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        for row_index in self.idxs:
            if not self.image_path(row_index).is_file():
                raise FileNotFoundError(self.image_path(row_index))

    def image_path(self, row_index):
        case = int(self.data.iloc[row_index].case_number)
        return self.root / "imgs" / f"{case}_0.png"

    def __len__(self):
        return len(self.idxs)

    def identity(self, idx):
        row_index = self.idxs[idx]
        return {"source_row": row_index,
                "case_number": int(self.data.iloc[row_index].case_number)}

    def __getitem__(self, idx):
        row_index = self.idxs[idx]
        row = self.data.iloc[row_index]
        with Image.open(self.image_path(row_index)) as image:
            image = self.transform(image.convert("RGB"))
        seed = int(row.evaluation_seed) if str(row.get("evaluation_seed", "")) else None
        guidance = float(row.get("evaluation_guidance", "") or 7.5)
        return image, str(row.prompt), seed, guidance
