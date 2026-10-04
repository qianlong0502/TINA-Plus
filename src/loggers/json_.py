"""Per-sample output writer. Completion is marked only after all outputs exist."""
import json
from pathlib import Path

import torch


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class JSONLogger:
    def __init__(self, root, config):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "noises").mkdir(exist_ok=True)
        write_json(self.root / "config.json", config)

    def image(self, name, image):
        path = self.root / "images" / f"{name}.png"
        image.save(path)
        return path

    def finish(self, baseline, attack=None, noise=None, index=None):
        records = [{"success": baseline["success"], "baseline": baseline}]
        if attack is not None:
            records.append({"attack_success": attack["success"], "attack": attack})
        if noise is not None:
            torch.save(noise.detach().cpu(), self.root / "noises" / f"optimized_noise_{index}.pt")
        write_json(self.root / "log.json", records)
        write_json(self.root / "complete.json", {"status": "complete"})
