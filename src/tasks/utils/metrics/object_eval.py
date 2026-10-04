import os
from pathlib import Path

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch


def _resolve_resnet50_source():
    """Local dir from TINA_RESNET50_MODEL_PATH, else HuggingFace hub id."""
    env_path = os.environ.get("TINA_RESNET50_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_dir():
            raise FileNotFoundError(
                f"TINA_RESNET50_MODEL_PATH is set but not a directory: {env_path}"
            )
        return str(p.resolve()), True
    return "microsoft/resnet-50", False


def imagenet_ResNet50(device):
    model_id_or_path, local_only = _resolve_resnet50_source()
    cache_dir = os.environ.get("TINA_RESNET50_CACHE_DIR", ".cache")
    load_kw = {"cache_dir": cache_dir}
    if local_only:
        load_kw["local_files_only"] = True

    processor = AutoImageProcessor.from_pretrained(model_id_or_path, **load_kw)
    model = ResNetForImageClassification.from_pretrained(model_id_or_path, **load_kw)
    model.to(device)
    return processor, model


def object_eval(classifier, img, processor, device):
    with torch.no_grad():
        inputs = processor(img, return_tensors="pt")
        inputs.to(device)
        logits = classifier(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return predicted_label, torch.softmax(logits, dim=-1).squeeze()
