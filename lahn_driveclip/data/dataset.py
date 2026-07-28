from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from .manifest import load_jsonl_manifest

class VisionLanguageDataset(Dataset):
    def __init__(self, manifest_path, preprocess, tokenizer, image_root=None):
        self.samples = load_jsonl_manifest(manifest_path)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = Path(sample.image_path)
        if not path.is_absolute() and self.image_root:
            path = self.image_root / path
        if not path.exists():
            raise FileNotFoundError(path)
        image = self.preprocess(Image.open(path).convert("RGB"))
        text = self.tokenizer([sample.text])[0]
        bbox = (torch.tensor(sample.bbox_xyxy, dtype=torch.float32)
                if sample.bbox_xyxy is not None
                else torch.full((4,), float("nan")))
        return {
            "image": image, "text": text, "bbox_xyxy": bbox,
            "sample_id": sample.sample_id, "raw_text": sample.text,
            "image_path": str(path),
        }
