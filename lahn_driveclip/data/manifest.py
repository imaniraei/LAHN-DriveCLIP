from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    text: str
    bbox_xyxy: tuple | None = None
    split: str | None = None
    dataset: str | None = None
    metadata: dict | None = None

def load_jsonl_manifest(path):
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            bbox = r.get("bbox_xyxy")
            try:
                samples.append(Sample(
                    sample_id=str(r["sample_id"]),
                    image_path=str(r["image_path"]),
                    text=str(r["text"]),
                    bbox_xyxy=tuple(map(float, bbox)) if bbox is not None else None,
                    split=r.get("split"),
                    dataset=r.get("dataset"),
                    metadata=r.get("metadata", {}),
                ))
            except Exception as exc:
                raise ValueError(f"Invalid manifest record at line {line_no}") from exc
    return samples

def save_jsonl_manifest(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "sample_id": s.sample_id,
                "image_path": s.image_path,
                "text": s.text,
                "bbox_xyxy": list(s.bbox_xyxy) if s.bbox_xyxy else None,
                "split": s.split,
                "dataset": s.dataset,
                "metadata": s.metadata or {},
            }, ensure_ascii=False) + "\n")
