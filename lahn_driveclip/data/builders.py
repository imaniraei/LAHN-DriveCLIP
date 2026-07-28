from pathlib import Path
import csv, json
from .manifest import Sample, save_jsonl_manifest

def first(row, names, default=""):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return default

def build_bddx_manifest(annotation_csv, split_txt, image_root, output_jsonl,
                        split_name, frame_template="{video_id}.jpg"):
    allowed = {x.strip() for x in Path(split_txt).read_text().splitlines() if x.strip()}
    samples = []
    with Path(annotation_csv).open("r", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            video_id = str(first(row, ["video_id","video","vid","file_name","name"])).strip()
            stem = Path(video_id).stem
            if allowed and video_id not in allowed and stem not in allowed:
                continue
            action = str(first(row, ["action","description","caption","sentence"])).strip()
            explanation = str(first(row, ["justification","explanation","reason"])).strip()
            text = " ".join(x for x in [action, explanation] if x)
            if not video_id or not text:
                continue
            samples.append(Sample(
                f"bddx-{split_name}-{i:06d}",
                str(Path(image_root) / frame_template.format(video_id=stem)),
                text, split=split_name, dataset="bddx",
                metadata={"video_id": video_id}
            ))
    save_jsonl_manifest(samples, output_jsonl)
    return len(samples)

def build_talk2car_manifest(commands_json, image_root, output_jsonl, split_name):
    payload = json.loads(Path(commands_json).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("commands", payload.get("data", []))
    samples = []
    for i, row in enumerate(records):
        text = first(row, ["command","text","description","caption"])
        if isinstance(text, dict):
            text = first(text, ["command","text"])
        image_name = first(row, ["image","image_name","file_name","img","frame"])
        bbox = first(row, ["bbox","box","bbox_xyxy"], None)
        if not text or not image_name:
            continue
        bbox_xyxy = None
        if bbox is not None and len(bbox) == 4:
            values = list(map(float, bbox))
            if row.get("bbox_format") == "xyxy":
                bbox_xyxy = tuple(values)
            else:
                x, y, w, h = values
                bbox_xyxy = (x, y, x+w, y+h)
        samples.append(Sample(
            str(first(row, ["id","command_id","sample_id"], f"talk2car-{i:06d}")),
            str(Path(image_root) / str(image_name)),
            str(text), bbox_xyxy, split_name, "talk2car", {"source": row}
        ))
    save_jsonl_manifest(samples, output_jsonl)
    return len(samples)

def build_localization_csv_manifest(csv_path, output_jsonl, dataset_name,
                                    split_name, image_root=None):
    root = Path(image_root) if image_root else None
    samples = []
    with Path(csv_path).open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image_path","text","x1","y1","x2","y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            path = Path(row["image_path"])
            if root and not path.is_absolute():
                path = root / path
            samples.append(Sample(
                row.get("sample_id") or f"{dataset_name}-{split_name}-{i:06d}",
                str(path), row["text"],
                tuple(float(row[k]) for k in ("x1","y1","x2","y2")),
                split_name, dataset_name, {}
            ))
    save_jsonl_manifest(samples, output_jsonl)
    return len(samples)
