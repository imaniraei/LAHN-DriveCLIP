import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from lahn_driveclip.data.manifest import load_jsonl_manifest
from lahn_driveclip.localization.gradcam import CLIPGradCAM
from lahn_driveclip.localization.gscorecam_adapter import GScoreCAMAdapter
from lahn_driveclip.metrics.localization import localization_metrics
from lahn_driveclip.models.clip import create_model
from lahn_driveclip.models.targets import get_last_visual_transformer_block
from lahn_driveclip.utils.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["gradcam","gscorecam"], default="gscorecam")
    parser.add_argument("--gscorecam-path", default="gScoreCAM")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer, _ = create_model(
        cfg["model"]["name"], cfg["model"]["pretrained"], cfg["lora"]
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()
    target = get_last_visual_transformer_block(model)

    if args.method == "gradcam":
        localizer = CLIPGradCAM(model, target)
    else:
        localizer = GScoreCAMAdapter(
            args.gscorecam_path, model, preprocess, tokenizer, target,
            cfg["localization"].get("cam_version", "gscorecam"),
            int(cfg["localization"].get("topk_channels", 300))
        )

    samples = load_jsonl_manifest(cfg["data"]["localization_manifest"])
    if args.limit:
        samples = samples[:args.limit]
    rows = []
    for sample in tqdm(samples):
        if sample.bbox_xyxy is None:
            continue
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tokens = tokenizer([sample.text]).to(device)
        if args.method == "gradcam":
            saliency = localizer(
                image_tensor, text_tokens, (image.height, image.width)
            ).cpu().numpy()
        else:
            saliency = localizer(image_tensor, text_tokens, image.size)
        rows.append({
            "sample_id": sample.sample_id,
            **localization_metrics(
                saliency, sample.bbox_xyxy,
                float(cfg["localization"].get("threshold_ratio", 0.5))
            )
        })

    if not rows:
        raise RuntimeError("No samples with bounding boxes were evaluated.")
    keys = ["ebpg","iou","iou_0.5","point_accuracy"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    result = {"summary": summary, "samples": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))
    if args.method == "gradcam":
        localizer.close()

if __name__ == "__main__":
    main()
