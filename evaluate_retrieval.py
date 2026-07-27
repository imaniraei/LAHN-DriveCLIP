import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lahn_driveclip.data.dataset import VisionLanguageDataset
from lahn_driveclip.metrics.retrieval import retrieval_metrics
from lahn_driveclip.models.clip import create_model
from lahn_driveclip.utils.config import load_config

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer, _ = create_model(
        cfg["model"]["name"], cfg["model"]["pretrained"], cfg["lora"]
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()

    dataset = VisionLanguageDataset(
        cfg["data"]["eval_manifest"], preprocess, tokenizer,
        cfg["data"].get("image_root")
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["evaluation"].get("batch_size", 128)),
        shuffle=False, num_workers=int(cfg["evaluation"].get("num_workers", 4))
    )
    images, texts = [], []
    for batch in loader:
        images.append(model.encode_image(batch["image"].to(device)).cpu())
        texts.append(model.encode_text(batch["text"].to(device)).cpu())
    similarity = torch.cat(images) @ torch.cat(texts).T
    result = retrieval_metrics(similarity)
    print(json.dumps(result, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
