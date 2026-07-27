import argparse
from pathlib import Path
import torch
from PIL import Image
from matplotlib import pyplot as plt
from lahn_driveclip.localization.gscorecam_adapter import GScoreCAMAdapter
from lahn_driveclip.models.clip import create_model
from lahn_driveclip.models.targets import get_last_visual_transformer_block

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/demo.png")
    parser.add_argument("--gscorecam-path", default="gScoreCAM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora = {
        "enabled": True, "rank": 8, "alpha": 16, "dropout": 0.05,
        "target_patterns": [
            r"^visual\.transformer\.resblocks\..*\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)$",
            r"^transformer\.resblocks\..*\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)$",
        ],
    }
    model, preprocess, tokenizer, _ = create_model("ViT-B-16", "openai", lora)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    tokens = tokenizer([args.prompt]).to(device)
    localizer = GScoreCAMAdapter(
        args.gscorecam_path, model, preprocess, tokenizer,
        get_last_visual_transformer_block(model)
    )
    saliency = localizer(image_tensor, tokens, image.size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.imshow(saliency, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title(args.prompt)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")

if __name__ == "__main__":
    main()
