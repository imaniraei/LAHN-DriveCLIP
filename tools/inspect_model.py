import argparse
from lahn_driveclip.models.clip import create_model

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", default="ViT-B-16")
parser.add_argument("--pretrained", default="openai")
args = parser.parse_args()
model, _, _, _ = create_model(args.model_name, args.pretrained)
for name, module in model.clip.named_modules():
    if module.__class__.__name__ == "Linear":
        print(name)
