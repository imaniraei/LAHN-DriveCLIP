import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora import inject_lora

class DualEncoderCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def encode_image(self, images):
        return F.normalize(self.clip.encode_image(images), dim=-1)

    def encode_text(self, texts):
        return F.normalize(self.clip.encode_text(texts), dim=-1)

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        scale = self.clip.logit_scale.exp().clamp(max=100)
        logits = scale * image_features @ text_features.T
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits,
            "logits_per_text": logits.T,
        }

def create_model(model_name, pretrained, lora_cfg=None):
    base, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = DualEncoderCLIP(base)
    report = None
    if lora_cfg and lora_cfg.get("enabled", True):
        report = inject_lora(
            model.clip,
            lora_cfg["target_patterns"],
            int(lora_cfg.get("rank", 8)),
            float(lora_cfg.get("alpha", 16)),
            float(lora_cfg.get("dropout", 0)),
        )
        if lora_cfg.get("train_logit_scale", False):
            model.clip.logit_scale.requires_grad = True
    return model, preprocess, tokenizer, report
