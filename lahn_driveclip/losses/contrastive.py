import torch
import torch.nn.functional as F

def symmetric_clip_loss(logits_per_image, logits_per_text):
    if logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ValueError("One aligned text per image is required.")
    targets = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return 0.5 * (
        F.cross_entropy(logits_per_image, targets) +
        F.cross_entropy(logits_per_text, targets)
    )
