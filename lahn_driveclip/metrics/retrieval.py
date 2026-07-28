import torch

def recall_at_k(similarity, k, direction):
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Expected a square paired similarity matrix.")
    scores = similarity if direction == "image_to_text" else similarity.T
    topk = scores.topk(min(k, scores.shape[1]), dim=1).indices
    target = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    return float((topk == target).any(dim=1).float().mean())

def retrieval_metrics(similarity):
    return {
        "image_to_text_top1": recall_at_k(similarity, 1, "image_to_text"),
        "image_to_text_top2": recall_at_k(similarity, 2, "image_to_text"),
        "text_to_image_top1": recall_at_k(similarity, 1, "text_to_image"),
        "text_to_image_top2": recall_at_k(similarity, 2, "text_to_image"),
    }
