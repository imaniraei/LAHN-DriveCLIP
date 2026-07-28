import numpy as np

def bbox_mask(shape, bbox):
    h, w = shape
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask

def ebpg(saliency, bbox, eps=1e-12):
    saliency = np.maximum(np.asarray(saliency, dtype=float), 0)
    mask = bbox_mask(saliency.shape, bbox)
    return float(saliency[mask].sum() / (saliency.sum() + eps))

def point_accuracy(saliency, bbox):
    y, x = np.unravel_index(np.asarray(saliency).argmax(), saliency.shape)
    x1, y1, x2, y2 = bbox
    return float(x1 <= x < x2 and y1 <= y < y2)

def localization_metrics(saliency, bbox, threshold_ratio=0.5):
    saliency = np.asarray(saliency, dtype=float)
    pred = saliency >= threshold_ratio * max(float(saliency.max()), 1e-12)
    target = bbox_mask(saliency.shape, bbox)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = float(inter / max(union, 1))
    return {
        "ebpg": ebpg(saliency, bbox),
        "iou": iou,
        "iou_0.5": float(iou >= 0.5),
        "point_accuracy": point_accuracy(saliency, bbox),
    }
