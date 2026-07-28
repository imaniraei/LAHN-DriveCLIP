import numpy as np
import torch
from lahn_driveclip.metrics.retrieval import retrieval_metrics
from lahn_driveclip.metrics.localization import localization_metrics

def test_retrieval():
    assert all(v == 1.0 for v in retrieval_metrics(torch.eye(4)).values())

def test_localization():
    sal = np.zeros((10,10), dtype=float)
    sal[2:6,3:7] = 1
    m = localization_metrics(sal, (3,2,7,6))
    assert m["ebpg"] == 1.0
    assert m["point_accuracy"] == 1.0
    assert m["iou_0.5"] == 1.0
