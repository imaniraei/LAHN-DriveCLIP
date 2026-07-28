import torch.nn as nn
from lahn_driveclip.models.lora import inject_lora, LoRALinear

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(4,4))
        self.text = nn.Sequential(nn.Linear(4,4))

def test_lora():
    model = Tiny()
    report = inject_lora(model, [r"visual\.0", r"text\.0"], rank=2)
    assert len(report.replaced_modules) == 2
    assert isinstance(model.visual[0], LoRALinear)
    assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
