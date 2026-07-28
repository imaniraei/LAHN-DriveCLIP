import math, re
from dataclasses import dataclass
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base_layer(x) + ((self.dropout(x) @ self.lora_A.T) @ self.lora_B.T) * self.scaling

@dataclass
class LoRAReport:
    replaced_modules: list
    trainable_parameters: int
    total_parameters: int

def _set_child(root, qualified_name, module):
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)

def inject_lora(model, target_patterns, rank=8, alpha=16.0, dropout=0.0):
    for p in model.parameters():
        p.requires_grad = False
    patterns = [re.compile(x) for x in target_patterns]
    candidates = list(model.named_modules())
    replaced = []
    for name, module in candidates:
        if isinstance(module, nn.Linear) and any(p.search(name) for p in patterns):
            _set_child(model, name, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    if not replaced:
        names = [n for n, m in candidates if isinstance(m, nn.Linear)]
        raise RuntimeError("No LoRA target matched. Run tools/inspect_model.py.\n" + "\n".join(names[:40]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return LoRAReport(replaced, trainable, total)

def lora_state_dict(model):
    return {n: t.detach().cpu() for n, t in model.state_dict().items()
            if "lora_A" in n or "lora_B" in n}
