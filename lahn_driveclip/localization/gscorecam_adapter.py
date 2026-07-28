from pathlib import Path
import sys
import numpy as np
import torch

class GScoreCAMAdapter:
    def __init__(self, repo_path, clip_model, preprocess, tokenizer, target_layer,
                 cam_version="gscorecam", topk=300):
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError("Run: bash scripts/setup_gscorecam.sh")
        sys.path.insert(0, str(repo_path))
        from tools.cam import CAMWrapper

        visual = clip_model.clip.visual
        image_size = getattr(visual, "image_size", 224)
        image_size = image_size[0] if isinstance(image_size, tuple) else image_size
        patch_size = getattr(visual, "patch_size", 16)
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        grid = int(image_size) // int(patch_size)

        def reshape(tensor):
            if tensor.shape[0] == 1 + grid * grid:
                tensor = tensor.permute(1, 0, 2)
            tensor = tensor[:, 1:, :]
            return tensor.reshape(tensor.shape[0], grid, grid, tensor.shape[2]).permute(0, 3, 1, 2)

        class Compatible(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def forward(self, images, texts):
                out = self.wrapped(images, texts)
                return out["logits_per_image"], out["logits_per_text"]

        self.cam = CAMWrapper(
            Compatible(clip_model), preprocess=preprocess,
            target_layers=[target_layer], tokenizer=tokenizer, drop=True,
            cam_version=cam_version.lower(), topk=topk, channels=None,
            is_transformer=True, cam_trans=reshape
        )

    def __call__(self, image_tensor, text_tokens, raw_size):
        return np.clip(np.asarray(
            self.cam.getCAM(image_tensor, text_tokens, raw_size, 0),
            dtype=np.float32
        ), 0, 1)
