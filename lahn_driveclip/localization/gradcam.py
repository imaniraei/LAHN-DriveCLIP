import torch
import torch.nn.functional as F

def tokens_to_map(tensor):
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D activation, got {tuple(tensor.shape)}")
    if tensor.shape[0] > tensor.shape[1] and tensor.shape[1] <= 16:
        tensor = tensor.permute(1, 0, 2)
    tokens = tensor[:, 1:, :]
    grid = int(tokens.shape[1] ** 0.5)
    if grid * grid != tokens.shape[1]:
        raise ValueError("Patch token count is not square.")
    return tokens.reshape(tokens.shape[0], grid, grid, tokens.shape[2]).permute(0, 3, 1, 2)

class CLIPGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._save_activation)
        self.bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image, text, output_size):
        self.model.zero_grad(set_to_none=True)
        score = self.model(image, text)["logits_per_image"][0, 0]
        score.backward()
        a = tokens_to_map(self.activations)
        g = tokens_to_map(self.gradients)
        weights = g.mean((2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, output_size, mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-12)
        return cam.detach()

    def close(self):
        self.fh.remove()
        self.bh.remove()
