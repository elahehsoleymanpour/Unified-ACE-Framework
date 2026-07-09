import torch
import torch.nn as nn
import math

class TopologicalFeatureExtractor(nn.Module):
    """Extracts non-parametric density using a differentiable Gaussian Kernel."""
    def __init__(self, device, n_points=100):
        super().__init__()
        self.device = device
        self.eval_points = torch.linspace(-4, 4, n_points).to(device)

    def forward(self, window_data: torch.Tensor) -> torch.Tensor:
        n = window_data.shape[0]
        sigma = torch.std(window_data) + 1e-8
        h = torch.clamp(1.06 * sigma * (n ** (-0.2)), min=1e-4) # Silverman's Adaptive Rule
        u = (self.eval_points.view(-1, 1) - window_data.view(1, -1)) / h
        kernel_vals = (1 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * (u ** 2))
        density = torch.sum(kernel_vals, dim=1) / (n * h)
        return density / torch.sum(density)
