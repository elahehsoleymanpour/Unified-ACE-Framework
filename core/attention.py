import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSynthesizer(nn.Module):
    """Neural Cross-Attention layer for Parameter Aggregation (APA)."""
    def __init__(self, device, hidden_dim=16):
        super().__init__()
        self.W1 = nn.Linear(6, hidden_dim).to(device)
        self.W2 = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, q, keys, params):
        concat = torch.cat([q.expand(keys.shape[0], -1), keys], dim=1)
        attn = F.softmax(self.W2(torch.tanh(self.W1(concat))).squeeze(-1), dim=0)
        synth = torch.sum(attn.unsqueeze(1) * params, dim=0)
        return synth[0].item(), synth[1].item()
