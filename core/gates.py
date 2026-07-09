import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedPatternMatcher(nn.Module):
    """Calculates enriched distances combining topology (KL/JSD) and semantics (Cosine)."""
    def __init__(self, device, weights=(0.4, 0.4, 0.2), decay_lambda=0.001):
        super().__init__()
        self.weights = torch.tensor(weights, device=device)
        self.decay_lambda = decay_lambda

    def jsd(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * torch.sum(p * torch.log(p / m)) + 0.5 * torch.sum(q * torch.log(q / m))

    def forward(self, target, hist, time_delta, mode):
        p, q = target['density'] + 1e-10, hist['density'] + 1e-10
        
        # Paper 2 explicitly replaces KL with JSD for symmetric distance
        d_dist = self.jsd(p, q) if mode == 'ST-ACE' else torch.sum(p * torch.log(p / q))
        d_vol = torch.abs(target['volatility'] - hist['volatility'])
        d_trend = torch.abs(target['trend'] - hist['trend'])
        
        dist_top = ((self.weights[0]*d_dist) + (self.weights[1]*d_vol) + (self.weights[2]*d_trend)) * (1 + self.decay_lambda * time_delta)
        
        if mode == 'F-ACE': 
            return dist_top
            
        sem_sim = F.cosine_similarity(target['semantic'], hist['semantic'])
        dist_sem = 1.0 - sem_sim # Convert similarity to distance
        return {'dist_top': dist_top, 'dist_sem': dist_sem}

class PredictiveFidelityGate(nn.Module):
    """Evaluates the predictive fidelity of historical analogies via future path correlation."""
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, future_paths):
        K = future_paths.shape[0]
        if K < 2: return False, 0.0
        p_norm = (future_paths - torch.mean(future_paths, dim=1, keepdim=True)) / (torch.std(future_paths, dim=1, keepdim=True) + 1e-8)
        corr = torch.matmul(p_norm, p_norm.T) / (future_paths.shape[1] - 1)
        triu = torch.triu_indices(row=K, col=K, offset=1)
        phi = torch.mean(corr[triu[0], triu[1]]).item()
        return (phi >= self.threshold), phi
