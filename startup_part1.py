import os

def create_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print(f"Created: {filepath}")

def main():
    print("Initializing Unified-ACE-Framework Architecture (Part 1)...")
    
    # =========================================================
    # 1. CORE MODULES
    # =========================================================
    
    # 1.1 Topology Module (100% Aligned with Eq 5-7, Paper 1)
    topology_code = """
import torch
import torch.nn as nn
import math

class TopologicalFeatureExtractor(nn.Module):
    \"\"\"Extracts non-parametric density using a differentiable Gaussian Kernel.\"\"\"
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
"""
    create_file("core/topology.py", topology_code)
    create_file("core/__init__.py", "")

    # 1.2 Semantic Module (100% Aligned with Eq 3, Paper 2)
    semantic_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

class SemanticFeatureExtractor(nn.Module):
    \"\"\"Extracts financial semantic embeddings E_t using FinBERT.\"\"\"
    def __init__(self, device, model_name="ProsusAI/finbert"):
        super().__init__()
        self.device = device
        logging.getLogger("transformers").setLevel(logging.ERROR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def forward(self, texts: list) -> torch.Tensor:
        if not texts or len(texts) == 0:
            return torch.zeros((1, 768), device=self.device)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extract [CLS] token representation and apply L2 Normalization for Cosine distance
        pooled = torch.mean(outputs.last_hidden_state[:, 0, :], dim=0, keepdim=True)
        return F.normalize(pooled, p=2, dim=1)
"""
    create_file("core/semantic.py", semantic_code)

    # 1.3 Gates Module (100% Aligned with JSD, Cosine, and Fidelity Gate)
    gates_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedPatternMatcher(nn.Module):
    \"\"\"Calculates enriched distances combining topology (KL/JSD) and semantics (Cosine).\"\"\"
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
    \"\"\"Evaluates the predictive fidelity of historical analogies via future path correlation.\"\"\"
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
"""
    create_file("core/gates.py", gates_code)

    # 1.4 Attention Module (100% Aligned with Cross-Attention Synthesis)
    attention_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSynthesizer(nn.Module):
    \"\"\"Neural Cross-Attention layer for Parameter Aggregation (APA).\"\"\"
    def __init__(self, device, hidden_dim=16):
        super().__init__()
        self.W1 = nn.Linear(6, hidden_dim).to(device)
        self.W2 = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, q, keys, params):
        concat = torch.cat([q.expand(keys.shape[0], -1), keys], dim=1)
        attn = F.softmax(self.W2(torch.tanh(self.W1(concat))).squeeze(-1), dim=0)
        synth = torch.sum(attn.unsqueeze(1) * params, dim=0)
        return synth[0].item(), synth[1].item()
"""
    create_file("core/attention.py", attention_code)


    # =========================================================
    # 2. UTILS MODULES
    # =========================================================
    
    # 2.1 Metrics (100% Aligned with evaluation standard of the papers)
    metrics_code = """
import numpy as np

def calc_rmse(pred, real): 
    return np.sqrt(np.mean((np.array(pred) - np.array(real))**2))

def calc_qlike(pred, real): 
    return np.mean(np.log(np.array(pred) + 1e-8) + (np.array(real) / (np.array(pred) + 1e-8)))

def calc_dd_curve(cum): 
    peaks = np.maximum.accumulate(cum)
    return (cum - peaks) / peaks
"""
    create_file("utils/metrics.py", metrics_code)
    create_file("utils/__init__.py", "")

    # 2.2 Data Loader
    data_loader_code = """
import pandas as pd
import numpy as np

def load_and_preprocess_unified(filepath):
    \"\"\"Aggregates news headlines by date and aligns them with market returns.\"\"\"
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df_grouped = df.groupby('Date').agg({'Title': list, 'CP': 'last'}).reset_index()
    df_grouped = df_grouped.sort_values('Date')
    
    prices = df_grouped['CP'].values
    dates = df_grouped['Date'].values[1:]
    news_lists = df_grouped['Title'].values[1:]
    returns = np.log(prices[1:] / prices[:-1])
    return returns, dates, news_lists
"""
    create_file("utils/data_loader.py", data_loader_code)
    
    print("\\nPart 1 Complete! Core and Utils folders generated successfully.")
    print("Please prompt for Part 2 to generate the Engine and the Main_Experiments.ipynb notebook.")

if __name__ == "__main__":
    main()