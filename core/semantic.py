import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

class SemanticFeatureExtractor(nn.Module):
    """Extracts financial semantic embeddings E_t using FinBERT."""
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
