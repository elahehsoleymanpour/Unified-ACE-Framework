‍‍‍‍
# Unified ACE Framework: Multi-Modal Semantic-Topological Parameter Estimation
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-green)

Official implementation repository for the Ph.D. dissertation on "Fidelity-Gated Analogy-Based Causal Estimation in Non-Stationary Financial Markets".

This repository provides a unified PyTorch architecture that seamlessly supports both **F-ACE** (Uni-Modal Topological) and **ST-ACE** (Multi-Modal Semantic-Topological) frameworks.

## 📖 Abstract
Estimating the governing parameters of continuous-time stochastic models (e.g., GBM) is a persistent challenge in quantitative finance due to abrupt regime shifts. While classical econometrics (like GARCH) fail during structural breaks, the **Unified ACE Framework** utilizes a **Quantitative Retrieval-Augmented Generation (Q-RAG)** architecture. 

It synthesizes parameters by retrieving historical analogues based on:
1. **Topological Price Geometry** (via Kernel Density Estimation & Jensen-Shannon Divergence).
2. **Semantic Market Narratives** (via FinBERT Embeddings & Cosine Similarity).

The synthesis is dynamically gated using **Predictive Fidelity Scores** and **Softplus-gated Elastic Mechanisms** to prevent parameter hallucination during random-walk regimes.

## 🚀 Key Features
- **Micro-Trained Cross-Attention (APA):** Learns dynamically to assign weights to retrieved analogues.
- **Asymmetric Volatility-Targeting:** A rigorous trading strategy module optimized for both Equities (S&P 500) and Jump-Diffusion assets (Bitcoin).
- **FinBERT Integration:** Extracts dense semantic embeddings from financial news headlines.

```text
Unified-ACE-Framework/
│
├── core/
│   ├── topology.py       # Non-parametric KDE extraction
│   ├── semantic.py       # FinBERT semantic feature extraction
│   ├── attention.py      # Cross-Attention Neural Synthesis
│   └── gates.py          # JSD/Cosine distances & Fidelity Gate
│
├── engine/
│   └── unified_model.py  # Joint-Training engine (Softplus & Dynamic P_t)
│
├── utils/
│   ├── data_loader.py    # Temporal alignment of news and prices
│   └── metrics.py        # QLIKE, RMSE, and Drawdown calculations
│
└── Main_Experiments.ipynb # Master notebook for Tables & Figures reproduction

## 💻 Usage
To run the experiments in Google Colab or your local machine:
1. Clone the repository.
2. Ensure `sp500_headlines_2008_2024.csv` is located in the root directory.
3. Open `Main_Experiments.ipynb` and set `CONFIG['MODE'] = 'ST-ACE'` or `'F-ACE'`.

## 📜 Citation
If you use this framework in your research, please cite the underlying dissertation/papers:
*(Citation details will be updated post-publication)*