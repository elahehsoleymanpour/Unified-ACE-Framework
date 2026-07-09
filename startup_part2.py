import os
import json

def create_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n')
    print(f"Created: {filepath}")

def main():
    print("Initializing Unified-ACE-Framework Architecture (Part 2)...")
    
    # =========================================================
    # 1. ENGINE MODULE (Unified F-ACE & ST-ACE with Softplus & Dynamic P_t)
    # =========================================================
    engine_code = """
import torch
import torch.nn as nn
import torch.optim as optim
from core.topology import TopologicalFeatureExtractor
from core.gates import UnifiedPatternMatcher, PredictiveFidelityGate
from core.attention import AttentionSynthesizer

class Unified_ACE_Engine(nn.Module):
    def __init__(self, db_records, tau, device, K, window_size, mode):
        super().__init__()
        self.records = db_records
        self.K = K
        self.window_size = window_size
        self.device = device
        self.mode = mode
        
        self.extractor = TopologicalFeatureExtractor(device)
        self.matcher = UnifiedPatternMatcher(device)
        self.gate = PredictiveFidelityGate(tau)
        self.apa_top = AttentionSynthesizer(device)
        
        if mode == 'ST-ACE':
            self.apa_sem = AttentionSynthesizer(device)
            # Softplus-gated synthesis mechanism (Overriding standard gating)
            self.softplus = nn.Softplus(beta=1)
            self.W_up = nn.Parameter(torch.tensor(0.5).to(device))
            self.W_down = nn.Parameter(torch.tensor(0.5).to(device))
            # Dynamic learnable parameter for news weight (P_t)
            self.P_weight = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid()).to(device)

    def _st_ace_synthesis(self, var_top, var_sem, P_t, garch_var):
        R_t = (P_t * var_sem) + ((1 - P_t) * var_top)
        lambda_up = self.W_up * self.softplus(R_t - garch_var) + (0.1 * P_t * garch_var)
        lambda_down = self.W_down * self.softplus(garch_var - R_t) * (1 - P_t)
        return torch.max(torch.tensor(1e-6).to(self.device), garch_var + lambda_up - lambda_down).item()

    def train_micro(self, q, keys, params, target_proxy, net):
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        target_t = torch.tensor(target_proxy, dtype=torch.float32).to(self.device)
        for _ in range(15):
            optimizer.zero_grad()
            _, synth_var = net(q, keys, params)
            loss = torch.log(torch.tensor(synth_var + 1e-8).to(self.device)) + (target_t / (synth_var + 1e-8))
            loss.backward()
            optimizer.step()

    def forward(self, current_win, current_idx, current_sem=None, garch_var=None, train_apa=True):
        norm = (current_win - torch.mean(current_win)) / (torch.std(current_win) + 1e-8)
        tgt = {'density': self.extractor(norm), 'volatility': torch.std(current_win), 
               'trend': (current_win[-1]-current_win[0])/self.window_size, 'semantic': current_sem}
        
        dists_top, dists_sem = [], []
        for r in self.records:
            if r['idx'] < current_idx - self.window_size:
                d = self.matcher(tgt, r, current_idx - r['idx'], self.mode)
                if self.mode == 'F-ACE': dists_top.append((d.item(), r))
                else:
                    dists_top.append((d['dist_top'].item(), r))
                    dists_sem.append((d['dist_sem'].item(), r))
                    
        dists_top.sort(key=lambda x: x[0])
        top_k = dists_top[:self.K]
        futs = torch.stack([x[1]['future'] for x in top_k])
        is_open, phi = self.gate(futs)
        mle_var = torch.var(current_win).item()
        
        if not is_open:
            return {'var_face': mle_var, 'var_stace': mle_var, 'phi': phi, 'status': 'MLE', 'top_face': top_k}
            
        params_t = torch.stack([x[1]['params'] for x in top_k])
        keys_t = torch.tensor([[torch.mean(x[1]['raw_win']).item(), x[1]['volatility'].item(), x[1]['trend']] for x in top_k]).to(self.device)
        q = torch.tensor([[torch.mean(current_win).item(), tgt['volatility'].item(), tgt['trend']]]).to(self.device)
        
        if train_apa and len(top_k) > 0:
            # Weighted average of future variances as target proxy for robust micro-training
            weights = 1.0 / torch.arange(1, len(params_t)+1).to(self.device)
            target_proxy = torch.sum(params_t[:, 1] * (weights / torch.sum(weights))).item()
            self.train_micro(q, keys_t, params_t, target_proxy, self.apa_top)
            
        _, var_top = self.apa_top(q, keys_t, params_t)
        
        if self.mode == 'F-ACE':
            return {'var_face': var_top, 'var_stace': var_top, 'phi': phi, 'status': 'F-ACE', 'top_face': top_k}
            
        dists_sem.sort(key=lambda x: x[0])
        top_sem = dists_sem[:self.K]
        params_sem = torch.stack([x[1]['params'] for x in top_sem])
        keys_sem = torch.tensor([[torch.mean(x[1]['raw_win']).item(), x[1]['volatility'].item(), x[1]['trend']] for x in top_sem]).to(self.device)
        
        if train_apa and len(top_sem) > 0:
            weights_sem = 1.0 / torch.arange(1, len(params_sem)+1).to(self.device)
            target_proxy_sem = torch.sum(params_sem[:, 1] * (weights_sem / torch.sum(weights_sem))).item()
            self.train_micro(q, keys_sem, params_sem, target_proxy_sem, self.apa_sem)
        _, var_sem = self.apa_sem(q, keys_sem, params_sem)
        
        # Dynamic P_t logic using ML representation learning
        P_t = self.P_weight(current_sem).item() if current_sem is not None else 0.5
        final_var = self._st_ace_synthesis(torch.tensor(var_top).to(self.device), torch.tensor(var_sem).to(self.device), P_t, torch.tensor(garch_var).to(self.device))
        
        return {'var_face': var_top, 'var_stace': final_var, 'phi': phi, 'status': 'ST-ACE', 'top_face': top_k, 'top_stace': top_sem, 'P_t': P_t}
"""
    create_file("engine/unified_model.py", engine_code)
    create_file("engine/__init__.py", "")

    # =========================================================
    # 2. MAIN EXPERIMENTS NOTEBOOK (JSON Generation)
    # =========================================================
    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<div dir=\"rtl\">\n",
                "<h1>چارچوب جامع یادگیری ماشین برای تخمین پارامتر مالی (Unified F-ACE & ST-ACE)</h1>\n",
                "<h2>تولید همزمان تمامی جداول و نمودارهای رساله</h2>\n",
                "<h3>۱. فرمول‌بندی فرآیندهای دیفیوژن</h3>\n",
                "<p>پویایی قیمت بر بستر حرکت براونی هندسی (GBM) با پارامترهای محلی:</p>\n",
                "$$dS_{t}=\\mu S_{t}dt+\\sigma S_{t}dW_{t} \\quad (1)$$\n",
                "$$r_{t}=\\ln(\\frac{S_{t}}{S_{t-1}}) \\sim \\mathcal{N}(\\nu\\Delta t,\\sigma^2\\Delta t) \\quad (2)$$\n",
                "</div>"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install transformers arch yfinance -q\n",
                "import torch, os, pickle, math\n",
                "import numpy as np, pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import matplotlib.dates as mdates\n",
                "from arch import arch_model\n",
                "\n",
                "from core.topology import TopologicalFeatureExtractor\n",
                "from core.semantic import SemanticFeatureExtractor\n",
                "from engine.unified_model import Unified_ACE_Engine\n",
                "from utils.data_loader import load_and_preprocess_unified\n",
                "from utils.metrics import calc_rmse, calc_qlike, calc_dd_curve\n",
                "\n",
                "CONFIG = {'DATA_PATH': 'sp500_headlines_2008_2024.csv', 'CACHE_DIR': './cache_db', 'WINDOW_SIZE': 60, 'FUTURE_SIZE': 20, 'K_NEIGHBORS': 10}\n",
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "def log_msg(msg): print(f\"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<div dir=\"rtl\">\n",
                "<h3>۲. استخراج ویژگی‌های معنایی با مدل زبانی (FinBERT)</h3>\n",
                "$$E_t = \\text{FinBERT}(\\text{News}_t) \\quad (3)$$\n",
                "</div>"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "returns, dates, news_lists = load_and_preprocess_unified(CONFIG['DATA_PATH'])\n",
                "news_embeddings = {}\n",
                "cache_file = os.path.join(CONFIG['CACHE_DIR'], 'semantic_cache.pkl')\n",
                "if os.path.exists(cache_file):\n",
                "    log_msg(\"Loading cached FinBERT embeddings...\")\n",
                "    with open(cache_file, 'rb') as f: news_embeddings = pickle.load(f)\n",
                "else:\n",
                "    log_msg(\"Extracting semantic embeddings (FinBERT)...\")\n",
                "    semantic_model = SemanticFeatureExtractor(DEVICE)\n",
                "    for i, d in enumerate(dates):\n",
                "        news_embeddings[d] = semantic_model(news_lists[i]).cpu()\n",
                "    with open(cache_file, 'wb') as f: pickle.dump(news_embeddings, f)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<div dir=\"rtl\">\n",
                "<h3>۳. واگرایی‌های متقارن (JSD و Cosine) و سنتز Q-RAG با گیت‌های الاستیک (Softplus)</h3>\n",
                "$$S_{top} \\leftarrow JSD(\\hat{p}_t(r), \\mathcal{D}_{hist}^{price}) \\quad (4)$$\n",
                "$$S_{sem} \\leftarrow Cosine(E_t, \\mathcal{D}_{hist}^{text}) \\quad (5)$$\n",
                "$$\\lambda_{t}^{UP} \\leftarrow W_{up} \\cdot \\Psi_{\\gamma}(\\mathcal{R}_{t} - \\sigma_{G,t}) + \\alpha \\cdot (\\mathcal{P}_{t} \\cdot \\sigma_{G,t}) \\quad (6)$$\n",
                "$$\\lambda_{t}^{DOWN} \\leftarrow W_{down} \\cdot \\Psi_{\\gamma}(\\sigma_{G,t} - \\mathcal{R}_{t}) \\cdot (1 - \\mathcal{P}_{t}) \\quad (7)$$\n",
                "$$\\sigma_{MTL}(t) \\leftarrow \\max(\\epsilon, \\sigma_{G,t} + \\lambda_{t}^{UP} - \\lambda_{t}^{DOWN}) \\quad (8)$$\n",
                "</div>"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "log_msg(\"Building Unified Historical Cache...\")\n",
                "history_end = np.where(pd.to_datetime(dates) == pd.to_datetime('2018-01-02'))[0][0]\n",
                "db_records = []\n",
                "extractor_dummy = TopologicalFeatureExtractor(DEVICE)\n",
                "for i in range(CONFIG['WINDOW_SIZE'], history_end, 5):\n",
                "    win = torch.tensor(returns[i-CONFIG['WINDOW_SIZE']:i], dtype=torch.float32).to(DEVICE)\n",
                "    fut = torch.tensor(returns[i:i+CONFIG['FUTURE_SIZE']], dtype=torch.float32).to(DEVICE)\n",
                "    w_norm = (win - torch.mean(win)) / (torch.std(win) + 1e-8)\n",
                "    db_records.append({'idx': i, 'date': dates[i], 'density': extractor_dummy(w_norm), 'volatility': torch.std(win), 'trend': (win[-1] - win[0]) / CONFIG['WINDOW_SIZE'], 'future': fut, 'raw_win': win, 'params': torch.tensor([torch.mean(fut).item(), torch.var(fut).item()]).to(DEVICE), 'semantic': news_embeddings.get(dates[i], torch.zeros((1, 768))).to(DEVICE)})\n",
                "\n",
                "engine = Unified_ACE_Engine(db_records, tau=0.15, device=DEVICE, K=CONFIG['K_NEIGHBORS'], window_size=CONFIG['WINDOW_SIZE'], mode='ST-ACE')\n",
                "oos_start, oos_end = history_end, len(returns) - 20\n",
                "res = {'real': [], 'mle': [], 'garch': [], 'face': [], 'stace': [], 'phi': [], 'dates': [], 'mkt_ret': [], 'strat_ret': []}\n",
                "\n",
                "log_msg(\"Executing Deep Out-of-Sample Backtest (2018-2023)...\")\n",
                "for t in range(oos_start, oos_end, 15):\n",
                "    win_np = returns[t-CONFIG['WINDOW_SIZE']:t]\n",
                "    win_t, fut_var = torch.tensor(win_np, dtype=torch.float32).to(DEVICE), np.var(returns[t:t+CONFIG['FUTURE_SIZE']]) * 252\n",
                "    try: v_garch = (arch_model(win_np * 100, vol='Garch', p=1, q=1).fit(disp='off').forecast(horizon=20).variance.values[-1, :].mean() / 10000) * 252\n",
                "    except: v_garch = np.var(win_np) * 252\n",
                "\n",
                "    out = engine(win_t, t, current_sem=news_embeddings.get(dates[t], torch.zeros((1, 768))).to(DEVICE), garch_var=v_garch, train_apa=True)\n",
                "    res['real'].append(fut_var); res['mle'].append(np.var(win_np) * 252); res['garch'].append(v_garch)\n",
                "    res['face'].append(out['var_face'] * 252); res['stace'].append(out['var_stace'] * 252)\n",
                "    res['phi'].append(out['phi']); res['dates'].append(dates[t])\n",
                "    \n",
                "    act_ret = np.sum(returns[t:t+15])\n",
                "    res['mkt_ret'].append(act_ret)\n",
                "    weight = min((0.15 / math.sqrt(252)) / (math.sqrt(out['var_stace']) + 1e-8), 2.0 if out['status'] != 'MLE' else 1.0)\n",
                "    res['strat_ret'].append(weight * act_ret)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<div dir=\"rtl\">\n",
                "<h3>۴. تولید جداول و نمودارهای مقایسه‌ای نهایی (تطبیق ۱۰۰٪)</h3>\n",
                "</div>"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "log_msg(\"Generating Tables and Figures...\")\n",
                "print(\"\\n--- Table 2 & 3: Multi-Modal vs Uni-Modal Estimation Results ---\")\n",
                "tbl_ablation = pd.DataFrame({\n",
                "    'Model': ['ST-ACE (Price + News)', 'F-ACE (Price Only)', 'GARCH(1,1)', 'Local MLE'],\n",
                "    'RMSE': [calc_rmse(res['stace'], res['real']), calc_rmse(res['face'], res['real']), calc_rmse(res['garch'], res['real']), calc_rmse(res['mle'], res['real'])],\n",
                "    'QLIKE': [calc_qlike(res['stace'], res['real']), calc_qlike(res['face'], res['real']), calc_qlike(res['garch'], res['real']), calc_qlike(res['mle'], res['real'])]\n",
                "})\n",
                "display(tbl_ablation.round(4))\n",
                "\n",
                "crisis_mask = (pd.to_datetime(res['dates']) >= '2020-02-15') & (pd.to_datetime(res['dates']) <= '2020-04-15')\n",
                "print(\"\\n--- Table 4: Performance Under Stress (COVID-19 Crash) ---\")\n",
                "tbl4 = pd.DataFrame({\n",
                "    'Model': ['ST-ACE (Multi-Modal)', 'F-ACE (Uni-Modal)', 'GARCH(1,1)'],\n",
                "    'Crisis RMSE': [calc_rmse(np.array(res['stace'])[crisis_mask], np.array(res['real'])[crisis_mask]), calc_rmse(np.array(res['face'])[crisis_mask], np.array(res['real'])[crisis_mask]), calc_rmse(np.array(res['garch'])[crisis_mask], np.array(res['real'])[crisis_mask])]\n",
                "})\n",
                "display(tbl4.round(4))\n",
                "\n",
                "cum_strat = np.exp(np.cumsum(res['strat_ret'])); cum_mkt = np.exp(np.cumsum(res['mkt_ret']))\n",
                "print(\"\\n--- Table 5: ST-ACE Financial Metrics ---\")\n",
                "tbl5 = pd.DataFrame({\n",
                "    'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown'],\n",
                "    'ST-ACE Strategy': [f\"{(cum_strat[-1]-1)*100:.2f}%\", round((np.mean(res['strat_ret'])/np.std(res['strat_ret']))*math.sqrt(252/15), 2), f\"{np.min(calc_dd_curve(cum_strat))*100:.2f}%\"],\n",
                "    'Market': [f\"{(cum_mkt[-1]-1)*100:.2f}%\", round((np.mean(res['mkt_ret'])/np.std(res['mkt_ret']))*math.sqrt(252/15), 2), f\"{np.min(calc_dd_curve(cum_mkt))*100:.2f}%\"]\n",
                "})\n",
                "display(tbl5)\n",
                "\n",
                "fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=150)\n",
                "plot_dates = pd.to_datetime(res['dates'])\n",
                "axes[0].plot(plot_dates, res['phi'], color='#8e44ad', label='$\Phi_t$ Score'); axes[0].axhline(0.15, color='red', linestyle='--'); axes[0].set_title('Figure 3: Structural Fidelity Score'); axes[0].legend()\n",
                "axes[1].plot(plot_dates[crisis_mask], np.array(res['real'])[crisis_mask], color='black', linewidth=2, label='Realized Volatility')\n",
                "axes[1].plot(plot_dates[crisis_mask], np.array(res['stace'])[crisis_mask], color='#27ae60', marker='o', label='ST-ACE Forecast')\n",
                "axes[1].plot(plot_dates[crisis_mask], np.array(res['garch'])[crisis_mask], color='#e74c3c', marker='x', label='GARCH Forecast')\n",
                "axes[1].set_title('Figure 5: Forecast Divergence (COVID-19 Crash)'); axes[1].legend()\n",
                "axes[2].plot(plot_dates, cum_strat, label='ST-ACE Strategy', color='#27ae60'); axes[2].plot(plot_dates, cum_mkt, label='Market', color='#7f8c8d', linestyle='--')\n",
                "axes[2].set_title('Figure 6: Wealth Generation'); axes[2].legend()\n",
                "plt.tight_layout(); plt.show()"
            ]
        }
    ]

    notebook_content = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open("Main_Experiments.ipynb", 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1, ensure_ascii=False)
    print("Created: Main_Experiments.ipynb")
    print("\\nPart 2 Complete! The Unified Architecture is now ready for deployment in Google Colab.")

if __name__ == "__main__":
    main()