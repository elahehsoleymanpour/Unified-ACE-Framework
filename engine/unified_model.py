import torch
import torch.nn as nn
import torch.optim as optim
from core.topology import TopologicalFeatureExtractor
from core.gates import UnifiedPatternMatcher, PredictiveFidelityGate
from core.attention import AttentionSynthesizer

class Fully_Optimized_Unified_Engine(nn.Module):
    def __init__(self, db_records, tau, device, K, window_size, mode='ST-ACE'):
        super().__init__()
        self.records = db_records
        self.K = K
        self.window_size = window_size
        self.device = device
        self.mode = mode  # رفع خطای هاردکد شدن
        
        self.extractor = TopologicalFeatureExtractor(device)
        self.matcher = UnifiedPatternMatcher(device)
        self.gate = PredictiveFidelityGate(tau)
        self.apa_top = AttentionSynthesizer(device)
        
        if self.mode == 'ST-ACE':
            self.apa_sem = AttentionSynthesizer(device)
            self.softplus = nn.Softplus(beta=1)
            self.W_up = nn.Parameter(torch.tensor(0.8).to(device))
            self.W_down = nn.Parameter(torch.tensor(0.8).to(device))
            # شبکه استخراج وزن پویای اخبار
            self.P_weight = nn.Sequential(
                nn.Linear(768, 16), 
                nn.ReLU(), 
                nn.Linear(16, 1), 
                nn.Sigmoid()
            ).to(device)

    def _st_ace_synthesis(self, var_top, var_sem, P_t, garch_var):
        R_t = (P_t * var_sem) + ((1 - P_t) * var_top)
        # مکانیسم Softplus-gated (معادلات مقاله دوم)
        lambda_up = self.W_up * self.softplus(R_t - garch_var) + (0.1 * P_t * garch_var)
        lambda_down = self.W_down * self.softplus(garch_var - R_t) * (1 - P_t)
        return torch.max(torch.tensor(1e-6).to(self.device), garch_var + lambda_up - lambda_down)

    def train_micro(self, q, keys, params, target_proxy, net):
        optimizer = optim.Adam(net.parameters(), lr=0.05) # افزایش Learning Rate برای خروج از Minima
        target_t = torch.tensor(target_proxy, dtype=torch.float32).to(self.device)
        for _ in range(10):
            optimizer.zero_grad()
            _, synth_var = net(q, keys, params)
            loss = torch.abs(synth_var - target_t) # استفاده از MAE برای پایداری بیشتر
            loss.backward()
            optimizer.step()

    def forward(self, current_win, current_idx, current_sem=None, garch_var=None):
        norm = (current_win - torch.mean(current_win)) / (torch.std(current_win) + 1e-8)
        tgt = {'density': self.extractor(norm), 'volatility': torch.std(current_win), 
               'trend': (current_win[-1]-current_win[0])/self.window_size, 'semantic': current_sem}
        
        dists_top, dists_sem = [], []
        for r in self.records:
            if r['idx'] < current_idx - self.window_size:
                # رفع خطای TypeError با استفاده از self.mode
                d = self.matcher(tgt, r, current_idx - r['idx'], mode=self.mode)
                if self.mode == 'F-ACE':
                    dists_top.append((d.item(), r))
                else:
                    dists_top.append((d['dist_top'].item(), r))
                    dists_sem.append((d['dist_sem'].item(), r))
                    
        dists_top.sort(key=lambda x: x[0])
        top_k = dists_top[:self.K]
        futs = torch.stack([x[1]['future'] for x in top_k])
        is_open, phi = self.gate(futs)
        mle_var = torch.var(current_win).item()
        
        if not is_open:
            return {'var_face': mle_var, 'var_stace': mle_var, 'phi': phi, 'status': 'MLE'}
            
        params_t = torch.stack([x[1]['params'] for x in top_k])
        keys_t = torch.tensor([[torch.mean(x[1]['raw_win']).item(), x[1]['volatility'].item(), x[1]['trend']] for x in top_k]).to(self.device)
        q = torch.tensor([[torch.mean(current_win).item(), tgt['volatility'].item(), tgt['trend']]]).to(self.device)
        
        # آموزش هدفمند روی واریانس GARCH (ترکیب اقتصادسنجی و ML)
        self.train_micro(q, keys_t, params_t, garch_var, self.apa_top)
        _, var_top = self.apa_top(q, keys_t, params_t)
        
        if self.mode == 'F-ACE':
            return {'var_face': var_top, 'var_stace': var_top, 'phi': phi, 'status': 'F-ACE'}
            
        # فاز پردازش ST-ACE
        dists_sem.sort(key=lambda x: x[0])
        top_sem = dists_sem[:self.K]
        params_sem = torch.stack([x[1]['params'] for x in top_sem])
        keys_sem = torch.tensor([[torch.mean(x[1]['raw_win']).item(), x[1]['volatility'].item(), x[1]['trend']] for x in top_sem]).to(self.device)
        
        self.train_micro(q, keys_sem, params_sem, garch_var, self.apa_sem)
        _, var_sem = self.apa_sem(q, keys_sem, params_sem)
        
        P_t = self.P_weight(current_sem).squeeze().item() if current_sem is not None else 0.5
        final_var = self._st_ace_synthesis(torch.tensor(var_top).to(self.device), torch.tensor(var_sem).to(self.device), P_t, garch_var)
        
        return {'var_face': var_top, 'var_stace': final_var.item(), 'phi': phi, 'status': 'ST-ACE'}