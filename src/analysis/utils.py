import os
import random
import torch
import numpy as np
import pandas as pd


def set_seed_all(seed: int):
    """
    Seed all RNGs for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_phase1_data(path: str):
    """
    Load Phase 1 sensitivity CSV into feature matrix and loss vector.
    Expects columns: 'axis', 'value', 'loss'.
    Returns:
      X: np.ndarray of shape (n_samples, 3) with [rank, bitwidth, sparsity]
      y: np.ndarray of shape (n_samples,) with losses
    """
    df = pd.read_csv(path)
    feats = []
    losses = []
    for _, row in df.iterrows():
        vec = [0, 0, 0]
        axis = row.get('axis')
        if axis == 'rank':
            vec[0] = row['value']
        elif axis == 'bitwidth':
            vec[1] = row['value']
        elif axis == 'sparsity':
            vec[2] = row['value']
        feats.append(vec)
        losses.append(row['loss'])
    return np.array(feats), np.array(losses)


def apply_compression(model, r_idx, b_idx, s_idx, cfg):
    """
    Apply LoRA, quantization, and pruning to a model using CPU for memory safety.
    """
    import torch.nn as nn
    from torch.nn.utils import prune

    class LoRAWrapper(nn.Module):
        def __init__(self, orig, rank, alpha=1.0):
            super().__init__()
            in_f, out_f = orig.in_features, orig.out_features
            self.orig = orig
            self.A = nn.Linear(in_f, rank, bias=False)
            self.B = nn.Linear(rank, out_f, bias=False)
            nn.init.zeros_(self.A.weight)
            nn.init.zeros_(self.B.weight)
            self.alpha = alpha

        def forward(self, x):
            return self.orig(x) + self.alpha * self.B(self.A(x))

    layer_idx = 0
    for name, module in list(model.named_modules()):
        if not hasattr(module, 'weight'):
            continue
        if layer_idx >= len(r_idx):
            break

        # Config values
        ri = int(r_idx[layer_idx])
        bi = int(b_idx[layer_idx])
        si = int(s_idx[layer_idx])

        rank = cfg['ranks'][ri]
        bitwidth = cfg['bitwidths'][bi]
        sparsity = cfg['sparsities'][si]

        # === LoRA injection ===
        if isinstance(module, nn.Linear) and 0 < rank < min(module.weight.shape):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRAWrapper(module, rank))

        # === Quantization (CPU side) ===
        if bitwidth < 32:
            with torch.no_grad():
                W = module.weight.data.float().cpu()
                max_abs = W.abs().max()
                scale = max_abs / ((2 ** bitwidth - 1) / 2) if max_abs >= 1e-8 else 1.0
                q = torch.clamp((W / scale).round(),
                                -(2 ** (bitwidth - 1)),
                                2 ** (bitwidth - 1) - 1)
                Wq = (q * scale).to(module.weight.dtype)
                module.weight.data = Wq.to(module.weight.device)

        # === Pruning (CPU side, then back to device) ===
                # === Pruning (manual CPU-side for Linear only) ===
        if sparsity > 0.0 and isinstance(module, nn.Linear):
            with torch.no_grad():
                W = module.weight.data.cpu()
                k = int(W.numel() * sparsity)
                if k > 0:
                    threshold = W.abs().view(-1).kthvalue(k).values
                    mask = (W.abs() >= threshold).float()
                    pruned_w = W * mask
                    module.weight.data = pruned_w.to(module.weight.device)
        layer_idx += 1

    return model
