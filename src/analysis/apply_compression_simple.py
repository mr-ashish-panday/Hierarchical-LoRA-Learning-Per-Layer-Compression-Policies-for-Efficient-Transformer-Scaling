import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.A = nn.Parameter(torch.randn(in_features, self.rank, device=device, dtype=dtype) * 0.02)
        self.B = nn.Parameter(torch.randn(self.rank, out_features, device=device, dtype=dtype) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    def forward(self, x):
        out = x @ self.A @ self.B
        if self.bias is not None:
            out = out + self.bias
        return out

def _iter_with_parent(module: nn.Module):
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, Conv1D)):
            yield module, name, child
        else:
            yield from _iter_with_parent(child)

def apply_compression(model, r_idx, b_idx, s_idx, cfg):
    linear_entries = list(_iter_with_parent(model))
    L = len(linear_entries)
    assert L == len(r_idx) == len(b_idx) == len(s_idx), f"Policy lengths must match #linears {L}"
    ranks, bitwidths, sparsities = cfg['ranks'], cfg['bitwidths'], cfg['sparsities']
    for i, (parent, name, lin) in enumerate(linear_entries):
        r, b, s = ranks[r_idx[i]], bitwidths[b_idx[i]], sparsities[s_idx[i]]
        if hasattr(lin, 'weight'):
            w = lin.weight
        if isinstance(lin, Conv1D):
            in_f, out_f = w.size(0), w.size(1)
        else:
            in_f, out_f = lin.in_features, lin.out_features
        device, dtype = w.device, w.dtype
        bias_flag = lin.bias is not None
        lr = LowRankLinear(in_f, out_f, rank=min(r, in_f, out_f), bias=bias_flag, device=device, dtype=dtype)
        if bias_flag:
            with torch.no_grad():
                lr.bias.data = lin.bias.data.clone()
        setattr(parent, name, lr)
    return model
