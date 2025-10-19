#!/usr/bin/env python3
import os, sys, copy, random, inspect
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
import matplotlib.pyplot as plt

# Project root
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# --------------------------------
# Config and small utilities
# --------------------------------
cfg_baseline = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9]
}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def fix_tokenizer_and_model(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def count_linears(model):
    return len([m for m in model.modules() if isinstance(m, nn.Linear)])

def ensure_list_indices(idx, L):
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().tolist()
    if isinstance(idx, (int, float, np.integer)):
        return [int(idx)] * L
    if not isinstance(idx, (list, tuple)):
        return [int(idx)] * L
    if len(idx) == 0:
        return [0] * L
    if len(idx) < L:
        return list(idx) + [idx[-1]] * (L - len(idx))
    if len(idx) > L:
        return list(idx)[:L]
    return list(map(int, idx))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).astype(np.float32).mean().item()
    return {'accuracy': float(acc)}

def cost_of_policy(r_idx, b_idx, s_idx):
    R, B, S = cfg_baseline['ranks'], cfg_baseline['bitwidths'], cfg_baseline['sparsities']
    return float(np.mean([R[r]*B[b]*(1-S[s]) for r,b,s in zip(r_idx,b_idx,s_idx)]))

# --------------------------------
# Feature extraction
# --------------------------------
def compute_task_features(task, tokenized_ds, max_len=128, sample_cap=512):
    # Build simple 6-dim task features from a small sample of the validation set
    ds = tokenized_ds['validation']
    n = min(len(ds), sample_cap)
    if n == 0:
        return torch.zeros(1, 6, dtype=torch.float32).to('cuda')
    input_ids = ds[:n]['input_ids']
    attention = ds[:n]['attention_mask']
    labels = ds[:n]['labels'] if 'labels' in ds.column_names else [0]*n

    # Sequence lengths
    lens = [int(a.sum().item()) for a in attention]
    avg_len = np.mean(lens) / max_len
    var_len = np.var(lens) / (max_len*max_len)

    # Unique tokens fraction (vs GPT2 vocab ~50k)
    uniq = set()
    for ids in input_ids:
        for t in ids:
            uniq.add(int(t))
    uniq_frac = min(len(uniq), 50000) / 50000.0

    # Label mean (normalized)
    try:
        lbl_mean = float(np.mean(labels))
    except Exception:
        lbl_mean = 0.5

    # Pair flag
    is_pair = task not in ['sst2','cola']
    pair_flag = 1.0 if is_pair else 0.0

    # Dataset size scale
    size_scale = min(len(tokenized_ds['train']), 10000) / 10000.0 if 'train' in tokenized_ds else 0.0

    vec = np.array([avg_len, var_len, uniq_frac, lbl_mean, pair_flag, size_scale], dtype=np.float32)
    return torch.tensor(vec[None, :], dtype=torch.float32).to('cuda')

def compute_layer_features(model):
    # 2-dim features per Linear layer: [depth_norm, weight_l2_norm_norm]
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    L = len(linear_layers)
    if L == 0:
        return torch.zeros(1, 1, 2, dtype=torch.float32).to('cuda')
    wnorms = []
    for layer in linear_layers:
        with torch.no_grad():
            w = layer.weight
            wnorms.append(float(torch.norm(w).item()))
    wnorms = np.array(wnorms, dtype=np.float32)
    wnorms_norm = (wnorms - wnorms.min()) / (wnorms.max() - wnorms.min() + 1e-8)
    depth = np.linspace(0.0, 1.0, L, dtype=np.float32)
    feats = np.stack([depth, wnorms_norm], axis=-1)  # [L, 2]
    return torch.tensor(feats[None, :, :], dtype=torch.float32).to('cuda')

# --------------------------------
# Policies
# --------------------------------
def uniform_policy(model, ep):
    L = count_linears(model)
    i = cfg_baseline['ranks'].index(8)
    j = cfg_baseline['bitwidths'].index(8)
    k = cfg_baseline['sparsities'].index(0.5)
    return [i]*L, [j]*L, [k]*L

def random_policy(model, ep):
    L = count_linears(model)
    return ([random.randrange(len(cfg_baseline['ranks'])) for _ in range(L)],
            [random.randrange(len(cfg_baseline['bitwidths'])) for _ in range(L)],
            [random.randrange(len(cfg_baseline['sparsities'])) for _ in range(L)])

class PolicyLoader:
    def __init__(self, ckpt_path, base_model, task_feat_dim=6, layer_feat_dim=2, hidden_dim=64):
        self.num_layers = count_linears(base_model)
        # Try multiple constructor signatures
        self.controller = None
        try:
            self.controller = MetaController(task_feat_dim, layer_feat_dim, hidden_dim,
                                             cfg_baseline['ranks'],
                                             cfg_baseline['bitwidths'],
                                             cfg_baseline['sparsities'],
                                             num_layers=self.num_layers)
        except TypeError:
            ctor_kwargs = {
                'task_feat_dim': task_feat_dim,
                'layer_feat_dim': layer_feat_dim,
                'ranks': cfg_baseline['ranks'],
                'bitwidths': cfg_baseline['bitwidths'],
                'sparsities': cfg_baseline['sparsities'],
                'num_layers': self.num_layers
            }
            sig = inspect.signature(MetaController.__init__)
            names = sig.parameters.keys()
            if 'hidden_dim' in names:
                ctor_kwargs['hidden_dim'] = hidden_dim
            elif 'controller_hidden' in names:
                ctor_kwargs['controller_hidden'] = hidden_dim
            self.controller = MetaController(**ctor_kwargs)
        self.controller = self.controller.to('cuda')
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cuda')
            self.controller.load_state_dict(sd)
            self.controller.eval()
            print(f"Loaded controller from {ckpt_path}")
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}; using random init.")

    def sample_actions_with_feats(self, tf, lf):
        L = lf.shape[1]
        with torch.no_grad():
            r_p, b_p, s_p = self.controller(tf, lf)
        r_idx = ensure_list_indices(torch.argmax(r_p, dim=-1).squeeze(), L)
        b_idx = ensure_list_indices(torch.argmax(b_p, dim=-1).squeeze(), L)
        s_idx = ensure_list_indices(torch.argmax(s_p, dim=-1).squeeze(), L)
        return r_idx, b_idx, s_idx

# --------------------------------
# Evaluation (kept compatible)
# --------------------------------
def tokenize_glue(task, tokenizer, max_len=128):
    raw = load_dataset('glue', task)
    is_pair = task not in ['sst2','cola']
    def prep(b):
        if is_pair:
            return tokenizer(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=max_len)
        return tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=max_len)
    data = raw.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    return data

def evaluate_policy(base_model, tokenizer, task, policy_kind, policy_fn=None, loader=None, epochs=5):
    data = tokenize_glue(task, tokenizer, 128)
    args = TrainingArguments(
        output_dir=f'output/baselines/{policy_kind}_{task}',
        per_device_eval_batch_size=16
    )

    # Pre-compute features once per task from the base model and tokenized data
    tf = compute_task_features(task, data, 128, 512)
    lf = compute_layer_features(base_model)

    losses, accs, costs = [], [], []
    for ep in range(1, epochs+1):
        m = copy.deepcopy(base_model)
        L = count_linears(m)
        if policy_kind == 'learned':
            r_idx, b_idx, s_idx = loader.sample_actions_with_feats(tf, lf)
        else:
            r_idx, b_idx, s_idx = policy_fn(m, ep)
        r_idx = ensure_list_indices(r_idx, L)
        b_idx = ensure_list_indices(b_idx, L)
        s_idx = ensure_list_indices(s_idx, L)

        m = apply_compression(m, r_idx, b_idx, s_idx, cfg_baseline)
        tokenizer, m = fix_tokenizer_and_model(tokenizer, m)
        trainer = Trainer(model=m, args=args, eval_dataset=data['validation'],
                          processing_class=tokenizer, compute_metrics=compute_metrics)
        res = trainer.evaluate()
        loss = res['eval_loss']; acc = res.get('eval_accuracy', float('nan'))
        cost = cost_of_policy(r_idx, b_idx, s_idx)
        losses.append(loss); accs.append(acc); costs.append(cost)
        print(f"{policy_kind} {task} Ep{ep}: Loss={loss:.4f} Acc={acc:.4f} Cost={cost:.2f}")
    return losses, accs, costs

if __name__ == '__main__':
    set_seed(42)
    os.makedirs('output/baselines', exist_ok=True)

    # Base model & tokenizer
    tok = AutoTokenizer.from_pretrained('gpt2-medium')
    base = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
    ).to('cuda')
    tok, base = fix_tokenizer_and_model(tok, base)

    # Controller
    loader = PolicyLoader('output/final_meta_controller_phase3e_corrected.pt', base)

    # MRPC: uniform, random, learned
    un = evaluate_policy(base, tok, 'mrpc', 'uniform', uniform_policy, loader, epochs=5)
    rd = evaluate_policy(base, tok, 'mrpc', 'random',  random_policy,  loader, epochs=5)
    ld = evaluate_policy(base, tok, 'mrpc', 'learned', None, loader, epochs=5)

    # Transfer: SST-2 and CoLA with learned
    sst = evaluate_policy(base, tok, 'sst2', 'learned', None, loader, epochs=5)
    cola = evaluate_policy(base, tok, 'cola', 'learned', None, loader, epochs=5)

    # Pareto (MRPC final epoch)
    pts = {'Uniform': (un[2][-1], un[0][-1]),
           'Random':  (rd[2][-1], rd[0][-1]),
           'Learned': (ld[2][-1], ld[0][-1])}
    plt.figure(figsize=(6,4))
    colors = {'Uniform':'blue','Random':'orange','Learned':'green'}
    for k,(cst,loss) in pts.items():
        plt.scatter(cst, loss, s=100, label=k, color=colors[k])
    plt.xlabel('Compression Cost (rank × bits × (1 - sparsity))')
    plt.ylabel('Eval Loss')
    plt.title('MRPC Pareto Frontier')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig('output/baselines/pareto_mrpc.png', dpi=300, bbox_inches='tight')
    print('Saved plot: output/baselines/pareto_mrpc.png')

    # Summary CSV
    import csv
    with open('output/baselines/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Policy','Task','FinalLoss','FinalAcc','FinalCost'])
        w.writerow(['Uniform','MRPC', un[0][-1], un[1][-1], un[2][-1]])
        w.writerow(['Random','MRPC',  rd[0][-1], rd[1][-1], rd[2][-1]])
        w.writerow(['Learned','MRPC',  ld[0][-1], ld[1][-1], ld[2][-1]])
        w.writerow(['Learned','SST2',  sst[0][-1], sst[1][-1], sst[2][-1]])
        w.writerow(['Learned','CoLA',  cola[0][-1], cola[1][-1], cola[2][-1]])
    print('Saved CSV: output/baselines/summary.csv')
