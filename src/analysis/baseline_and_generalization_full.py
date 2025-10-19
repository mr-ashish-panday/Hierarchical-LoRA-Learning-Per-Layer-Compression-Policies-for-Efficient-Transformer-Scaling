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

# -----------------------
# Config and utilities
# -----------------------
cfg_baseline = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9]
}

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
    if isinstance(idx, (int, np.integer, float)):
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
    return float(np.mean([R[r] * B[b] * (1 - S[s]) for r,b,s in zip(r_idx, b_idx, s_idx)]))

# -----------------------
# Policies
# -----------------------
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

        # Adaptive constructor
        self.controller = None
        try:
            # Positional attempt
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
            param_names = set(sig.parameters.keys())
            if 'hidden_dim' in param_names:
                ctor_kwargs['hidden_dim'] = hidden_dim
            elif 'controller_hidden' in param_names:
                ctor_kwargs['controller_hidden'] = hidden_dim
            self.controller = MetaController(**ctor_kwargs)

        self.controller = self.controller.to('cuda')
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cuda')
            self.controller.load_state_dict(sd)
            self.controller.eval()
            print(f"Loaded controller from {ckpt_path}")
        else:
            print(f"Warning: checkpoint not found: {ckpt_path}")

    def sample_actions(self, model, ep):
        # Simulate real features: task features (e.g., GLUE stats for MRPC/SST2)
        # Replace with actual Phase 2 features when available
        tf = torch.tensor([[0.5, 0.3, 0.2, 0.4, 0.1, 0.6]]).to('cuda')  # Example MRPC-like

        # Per-layer features (vary by layer index to simulate gradient norms, etc.)
        layer_indices = torch.arange(self.num_layers, dtype=torch.float32).to('cuda')
        lf = torch.stack([
            layer_indices / self.num_layers,  # Normalized layer position
            torch.sin(layer_indices / 10)     # Simulated layer stat variation
        ], dim=-1).unsqueeze(0).to('cuda')  # Shape: [1, num_layers, 2]

        with torch.no_grad():
            r_p, b_p, s_p = self.controller(tf, lf)
        r_idx = ensure_list_indices(torch.argmax(r_p, dim=-1).squeeze(), self.num_layers)
        b_idx = ensure_list_indices(torch.argmax(b_p, dim=-1).squeeze(), self.num_layers)
        s_idx = ensure_list_indices(torch.argmax(s_p, dim=-1).squeeze(), self.num_layers)
        print(f"Learned policy sample (first 5 layers): r={r_idx[:5]}, b={b_idx[:5]}, s={s_idx[:5]}")
        return r_idx, b_idx, s_idx

# -----------------------
# Evaluation (compatible with older Transformers)
# -----------------------
def evaluate_policy(base_model, tokenizer, task, policy_fn, policy_name, epochs=5):
    ds = load_dataset('glue', task)
    is_pair = task not in ['sst2']
    def prep(b):
        if is_pair:
            return tokenizer(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=128)
        return tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=128)
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    val_ds = data['validation']

    # Minimal compatible TrainingArguments
    args = TrainingArguments(
        output_dir=f'output/baselines/{policy_name}_{task}',
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        logging_dir=None
    )

    losses, accs, costs = [], [], []
    for ep in range(1, epochs+1):
        m = copy.deepcopy(base_model)
        L = count_linears(m)
        r_idx, b_idx, s_idx = policy_fn(m, ep)
        r_idx = ensure_list_indices(r_idx, L)
        b_idx = ensure_list_indices(b_idx, L)
        s_idx = ensure_list_indices(s_idx, L)
        m = apply_compression(m, r_idx, b_idx, s_idx, cfg_baseline)
        tokenizer, m = fix_tokenizer_and_model(tokenizer, m)
        trainer = Trainer(model=m, args=args, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
        res = trainer.evaluate()
        losses.append(res['eval_loss'])
        accs.append(res.get('eval_accuracy', float('nan')))
        costs.append(cost_of_policy(r_idx, b_idx, s_idx))
        print(f"{policy_name} {task} Ep{ep}: Loss={losses[-1]:.4f} Acc={accs[-1]:.4f} Cost={costs[-1]:.2f}")
    return losses, accs, costs

if __name__ == '__main__':
    os.makedirs('output/baselines', exist_ok=True)
    # Base model and tokenizer (binary classification for MRPC/SST2)
    tok = AutoTokenizer.from_pretrained('gpt2-medium')
    base = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
    ).to('cuda')
    tok, base = fix_tokenizer_and_model(tok, base)

    # Baselines and learned on MRPC
    loader = PolicyLoader('output/final_meta_controller_phase3e_corrected.pt', base)
    un = evaluate_policy(base, tok, 'mrpc', uniform_policy, 'uniform', epochs=5)
    rd = evaluate_policy(base, tok, 'mrpc', random_policy, 'random', epochs=5)
    ld = evaluate_policy(base, tok, 'mrpc', loader.sample_actions, 'learned', epochs=5)

    # Transfer to SST-2 with learned policy
    sst = evaluate_policy(base, tok, 'sst2', loader.sample_actions, 'learned', epochs=5)

    # Pareto on MRPC (final epoch)
    mrpc_points = {
        'Uniform': (un[2][-1], un[0][-1]),
        'Random':  (rd[2][-1], rd[0][-1]),
        'Learned': (ld[2][-1], ld[0][-1])
    }
    plt.figure(figsize=(6,4))
    plt.scatter(mrpc_points['Uniform'][0], mrpc_points['Uniform'][1], label='Uniform', color='blue', s=100)
    plt.scatter(mrpc_points['Random'][0], mrpc_points['Random'][1], label='Random', color='orange', s=100)
    plt.scatter(mrpc_points['Learned'][0], mrpc_points['Learned'][1], label='Learned', color='green', s=100)
    plt.xlabel('Compression Cost (rank × bits × (1 - sparsity))')
    plt.ylabel('Eval Loss')
    plt.title('MRPC Pareto Frontier')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('output/baselines/pareto_mrpc.png', dpi=300, bbox_inches='tight')
    print('Saved MRPC Pareto to output/baselines/pareto_mrpc.png')

    # Summary CSV
    import csv
    with open('output/baselines/summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Policy','Task','FinalLoss','FinalAcc','FinalCost'])
        w.writerow(['Uniform','MRPC', un[0][-1], un[1][-1], un[2][-1]])
        w.writerow(['Random','MRPC', rd[0][-1], rd[1][-1], rd[2][-1]])
        w.writerow(['Learned','MRPC', ld[0][-1], ld[1][-1], ld[2][-1]])
        w.writerow(['Learned','SST2', sst[0][-1], sst[1][-1], sst[2][-1]])
    print('Saved summary to output/baselines/summary.csv')
