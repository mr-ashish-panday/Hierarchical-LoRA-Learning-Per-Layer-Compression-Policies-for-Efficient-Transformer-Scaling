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

# Project root for imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Compression config
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
    if isinstance(idx, (int, float, np.integer)):
        return [int(idx)] * L
    if not isinstance(idx, (list, tuple)):
        return [int(idx)] * L
    if len(idx) < L:
        return list(idx) + [idx[-1]]*(L-len(idx))
    if len(idx) > L:
        return idx[:L]
    return list(map(int, idx))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).astype(np.float32).mean().item()
    return {'accuracy': float(acc)}

def cost_of_policy(r_idx, b_idx, s_idx):
    R,B,S = cfg_baseline['ranks'], cfg_baseline['bitwidths'], cfg_baseline['sparsities']
    return float(np.mean([R[r]*B[b]*(1-S[s]) for r,b,s in zip(r_idx,b_idx,s_idx)]))

# Uniform & random policies
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

# Learned policy loader
class PolicyLoader:
    def __init__(self, ckpt, base_model):
        self.num_layers = count_linears(base_model)
        self.controller = MetaController(
            task_feat_dim=6,
            layer_feat_dim=2,
            hidden_dim=64,
            ranks=cfg_baseline['ranks'],
            bitwidths=cfg_baseline['bitwidths'],
            sparsities=cfg_baseline['sparsities'],
            num_layers=self.num_layers
        ).to('cuda')
        sd = torch.load(ckpt, map_location='cuda')
        self.controller.load_state_dict(sd)
        self.controller.eval()
        print(f"Loaded controller from {ckpt}")

    def sample_actions(self, model, ep):
        # TODO: Replace this with your actual Phase 2 MRPC features
        tf = torch.tensor([[/* your 6-dim MRPC task feature vector */]], device='cuda')

        # TODO: Replace this with your actual per-layer features (num_layers × 2)
        lf = torch.tensor([[[/* layer_feat_dim values per layer */]]], device='cuda')

        with torch.no_grad():
            r_p, b_p, s_p = self.controller(tf, lf)
        r_idx = ensure_list_indices(torch.argmax(r_p, -1).squeeze(), self.num_layers)
        b_idx = ensure_list_indices(torch.argmax(b_p, -1).squeeze(), self.num_layers)
        s_idx = ensure_list_indices(torch.argmax(s_p, -1).squeeze(), self.num_layers)
        print("Learned indices sample:", r_idx[:5], b_idx[:5], s_idx[:5])
        return r_idx, b_idx, s_idx

# Evaluation
def evaluate_policy(base, tok, task, policy_fn, name, epochs=5):
    ds = load_dataset('glue', task)
    is_pair = task not in ['sst2','cola']
    def prep(b):
        if is_pair:
            return tok(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=128)
        return tok(b['sentence'], padding='max_length', truncation=True, max_length=128)
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    val_ds = data['validation']

    args = TrainingArguments(
        output_dir=f'output/baselines/{name}_{task}',
        per_device_eval_batch_size=16,
        do_train=False, do_eval=True, logging_dir=None
    )

    Ls, As, Cs = [], [], []
    for ep in range(1, epochs+1):
        m = copy.deepcopy(base)
        r,b,s = policy_fn(m, ep)
        L = count_linears(m)
        r, b, s = ensure_list_indices(r,L), ensure_list_indices(b,L), ensure_list_indices(s,L)
        m = apply_compression(m, r, b, s, cfg_baseline)
        tok, m = fix_tokenizer_and_model(tok, m)
        trainer = Trainer(model=m, args=args, eval_dataset=val_ds, tokenizer=tok, compute_metrics=compute_metrics)
        res = trainer.evaluate()
        Ls.append(res['eval_loss'])
        As.append(res.get('eval_accuracy', float('nan')))
        Cs.append(cost_of_policy(r,b,s))
        print(f"{name} {task} Ep{ep}: Loss={Ls[-1]:.4f}, Acc={As[-1]:.4f}, Cost={Cs[-1]:.2f}")
    return Ls, As, Cs

if __name__ == '__main__':
    os.makedirs('output/baselines', exist_ok=True)
    tok = AutoTokenizer.from_pretrained('gpt2-medium')
    base = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2, ignore_mismatched_sizes=True).to('cuda')
    tok, base = fix_tokenizer_and_model(tok, base)

    loader = PolicyLoader('output/final_meta_controller_phase3e_corrected.pt', base)

    # MRPC baselines & learned
    evaluate_policy(base, tok, 'mrpc', uniform_policy, 'uniform')
    evaluate_policy(base, tok, 'mrpc', random_policy,  'random')
    evaluate_policy(base, tok, 'mrpc', loader.sample_actions, 'learned')

    # Transfer SST-2
    evaluate_policy(base, tok, 'sst2', loader.sample_actions, 'learned')
