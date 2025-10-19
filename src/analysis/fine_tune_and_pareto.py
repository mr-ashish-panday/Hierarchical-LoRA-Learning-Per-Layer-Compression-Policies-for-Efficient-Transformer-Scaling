#!/usr/bin/env python3
"""
Fine-tune MetaController with real performance loss + cost penalty,
then evaluate at multiple temperatures for a true Pareto frontier.
Optimized for 12GB GPU.
"""
import os, sys, random, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import csv

# Project imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)

CFG = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9],
}

def count_linears(m): return len([l for l in m.modules() if isinstance(l, nn.Linear)])
def calc_cost(r,b,s):
    R,B,S = CFG['ranks'], CFG['bitwidths'], CFG['sparsities']
    return float(np.mean([R[ri]*B[bi]*(1-S[si]) for ri,bi,si in zip(r,b,s)]))
def ensure_list(idx,L):
    if isinstance(idx,int): return [idx]*L
    if isinstance(idx,torch.Tensor): idx=idx.detach().cpu().tolist()
    if isinstance(idx,(list,tuple)):
        if len(idx)>=L: return idx[:L]
        return idx+[idx[-1]]*(L-len(idx))
    return [0]*L

# Prepare tokenizer & base model
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
base_model = AutoModelForSequenceClassification.from_pretrained(
    'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
).to(DEVICE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
base_model.config.pad_token_id = tokenizer.pad_token_id

# Load & tokenize MRPC
raw = load_dataset('glue','mrpc')
def tok_fn(ex):
    return tokenizer(ex['sentence1'], ex['sentence2'],
                     padding='max_length', truncation=True, max_length=128)
tokd = raw.map(tok_fn, batched=True, remove_columns=['sentence1','sentence2','idx'])
tokd = tokd.rename_column('label','labels')
tokd.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
train_ds = tokd['train']
val_ds   = tokd['validation']

# Small subset for fine-tuning
subset_idx = list(range(min(32, len(train_ds))))
train_small = Subset(train_ds, subset_idx)
train_loader = DataLoader(train_small, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds,    batch_size=8)

num_layers = count_linears(base_model)

# Load pretrained controller
checkpoint = 'output/final_meta_controller_phase3e_corrected.pt'
controller_base = MetaController(
    task_feat_dim=6, layer_feat_dim=2, hidden_dim=64,
    ranks=CFG['ranks'], bitwidths=CFG['bitwidths'], sparsities=CFG['sparsities'],
    num_layers=num_layers
).to(DEVICE)
controller_base.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

# Fine-tune controller with cost penalty
lambdas = [0.01, 0.05, 0.1]
trained_ckpts = []
for lam in lambdas:
    ctrl = copy.deepcopy(controller_base).train()
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-4)
    for epoch in range(5):
        for batch in train_loader:
            inp = batch['input_ids'].to(DEVICE)
            att = batch['attention_mask'].to(DEVICE)
            lab = batch['labels'].to(DEVICE)

            # Create dummy real features
            task_feat = torch.tensor([[128/128,0.0,1.0, torch.mean(lab.float()).item(), 1.0, len(train_ds)/10000.0]],
                                     dtype=torch.float32, device=DEVICE)
            lf = torch.zeros(1, num_layers, 2, device=DEVICE, dtype=torch.float32)

            r_log,b_log,s_log = ctrl(task_feat, lf)
            r_idx = torch.softmax(r_log, -1).argmax(-1).squeeze()
            b_idx = torch.softmax(b_log, -1).argmax(-1).squeeze()
            s_idx = torch.softmax(s_log, -1).argmax(-1).squeeze()
            r_list = ensure_list(r_idx, num_layers)
            b_list = ensure_list(b_idx, num_layers)
            s_list = ensure_list(s_idx, num_layers)

            # Compress and forward
            m = apply_compression(copy.deepcopy(base_model), r_list, b_list, s_list, CFG).to(DEVICE)
            out = m(input_ids=inp, attention_mask=att)
            perf_loss = nn.CrossEntropyLoss()(out.logits, lab)
            cost = calc_cost(r_list, b_list, s_list)
            loss = perf_loss + lam * cost

            opt.zero_grad()
            loss.backward()
            opt.step()

    out_ckpt = f'output/baselines/controller_lam{lam}.pt'
    torch.save(ctrl.state_dict(), out_ckpt)
    trained_ckpts.append((lam, out_ckpt))
    print(f"Saved controller λ={lam} -> {out_ckpt}")

# Evaluation function
def eval_ctrl(ctrl, temp):
    ctrl.eval()
    task_feat = torch.tensor([[128/128,0.0,1.0,0.5,1.0, len(val_ds)/10000.0]],
                             dtype=torch.float32, device=DEVICE)
    lf = torch.zeros(1, num_layers, 2, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        r_log,b_log,s_log = ctrl(task_feat, lf)
    r_idx = torch.softmax(r_log/temp, -1).argmax(-1).squeeze()
    b_idx = torch.softmax(b_log/temp, -1).argmax(-1).squeeze()
    s_idx = torch.softmax(s_log/temp, -1).argmax(-1).squeeze()
    r_list = ensure_list(r_idx, num_layers)
    b_list = ensure_list(b_idx, num_layers)
    s_list = ensure_list(s_idx, num_layers)

    m = apply_compression(copy.deepcopy(base_model), r_list, b_list, s_list, CFG).to(DEVICE)
    m.eval()
    total_loss, total = 0.0, 0
    for batch in val_loader:
        inp = batch['input_ids'].to(DEVICE)
        att = batch['attention_mask'].to(DEVICE)
        lab = batch['labels'].to(DEVICE)
        out = m(input_ids=inp, attention_mask=att)
        l = nn.CrossEntropyLoss()(out.logits, lab)
        total_loss += l.item()*lab.size(0); total += lab.size(0)
    return total_loss/total, calc_cost(r_list, b_list, s_list)

# Compute results
results = [("Uniform, T=1.0", *eval_ctrl(controller_base, 1.0))]
for lam, ck in trained_ckpts:
    for temp in [0.5, 1.0, 2.0]:
        ctrl = MetaController(
            6,2,64, CFG['ranks'], CFG['bitwidths'], CFG['sparsities'],
            num_layers=num_layers
        ).to(DEVICE)
        ctrl.load_state_dict(torch.load(ck, map_location=DEVICE))
        loss, cost = eval_ctrl(ctrl, temp)
        results.append((f"λ={lam}, T={temp}", loss, cost))

# Plot & save
plt.figure(figsize=(6,4))
for name, loss, cost in results:
    plt.scatter(cost, loss, label=name)
plt.xlabel('Cost'); plt.ylabel('Loss'); plt.legend(fontsize=8); plt.grid(True)
os.makedirs('output/baselines', exist_ok=True)
plt.savefig('output/baselines/paper_ready_pareto.png', dpi=300)
with open('output/baselines/paper_ready.csv','w') as f:
    w=csv.writer(f); w.writerow(['Policy','Loss','Cost']); w.writerows(results)
print("Done. Plot and CSV in output/baselines/")
