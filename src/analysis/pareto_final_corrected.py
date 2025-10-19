#!/usr/bin/env python3
import os, sys, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt

# Add src path
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Config
cfg = {'ranks':[1,2,4,8,16],'bitwidths':[2,4,8,16,32],'sparsities':[0.0,0.2,0.5,0.7,0.9]}

# Utilities
def count_lin(m): return len([x for x in m.modules() if isinstance(x, nn.Linear)])
def cost(r,b,s): return np.mean([cfg['ranks'][ri]*cfg['bitwidths'][bi]*(1-cfg['sparsities'][si]) for ri,bi,si in zip(r,b,s)])

def ensure_list(idx, L):
    # Ensure idx is a list of length L
    if isinstance(idx, int):
        return [idx]*L
    if isinstance(idx, (list,tuple)):
        if len(idx) == 0:
            return [0]*L
        if len(idx) < L:
            return list(idx) + [idx[-1]]*(L-len(idx))
        return list(idx)[:L]
    # idx might be scalar tensor -> convert
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().tolist()
        if isinstance(idx, int):
            return [idx]*L
        return list(idx)[:L]
    return [0]*L

def collate(batch):
    input_ids = torch.stack([b['input_ids'] if isinstance(b['input_ids'], torch.Tensor) else torch.tensor(b['input_ids']) for b in batch])
    attention = torch.stack([b['attention_mask'] if isinstance(b['attention_mask'], torch.Tensor) else torch.tensor(b['attention_mask']) for b in batch])
    labels = torch.tensor([b['labels'] for b in batch])
    return {'input_ids':input_ids, 'attention_mask':attention, 'labels':labels}

# Load tokenizer and base model
tok = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2, ignore_mismatched_sizes=True).to('cuda')
# Ensure pad token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

# Prepare MRPC validation dataset
raw = load_dataset('glue','mrpc')
def prep_fn(ex):
    return tok(ex['sentence1'], ex['sentence2'], padding='max_length', truncation=True, max_length=128)
val = raw['validation'].map(prep_fn, batched=True).rename_column('label','labels')
val.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
loader = DataLoader(val, batch_size=16, collate_fn=collate)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Baseline policies
L = count_lin(model)
uniform = ([cfg['ranks'].index(8)]*L, [cfg['bitwidths'].index(8)]*L, [cfg['sparsities'].index(0.5)]*L)
rand = (
    [random.randrange(len(cfg['ranks'])) for _ in range(L)],
    [random.randrange(len(cfg['bitwidths'])) for _ in range(L)],
    [random.randrange(len(cfg['sparsities'])) for _ in range(L)]
)

# Load controller
ctrl = MetaController(6,2,64, cfg['ranks'],cfg['bitwidths'],cfg['sparsities'], num_layers=L).to('cuda')
ctrl.load_state_dict(torch.load('output/final_meta_controller_phase3e_corrected.pt', map_location='cuda'))
ctrl.eval()

# Fixed features (replace with real ones if available)
tf = torch.zeros(1,6).to('cuda')
lf = torch.zeros(1,L,2).to('cuda')

# Evaluate a policy given rank,bit,sparsity lists
def eval_policy(r_idx,b_idx,s_idx):
    # Ensure lists
    r_idx = ensure_list(r_idx, L)
    b_idx = ensure_list(b_idx, L)
    s_idx = ensure_list(s_idx, L)
    m = copy.deepcopy(model)
    m = apply_compression(m, r_idx, b_idx, s_idx, cfg).to('cuda')
    m.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to('cuda')
            att_mask  = batch['attention_mask'].to('cuda')
            labels    = batch['labels'].to('cuda')
            outputs = m(input_ids, attention_mask=att_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()*labels.size(0)
            total += labels.size(0)
    return total_loss/total, cost(r_idx,b_idx,s_idx)

# Baseline results
print("Evaluating uniform policy...")
u_loss,u_cost = eval_policy(*uniform)
print(f"Uniform: Loss={u_loss:.4f}, Cost={u_cost:.2f}")
print("Evaluating random policy...")
r_loss,r_cost = eval_policy(*rand)
print(f"Random: Loss={r_loss:.4f}, Cost={r_cost:.2f}")

# Temperature sweep
temps = [0.5,1.0,2.0,5.0]
sweep = []
with torch.no_grad():
    logits_r,logits_b,logits_s = ctrl(tf, lf)
for T in temps:
    pr = torch.softmax(logits_r / T, dim=-1)
    pb = torch.softmax(logits_b / T, dim=-1)
    ps = torch.softmax(logits_s / T, dim=-1)
    r_idx = pr.argmax(-1).squeeze()
    b_idx = pb.argmax(-1).squeeze()
    s_idx = ps.argmax(-1).squeeze()
    # Convert to lists
    r_idx = ensure_list(r_idx, L)
    b_idx = ensure_list(b_idx, L)
    s_idx = ensure_list(s_idx, L)
    print(f"Evaluating learned T={T}...")
    l,c = eval_policy(r_idx,b_idx,s_idx)
    sweep.append((T,l,c))
    print(f"T={T}: Loss={l:.4f}, Cost={c:.2f}")

# Plot
plt.figure(figsize=(6,4))
plt.scatter(u_cost,u_loss,label='Uniform',color='blue',s=100)
plt.scatter(r_cost,r_loss,label='Random',color='orange',s=100)
for T,l,c in sweep:
    plt.scatter(c,l,label=f'Learned T={T}',marker='x',s=100)
plt.xlabel('Compression Cost')
plt.ylabel('Eval Loss')
plt.title('MRPC Pareto Frontier (Temperature Sweep)')
plt.legend()
plt.grid(True,alpha=0.3)
os.makedirs('output/baselines', exist_ok=True)
plt.savefig('output/baselines/pareto_final.png', dpi=300, bbox_inches='tight')
print('Saved plot to output/baselines/pareto_final.png')
