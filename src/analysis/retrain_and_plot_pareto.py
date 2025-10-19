#!/usr/bin/env python3
import os, sys, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import csv

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
    if isinstance(idx, int):
        return [idx]*L
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().tolist()
        if isinstance(idx, int):
            return [idx]*L
        return list(idx)[:L]
    if isinstance(idx, (list,tuple)):
        if len(idx)==0: return [0]*L
        if len(idx)<L: return list(idx)+[idx[-1]]*(L-len(idx))
        return list(idx)[:L]
    return [0]*L

def collate(batch):
    input_ids = torch.stack([b['input_ids'] if isinstance(b['input_ids'],torch.Tensor) else torch.tensor(b['input_ids']) for b in batch])
    attention = torch.stack([b['attention_mask'] if isinstance(b['attention_mask'],torch.Tensor) else torch.tensor(b['attention_mask']) for b in batch])
    labels = torch.tensor([b['labels'] for b in batch])
    return {'input_ids':input_ids,'attention_mask':attention,'labels':labels}

# Load tokenizer and base model
tok = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium',num_labels=2,ignore_mismatched_sizes=True).to('cuda')
if tok.pad_token is None:
    tok.pad_token=tok.eos_token; tok.pad_token_id=tok.eos_token_id
model.config.pad_token_id=tok.pad_token_id

# Prepare MRPC validation dataset
raw=load_dataset('glue','mrpc')
def prep_fn(ex): return tok(ex['sentence1'],ex['sentence2'],padding='max_length',truncation=True,max_length=128)
val=raw['validation'].map(prep_fn,batched=True).rename_column('label','labels').set_format(type='torch',columns=['input_ids','attention_mask','labels'])
loader=DataLoader(val,batch_size=16,collate_fn=collate)

loss_fn=nn.CrossEntropyLoss()
L=count_lin(model)

# Baseline policies
uniform=([cfg['ranks'].index(8)]*L,[cfg['bitwidths'].index(8)]*L,[cfg['sparsities'].index(0.5)]*L)
rand=([random.randrange(len(cfg['ranks'])) for _ in range(L)],[random.randrange(len(cfg['bitwidths'])) for _ in range(L)],[random.randrange(len(cfg['sparsities'])) for _ in range(L)])

# Fixed features
tf=torch.zeros(1,6).to('cuda')
lf=torch.zeros(1,L,2).to('cuda')

# Evaluate helper
def eval_policy(r_idx,b_idx,s_idx):
    r_idx,b_idx,s_idx=ensure_list(r_idx,L),ensure_list(b_idx,L),ensure_list(s_idx,L)
    m=copy.deepcopy(model)
    m=apply_compression(m,r_idx,b_idx,s_idx,cfg).to('cuda')
    m.eval()
    total_loss,total=0.0,0
    with torch.no_grad():
        for batch in loader:
            out=m(batch['input_ids'].to('cuda'),attention_mask=batch['attention_mask'].to('cuda'))
            loss=loss_fn(out.logits,batch['labels'].to('cuda'))
            total_loss+=loss.item()*batch['labels'].size(0); total+=batch['labels'].size(0)
    return total_loss/total,cost(r_idx,b_idx,s_idx)

# Evaluate baselines
print("Evaluating uniform...")
u_loss,u_cost=eval_policy(*uniform)
print(f"Uniform: Loss={u_loss:.4f}, Cost={u_cost:.2f}")
print("Evaluating random...")
r_loss,r_cost=eval_policy(*rand)
print(f"Random: Loss={r_loss:.4f}, Cost={r_cost:.2f}")

# Retrain controller with cost penalty for multiple lambdas
lambdas=[0.0,0.01,0.05,0.1]
learned_policies=[]

for lam in lambdas:
    print(f"\nTraining controller with lambda={lam}...")
    ctrl=MetaController(6,2,64,cfg['ranks'],cfg['bitwidths'],cfg['sparsities'],num_layers=L).to('cuda')
    opt=torch.optim.Adam(ctrl.parameters(),lr=1e-3)
    
    # Simple synthetic training loop (replace with your real training data/loop)
    for epoch in range(50):
        opt.zero_grad()
        # Synthetic task/layer features
        tf_train=torch.rand(1,6).to('cuda')
        lf_train=torch.rand(1,L,2).to('cuda')
        r_p,b_p,s_p=ctrl(tf_train,lf_train)
        # Sample indices
        r_idx=torch.argmax(r_p,dim=-1).squeeze()
        b_idx=torch.argmax(b_p,dim=-1).squeeze()
        s_idx=torch.argmax(s_p,dim=-1).squeeze()
        # Compute cost
        r_list=ensure_list(r_idx,L)
        b_list=ensure_list(b_idx,L)
        s_list=ensure_list(s_idx,L)
        c=cost(r_list,b_list,s_list)
        # Synthetic performance loss (random for demo; replace with real validation loss)
        perf_loss=torch.rand(1).to('cuda')
        total_loss=perf_loss+lam*c
        total_loss.backward()
        opt.step()
    
    # Evaluate this trained controller
    ctrl.eval()
    with torch.no_grad():
        r_p,b_p,s_p=ctrl(tf,lf)
    r_idx=ensure_list(torch.argmax(r_p,dim=-1).squeeze(),L)
    b_idx=ensure_list(torch.argmax(b_p,dim=-1).squeeze(),L)
    s_idx=ensure_list(torch.argmax(s_p,dim=-1).squeeze(),L)
    l_loss,l_cost=eval_policy(r_idx,b_idx,s_idx)
    learned_policies.append((lam,l_loss,l_cost))
    print(f"Learned (lambda={lam}): Loss={l_loss:.4f}, Cost={l_cost:.2f}")

# Plot Pareto frontier
plt.figure(figsize=(7,5))
plt.scatter(u_cost,u_loss,label='Uniform',color='blue',s=120,marker='o')
plt.scatter(r_cost,r_loss,label='Random',color='orange',s=120,marker='s')
for lam,l,c in learned_policies:
    plt.scatter(c,l,label=f'Learned λ={lam}',s=120,marker='x')
plt.xlabel('Compression Cost (rank × bits × (1 - sparsity))')
plt.ylabel('Validation Loss')
plt.title('MRPC Pareto Frontier: Baselines vs. Learned Policies')
plt.legend()
plt.grid(True,alpha=0.3)
os.makedirs('output/baselines',exist_ok=True)
plt.savefig('output/baselines/pareto_paper_ready.png',dpi=300,bbox_inches='tight')
print('\nSaved Pareto plot: output/baselines/pareto_paper_ready.png')

# Save CSV summary
with open('output/baselines/pareto_summary.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['Policy','Lambda','Loss','Cost'])
    w.writerow(['Uniform','N/A',u_loss,u_cost])
    w.writerow(['Random','N/A',r_loss,r_cost])
    for lam,l,c in learned_policies:
        w.writerow(['Learned',lam,l,c])
print('Saved CSV: output/baselines/pareto_summary.csv')
