#!/usr/bin/env python3
"""
Hierarchical LoRA Compression Controller — GPT-2 Medium
Detects and compresses all feed-forward layers (Conv1D/Linear) in GPT-2 Medium,
training a meta-controller with policy gradients for Pareto-optimal performance vs cost.
"""

import os, sys, copy, random, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from transformers.models.gpt2.modeling_gpt2 import Conv1D

# Ensure src on path
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(proj, 'src'))

from analysis.apply_compression_simple import apply_compression, LowRankLinear
from models.meta_controller import MetaController

# Config & Seeds
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)

CFG = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9],
}
TRAIN_CFG = {
    'lr':1e-3, 'epochs':8, 'batch_size':4, 'val_batch_size':8,
    'entropy_coef':0.01,'baseline_momentum':0.9,'log_interval':25,'max_train_samples':200
}

# Utility functions
def count_linears(model):
    cnt=0
    for _,m in model.named_modules():
        if isinstance(m,(nn.Linear,LowRankLinear,Conv1D)): cnt+=1
    return cnt

def ensure_list(x,L):
    if isinstance(x,torch.Tensor):
        lst=x.squeeze().detach().cpu().tolist()
        vals=[int(lst)] if isinstance(lst,(int,float)) else [int(v) for v in lst]
    elif isinstance(x,(int,float)):
        vals=[int(x)]
    else:
        vals=[int(v) for v in x]
    return vals if len(vals)==L else vals+[vals[-1]]*(L-len(vals))

def normalize_logits(logits,L):
    if isinstance(logits,(list,tuple)): logits=logits[0]
    if logits.dim()==1: return logits.unsqueeze(0).repeat(L,1)
    if logits.dim()==2: return logits.repeat(L,1) if logits.size(0)==1 else logits
    if logits.dim()==3: return logits.squeeze(0)[:L]
    raise RuntimeError(f"Unexpected logits shape {logits.shape}")

def compute_cost(r_idx,b_idx,s_idx):
    return float(np.mean([CFG['ranks'][ri]*CFG['bitwidths'][bi]*(1-CFG['sparsities'][si])
        for ri,bi,si in zip(r_idx,b_idx,s_idx)]))

# Data pipeline
tokenizer=AutoTokenizer.from_pretrained('gpt2-medium')
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token;tokenizer.pad_token_id=tokenizer.eos_token_id
raw=load_dataset('glue','sst2')
def tok_fn(ex): return tokenizer(ex['sentence'],padding='max_length',truncation=True,max_length=128)
tokd=raw.map(tok_fn,batched=True,remove_columns=['sentence','idx'])
tokd=tokd.rename_column('label','labels'); tokd.set_format('torch',['input_ids','attention_mask','labels'])
train_ds=Subset(tokd['train'],list(range(min(TRAIN_CFG['max_train_samples'],len(tokd['train'])))))
train_loader=DataLoader(train_ds,batch_size=TRAIN_CFG['batch_size'],shuffle=True)
val_loader=DataLoader(tokd['validation'],batch_size=TRAIN_CFG['val_batch_size'],shuffle=False)

# Model setup
base_model=AutoModelForSequenceClassification.from_pretrained(
    'gpt2-medium',num_labels=2,ignore_mismatched_sizes=True
).to(DEVICE)
if base_model.config.pad_token_id is None: base_model.config.pad_token_id=tokenizer.pad_token_id
NL=count_linears(base_model)
print(f"[INFO] Detected {NL} compressible modules.")
ctrl_template=MetaController(6,2,64,CFG['ranks'],CFG['bitwidths'],CFG['sparsities'],num_layers=NL).to(DEVICE)
tpl_ckpt='output/final_meta_controller_phase3e_corrected.pt'
if os.path.exists(tpl_ckpt): ctrl_template.load_state_dict(torch.load(tpl_ckpt,map_location=DEVICE))

# Sanity check
def sanity_check():
    m=copy.deepcopy(base_model).cpu()
    orig=sum(p.numel() for p in m.parameters());L=NL
    policies=[([0]*L,[0]*L,[0]*L),([len(CFG['ranks'])-1]*L,[len(CFG['bitwidths'])-1]*L,[len(CFG['sparsities'])-1]*L)]
    for name,(r,b,s) in zip(['min','max'],policies):
        cm=apply_compression(copy.deepcopy(m),r,b,s,CFG)
        comp=sum(q.numel() for q in cm.parameters())
        print(f"[Sanity:{name}] orig={orig},comp={comp},ratio={comp/orig:.4f}")
sanity_check()

# Training function
def train_budget(target):
    ctrl=copy.deepcopy(ctrl_template).train().to(DEVICE)
    opt=torch.optim.Adam(ctrl.parameters(),lr=TRAIN_CFG['lr']);baseline=0.0
    for ep in range(TRAIN_CFG['epochs']):
        for step,batch in enumerate(train_loader):
            inp,att,lbl=[batch[k].to(DEVICE) for k in ['input_ids','attention_mask','labels']]
            tf=torch.zeros(1,6,device=DEVICE); lf=torch.zeros(1,NL,2,device=DEVICE)
            r_log,b_log,s_log=ctrl(tf,lf)
            r_log, b_log, s_log = normalize_logits(r_log,NL), normalize_logits(b_log,NL), normalize_logits(s_log,NL)
            r_cat,b_cat,s_cat=Categorical(logits=r_log),Categorical(logits=b_log),Categorical(logits=s_log)
            r_s,b_s,s_s=r_cat.sample().squeeze(),b_cat.sample().squeeze(),s_cat.sample().squeeze()
            r_idx,b_idx,s_idx=ensure_list(r_s,NL),ensure_list(b_s,NL),ensure_list(s_s,NL)
            M=apply_compression(copy.deepcopy(base_model).cpu(),r_idx,b_idx,s_idx,CFG).to(DEVICE);M.eval()
            with torch.no_grad(): perf=F.cross_entropy(M(input_ids=inp,attention_mask=att).logits,lbl).item()
            cost=compute_cost(r_idx,b_idx,s_idx)
            reward=-perf - (target/max(target,1e-6))*cost
            adv=reward-baseline; baseline=TRAIN_CFG['baseline_momentum']*baseline+(1-TRAIN_CFG['baseline_momentum'])*reward
            logp=r_cat.log_prob(r_s).sum()+b_cat.log_prob(b_s).sum()+s_cat.log_prob(s_s).sum()
            ent=r_cat.entropy().sum()+b_cat.entropy().sum()+s_cat.entropy().sum()
            loss=-adv*logp - TRAIN_CFG['entropy_coef']*ent
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ctrl.parameters(),0.5); opt.step()
            if step%TRAIN_CFG['log_interval']==0:
                print(f"[Train] ep{ep} step{step} perf={perf:.4f} cost={cost:.2f} reward={reward:.4f}")
    return ctrl.eval()

# Main run
os.makedirs('output/baselines',exist_ok=True)
results=[]
for T in [2.0,8.0,16.0,32.0]:
    print(f"**** Training for budget {T} ****")
    ctrl=train_budget(T); torch.save(ctrl.state_dict(),f'output/baselines/ctrl_{int(T)}.pt')
    for temp in [0.5,1.0,2.0]:
        tf,lf=torch.zeros(1,6,device=DEVICE),torch.zeros(1,NL,2,device=DEVICE)
        with torch.no_grad(): r,b,s=ctrl(tf,lf)
        r,b,s=normalize_logits(r,NL),normalize_logits(b,NL),normalize_logits(s,NL)
        r_idx,b_idx,s_idx=ensure_list(r.argmax(-1).squeeze(),NL),ensure_list(b.argmax(-1).squeeze(),NL),ensure_list(s.argmax(-1).squeeze(),NL)
        M=apply_compression(copy.deepcopy(base_model).cpu(),r_idx,b_idx,s_idx,CFG).to(DEVICE);M.eval()
        tot,n=0,0
        for batch in val_loader:
            out=M(input_ids=batch['input_ids'].to(DEVICE),attention_mask=batch['attention_mask'].to(DEVICE))
            l=F.cross_entropy(out.logits,batch['labels'].to(DEVICE)).item()
            tot+=l*batch['labels'].size(0); n+=batch['labels'].size(0)
        avg_loss=tot/n; cost=compute_cost(r_idx,b_idx,s_idx)
        results.append((f"T={int(T)},T={temp}",avg_loss,cost))
        print(f"[Eval] T={T},temp={temp} loss={avg_loss:.4f} cost={cost:.2f}")

plt.figure(figsize=(6,4))
for name,loss,cost in results:
    plt.scatter(cost,loss,label=name)
plt.xlabel('Cost'); plt.ylabel('Loss'); plt.legend(fontsize=8); plt.grid(); plt.tight_layout()
plt.savefig('output/baselines/final_pareto.png',dpi=300)
with open('output/baselines/final_pareto.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['Policy','Loss','Cost']); w.writerows(results)
print("Done. See output/baselines/")
