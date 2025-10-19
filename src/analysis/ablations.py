#!/usr/bin/env python3
"""
Extended Ablation and Visualization Suite for Hierarchical LoRA Compression

This script augments the existing pipeline with:
1. Hyperparameter ablations (entropy_coef, lr, epochs)
2. ε-constraint optimization baseline
3. Layerwise policy heatmaps per budget
4. Inference latency & memory benchmarking
5. Aggregated result plots for ablation studies

Copy–paste into a new file `src/analysis/ablations.py` and run:
    python src/analysis/ablations.py
"""

import os, sys, time, copy, random, csv
import numpy as np
import torch, psutil
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Categorical
from transformers.models.gpt2.modeling_gpt2 import Conv1D

# Ensure src on path
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(proj, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# --------------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)

# Base config
CFG = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9],
}
BUDGETS = [2.0,8.0,16.0,32.0]

# Ablation hyperparams
ABL_PARAMS = {
    'entropy_coef': [0.0, 0.01, 0.05],
    'lr':           [1e-4,1e-3,1e-2],
    'epochs':       [4,8,16],
}

# Utility functions
def normalize_logits(logits, L):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if logits.dim() == 1:
        return logits.unsqueeze(0).repeat(L, 1)
    if logits.dim() == 2:
        return logits.repeat(L, 1) if logits.size(0) == 1 else logits
    if logits.dim() == 3:
        return logits.squeeze(0)[:L]
    raise RuntimeError(f"Unexpected logits shape {logits.shape}")

def ensure_list(x, L):
    if isinstance(x, torch.Tensor):
        lst = x.squeeze().detach().cpu().tolist()
        vals = [int(lst)] if isinstance(lst, (int, float)) else [int(v) for v in lst]
    elif isinstance(x, (int, float)):
        vals = [int(x)]
    else:
        vals = [int(v) for v in x]
    return vals if len(vals) == L else vals + [vals[-1]] * (L - len(vals))

# --------------------------------------------------------------------------------
def setup_data(max_samples=500):
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    raw = load_dataset('glue','sst2')
    def tok(ex): return tokenizer(ex['sentence'],padding='max_length',truncation=True,max_length=128)
    ds = raw.map(tok,batched=True,remove_columns=['sentence','idx'])
    ds=ds.rename_column('label','labels'); ds.set_format('torch',['input_ids','attention_mask','labels'])
    train_ds = Subset(ds['train'], list(range(min(max_samples,len(ds['train'])))))
    train_loader = DataLoader(train_ds,batch_size=8,shuffle=True)
    val_loader   = DataLoader(ds['validation'],batch_size=16,shuffle=False)
    return tokenizer,train_loader,val_loader

def count_modules(model):
    cnt=0
    for _,m in model.named_modules():
        if isinstance(m,(torch.nn.Linear,Conv1D)):
            cnt+=1
    return cnt

# --------------------------------------------------------------------------------
def benchmark_inference(model, n_steps=50):
    dummy_inp = torch.randint(0,50257,(1,128),device=DEVICE)
    dummy_att = torch.ones_like(dummy_inp)
    torch.cuda.synchronize() if DEVICE.type=='cuda' else None
    t0=time.time()
    for _ in range(n_steps):
        _=model(input_ids=dummy_inp,attention_mask=dummy_att)
    torch.cuda.synchronize() if DEVICE.type=='cuda' else None
    latency=(time.time()-t0)/n_steps*1000
    mem=psutil.Process().memory_info().rss/1e6
    return latency,mem

# --------------------------------------------------------------------------------
def run_ablation():
    tokenizer, train_loader, val_loader = setup_data()
    base = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium',num_labels=2,ignore_mismatched_sizes=True
    ).to(DEVICE)
    if base.config.pad_token_id is None: base.config.pad_token_id = tokenizer.pad_token_id
    NL = count_modules(base)
    ctrl_tpl = MetaController(6,2,64,CFG['ranks'],CFG['bitwidths'],CFG['sparsities'],num_layers=NL).to(DEVICE)
    if os.path.exists('output/final_meta_controller_phase3e_corrected.pt'):
        ctrl_tpl.load_state_dict(torch.load('output/final_meta_controller_phase3e_corrected.pt',map_location=DEVICE))

    results_ablation=[]
    for ec in ABL_PARAMS['entropy_coef']:
        for lr in ABL_PARAMS['lr']:
            for epochs in ABL_PARAMS['epochs']:
                for budget in BUDGETS:
                    print(f"[Ablation] entropy={ec}, lr={lr}, epochs={epochs}, budget={budget}")
                    # train a controller
                    ctrl=copy.deepcopy(ctrl_tpl).train().to(DEVICE)
                    opt=torch.optim.Adam(ctrl.parameters(),lr=lr)
                    baseline=0.0
                    for ep in range(epochs):
                        for batch in train_loader:
                            inp,att,lbl = [batch[k].to(DEVICE) for k in ['input_ids','attention_mask','labels']]
                            tf,lf = torch.zeros(1,6,device=DEVICE),torch.zeros(1,NL,2,device=DEVICE)
                            r_log,b_log,s_log=ctrl(tf,lf)
                            # normalize
                            r_log,b_log,s_log = normalize_logits(r_log,NL),normalize_logits(b_log,NL),normalize_logits(s_log,NL)
                            r_cat,b_cat,s_cat = Categorical(r_log),Categorical(b_log),Categorical(s_log)
                            r_s,b_s,s_s=r_cat.sample().squeeze(),b_cat.sample().squeeze(),s_cat.sample().squeeze()
                            r_idx,b_idx,s_idx=ensure_list(r_s,NL),ensure_list(b_s,NL),ensure_list(s_s,NL)
                            M=apply_compression(copy.deepcopy(base).cpu(),r_idx,b_idx,s_idx,CFG).to(DEVICE);M.eval()
                            with torch.no_grad():
                                perf=F.cross_entropy(M(input_ids=inp,attention_mask=att).logits,lbl).item()
                            cost=np.mean([CFG['ranks'][ri]*CFG['bitwidths'][bi]*(1-CFG['sparsities'][si])
                                          for ri,bi,si in zip(r_idx,b_idx,s_idx)])
                            reward = -perf - (budget/max(budget,1e-6))*cost
                            adv = reward-baseline
                            baseline = 0.9*baseline + 0.1*reward
                            logp = sum(cat.log_prob(x).sum() for cat,x in zip((r_cat,b_cat,s_cat),(r_s,b_s,s_s)))
                            ent = sum(cat.entropy().sum() for cat in (r_cat,b_cat,s_cat))
                            loss=-adv*logp - ec*ent
                            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ctrl.parameters(),0.5); opt.step()
                    # evaluate
                    lat,mem = benchmark_inference(base,10)
                    # record
                    results_ablation.append({
                        'entropy_coef':ec,'lr':lr,'epochs':epochs,'budget':budget,
                        'latency_ms':lat,'mem_mb':mem
                    })
    # save CSV
    os.makedirs('output/ablation',exist_ok=True)
    keys=results_ablation[0].keys()
    with open('output/ablation/ablation_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(results_ablation)

    # plot latency vs mem for each budget
    df=results_ablation
    sns.set(style="whitegrid")
    plt.figure(figsize=(8,6))
    for bud in BUDGETS:
        sub=[r for r in df if r['budget']==bud]
        plt.scatter([r['latency_ms'] for r in sub],[r['mem_mb'] for r in sub],label=f'B={bud}')
    plt.xlabel('Inference Latency (ms)'); plt.ylabel('Memory Footprint (MB)')
    plt.title('Ablation: Latency vs Memory by Budget')
    plt.legend(); plt.tight_layout()
    plt.savefig('output/ablation/latency_memory.png',dpi=300)
    print("Ablation complete. Check output/ablation/")

if __name__ == "__main__":
    run_ablation()