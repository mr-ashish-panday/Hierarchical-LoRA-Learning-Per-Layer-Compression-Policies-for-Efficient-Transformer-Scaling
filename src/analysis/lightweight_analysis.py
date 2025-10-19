#!/usr/bin/env python3
"""
Lightweight Research Analysis Suite for Hierarchical LoRA
========================================================

Focused ablation study with minimal computational overhead:
1. Single entropy coefficient test (0.01 vs 0.05)
2. Layer-wise policy visualization
3. Performance vs compression trade-off analysis
4. Publication-ready plots and tables

Runs in ~10-15 minutes instead of hours.
"""

import os, sys, copy, random, csv
import numpy as np
import torch
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

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)

CFG = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9],
}

# Lightweight ablation - only test key parameters
ENTROPY_VALS = [0.01, 0.05]  # Just 2 values
BUDGETS = [2.0, 16.0]        # Just 2 budgets
EPOCHS = 6                   # Reduced epochs
MAX_SAMPLES = 100            # Fewer training samples

def normalize_logits(logits, L):
    if isinstance(logits, (list,tuple)): logits = logits[0]
    if logits.dim()==1: return logits.unsqueeze(0).repeat(L,1)
    if logits.dim()==2: return logits.repeat(L,1) if logits.size(0)==1 else logits
    if logits.dim()==3: return logits.squeeze(0)[:L]
    raise RuntimeError(f"Unexpected logits shape {logits.shape}")

def ensure_list(x, L):
    if isinstance(x, torch.Tensor):
        lst = x.squeeze().detach().cpu().tolist()
        vals = [int(lst)] if isinstance(lst, (int,float)) else [int(v) for v in lst]
    elif isinstance(x, (int,float)): vals = [int(x)]
    else: vals = [int(v) for v in x]
    return vals if len(vals)==L else vals+[vals[-1]]*(L-len(vals))

def count_modules(model):
    cnt=0
    for _,m in model.named_modules():
        if isinstance(m,(torch.nn.Linear,Conv1D)): cnt+=1
    return cnt

def setup_data():
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    raw = load_dataset('glue','sst2')
    def tok(ex): return tokenizer(ex['sentence'],padding='max_length',truncation=True,max_length=128)
    ds = raw.map(tok,batched=True,remove_columns=['sentence','idx'])
    ds = ds.rename_column('label','labels')
    ds.set_format('torch',['input_ids','attention_mask','labels'])
    train_ds = Subset(ds['train'], list(range(min(MAX_SAMPLES,len(ds['train'])))))
    train_loader = DataLoader(train_ds,batch_size=4,shuffle=True)
    val_loader = DataLoader(ds['validation'],batch_size=8,shuffle=False)
    return tokenizer, train_loader, val_loader

def train_controller(base_model, NL, entropy_coef, budget, train_loader):
    """Train a single controller quickly."""
    ctrl = MetaController(6,2,64,CFG['ranks'],CFG['bitwidths'],CFG['sparsities'],num_layers=NL).to(DEVICE)
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-3)
    baseline = 0.0
    
    print(f"  Training: entropy={entropy_coef}, budget={budget}")
    
    for ep in range(EPOCHS):
        for step, batch in enumerate(train_loader):
            inp,att,lbl = [batch[k].to(DEVICE) for k in ['input_ids','attention_mask','labels']]
            tf,lf = torch.zeros(1,6,device=DEVICE), torch.zeros(1,NL,2,device=DEVICE)
            r_log,b_log,s_log = ctrl(tf,lf)
            r_log,b_log,s_log = normalize_logits(r_log,NL), normalize_logits(b_log,NL), normalize_logits(s_log,NL)
            r_cat,b_cat,s_cat = Categorical(r_log), Categorical(b_log), Categorical(s_log)
            r_s,b_s,s_s = r_cat.sample().squeeze(), b_cat.sample().squeeze(), s_cat.sample().squeeze()
            r_idx,b_idx,s_idx = ensure_list(r_s,NL), ensure_list(b_s,NL), ensure_list(s_s,NL)
            
            M = apply_compression(copy.deepcopy(base_model).cpu(),r_idx,b_idx,s_idx,CFG).to(DEVICE)
            M.eval()
            with torch.no_grad():
                perf = F.cross_entropy(M(input_ids=inp,attention_mask=att).logits,lbl).item()
            
            cost = np.mean([CFG['ranks'][ri]*CFG['bitwidths'][bi]*(1-CFG['sparsities'][si])
                           for ri,bi,si in zip(r_idx,b_idx,s_idx)])
            reward = -perf - (budget/max(budget,1e-6))*cost
            adv = reward - baseline
            baseline = 0.9*baseline + 0.1*reward
            
            logp = sum(cat.log_prob(x).sum() for cat,x in zip((r_cat,b_cat,s_cat),(r_s,b_s,s_s)))
            ent = sum(cat.entropy().sum() for cat in (r_cat,b_cat,s_cat))
            loss = -adv*logp - entropy_coef*ent
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ctrl.parameters(),0.5)
            opt.step()
    
    return ctrl.eval()

def evaluate_controller(ctrl, base_model, NL, val_loader):
    """Get policy and evaluate performance."""
    tf,lf = torch.zeros(1,6,device=DEVICE), torch.zeros(1,NL,2,device=DEVICE)
    with torch.no_grad(): r,b,s = ctrl(tf,lf)
    r,b,s = normalize_logits(r,NL), normalize_logits(b,NL), normalize_logits(s,NL)
    r_idx = ensure_list(r.argmax(-1).squeeze(),NL)
    b_idx = ensure_list(b.argmax(-1).squeeze(),NL)
    s_idx = ensure_list(s.argmax(-1).squeeze(),NL)
    
    M = apply_compression(copy.deepcopy(base_model).cpu(),r_idx,b_idx,s_idx,CFG).to(DEVICE)
    M.eval()
    
    tot,n = 0,0
    for batch in val_loader:
        out = M(input_ids=batch['input_ids'].to(DEVICE),attention_mask=batch['attention_mask'].to(DEVICE))
        l = F.cross_entropy(out.logits,batch['labels'].to(DEVICE)).item()
        tot += l*batch['labels'].size(0); n += batch['labels'].size(0)
    
    avg_loss = tot/n
    cost = np.mean([CFG['ranks'][ri]*CFG['bitwidths'][bi]*(1-CFG['sparsities'][si])
                   for ri,bi,si in zip(r_idx,b_idx,s_idx)])
    
    return avg_loss, cost, r_idx, b_idx, s_idx

def create_policy_heatmap(policies_data, filename):
    """Create layerwise policy visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (policy_type, title) in enumerate([('ranks', 'LoRA Ranks'), ('bitwidths', 'Quantization Bits'), ('sparsities', 'Sparsity Levels')]):
        data = []
        labels = []
        for entropy, budget, r_idx, b_idx, s_idx in policies_data:
            if policy_type == 'ranks': vals = [CFG['ranks'][r] for r in r_idx[:20]]  # First 20 layers
            elif policy_type == 'bitwidths': vals = [CFG['bitwidths'][b] for b in b_idx[:20]]
            else: vals = [CFG['sparsities'][s] for s in s_idx[:20]]
            data.append(vals)
            labels.append(f"E={entropy},B={budget}")
        
        sns.heatmap(data, annot=True, fmt='.1f' if policy_type=='sparsities' else 'd',
                   cmap='viridis', ax=axes[i], yticklabels=labels)
        axes[i].set_title(f'{title} per Layer')
        axes[i].set_xlabel('Layer Index')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Setting up lightweight analysis...")
    tokenizer, train_loader, val_loader = setup_data()
    base_model = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    
    NL = count_modules(base_model)
    print(f"Found {NL} compressible modules")
    
    # Run lightweight ablation
    results = []
    policies_data = []
    
    for entropy in ENTROPY_VALS:
        for budget in BUDGETS:
            ctrl = train_controller(base_model, NL, entropy, budget, train_loader)
            loss, cost, r_idx, b_idx, s_idx = evaluate_controller(ctrl, base_model, NL, val_loader)
            
            results.append({
                'entropy_coef': entropy,
                'budget': budget,
                'val_loss': loss,
                'compression_cost': cost,
                'param_reduction': f"{(1-cost/32.0)*100:.1f}%"  # Rough estimate
            })
            
            policies_data.append((entropy, budget, r_idx, b_idx, s_idx))
            print(f"  Result: E={entropy}, B={budget} -> Loss={loss:.4f}, Cost={cost:.2f}")
    
    # Save results
    os.makedirs('output/lightweight', exist_ok=True)
    
    # CSV results
    with open('output/lightweight/results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, results[0].keys())
        w.writeheader()
        w.writerows(results)
    
    # Performance plot
    plt.figure(figsize=(8, 6))
    colors = {'2.0': 'red', '16.0': 'blue'}
    markers = {'0.01': 'o', '0.05': 's'}
    
    for r in results:
        plt.scatter(r['compression_cost'], r['val_loss'], 
                   color=colors[str(r['budget'])], marker=markers[str(r['entropy_coef'])],
                   s=100, alpha=0.7, 
                   label=f"B={r['budget']}, E={r['entropy_coef']}")
    
    plt.xlabel('Compression Cost')
    plt.ylabel('Validation Loss')
    plt.title('Lightweight Ablation: Performance vs Compression Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/lightweight/performance_tradeoff.png', dpi=300)
    plt.close()
    
    # Policy heatmap
    create_policy_heatmap(policies_data, 'output/lightweight/policy_heatmap.png')
    
    # Summary table
    print("\n" + "="*60)
    print("LIGHTWEIGHT ABLATION RESULTS")
    print("="*60)
    print(f"{'Entropy':<8} {'Budget':<8} {'Val Loss':<10} {'Cost':<8} {'Param Reduction':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['entropy_coef']:<8} {r['budget']:<8} {r['val_loss']:<10.4f} {r['compression_cost']:<8.2f} {r['param_reduction']:<15}")
    
    print(f"\nResults saved to output/lightweight/")
    print("Files generated:")
    print("- results.csv (raw data)")
    print("- performance_tradeoff.png (Pareto frontier)")  
    print("- policy_heatmap.png (layer-wise policies)")

if __name__ == "__main__":
    main()
