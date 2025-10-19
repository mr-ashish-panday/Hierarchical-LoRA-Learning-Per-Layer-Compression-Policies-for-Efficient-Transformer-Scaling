#!/usr/bin/env python3
import os, sys, copy, random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load project src
proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Config
cfg = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9]
}

def fix_tok(t, m):
    if t.pad_token is None:
        t.pad_token = t.eos_token
        t.pad_token_id = t.eos_token_id
    m.config.pad_token_id = t.pad_token_id
    return t, m

def count_lin(m): return len([x for x in m.modules() if isinstance(x, nn.Linear)])
def avg_cost(r,b,s): return np.mean([cfg['ranks'][ri]*cfg['bitwidths'][bi]*(1-cfg['sparsities'][si]) for ri,bi,si in zip(r,b,s)])

# Prepare data
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium',num_labels=2,ignore_mismatched_sizes=True).to('cuda')
tokenizer, model = fix_tok(tokenizer, model)
raw = load_dataset('glue','mrpc')
def tok_fn(ex): return tokenizer(ex['sentence1'], ex['sentence2'], padding='max_length', truncation=True, max_length=128)
train = raw['train'].map(tok_fn, batched=True).rename_column('label','labels').set_format('torch',['input_ids','attention_mask','labels'])
val   = raw['validation'].map(tok_fn, batched=True).rename_column('label','labels').set_format('torch',['input_ids','attention_mask','labels'])

# Load controller
ctrl = MetaController(6,2,64, cfg['ranks'],cfg['bitwidths'],cfg['sparsities'], num_layers=count_lin(model)).to('cuda')
ctrl.load_state_dict(torch.load('output/final_meta_controller_phase3e_corrected.pt', map_location='cuda'))
ctrl.eval()

# Eval helper
def eval_policy(r,b,s, tokenizer):
    m = copy.deepcopy(model)
    m = apply_compression(m, r, b, s, cfg)
    _, m = fix_tok(tokenizer, m)
    tr = Trainer(model=m, args=TrainingArguments(output_dir='tmp',per_device_eval_batch_size=16,do_eval=True,do_train=False), eval_dataset=val)
    res = tr.evaluate()
    return res['eval_loss'], avg_cost(r,b,s)

# Baselines
L = count_lin(model)
u = ([cfg['ranks'].index(8)]*L,[cfg['bitwidths'].index(8)]*L,[cfg['sparsities'].index(0.5)]*L)
r = ([random.randrange(len(cfg['ranks'])) for _ in range(L)],[random.randrange(len(cfg['bitwidths'])) for _ in range(L)],[random.randrange(len(cfg['sparsities'])) for _ in range(L)])
u_loss,u_cost = eval_policy(*u, tokenizer)
r_loss,r_cost = eval_policy(*r, tokenizer)

# Temperature sweep
with torch.no_grad():
    tf = torch.zeros(1,6).to('cuda')
    lf = torch.zeros(1,L,2).to('cuda')
temps=[0.5,1.0,2.0,5.0]
sweep=[]
for T in temps:
    with torch.no_grad():
        pr,pb,ps = ctrl(tf,lf)
    sr = torch.softmax(pr/T,-1).argmax(-1).squeeze().tolist()
    sb = torch.softmax(pb/T,-1).argmax(-1).squeeze().tolist()
    ss = torch.softmax(ps/T,-1).argmax(-1).squeeze().tolist()
    loss,cst = eval_policy(sr,sb,ss, tokenizer)
    sweep.append((T,loss,cst))
    print(f"T={T}: Loss={loss:.4f}, Cost={cst:.2f}")

# Plot
plt.figure(figsize=(6,4))
plt.scatter(u_cost,u_loss,label='Uniform',color='blue',s=100)
plt.scatter(r_cost,r_loss,label='Random',color='orange',s=100)
for T,l,c in sweep:
    plt.scatter(c,l,label=f'Learned T={T}',marker='x',s=100)
plt.xlabel('Cost');plt.ylabel('Loss');plt.title('MRPC Pareto Temperature Sweep');plt.legend();plt.grid(True)
plt.savefig('output/baselines/pareto_temp_sweep.png',bbox_inches='tight',dpi=300)
print("Saved pareto_temp_sweep.png")
