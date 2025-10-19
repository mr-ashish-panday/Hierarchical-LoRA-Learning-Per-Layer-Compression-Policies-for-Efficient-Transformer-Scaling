#!/usr/bin/env python3
import os, sys, copy, random
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

# Add project src
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Baseline config
cfg = {
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

def count_linears(m): return len([x for x in m.modules() if isinstance(x, nn.Linear)])
def cost(r,b,s): return np.mean([cfg['ranks'][ri]*cfg['bitwidths'][bi]*(1-cfg['sparsities'][si]) for ri,bi,si in zip(r,b,s)])

# Load model and controller
tok = AutoTokenizer.from_pretrained('gpt2-medium')
base = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2, ignore_mismatched_sizes=True).to('cuda')
tok, base = fix_tokenizer_and_model(tok, base)
policy_ckpt = 'output/final_meta_controller_phase3e_corrected.pt'
controller = MetaController(6,2,64, cfg['ranks'], cfg['bitwidths'], cfg['sparsities'], num_layers=count_linears(base)).to('cuda')
controller.load_state_dict(torch.load(policy_ckpt, map_location='cuda'))
controller.eval()

# Prepare MRPC eval data
raw = load_dataset('glue','mrpc')
# Tokenize each split and set format
def prep(b):
    return tok(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=128)
for split in ['train','validation','test']:
    raw[split] = raw[split].map(prep, batched=True)
raw['validation'] = raw['validation'].rename_column('label','labels').set_format('torch', columns=['input_ids','attention_mask','labels'])
val_ds = raw['validation']

# Compute fixed features
with torch.no_grad():
    tf = torch.zeros(1,6).to('cuda')  # replace with real task features if available
    lf = torch.zeros(1,count_linears(base),2).to('cuda')  # replace with real per-layer features

# Eval helper
def eval_policy(r_idx,b_idx,s_idx, tokenizer):
    m = copy.deepcopy(base)
    m = apply_compression(m, r_idx, b_idx, s_idx, cfg)
    _, m = fix_tokenizer_and_model(tokenizer, m)
    trainer = Trainer(model=m, args=TrainingArguments(output_dir='tmp', per_device_eval_batch_size=16, do_train=False, do_eval=True), eval_dataset=val_ds)
    res = trainer.evaluate()
    return res['eval_loss'], cost(r_idx,b_idx,s_idx)

# Baselines
# Uniform
L = count_linears(base)
u_r = [cfg['ranks'].index(8)]*L
u_b = [cfg['bitwidths'].index(8)]*L
u_s = [cfg['sparsities'].index(0.5)]*L
u_loss, u_cost = eval_policy(u_r,u_b,u_s, tok)
# Random
r_r,r_b,r_s = [random.randrange(len(cfg['ranks'])) for _ in range(L)], [random.randrange(len(cfg['bitwidths'])) for _ in range(L)], [random.randrange(len(cfg['sparsities'])) for _ in range(L)]
r_loss, r_cost = eval_policy(r_r,r_b,r_s, tok)

# Temperature sweep
temps = [0.5,1.0,2.0,5.0]
sweep_points = []
for T in temps:
    with torch.no_grad():
        logits_r, logits_b, logits_s = controller(tf, lf)
    # apply temperature
    r_p = torch.softmax(logits_r / T, dim=-1)
    b_p = torch.softmax(logits_b / T, dim=-1)
    s_p = torch.softmax(logits_s / T, dim=-1)
    r_idx = r_p.argmax(-1).squeeze().tolist()
    b_idx = b_p.argmax(-1).squeeze().tolist()
    s_idx = s_p.argmax(-1).squeeze().tolist()
    l, c = eval_policy(r_idx, b_idx, s_idx, tok)
    sweep_points.append((T, l, c))
    print(f"T={T}: Loss={l:.4f}, Cost={c:.2f}")

# Plot
plt.figure(figsize=(6,4))
plt.scatter(u_cost, u_loss, label='Uniform', color='blue', s=100)
plt.scatter(r_cost, r_loss, label='Random', color='orange', s=100)
for T,l,c in sweep_points:
    plt.scatter(c, l, label=f'Learned T={T}', s=100, marker='x')
plt.xlabel('Compression Cost')
plt.ylabel('Eval Loss')
plt.title('MRPC Pareto Frontier with Temperature Sweep')
plt.legend()
plt.grid(True,alpha=0.3)
plt.savefig('output/baselines/pareto_temp_sweep.png', dpi=300,bbox_inches='tight')
print('Saved', 'output/baselines/pareto_temp_sweep.png')
