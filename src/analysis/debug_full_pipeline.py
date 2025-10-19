#!/usr/bin/env python3
import os, sys, copy, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)

CFG = {'ranks':[1,2,4,8,16],'bitwidths':[2,4,8,16,32],'sparsities':[0.0,0.2,0.5,0.7,0.9]}

def count_linears(m): return len([l for l in m.modules() if isinstance(l, nn.Linear)])

def test_compression_param_counts():
    print("== Test compression param counts ==")
    model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
    L = count_linears(model)
    cfg = CFG
    # min vs max rank, bits; no sparsity to isolate effects
    policies = [([0]*L,[0]*L,[0]*L), ([4]*L,[4]*L,[0]*L)]
    for i,(r,b,s) in enumerate(policies):
        m2 = copy.deepcopy(model)
        comp = apply_compression(m2, r, b, s, cfg)
        orig = sum(p.numel() for p in model.parameters())
        new  = sum(p.numel() for p in comp.parameters())
        print(f"Policy {i}: orig={orig}, comp={new}, ratio={new/orig:.4f}")

def load_data():
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    # Force a valid pad token using eos to avoid any new token resizing complexity
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw = load_dataset('glue','mrpc')
    def tok(ex):
        return tokenizer(ex['sentence1'], ex['sentence2'], padding='max_length', truncation=True, max_length=128)
    mapped = raw.map(tok, batched=True, remove_columns=['sentence1','sentence2','idx'])
    mapped = mapped.rename_column('label','labels')
    mapped.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    return mapped['train'], mapped['validation'], tokenizer

def log_logits_sample(ctrl, num_layers, title):
    with torch.no_grad():
        tf = torch.zeros(1,6,device=DEVICE)
        lf = torch.zeros(1,num_layers,2,device=DEVICE)
        r,b,s = ctrl(tf, lf)
        print(title)
        print(" r[0,:5]:", r[0,:5].detach().cpu().numpy())
        print(" b[0,:5]:", b[0,:5].detach().cpu().numpy())
        print(" s[0,:5]:", s[0,:5].detach().cpu().numpy())

def main():
    test_compression_param_counts()

    print("\n== Load controller and log logits ==")
    base_model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
    num_layers = count_linears(base_model)
    ctrl = MetaController(6,2,64,CFG['ranks'],CFG['bitwidths'],CFG['sparsities'], num_layers=num_layers).to(DEVICE)
    ckpt = 'output/final_meta_controller_phase3e_corrected.pt'
    ctrl.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    ctrl.eval()
    log_logits_sample(ctrl, num_layers, "Initial controller logits:")

    print("\n== Fine-tune one batch (cost-augmented) ==")
    train_ds, val_ds, tokenizer = load_data()
    loader = DataLoader(Subset(train_ds, list(range(32))), batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(ctrl.parameters(), lr=1e-4)
    for batch in loader:
        inp = batch['input_ids'].to(DEVICE)
        att = batch['attention_mask'].to(DEVICE)
        lab = batch['labels'].to(DEVICE)
        tf = torch.zeros(1,6,device=DEVICE)
        lf = torch.zeros(1,num_layers,2,device=DEVICE)
        r_log,b_log,s_log = ctrl(tf, lf)
        r_idx = r_log.argmax(-1).squeeze().tolist()
        b_idx = b_log.argmax(-1).squeeze().tolist()
        s_idx = s_log.argmax(-1).squeeze().tolist()
        m = apply_compression(copy.deepcopy(base_model), r_idx, b_idx, s_idx, CFG).to(DEVICE)
        out = m(input_ids=inp, attention_mask=att)
        perf_loss = nn.CrossEntropyLoss()(out.logits, lab)
        # cost per layer average
        ranks, bits, spars = CFG['ranks'], CFG['bitwidths'], CFG['sparsities']
        cost = sum(ranks[ri]*bits[bi]*(1-spars[si]) for ri,bi,si in zip(r_idx,b_idx,s_idx)) / num_layers
        loss = perf_loss + 0.1*cost
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        break
    ctrl.eval()
    log_logits_sample(ctrl, num_layers, "Updated controller logits:")

    print("\n== Sample policy and verify compression ==")
    tf = torch.zeros(1,6,device=DEVICE)
    lf = torch.zeros(1,num_layers,2,device=DEVICE)
    with torch.no_grad():
        r_log,b_log,s_log = ctrl(tf, lf)
    r_idx = r_log.argmax(-1).squeeze().tolist()
    b_idx = b_log.argmax(-1).squeeze().tolist()
    s_idx = s_log.argmax(-1).squeeze().tolist()
    m = apply_compression(copy.deepcopy(base_model), r_idx, b_idx, s_idx, CFG).to(DEVICE)
    orig = sum(p.numel() for p in base_model.parameters())
    new  = sum(p.numel() for p in m.parameters())
    print(f"Sampled policy: r[:5]={r_idx[:5]}, b[:5]={b_idx[:5]}, s[:5]={s_idx[:5]}")
    print(f"Params: orig={orig}, comp={new}, ratio={new/orig:.4f}")

if __name__ == "__main__":
    main()
