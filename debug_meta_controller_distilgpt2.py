#!/usr/bin/env python3
import os, sys, copy, random, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Project path (adjust if needed)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import LowRankLinear, apply_compression
from models.meta_controller import MetaController

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(SEED)

CFG = {'ranks':[1,2,4,8,16],'bitwidths':[2,4,8,16,32],'sparsities':[0.0,0.2,0.5,0.7,0.9]}

# ----------------------------
# Helpers
# ----------------------------
def count_linears(m):
    return len([l for l in m.modules() if isinstance(l, (nn.Linear, LowRankLinear))])

def ensure_list(x, L):
    if isinstance(x, torch.Tensor):
        x = [int(x.item())]*L if x.ndim==0 else x.detach().cpu().tolist()
    if isinstance(x, int): return [x]*L
    return x if len(x)==L else x + [x[-1]]*(L-len(x))

# ----------------------------
# Tokenizer & Dataset
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

raw = load_dataset('glue', 'sst2')
def tok_fn(ex): return tokenizer(ex['sentence'], padding='max_length', truncation=True, max_length=64)
tokd = raw.map(tok_fn, batched=True, remove_columns=['sentence','idx'])
tokd = tokd.rename_column('label','labels')
tokd.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

train_ds, val_ds = tokd['train'], tokd['validation']
train_loader = DataLoader(Subset(train_ds, list(range(200))), batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# ----------------------------
# Base Model
# ----------------------------
print("[INFO] Loading DistilGPT2 model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    'distilgpt2', num_labels=2, ignore_mismatched_sizes=True
).to(DEVICE)
base_model.config.pad_token_id = tokenizer.pad_token_id

NL = count_linears(base_model)
print(f"[INFO] NL (linear/lowrank layers) = {NL}")

# ----------------------------
# Controller
# ----------------------------
controller_template = MetaController(
    6,2,64,CFG['ranks'],CFG['bitwidths'],CFG['sparsities'], num_layers=NL
).to(DEVICE)

# ----------------------------
# Sanity Check
# ----------------------------
def sanity_check():
    m = copy.deepcopy(base_model)
    L = NL
    policies = [([0]*L,[0]*L,[0]*L), ([4]*L,[4]*L,[0]*L)]
    for i,p in enumerate(policies):
        cm = apply_compression(copy.deepcopy(m), p[0],p[1],p[2], CFG)
        orig = sum(param.numel() for param in m.parameters())
        comp = sum(param.numel() for param in cm.parameters())
        print(f"[Sanity] Policy {i}: Original={orig}, Compressed={comp}, Ratio={comp/orig:.6f}")
sanity_check()

# ----------------------------
# Train Controller
# ----------------------------
def train_for_budget(target_cost):
    ctrl = copy.deepcopy(controller_template).train()
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-3)
    baseline = 0.0; beta=0.9

    for epoch in range(8):
        for batch in train_loader:
            inp = batch['input_ids'].to(DEVICE)
            att = batch['attention_mask'].to(DEVICE)
            lab = batch['labels'].to(DEVICE)

            tf, lf = torch.zeros(1,6,device=DEVICE), torch.zeros(1,NL,2,device=DEVICE)
            r_log, b_log, s_log = ctrl(tf, lf)

            r_cat, b_cat, s_cat = Categorical(logits=r_log), Categorical(logits=b_log), Categorical(logits=s_log)
            r_s, b_s, s_s = r_cat.sample().squeeze(), b_cat.sample().squeeze(), s_cat.sample().squeeze()
            r_idx, b_idx, s_idx = ensure_list(r_s, NL), ensure_list(b_s, NL), ensure_list(s_s, NL)

            m = apply_compression(copy.deepcopy(base_model), r_idx, b_idx, s_idx, CFG).to(DEVICE)
            out = m(input_ids=inp, attention_mask=att)
            perf = F.cross_entropy(out.logits, lab)

            cost = np.mean([CFG['ranks'][ri]*CFG['bitwidths'][bi]*(1-CFG['sparsities'][si]) for ri,bi,si in zip(r_idx,b_idx,s_idx)])
            reward = -perf.item() - abs(cost - target_cost)
            advantage = reward - baseline
            baseline = beta*baseline + (1-beta)*reward

            logp = (r_cat.log_prob(r_s) + b_cat.log_prob(b_s) + s_cat.log_prob(s_s)).sum()
            ent = r_cat.entropy().mean() + b_cat.entropy().mean() + s_cat.entropy().mean()
            loss = -advantage*logp - 0.01*ent

            opt.zero_grad(); loss.backward(); opt.step()
            break
    return ctrl

# ----------------------------
# Main: Train & Evaluate
# ----------------------------
budgets = [2.0,8.0,16.0,32.0]
os.makedirs('output/baselines', exist_ok=True)
paths=[]

for TGT in budgets:
    C = train_for_budget(TGT)
    p = f'output/baselines/ctrl_budget_{int(TGT)}_distil.pt'
    torch.save(C.state_dict(),p)
    paths.append((TGT,p))
    print(f"[Saved] {p}")

results=[]
for TGT,p in paths:
    C = MetaController(6,2,64,CFG['ranks'],CFG['bitwidths'],CFG['sparsities'],num_layers=NL).to(DEVICE)
    C.load_state_dict(torch.load(p,map_location=DEVICE)); C.eval()

    for temp in [0.5,1.0,2.0]:
        tf, lf = torch.zeros(1,6,device=DEVICE), torch.zeros(1,NL,2,device=DEVICE)
        with torch.no_grad(): r_log,b_log,s_log = C(tf,lf)

        r_idx = ensure_list(F.softmax(r_log/temp,-1).argmax(-1).squeeze(),NL)
        b_idx = ensure_list(F.softmax(b_log/temp,-1).argmax(-1).squeeze(),NL)
        s_idx = ensure_list(F.softmax(s_log/temp,-1).argmax(-1).squeeze(),NL)

        M = apply_compression(copy.deepcopy(base_model), r_idx,b_idx,s_idx,CFG).to(DEVICE); M.eval()
        total,n=0.0,0
        for batch in val_loader:
            out = M(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            l = F.cross_entropy(out.logits, batch['labels'].to(DEVICE))
            total += l.item()*batch['labels'].size(0); n+=batch['labels'].size(0)
        avg_loss = total/n
        cost = np.mean([CFG['ranks'][ri]*CFG['bitwidths'][bi]*(1-CFG['sparsities'][si]) for ri,bi,si in zip(r_idx,b_idx,s_idx)])
        results.append((f"target={int(TGT)},T={temp}",avg_loss,cost))
        print(f"[Eval] target={TGT}, T={temp}: loss={avg_loss:.4f}, cost={cost:.2f}")

# Plot
plt.figure(figsize=(7,5))
for name,loss,cost in results:
    plt.scatter(cost,loss,label=name,s=60)
plt.xlabel('Cost'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
plt.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
plt.tight_layout(); plt.savefig('output/baselines/final_pareto_distil.png',dpi=300)

# CSV
with open('output/baselines/final_pareto_distil.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['Policy','Loss','Cost']); w.writerows(results)

print("[DONE] debug_meta_controller_distilgpt2.py finished.")
