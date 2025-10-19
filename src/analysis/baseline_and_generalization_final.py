#!/usr/bin/env python3
import os, sys, copy, random, inspect
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

# Project root for imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# ------------------------
# Configuration & Utils
# ------------------------
cfg_baseline = {
    'ranks': [1,2,4,8,16],
    'bitwidths': [2,4,8,16,32],
    'sparsities': [0.0,0.2,0.5,0.7,0.9]
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fix_tokenizer_and_model(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def count_linears(model):
    return len([m for m in model.modules() if isinstance(m, nn.Linear)])

def ensure_list_indices(idx, L):
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().tolist()
    if isinstance(idx, (int, float, np.integer)):
        return [int(idx)] * L
    if not isinstance(idx, (list, tuple)):
        return [int(idx)] * L
    if len(idx) < L:
        return list(idx) + [idx[-1]] * (L - len(idx))
    if len(idx) > L:
        return list(idx)[:L]
    return list(map(int, idx))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).astype(np.float32).mean().item()
    return {'accuracy': float(acc)}

def cost_of_policy(r_idx, b_idx, s_idx):
    R, B, S = cfg_baseline['ranks'], cfg_baseline['bitwidths'], cfg_baseline['sparsities']
    return float(np.mean([R[r]*B[b]*(1-S[s]) for r,b,s in zip(r_idx,b_idx,s_idx)]))

# ------------------------
# Real Feature Extraction
# ------------------------
def compute_task_features(task, tokenizer, max_len=128, sample_cap=512):
    raw = load_dataset('glue', task)
    is_pair = task not in ['sst2','cola']
    def prep(b):
        if is_pair:
            return tokenizer(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=max_len)
        return tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=max_len)
    data = raw.map(prep, batched=True)
    n = min(len(data['validation']), sample_cap)
    attention = data['validation'][:n]['attention_mask']
    labels = data['validation'][:n]['label']
    # sequence length stats
    lens = [int(torch.tensor(a).sum().item()) for a in attention]
    avg_len = np.mean(lens)/max_len
    var_len = np.var(lens)/(max_len**2)
    # vocabulary coverage
    uniq = set()
    for ids in data['validation'][:n]['input_ids']:
        uniq.update(ids)
    uniq_frac = min(len(uniq),50000)/50000.0
    # label distribution
    lbl_mean = float(np.mean(labels))
    # pair flag
    pair_flag = 1.0 if is_pair else 0.0
    # train size scale
    train_size = min(len(raw['train']),10000)/10000.0 if 'train' in raw else 0.0
    vec = np.array([avg_len, var_len, uniq_frac, lbl_mean, pair_flag, train_size], dtype=np.float32)
    return torch.tensor(vec[None,:], dtype=torch.float32).to('cuda')

def compute_layer_features(model):
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    L = len(linear_layers)
    if L == 0:
        return torch.zeros(1,1,2, dtype=torch.float32).to('cuda')
    norms = []
    for layer in linear_layers:
        wnorm = float(torch.norm(layer.weight).item())
        norms.append(wnorm)
    norms = np.array(norms, dtype=np.float32)
    norms_norm = (norms - norms.min())/(norms.max()-norms.min()+1e-8)
    depths = np.linspace(0.0,1.0,L, dtype=np.float32)
    feats = np.stack([depths, norms_norm], axis=-1)
    return torch.tensor(feats[None,:,:], dtype=torch.float32).to('cuda')

# ------------------------
# Policies
# ------------------------
def uniform_policy(model, ep):
    L = count_linears(model)
    i = cfg_baseline['ranks'].index(8)
    j = cfg_baseline['bitwidths'].index(8)
    k = cfg_baseline['sparsities'].index(0.5)
    return [i]*L, [j]*L, [k]*L

def random_policy(model, ep):
    L = count_linears(model)
    return ([random.randrange(len(cfg_baseline['ranks'])) for _ in range(L)],
            [random.randrange(len(cfg_baseline['bitwidths'])) for _ in range(L)],
            [random.randrange(len(cfg_baseline['sparsities'])) for _ in range(L)])

class PolicyLoader:
    def __init__(self, ckpt_path, base_model, hidden_dim=64):
        self.num_layers = count_linears(base_model)
        # adaptive constructor
        try:
            self.controller = MetaController(6,2,hidden_dim,
                                             cfg_baseline['ranks'],
                                             cfg_baseline['bitwidths'],
                                             cfg_baseline['sparsities'],
                                             num_layers=self.num_layers)
        except TypeError:
            ctor_kwargs = {
                'task_feat_dim':6,'layer_feat_dim':2,
                'ranks':cfg_baseline['ranks'],
                'bitwidths':cfg_baseline['bitwidths'],
                'sparsities':cfg_baseline['sparsities'],
                'num_layers':self.num_layers,
                'hidden_dim':hidden_dim
            }
            self.controller = MetaController(**ctor_kwargs)
        self.controller.to('cuda')
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cuda')
            self.controller.load_state_dict(sd)
            self.controller.eval()
            print(f"Loaded controller from {ckpt_path}")
        else:
            print(f"Warning: checkpoint not found: {ckpt_path}")

    def sample_actions(self, tf, lf):
        with torch.no_grad():
            r_p, b_p, s_p = self.controller(tf, lf)
        L = lf.shape[1]
        r_idx = ensure_list_indices(torch.argmax(r_p, dim=-1).squeeze(), L)
        b_idx = ensure_list_indices(torch.argmax(b_p, dim=-1).squeeze(), L)
        s_idx = ensure_list_indices(torch.argmax(s_p, dim=-1).squeeze(), L)
        return r_idx, b_idx, s_idx

# ------------------------
# Evaluation
# ------------------------
def evaluate_policy(base, tokenizer, task, policy_kind, policy_fn=None, loader=None, epochs=5):
    # Precompute features
    tf = compute_task_features(task, tokenizer)
    lf = compute_layer_features(base)

    # Prepare data
    raw = load_dataset('glue', task)
    is_pair = task not in ['sst2','cola']
    def prep(b):
        if is_pair:
            return tokenizer(b['sentence1'], b['sentence2'],
                             padding='max_length', truncation=True, max_length=128)
        return tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=128)
    data = raw.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])

    args = TrainingArguments(
        output_dir=f'output/baselines/{policy_kind}_{task}',
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        logging_dir=None
    )

    losses, accs, costs = [], [], []
    for ep in range(1, epochs+1):
        m = copy.deepcopy(base)
        if policy_kind == 'learned':
            r,b,s = loader.sample_actions(tf, lf)
        else:
            r,b,s = policy_fn(m, ep)
        L = count_linears(m)
        r = ensure_list_indices(r, L)
        b = ensure_list_indices(b, L)
        s = ensure_list_indices(s, L)

        m = apply_compression(m, r, b, s, cfg_baseline)
        tokenizer, m = fix_tokenizer_and_model(tokenizer, m)
        trainer = Trainer(model=m, args=args,
                          eval_dataset=data['validation'],
                          processing_class=tokenizer,
                          compute_metrics=compute_metrics)
        res = trainer.evaluate()
        loss = res['eval_loss']; acc = res.get('eval_accuracy', float('nan'))
        cost = cost_of_policy(r,b,s)
        losses.append(loss); accs.append(acc); costs.append(cost)
        print(f"{policy_kind} {task} Ep{ep}: Loss={loss:.4f} Acc={acc:.4f} Cost={cost:.2f}")
    return losses, accs, costs

if __name__ == '__main__':
    set_seed(42)
    os.makedirs('output/baselines', exist_ok=True)

    tok = AutoTokenizer.from_pretrained('gpt2-medium')
    base = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
    ).to('cuda')
    tok, base = fix_tokenizer_and_model(tok, base)

    loader = PolicyLoader('output/final_meta_controller_phase3e_corrected.pt', base)

    un = evaluate_policy(base, tok, 'mrpc',  'uniform', uniform_policy, loader, epochs=5)
    rd = evaluate_policy(base, tok, 'mrpc',  'random',  random_policy,  loader, epochs=5)
    ld = evaluate_policy(base, tok, 'mrpc',  'learned', None,          loader, epochs=5)

    sst = evaluate_policy(base, tok, 'sst2', 'learned', None, loader, epochs=5)
    cola = evaluate_policy(base, tok, 'cola', 'learned', None, loader, epochs=5)

    # Pareto plot
    pts = {'Uniform':(un[2][-1],un[0][-1]),
           'Random':(rd[2][-1],rd[0][-1]),
           'Learned':(ld[2][-1],ld[0][-1])}
    plt.figure(figsize=(6,4))
    cols={'Uniform':'blue','Random':'orange','Learned':'green'}
    for k,(cst,loss) in pts.items():
        plt.scatter(cst, loss, s=100, label=k, color=cols[k])
    plt.xlabel('Compression Cost'); plt.ylabel('Eval Loss')
    plt.title('MRPC Pareto Frontier'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig('output/baselines/pareto_mrpc.png', dpi=300, bbox_inches='tight')
    print('Saved Pareto: output/baselines/pareto_mrpc.png')

    # CSV summary
    import csv
    with open('output/baselines/summary.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Policy','Task','FinalLoss','FinalAcc','FinalCost'])
        w.writerow(['Uniform','MRPC', un[0][-1], un[1][-1], un[2][-1]])
        w.writerow(['Random','MRPC',  rd[0][-1], rd[1][-1], rd[2][-1]])
        w.writerow(['Learned','MRPC',  ld[0][-1], ld[1][-1], ld[2][-1]])
        w.writerow(['Learned','SST2',  sst[0][-1], sst[1][-1], sst[2][-1]])
        w.writerow(['Learned','CoLA',  cola[0][-1], cola[1][-1], cola[2][-1]])
    print('Saved CSV: output/baselines/summary.csv')
