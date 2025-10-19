#!/usr/bin/env python3
"""
Budgeted MetaController training for a true Pareto frontier.
- Trains separate controllers toward target average costs: [2, 8, 16, 32]
- Uses REINFORCE (policy gradient) with entropy regularization and moving baseline
- Evaluates each trained controller at multiple temperatures
- Plots a Pareto frontier that should show distinct cost–loss trade-offs
"""

import os, sys, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import csv

# Project paths
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))

# Project modules
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Reproducibility and device
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

# Compression configuration
CFG = {
    'ranks': [1, 2, 4, 8, 16],
    'bitwidths': [2, 4, 8, 16, 32],
    'sparsities': [0.0, 0.2, 0.5, 0.7, 0.9],
}

def count_linears(model):
    return len([m for m in model.modules() if isinstance(m, nn.Linear)])

def calc_cost(r_idx, b_idx, s_idx):
    R, B, S = CFG['ranks'], CFG['bitwidths'], CFG['sparsities']
    return float(np.mean([R[r]*B[b]*(1 - S[s]) for r, b, s in zip(r_idx, b_idx, s_idx)]))

def ensure_list(idx, L):
    if isinstance(idx, int):
        return [idx]*L
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().tolist()
    if isinstance(idx, (list, tuple)):
        if len(idx) >= L:
            return list(idx)[:L]
        return list(idx) + [idx[-1]]*(L - len(idx))
    return [0]*L

def verify_compression(original_model, compressed_model, verbose=False):
    # Note: quant/sparsity may not change param count; still keep as sanity check
    orig_params = sum(p.numel() for p in original_model.parameters())
    comp_params = sum(p.numel() for p in compressed_model.parameters())
    if verbose:
        print(f"Param count: original={orig_params}, compressed={comp_params}")
    return True  # do not block training; different mechanisms may keep numel same

def load_base_model_and_data():
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    model = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    raw = load_dataset('glue', 'mrpc')
    def tok_fn(ex):
        return tokenizer(ex['sentence1'], ex['sentence2'], padding='max_length', truncation=True, max_length=128)
    tokd = raw.map(tok_fn, batched=True, remove_columns=['sentence1','sentence2','idx'])
    tokd = tokd.rename_column('label', 'labels')
    tokd.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # Larger subset for signal, but still 12GB friendly
    train_ds = tokd['train']
    val_ds   = tokd['validation']
    train_idx = list(range(min(512, len(train_ds))))
    train_subset = Subset(train_ds, train_idx)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,    batch_size=8, shuffle=False)
    return tokenizer, model, train_subset, val_ds, train_loader, val_loader

def build_features(model, val_ds):
    # Task features: [avg_len, var_len, vocab_cov, label_balance, is_pair, norm_size]
    avg_len = 1.0
    var_len = 0.1
    vocab_cov = 0.5
    label_balance = 0.5
    is_pair = 1.0
    norm_size = len(val_ds)/10000.0
    task_feat = torch.tensor([[avg_len, var_len, vocab_cov, label_balance, is_pair, norm_size]],
                             dtype=torch.float32, device=DEVICE)

    # Layer features: [depth_position, pseudo_norm]
    L = count_linears(model)
    depths = torch.linspace(0, 1, steps=L, device=DEVICE).unsqueeze(-1)
    pseudo_norm = torch.ones_like(depths) * 0.5
    layer_feat = torch.cat([depths, pseudo_norm], dim=-1).unsqueeze(0)  # [1, L, 2]
    return task_feat, layer_feat

def make_controller(num_layers):
    # Constructor flexibility
    try:
        ctrl = MetaController(6, 2, 64, CFG['ranks'], CFG['bitwidths'], CFG['sparsities'], num_layers=num_layers)
    except TypeError:
        try:
            ctrl = MetaController(task_feat_dim=6, layer_feat_dim=2, hidden_dim=64,
                                  ranks=CFG['ranks'], bitwidths=CFG['bitwidths'],
                                  sparsities=CFG['sparsities'], num_layers=num_layers)
        except TypeError:
            ctrl = MetaController(6, 2, 64, CFG['ranks'], CFG['bitwidths'], CFG['sparsities'])
    return ctrl.to(DEVICE)

def policy_gradient_train_for_budget(base_model, train_loader, task_feat, layer_feat,
                                     num_layers, target_cost, epochs=8,
                                     lr=1e-3, entropy_w=0.05, budget_w=2.0):
    """
    Train a controller to match a target average cost using REINFORCE with
    a budget-violation penalty and entropy bonus.
    reward = - perf_loss - budget_w * (abs(cost - target)/target)
    loss = - (reward - baseline) * logprob - entropy_w * entropy
    """
    controller = make_controller(num_layers)
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Moving average baseline for variance reduction
    baseline = None
    beta = 0.9

    for epoch in range(epochs):
        controller.train()
        epoch_obj = 0.0
        batches = 0

        for batch in train_loader:
            inp = batch['input_ids'].to(DEVICE)
            att = batch['attention_mask'].to(DEVICE)
            lab = batch['labels'].to(DEVICE)

            # Controller forward
            r_logits, b_logits, s_logits = controller(task_feat, layer_feat)

            # Categorical sampling per layer with log-probs
            r_cat = Categorical(logits=r_logits)
            b_cat = Categorical(logits=b_logits)
            s_cat = Categorical(logits=s_logits)

            r_sample = r_cat.sample()  # [1, L]
            b_sample = b_cat.sample()
            s_sample = s_cat.sample()

            # log prob sum over layers and actions
            logprob = (r_cat.log_prob(r_sample) + b_cat.log_prob(b_sample) + s_cat.log_prob(s_sample)).sum()

            # Build lists of indices
            r_list = ensure_list(r_sample.squeeze(), num_layers)
            b_list = ensure_list(b_sample.squeeze(), num_layers)
            s_list = ensure_list(s_sample.squeeze(), num_layers)

            # Apply compression to a fresh copy
            m = apply_compression(copy.deepcopy(base_model), r_list, b_list, s_list, CFG).to(DEVICE)
            m.eval()

            # Sanity: allow even if param count unchanged (quant/sparsity don't change numel)
            verify_compression(base_model, m, verbose=False)

            # Performance loss on this mini-batch
            out = m(input_ids=inp, attention_mask=att)
            perf_loss = F.cross_entropy(out.logits, lab)

            # Cost and budget penalty
            cost = calc_cost(r_list, b_list, s_list)
            budget_pen = budget_w * (abs(cost - target_cost) / max(target_cost, 1e-6))

            # Reward to maximize
            reward = - perf_loss.item() - budget_pen

            # Moving baseline
            if baseline is None:
                baseline = reward
            else:
                baseline = beta * baseline + (1 - beta) * reward

            # Entropy bonus for exploration
            r_ent = r_cat.entropy().mean()
            b_ent = b_cat.entropy().mean()
            s_ent = s_cat.entropy().mean()
            entropy = r_ent + b_ent + s_ent

            # Policy gradient loss (maximize reward)
            loss = - (reward - baseline) * logprob - entropy_w * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_obj += loss.item()
            batches += 1

            # Memory cleanup
            del m, out
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = epoch_obj / max(batches, 1)
        print(f"[Budget {target_cost}] Epoch {epoch+1}/{epochs} - PG Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

    return controller

def evaluate_controller_greedy(base_model, controller, task_feat, layer_feat, val_loader, temps=(0.5,1.0,2.0)):
    controller.eval()
    results = []
    with torch.no_grad():
        r_logits, b_logits, s_logits = controller(task_feat, layer_feat)
        for T in temps:
            r_probs = F.softmax(r_logits / T, dim=-1)
            b_probs = F.softmax(b_logits / T, dim=-1)
            s_probs = F.softmax(s_logits / T, dim=-1)
            r_idx = r_probs.argmax(-1).squeeze()
            b_idx = b_probs.argmax(-1).squeeze()
            s_idx = s_probs.argmax(-1).squeeze()

            r_list = ensure_list(r_idx, r_logits.size(1))
            b_list = ensure_list(b_idx, b_logits.size(1))
            s_list = ensure_list(s_idx, s_logits.size(1))

            m = apply_compression(copy.deepcopy(base_model), r_list, b_list, s_list, CFG).to(DEVICE)
            m.eval()

            total_loss, total = 0.0, 0
            for batch in val_loader:
                inp = batch['input_ids'].to(DEVICE)
                att = batch['attention_mask'].to(DEVICE)
                lab = batch['labels'].to(DEVICE)
                out = m(input_ids=inp, attention_mask=att)
                l = F.cross_entropy(out.logits, lab)
                total_loss += l.item() * lab.size(0)
                total += lab.size(0)
            avg_loss = total_loss / max(total, 1)
            c = calc_cost(r_list, b_list, s_list)
            results.append((T, avg_loss, c))
            del m
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    return results

def main():
    print("Loading base model and data...")
    tokenizer, base_model, train_subset, val_ds, train_loader, val_loader = load_base_model_and_data()
    task_feat, layer_feat = build_features(base_model, val_ds)
    num_layers = count_linears(base_model)

    # Define target budgets to span the frontier
    target_costs = [2.0, 8.0, 16.0, 32.0]
    epochs = 8
    controllers = []

    # Train one controller per target budget
    for target in target_costs:
        print(f"\n=== Training controller for target cost {target:.1f} ===")
        ctrl = policy_gradient_train_for_budget(
            base_model=base_model,
            train_loader=train_loader,
            task_feat=task_feat,
            layer_feat=layer_feat,
            num_layers=num_layers,
            target_cost=target,
            epochs=epochs,
            lr=1e-3,
            entropy_w=0.05,
            budget_w=2.0
        )
        save_path = f'output/baselines/controller_budget_{int(target)}.pt'
        os.makedirs('output/baselines', exist_ok=True)
        torch.save(ctrl.state_dict(), save_path)
        controllers.append((target, save_path))
        print(f"Saved controller for cost {target} -> {save_path}")

    # Evaluate all controllers with temperature sweep
    all_results = []
    for target, path in controllers:
        print(f"\n--- Evaluating controller for target {target} ---")
        ctrl = make_controller(num_layers)
        sd = torch.load(path, map_location=DEVICE)
        ctrl.load_state_dict(sd)
        res = evaluate_controller_greedy(base_model, ctrl, task_feat, layer_feat, val_loader, temps=(0.5,1.0,2.0))
        for T, loss, c in res:
            all_results.append((f"target={target},T={T}", loss, c))
            print(f"target={target}, T={T}: Loss={loss:.4f}, Cost={c:.2f}")

    # Plot Pareto frontier
    plt.figure(figsize=(8,5))
    for name, loss, cost in all_results:
        plt.scatter(cost, loss, label=name, s=80)
    plt.xlabel('Compression Cost (rank × bits × (1 - sparsity))')
    plt.ylabel('Validation Loss')
    plt.title('Budgeted Pareto Frontier: MetaController (Policy Gradient)')
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    os.makedirs('output/baselines', exist_ok=True)
    plt.tight_layout()
    plot_path = 'output/baselines/budgeted_pareto_frontier.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")

    # Save CSV
    csv_path = 'output/baselines/budgeted_pareto_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Policy', 'Loss', 'Cost'])
        for name, loss, cost in all_results:
            w.writerow([name, f"{loss:.6f}", f"{cost:.2f}"])
    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()
