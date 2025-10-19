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

# Project path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

# Baseline compression config
cfg_baseline = {
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

def count_linear_layers(model):
    return len([m for m in model.modules() if isinstance(m, nn.Linear)])

def ensure_list_indices(idx, L):
    # Convert tensors to Python
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().tolist()
    # Scalars -> repeat
    if isinstance(idx, (int, np.integer, float)):
        return [int(idx)] * L
    # Non-list types -> force single then repeat
    if not isinstance(idx, (list, tuple)):
        return [int(idx)] * L
    # Empty -> default zeros
    if len(idx) == 0:
        return [0] * L
    # If length mismatch, fix by repeating or truncating
    if len(idx) < L:
        return list(idx) + [idx[-1]] * (L - len(idx))
    if len(idx) > L:
        return list(idx)[:L]
    return list(map(int, idx))

def evaluate_policy(model_base, tokenizer, ds_name, policy_fn, policy_name, epochs, output_dir, losses_dict=None):
    ds = load_dataset('glue', ds_name)
    if ds_name == 'sst2':
        prep = lambda b: tokenizer(b['sentence'], padding='max_length', truncation=True, max_length=128)
    else:
        prep = lambda b: tokenizer(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=128)
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    val_ds = data['validation']
    args = TrainingArguments(
        output_dir=os.path.join(output_dir, policy_name+'_'+ds_name),
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        logging_strategy='no',
        save_strategy='no',
        eval_strategy='epoch',
        disable_tqdm=True,
        report_to=None,
    )
    epoch_losses = []
    for ep in range(1, epochs+1):
        m = copy.deepcopy(model_base)
        L = count_linear_layers(m)
        r_idx, b_idx, s_idx = policy_fn(m, ep)
        r_idx = ensure_list_indices(r_idx, L)
        b_idx = ensure_list_indices(b_idx, L)
        s_idx = ensure_list_indices(s_idx, L)
        m = apply_compression(m, r_idx, b_idx, s_idx, cfg_baseline)
        tokenizer, m = fix_tokenizer_and_model(tokenizer, m)
        trainer = Trainer(model=m, args=args, eval_dataset=val_ds, processing_class=tokenizer)
        ev = trainer.evaluate()
        loss = ev['eval_loss']
        epoch_losses.append(loss)
        print(f"{policy_name} {ds_name} Epoch {ep} Loss {loss:.4f}")
    if losses_dict is not None:
        losses_dict[f"{policy_name}_{ds_name}"] = epoch_losses
    return epoch_losses

def uniform_policy(model, ep):
    L = count_linear_layers(model)
    i = cfg_baseline['ranks'].index(8)
    j = cfg_baseline['bitwidths'].index(8)
    k = cfg_baseline['sparsities'].index(0.5)
    return ([i]*L, [j]*L, [k]*L)

def random_policy(model, ep):
    L = count_linear_layers(model)
    return ([random.randrange(len(cfg_baseline['ranks'])) for _ in range(L)],
            [random.randrange(len(cfg_baseline['bitwidths'])) for _ in range(L)],
            [random.randrange(len(cfg_baseline['sparsities'])) for _ in range(L)])

class PolicyLoader:
    def __init__(self, ckpt_path, base_model):
        num_layers = count_linear_layers(base_model)
        # IMPORTANT: match MetaController signature (includes hidden_dim here)
        self.controller = MetaController(
            task_feat_dim=6,
            layer_feat_dim=2,
            hidden_dim=64,
            ranks=cfg_baseline['ranks'],
            bitwidths=cfg_baseline['bitwidths'],
            sparsities=cfg_baseline['sparsities'],
            num_layers=num_layers
        ).to('cuda')
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cuda')
            self.controller.load_state_dict(sd)
            self.controller.eval()
            print(f"Loaded controller from {ckpt_path}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")
        self.num_layers = num_layers

    def sample_actions(self, model):
        # Replace zero features with real ones when available
        tf = torch.zeros(1, 6).to('cuda')
        lf = torch.zeros(1, self.num_layers, 2).to('cuda')
        with torch.no_grad():
            r_p, b_p, s_p = self.controller(tf, lf)
        # Defensive conversion to per-layer lists
        r_idx = ensure_list_indices(torch.argmax(r_p, dim=-1).squeeze(), self.num_layers)
        b_idx = ensure_list_indices(torch.argmax(b_p, dim=-1).squeeze(), self.num_layers)
        s_idx = ensure_list_indices(torch.argmax(s_p, dim=-1).squeeze(), self.num_layers)
        return r_idx, b_idx, s_idx

if __name__ == '__main__':
    os.makedirs('output/baselines', exist_ok=True)
    losses = {}

    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    base = AutoModelForSequenceClassification.from_pretrained(
        'gpt2-medium', num_labels=2, ignore_mismatched_sizes=True
    ).to('cuda')
    tokenizer, base = fix_tokenizer_and_model(tokenizer, base)

    evaluate_policy(base, tokenizer, 'mrpc', uniform_policy, 'uniform', 10, 'output/baselines', losses)
    evaluate_policy(base, tokenizer, 'mrpc', random_policy, 'random', 10, 'output/baselines', losses)

    loader = PolicyLoader('output/final_meta_controller_phase3e_corrected.pt', base)
    evaluate_policy(base, tokenizer, 'sst2', lambda m,ep: loader.sample_actions(m),
                    'learned', 10, 'output/baselines', losses)

    # Optional: quick Pareto using final losses (costs for learned are placeholders until computed from sampled indices)
    final_losses = {k: v[-1] for k, v in losses.items()}
    costs = {
        'uniform_mrpc': 8 * 8 * (1 - 0.5),
        'random_mrpc': np.mean([cfg_baseline['ranks'][random.randrange(5)] * cfg_baseline['bitwidths'][random.randrange(5)] * (1 - cfg_baseline['sparsities'][random.randrange(5)]) for _ in range(50)]),
        'learned_sst2': np.mean([cfg_baseline['ranks'][random.randrange(5)] * cfg_baseline['bitwidths'][random.randrange(5)] * (1 - cfg_baseline['sparsities'][random.randrange(5)]) for _ in range(50)])
    }
    plt.figure(figsize=(8, 6))
    if 'uniform_mrpc' in final_losses:
        plt.scatter(costs['uniform_mrpc'], final_losses['uniform_mrpc'], label='Uniform MRPC', s=100, color='blue')
    if 'random_mrpc' in final_losses:
        plt.scatter(costs['random_mrpc'], final_losses['random_mrpc'], label='Random MRPC', s=100, color='orange')
    if 'learned_sst2' in final_losses:
        plt.scatter(costs['learned_sst2'], final_losses['learned_sst2'], label='Learned SST-2', s=100, color='green')
    plt.xlabel('Avg Compression Cost (rank × bits × (1-sparsity))')
    plt.ylabel('Final Eval Loss')
    plt.title('Pareto Frontier: Baselines vs. Learned Policy')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('output/baselines/pareto_frontier.png', dpi=300, bbox_inches='tight')
    print('Pareto plot saved to output/baselines/pareto_frontier.png')
