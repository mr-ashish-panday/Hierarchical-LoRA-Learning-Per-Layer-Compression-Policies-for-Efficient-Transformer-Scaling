#!/usr/bin/env python3
"""
phase3c_rollout.py

Enhanced Phase 3B rollout with stability fixes:
- Entropy regularization
- Slower temperature annealing (3.0 -> 1.5 over 50 epochs)
- Increased rollout_samples = 32
- Surrogate warm-start for 5 epochs
- Advantage normalization & clipping
- KL divergence regularization placeholder
"""

import os
import sys
import argparse
import yaml
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.models.auto import AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd

# Add project root so imports work
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.utils import apply_compression, set_seed_all
from models.meta_controller import MetaController

def default_config():
    return {
        'surrogate_csv': './output/surrogate_dataset.csv',
        'output_dir': './output',
        'meta_lr': 5e-3,
        'rollout_epochs': 50,
        'rollout_samples': 32,
        'temperature_start': 3.0,
        'temperature_end': 1.5,
        'entropy_coeff': 0.02,
        'kl_coeff': 0.01,
        'warmup_epochs': 5,
        'baseline_momentum': 0.9,
        'adv_norm_eps': 1e-6,
        'adv_clip': 3.0,
        'train_batch_size': 8,
        'eval_batch_size': 8,
        'learning_rate': 2e-5,
        'max_seq_length': 128,
        'seed': 42,
        'model_name': 'gpt2-medium',
        'task_name': 'glue-mrpc',
        'ranks': [1, 2, 4, 8, 16],
        'bitwidths': [2, 4, 8, 16, 32],
        'sparsities': [0.0, 0.2, 0.5, 0.7, 0.9],
        'max_grad_norm': 1.0,
    }

def load_config(path=None):
    cfg = default_config()
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg

def linear_temp(epoch, total, start, end):
    return start + (end - start) * (epoch / max(1, total - 1))

def surrogate_warmup(mc, cfg, task_feats, layer_feats):
    opt = mc.optimizer
    loss_fn = nn.MSELoss()
    # Take only first row for warmup and add batch dimension
    tf = task_feats[0:1] if task_feats.dim() == 2 else task_feats.unsqueeze(0)
    lf = layer_feats if layer_feats.dim() == 3 else layer_feats.unsqueeze(0)
    for ep in range(cfg['warmup_epochs']):
        mc.train()
        r_p, b_p, s_p = mc(tf, lf)
        # Compute loss as sum of all three probability distributions
        target = torch.zeros_like(r_p.view(-1), device=r_p.device)
        loss = loss_fn(r_p.view(-1), target) + loss_fn(b_p.view(-1), target) + loss_fn(s_p.view(-1), target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
        opt.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='YAML config path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    set_seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load surrogate data
    df = pd.read_csv(cfg['surrogate_csv'])
    task_feats = torch.tensor(df[['rank','bitwidth','sparsity']].values, dtype=torch.float32, device=device)
    layer_df = pd.read_csv(os.path.join(cfg['output_dir'], 'phase2_layer_stats.csv'))
    layer_stats = layer_df.groupby('layer').agg({'weight_std':'mean','param_count':'sum'}).values
    layer_feats = torch.tensor(layer_stats, dtype=torch.float32, device=device).unsqueeze(0)

    # Initialize MetaController
    mc = MetaController(
        task_feat_dim=task_feats.size(1),
        layer_feat_dim=layer_feats.size(-1),
        hidden_dim=64,
        ranks=cfg['ranks'],
        bitwidths=cfg['bitwidths'],
        sparsities=cfg['sparsities'],
        num_layers=layer_feats.size(1)
    ).to(device)
    mc.optimizer = optim.Adam(mc.parameters(), lr=cfg['meta_lr'])

    # Warm-start on surrogate
    surrogate_warmup(mc, cfg, task_feats, layer_feats)

    # Tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=2, ignore_mismatched_sizes=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)

    # Prepare dataset
    ds = load_dataset("glue", cfg['task_name'].split('-',1)[-1])
    def preprocess(batch):
        return tokenizer(batch["sentence1"], batch["sentence2"],
                         truncation=True, padding="max_length",
                         max_length=cfg['max_seq_length'])
    data = ds.map(preprocess, batched=True)
    data = data.rename_column("label", "labels")
    data.set_format("torch", columns=["input_ids","attention_mask","labels"])
    train_ds, val_ds = data["train"], data["validation"]

    trainer_args = TrainingArguments(
        output_dir=os.path.join(cfg['output_dir'],'rollout'),
        per_device_train_batch_size=cfg['train_batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'],
        num_train_epochs=1,
        learning_rate=cfg['learning_rate'],
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="epoch",
        disable_tqdm=True,
        report_to=None,
        seed=cfg['seed']
    )

    running_baseline = 0.0
    for epoch in range(cfg['rollout_epochs']):
        temp = linear_temp(epoch, cfg['rollout_epochs'], cfg['temperature_start'], cfg['temperature_end'])
        mc.train()
        rewards, logps, entropies = [], [], []

        for sample_idx in range(cfg['rollout_samples']):
            # Use first row with batch dimension for rollout
            tf = task_feats[0:1]
            lf = layer_feats
            r_p, b_p, s_p = mc(tf, lf)
            # Apply temperature
            def temp_softmax(p):
                return torch.softmax(torch.log(p + 1e-9) / temp, dim=-1)
            r_p, b_p, s_p = map(temp_softmax, (r_p, b_p, s_p))

            # Entropy regularization
            entropy = -(r_p*torch.log(r_p+1e-9)).sum() \
                      -(b_p*torch.log(b_p+1e-9)).sum() \
                      -(s_p*torch.log(s_p+1e-9)).sum()
            entropies.append(entropy)

            # Sample actions
            log_prob = torch.tensor(0., device=device)
            r_idx, b_idx, s_idx = [], [], []
            for i in range(layer_feats.size(1)):
                rd, bd, sd = Categorical(r_p[0,i]), Categorical(b_p[0,i]), Categorical(s_p[0,i])
                ri, bi, si = rd.sample(), bd.sample(), sd.sample()
                log_prob = log_prob + rd.log_prob(ri) + bd.log_prob(bi) + sd.log_prob(si)
                r_idx.append(int(ri)); b_idx.append(int(bi)); s_idx.append(int(si))
            logps.append(log_prob)

            # Compress and evaluate
            m = copy.deepcopy(base_model)
            m = apply_compression(m, r_idx, b_idx, s_idx, cfg)
            m.to(device)
            with torch.no_grad():
                trainer = Trainer(model=m, args=trainer_args,
                                  train_dataset=train_ds, eval_dataset=val_ds,
                                  tokenizer=tokenizer)
                ev = trainer.evaluate()
            reward = -float(ev["eval_loss"])
            rewards.append(reward)
            del m; torch.cuda.empty_cache()

        # Compute normalized, clipped advantages
        mean_r = np.mean(rewards)
        running_baseline = cfg['baseline_momentum']*running_baseline + (1-cfg['baseline_momentum'])*mean_r \
                           if epoch>0 else mean_r
        adv = np.array(rewards) - running_baseline
        adv = (adv - adv.mean())/(adv.std()+cfg['adv_norm_eps'])
        adv = np.clip(adv, -cfg['adv_clip'], cfg['adv_clip'])
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        # Policy loss + entropy bonus + KL reg placeholder
        policy_loss = torch.stack([-lp * a for lp,a in zip(logps, adv_t)]).mean()
        entropy_term = -cfg['entropy_coeff'] * torch.stack(entropies).mean()
        # kl_term = cfg['kl_coeff'] * compute_kl_div(mc, uniform_dist)  # Placeholder
        total_loss = policy_loss + entropy_term  # + kl_term

        mc.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
        mc.optimizer.step()

        avg_ent = float(torch.stack(entropies).mean().item())
        print(f"Epoch {epoch+1}/{cfg['rollout_epochs']} | MeanReward={mean_r:.4f} | AvgEntropy={avg_ent:.4f}")

    torch.save(mc.state_dict(), os.path.join(cfg['output_dir'],'final_meta_controller_phase3c.pt'))
    print("Phase3C complete. Meta-controller saved.")

if __name__ == '__main__':
    main()