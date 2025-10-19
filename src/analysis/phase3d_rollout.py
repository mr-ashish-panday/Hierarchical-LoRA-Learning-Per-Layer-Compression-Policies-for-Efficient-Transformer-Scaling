#!/usr/bin/env python3
"""
phase3d_rollout.py - Fixed exploration and checkpointing
"""

import os, sys, argparse, yaml, copy, time, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.utils import apply_compression, set_seed_all
from models.meta_controller import MetaController

def default_config():
    return {
        'surrogate_csv': './output/surrogate_dataset.csv',
        'output_dir': './output',
        'meta_lr': 1e-3,
        'rollout_epochs': 25,
        'rollout_samples': 16,
        'temperature_start': 4.0,
        'temperature_end': 2.5,
        'entropy_coeff': 0.05,
        'baseline_momentum': 0.9,
        'adv_norm_eps': 1e-6,
        'adv_clip': 2.0,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='YAML config path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    set_seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data - simplified approach
    task_path = os.path.join(cfg['output_dir'], 'phase2_task_features.csv')
    layer_path = os.path.join(cfg['output_dir'], 'phase2_layer_stats.csv')
    
    task_feats = torch.tensor(pd.read_csv(task_path).values, dtype=torch.float32, device=device)
    layer_df = pd.read_csv(layer_path)
    layer_stats = layer_df.groupby('layer').agg({'weight_std':'mean','param_count':'sum'}).values
    layer_feats = torch.tensor(layer_stats, dtype=torch.float32).unsqueeze(0).to(device)

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
    opt = optim.Adam(mc.parameters(), lr=cfg['meta_lr'])

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

        print(f"=== Epoch {epoch+1}/{cfg['rollout_epochs']} | Temp={temp:.3f} ===")

        for sample_idx in range(cfg['rollout_samples']):
            r_p, b_p, s_p = mc(task_feats, layer_feats)
            
            # Apply temperature to logits
            def temp_softmax(p):
                return torch.softmax(torch.log(p + 1e-9) / temp, dim=-1)
            r_p, b_p, s_p = map(temp_softmax, (r_p, b_p, s_p))

            # Calculate entropy
            entropy = -(r_p*torch.log(r_p+1e-9)).sum() \
                      -(b_p*torch.log(b_p+1e-9)).sum() \
                      -(s_p*torch.log(s_p+1e-9)).sum()
            entropies.append(entropy)

            # Sample actions and compute log prob
            log_prob = torch.tensor(0., device=device, requires_grad=True)
            r_idx, b_idx, s_idx = [], [], []
            for i in range(layer_feats.size(1)):
                rd, bd, sd = Categorical(r_p[0,i]), Categorical(b_p[0,i]), Categorical(s_p[0,i])
                ri, bi, si = rd.sample(), bd.sample(), sd.sample()
                log_prob = log_prob + rd.log_prob(ri) + bd.log_prob(bi) + sd.log_prob(si)
                r_idx.append(int(ri)); b_idx.append(int(bi)); s_idx.append(int(si))
            logps.append(log_prob)

            # Debug: print first few compression choices
            if sample_idx < 3:
                print(f"  Sample {sample_idx+1}: ranks={r_idx[:3]}, bits={b_idx[:3]}, sparsity={s_idx[:3]}")

            # Apply compression and evaluate
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
            print(f"    Loss: {ev['eval_loss']:.6f}, Reward: {reward:.6f}")
            del m; torch.cuda.empty_cache()

        # Compute policy update
        mean_r = np.mean(rewards)
        running_baseline = cfg['baseline_momentum']*running_baseline + (1-cfg['baseline_momentum'])*mean_r \
                           if epoch>0 else mean_r
        adv = np.array(rewards) - running_baseline
        adv = (adv - adv.mean())/(adv.std()+cfg['adv_norm_eps'])
        adv = np.clip(adv, -cfg['adv_clip'], cfg['adv_clip'])
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        policy_loss = torch.stack([-lp * a for lp,a in zip(logps, adv_t)]).mean()
        entropy_term = -cfg['entropy_coeff'] * torch.stack(entropies).mean()
        total_loss = policy_loss + entropy_term

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
        opt.step()

        avg_ent = float(torch.stack(entropies).mean().item())
        print(f"MeanReward={mean_r:.4f} | Baseline={running_baseline:.4f} | AvgEntropy={avg_ent:.4f} | PolicyLoss={policy_loss.item():.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(mc.state_dict(), f"{cfg['output_dir']}/phase3d_epoch{epoch+1}.pt")

    torch.save(mc.state_dict(), os.path.join(cfg['output_dir'],'final_meta_controller_phase3d.pt'))
    print("Phase3D complete.")

if __name__ == '__main__':
    main()
