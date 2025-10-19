#!/usr/bin/env python3
import os, sys, glob, time, copy, random
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset
import pandas as pd
import logging

# Project root for imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))

# -----------------------------
# Built-in simple compressor (no external dependency)
# -----------------------------
import torch.nn as nn
def _quantize_sym(w: torch.Tensor, bits: int) -> torch.Tensor:
    bits = int(bits)
    if bits >= 32: return w
    qmax = float(2**(bits-1) - 1)
    scale = w.abs().max()
    if scale.item() == 0.0: return w
    step = scale / qmax
    return torch.round(w / step) * step

def _prune_mag(w: torch.Tensor, sparsity: float) -> torch.Tensor:
    s = float(sparsity)
    if s <= 0.0: return w
    numel = w.numel()
    k = int(s * numel)
    if k <= 0: return w
    if k >= numel: return torch.zeros_like(w)
    flat = w.abs().view(-1)
    thresh = torch.topk(flat, k, largest=False).values.max()
    mask = (w.abs() > thresh).to(w.dtype)
    return w * mask

def _lora_like_update(w: torch.Tensor, rank: int, scale: float = 0.05) -> torch.Tensor:
    if rank <= 0: return w
    out_f, in_f = w.shape
    r = min(rank, out_f, in_f)
    A = torch.randn(out_f, r, device=w.device, dtype=w.dtype)
    B = torch.randn(in_f, r, device=w.device, dtype=w.dtype)
    delta = scale * (A @ B.t()) / (r ** 0.5)
    return w + delta

def apply_compression(model: nn.Module,
                      r_idx, b_idx, s_idx, cfg) -> nn.Module:
    ranks = cfg['ranks']; bits = cfg['bitwidths']; spars = cfg['sparsities']
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    L = min(len(linear_layers), len(r_idx), len(b_idx), len(s_idx))
    for i in range(L):
        layer = linear_layers[i]
        W = layer.weight.data
        W = _lora_like_update(W, ranks[r_idx[i]])
        W = _quantize_sym(W, bits[b_idx[i]])
        W = _prune_mag(W, spars[s_idx[i]])
        layer.weight.data.copy_(W)
    return model

# -----------------------------
# MetaController import
# -----------------------------
from models.meta_controller import MetaController

def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def linear_anneal(ep, total, start, end):
    return start + (end - start) * (ep / max(1, total - 1))

def snapshot_named_modules(model):
    return [(n, m) for n, m in model.named_modules()]

def total_l2_delta(before, after):
    tot = 0.0
    for (n1, m1), (n2, m2) in zip(before, after):
        if hasattr(m1, 'weight') and hasattr(m2, 'weight'):
            if (m1.weight is not None) and (m2.weight is not None):
                w1, w2 = m1.weight.data.float().cpu(), m2.weight.data.float().cpu()
                if w1.ndim == 2 and w1.shape == w2.shape:
                    tot += torch.norm(w1 - w2).item()
    return tot

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"phase3e_full_corrected_{ts}.log")
    logger = logging.getLogger("phase3e_full_corrected")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger, log_path

def default_config():
    return {
        'output_dir': './output',
        'log_dir': './output/logs',
        'meta_lr': 1e-3,
        'rollout_epochs': 30,  # Reduced for overnight run
        'rollout_samples': 24,
        'warmstart_epochs': 2,  # New: supervised warm-start epochs
        'temperature_start': 4.0,
        'temperature_end': 2.0,
        'epsilon_start': 0.20,
        'epsilon_end': 0.05,
        'action_epsilon_start': 0.30,
        'action_epsilon_end': 0.05,
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
        'ranks': [1,2,4,8,16],
        'bitwidths': [2,4,8,16,32],
        'sparsities': [0.0,0.2,0.5,0.7,0.9],
        'max_grad_norm': 1.0,
        'ckpt_every': 5,
        'resume': False,  # Start fresh for warm-start
        'phase2_task_features_csv': './output/phase2_task_features.csv',
        'phase2_layer_stats_csv': './output/phase2_layer_stats.csv',
        'min_delta': 1e-3,
        'resample_tries': 5,
        'warmstart_lr': 5e-4,  # Learning rate for warm-start
        'shard_size': 50,  # New: size of eval shards for variance
    }

def find_latest_checkpoint(output_dir):
    files = glob.glob(os.path.join(output_dir, "phase3e_full_corrected_epoch*.pt"))
    if not files: return None, 0
    def epoch_num(p):
        b = os.path.basename(p)
        try: return int(b.split("epoch")[1].split(".pt")[0])
        except: return -1
    files = [(f, epoch_num(f)) for f in files]
    files = [x for x in files if x[1] >= 1]
    if not files: return None, 0
    latest = max(files, key=lambda z: z[1])
    return latest[0], latest[1]

def supervised_warmstart(mc, task_mat, layer_feats, cfg, device, logger):
    """Warm-start controller with supervised loss toward mid-level compressions on important layers."""
    mc.train()
    opt_ws = optim.Adam(mc.parameters(), lr=cfg['warmstart_lr'])
    num_layers = layer_feats.size(1)
    ranks = cfg['ranks']; bits = cfg['bitwidths']; spars = cfg['sparsities']
    
    # Target mid-level indices for layers with higher param_count/weight_std
    layer_importance = layer_feats[0, :, 1]  # param_count as importance
    mid_r_idx = torch.tensor([len(ranks)//2] * num_layers, device=device)  # Mid rank
    mid_b_idx = torch.tensor([len(bits)//2] * num_layers, device=device)  # Mid bits
    mid_s_idx = torch.tensor([1] * num_layers, device=device)  # Low sparsity (mid=0.2)
    
    # Weight targets by layer importance (normalize)
    importance_weights = layer_importance / (layer_importance.sum() + 1e-8)
    
    for ws_ep in range(cfg['warmstart_epochs']):
        total_loss = 0.0
        for _ in range(100):  # Quick 100 pseudo-samples for warm-start
            # Sample task feature
            ridx = torch.randint(0, task_mat.size(0), (1,)).item()
            tf = task_mat[ridx:ridx+1].to(device)
            lf = layer_feats
            
            # Forward
            r_logits, b_logits, s_logits = mc(tf, lf)
            
            # Target logits: mid-level, weighted by importance
            target_r = torch.full((1, num_layers), float(len(ranks)//2), device=device)
            target_b = torch.full((1, num_layers), float(len(bits)//2), device=device)
            target_s = torch.full((1, num_layers), 1.0, device=device)  # sparsity index 1 (0.2)
            
            # Cross-entropy loss, weighted
            r_loss = torch.nn.functional.cross_entropy(r_logits.transpose(1,2), target_r.long(), reduction='none')
            b_loss = torch.nn.functional.cross_entropy(b_logits.transpose(1,2), target_b.long(), reduction='none')
            s_loss = torch.nn.functional.cross_entropy(s_logits.transpose(1,2), target_s.long(), reduction='none')
            
            # Weight by layer importance
            r_loss = (r_loss * importance_weights.unsqueeze(0)).mean()
            b_loss = (b_loss * importance_weights.unsqueeze(0)).mean()
            s_loss = (s_loss * importance_weights.unsqueeze(0)).mean()
            
            loss = (r_loss + b_loss + s_loss) / 3
            total_loss += loss.item()
            
            opt_ws.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
            opt_ws.step()
        
        avg_loss = total_loss / 100
        logger.info(f"Warm-start epoch {ws_ep+1}/{cfg['warmstart_epochs']} | AvgLoss={avg_loss:.4f}")
    
    logger.info("Warm-start complete")

def evaluate_with_shard(trainer, shard_start, shard_end):
    """Evaluate on a specific shard of the validation dataset to induce variance."""
    val_ds_shard = trainer.eval_dataset.select(range(shard_start, min(shard_end, len(trainer.eval_dataset))))
    with torch.no_grad():
        ev = trainer.evaluate(eval_dataset=val_ds_shard)
    return ev

def main():
    cfg = default_config()
    os.makedirs(cfg['output_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)
    logger, log_path = setup_logger(cfg['log_dir'])
    set_seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Phase2 features
    task_mat_np = pd.read_csv(cfg['phase2_task_features_csv']).values
    task_mat = torch.tensor(task_mat_np, dtype=torch.float32)
    layer_df = pd.read_csv(cfg['phase2_layer_stats_csv'])
    layer_stats = layer_df.groupby('layer').agg({'weight_std':'mean', 'param_count':'sum'}).values
    layer_feats = torch.tensor(layer_stats, dtype=torch.float32, device=device).unsqueeze(0)

    num_layers = layer_feats.size(1)
    task_feat_dim = task_mat.size(1)
    layer_feat_dim = layer_feats.size(-1)
    logger.info(f"Features: task_dim={task_feat_dim}, layers={num_layers}, layer_dim={layer_feat_dim}")

    # Controller + optimizer
    mc = MetaController(task_feat_dim, layer_feat_dim, 64,
                        cfg['ranks'], cfg['bitwidths'], cfg['sparsities'],
                        num_layers=num_layers).to(device)
    opt = optim.Adam(mc.parameters(), lr=cfg['meta_lr'])

    # No resume for corrected run; start fresh with warm-start
    start_ep = 0

    # New: Supervised warm-start
    supervised_warmstart(mc, task_mat, layer_feats, cfg, device, logger)

    # HF stack
    tok = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=2, ignore_mismatched_sizes=True)
    base.config.pad_token_id = tok.pad_token_id
    base.to(device)

    ds = load_dataset('glue', cfg['task_name'].split('-',1)[-1])
    def prep(b):
        return tok(b['sentence1'], b['sentence2'],
                   padding='max_length', truncation=True,
                   max_length=cfg['max_seq_length'])
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    train_ds, val_ds = data['train'], data['validation']

    args_tr = TrainingArguments(
        output_dir=os.path.join(cfg['output_dir'],'rollout_full'),
        per_device_eval_batch_size=cfg['eval_batch_size'],
        per_device_train_batch_size=cfg['train_batch_size'],
        num_train_epochs=1,
        learning_rate=cfg['learning_rate'],
        logging_strategy='no',
        save_strategy='no',
        eval_strategy='no',  # Disable default eval
        disable_tqdm=True,
        report_to=None,
        seed=cfg['seed']
    )

    running_baseline = 0.0
    total_ep = cfg['rollout_epochs']
    num_shards = len(val_ds) // cfg['shard_size'] + 1
    logger.info(f"Eval shards: {num_shards} (size={cfg['shard_size']})")

    for ep in range(start_ep, total_ep):
        temp = linear_anneal(ep, total_ep, cfg['temperature_start'], cfg['temperature_end'])
        eps = linear_anneal(ep, total_ep, cfg['epsilon_start'], cfg['epsilon_end'])
        aeps = linear_anneal(ep, total_ep, cfg['action_epsilon_start'], cfg['action_epsilon_end'])
        mc.train()
        rewards, logps, ents = [], [], []

        logger.info(f"Epoch {ep+1}/{total_ep} | Temp={temp:.2f} | Eps={eps:.2f} | ActionEps={aeps:.2f}")

        for sidx in range(cfg['rollout_samples']):
            # Random real task feature row
            ridx = torch.randint(0, task_mat.size(0), (1,)).item()
            tf = task_mat[ridx:ridx+1].to(device)
            lf = layer_feats

            # Controller forward
            r_p, b_p, s_p = mc(tf, lf)

            # Temperature softmax and epsilon mixing toward uniform
            def tsoft(p): return torch.softmax(torch.log(p + 1e-9)/temp, dim=-1)
            def mix(p):
                K = p.size(-1)
                return (1.0 - eps)*p + eps*(1.0/K)
            r_p, b_p, s_p = map(lambda p: mix(tsoft(p)), (r_p, b_p, s_p))

            # Entropy
            ent = -(r_p*torch.log(r_p+1e-9)).sum() \
                  -(b_p*torch.log(b_p+1e-9)).sum() \
                  -(s_p*torch.log(s_p+1e-9)).sum()
            ents.append(ent)

            # Sample actions with action-epsilon (random override)
            r_idx, b_idx, s_idx = [], [], []
            logp = torch.tensor(0., device=device)
            for li in range(num_layers):
                if random.random() < aeps:
                    ri = random.randrange(r_p.size(-1))
                    bi = random.randrange(b_p.size(-1))
                    si = random.randrange(s_p.size(-1))
                    logp = logp + torch.log(r_p[0,li,ri] + 1e-9) \
                                + torch.log(b_p[0,li,bi] + 1e-9) \
                                + torch.log(s_p[0,li,si] + 1e-9)
                else:
                    rd, bd, sd = Categorical(r_p[0,li]), Categorical(b_p[0,li]), Categorical(s_p[0,li])
                    ri, bi, si = rd.sample(), bd.sample(), sd.sample()
                    logp = logp + rd.log_prob(ri) + bd.log_prob(bi) + sd.log_prob(si)
                    ri, bi, si = int(ri), int(bi), int(si)
                r_idx.append(ri); b_idx.append(bi); s_idx.append(si)
            logps.append(logp)

            # Ensure non-zero compression delta (resample up to N tries)
            tries = 0
            delta = 0.0
            while tries == 0 or (delta < cfg['min_delta'] and tries < cfg['resample_tries']):
                before = snapshot_named_modules(base)
                m = copy.deepcopy(base)
                m = apply_compression(m, r_idx, b_idx, s_idx, cfg)
                after = snapshot_named_modules(m)
                delta = total_l2_delta(before, after)
                if delta >= cfg['min_delta']: break
                r_idx = [random.randrange(len(cfg['ranks'])) for _ in range(num_layers)]
                b_idx = [random.randrange(len(cfg['bitwidths'])) for _ in range(num_layers)]
                s_idx = [random.randrange(len(cfg['sparsities'])) for _ in range(num_layers)]
                tries += 1

            if sidx < 3:
                logger.info(f"  Sample {sidx+1}: delta={delta:.4f} tries={tries}")

            # New: Evaluate on random shard for variance
            shard_idx = (sidx * 12345) % num_shards  # Deterministic but unique per sample
            shard_start = shard_idx * cfg['shard_size']
            shard_end = shard_start + cfg['shard_size']
            m.to(device)
            tr = Trainer(model=m, args=args_tr,
                         train_dataset=train_ds, eval_dataset=val_ds,
                         tokenizer=tok)
            ev = evaluate_with_shard(tr, shard_start, shard_end)
            reward = -float(ev['eval_loss'])
            rewards.append(reward)
            if sidx < 3:
                logger.info(f"    eval_loss={ev['eval_loss']:.6f} reward={reward:.6f} (shard {shard_idx})")
            del m; torch.cuda.empty_cache()

        # Policy update
        mean_r = float(np.mean(rewards))
        running_baseline = (cfg['baseline_momentum']*running_baseline
                            + (1-cfg['baseline_momentum'])*mean_r) if ep>0 else mean_r
        adv = np.array(rewards) - running_baseline
        std = adv.std()
        if std > 1e-9:
            adv = (adv - adv.mean())/(std + cfg['adv_norm_eps'])
        else:
            adv = adv - adv.mean()
        adv = np.clip(adv, -cfg['adv_clip'], cfg['adv_clip'])
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        policy_loss = torch.stack([-lp*a for lp,a in zip(logps, adv_t)]).mean()
        entropy_term = -cfg['entropy_coeff']*torch.stack(ents).mean()
        loss = policy_loss + entropy_term

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
        opt.step()

        avg_ent = float(torch.stack(ents).mean().item())
        std_reward = float(np.std(rewards))
        logger.info(f"Epoch {ep+1} done | MeanReward={mean_r:.4f} | RewardStd={std_reward:.4f} | Entropy={avg_ent:.4f} | PolicyLoss={policy_loss.item():.6f}")

        if (ep + 1) % cfg['ckpt_every'] == 0:
            torch.save(mc.state_dict(), os.path.join(cfg['output_dir'], f"phase3e_full_corrected_epoch{ep+1}.pt"))
            torch.save(opt.state_dict(), os.path.join(cfg['output_dir'], f"opt_epoch{ep+1}_corrected.pt"))
            logger.info(f"Saved checkpoint epoch {ep+1}")

    torch.save(mc.state_dict(), os.path.join(cfg['output_dir'], "final_meta_controller_phase3e_corrected.pt"))
    logger.info(f"COMPLETE | Logs in {log_path}")

if __name__ == '__main__':
    main()
