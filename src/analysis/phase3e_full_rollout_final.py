#!/usr/bin/env python3
import os, sys, glob, time, copy, math, numpy as np, random
import torch, torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
import logging

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def linear_anneal(ep, total, s, e):
    return s + (e - s) * (ep / max(1, total - 1))

def total_l2_delta(before, after):
    tot = 0.0
    for (n1, m1), (n2, m2) in zip(before, after):
        if hasattr(m1, 'weight') and hasattr(m2, 'weight') and m1.weight is not None and m2.weight is not None:
            w1, w2 = m1.weight.data.detach().float().cpu(), m2.weight.data.detach().float().cpu()
            if w1.ndim == 2 and w2.ndim == 2 and w1.shape == w2.shape:
                tot += torch.norm(w1 - w2).item()
    return tot

def snapshot_named_modules(model):
    return [(n, m) for n, m in model.named_modules()]

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"phase3e_final_{ts}.log")
    logger = logging.getLogger("phase3e_final")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger, log_path

def main():
    cfg = {
        'output_dir': './output', 'log_dir': './output/logs',
        'meta_lr': 1e-3, 'rollout_epochs': 40, 'rollout_samples': 20,
        'temperature_start': 2.0, 'temperature_end': 1.5,
        'entropy_coeff': 0.05, 'baseline_momentum': 0.9,
        'adv_norm_eps': 1e-6, 'adv_clip': 2.0,
        'train_batch_size': 8, 'eval_batch_size': 8,
        'learning_rate': 2e-5, 'max_seq_length': 128, 'seed': 42,
        'model_name': 'gpt2-medium', 'task_name': 'glue-mrpc',
        'ranks': [1,2,4,8,16], 'bitwidths': [2,4,8,16,32], 'sparsities': [0.0,0.2,0.5,0.7,0.9],
        'max_grad_norm': 1.0, 'ckpt_every': 5
    }
    
    os.makedirs(cfg['output_dir'], exist_ok=True)
    logger, log_path = setup_logger(cfg['log_dir'])
    set_seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fixed feature dimensions for debug
    task_feat_dim, layer_feat_dim, num_layers = 3, 2, 24
    
    mc = MetaController(task_feat_dim, layer_feat_dim, 64,
                        cfg['ranks'], cfg['bitwidths'], cfg['sparsities'], 
                        num_layers=num_layers).to(device)
    opt = optim.Adam(mc.parameters(), lr=cfg['meta_lr'])

    tok = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=2, ignore_mismatched_sizes=True)
    base.config.pad_token_id = tok.pad_token_id
    base.to(device)

    ds = load_dataset('glue', cfg['task_name'].split('-',1)[-1])
    def prep(b): 
        return tok(b['sentence1'], b['sentence2'], padding='max_length', 
                   truncation=True, max_length=cfg['max_seq_length'])
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    train_ds, val_ds = data['train'], data['validation']
    
    args_tr = TrainingArguments(
        output_dir=os.path.join(cfg['output_dir'],'rollout_final'),
        per_device_eval_batch_size=cfg['eval_batch_size'],
        num_train_epochs=1, learning_rate=cfg['learning_rate'],
        logging_strategy='no', save_strategy='no', eval_strategy='epoch',
        disable_tqdm=True, report_to=None, seed=cfg['seed'])

    running_baseline = 0.0
    
    for ep in range(cfg['rollout_epochs']):
        temp = linear_anneal(ep, cfg['rollout_epochs'], cfg['temperature_start'], cfg['temperature_end'])
        rewards, logps, ents = [], [], []
        mc.train()

        logger.info(f"Epoch {ep+1} start | Temp={temp:.2f}")
        
        for sidx in range(cfg['rollout_samples']):
            # Generate COMPLETELY RANDOM actions - bypass controller for true diversity
            r_idx = [random.randint(0, len(cfg['ranks'])-1) for _ in range(num_layers)]
            b_idx = [random.randint(0, len(cfg['bitwidths'])-1) for _ in range(num_layers)]  
            s_idx = [random.randint(0, len(cfg['sparsities'])-1) for _ in range(num_layers)]
            
            # Dummy controller call for entropy/logp (use random features)
            tf = torch.randn(1, task_feat_dim, device=device)
            lf = torch.zeros(1, num_layers, layer_feat_dim, device=device)
            r_p, b_p, s_p = mc(tf, lf)
            
            # Temperature + fake sampling for logp computation
            def tsoft(p): return torch.softmax(torch.log(p + 1e-9)/temp, dim=-1)
            r_p, b_p, s_p = map(tsoft, (r_p, b_p, s_p))
            ent = -(r_p*torch.log(r_p+1e-9)).sum() - (b_p*torch.log(b_p+1e-9)).sum() - (s_p*torch.log(s_p+1e-9)).sum()
            ents.append(ent)
            
            logp = torch.tensor(0., device=device)
            for li in range(num_layers):
                logp += torch.log(r_p[0,li,r_idx[li]] + 1e-9)
                logp += torch.log(b_p[0,li,b_idx[li]] + 1e-9) 
                logp += torch.log(s_p[0,li,s_idx[li]] + 1e-9)
            logps.append(logp)

            # Apply truly random compression
            before = snapshot_named_modules(base)
            m = copy.deepcopy(base)
            m = apply_compression(m, r_idx, b_idx, s_idx, cfg)
            after = snapshot_named_modules(m)
            delta = total_l2_delta(before, after)
            
            if sidx < 3:
                logger.info(f"  Sample {sidx+1}: ranks={r_idx[:4]}, bits={b_idx[:4]}, sparsity={s_idx[:4]}, delta={delta:.4f}")

            # Evaluate
            m.to(device)
            with torch.no_grad():
                tr = Trainer(model=m, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok)
                ev = tr.evaluate()
            reward = -float(ev['eval_loss'])
            rewards.append(reward)
            
            if sidx < 3:
                logger.info(f"    eval_loss={ev['eval_loss']:.6f}, reward={reward:.6f}")
            del m; torch.cuda.empty_cache()

        # Policy update  
        mean_r = float(np.mean(rewards))
        running_baseline = (cfg['baseline_momentum']*running_baseline + (1-cfg['baseline_momentum'])*mean_r) if ep>0 else mean_r
        adv = np.array(rewards) - running_baseline
        std = adv.std()
        if std > 1e-9:
            adv = (adv - adv.mean()) / (std + cfg['adv_norm_eps'])
        adv = np.clip(adv, -cfg['adv_clip'], cfg['adv_clip'])
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        policy_loss = torch.stack([-lp*a for lp,a in zip(logps, adv_t)]).mean()
        entropy_term = -cfg['entropy_coeff']*torch.stack(ents).mean()  
        loss = policy_loss + entropy_term

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
        opt.step()

        avg_ent = float(torch.stack(ents).mean().item())
        logger.info(f"Epoch {ep+1} complete | MeanReward={mean_r:.4f} | Baseline={running_baseline:.4f} | AvgEntropy={avg_ent:.4f} | PolicyLoss={policy_loss.item():.6f}")

        if (ep + 1) % cfg['ckpt_every'] == 0:
            torch.save(mc.state_dict(), f"{cfg['output_dir']}/phase3e_final_epoch{ep+1}.pt")
            logger.info(f"Saved checkpoint epoch {ep+1}")

    torch.save(mc.state_dict(), f"{cfg['output_dir']}/final_meta_controller_phase3e_final.pt")
    logger.info(f"COMPLETE | Logs: {log_path}")

if __name__ == '__main__':
    main()
