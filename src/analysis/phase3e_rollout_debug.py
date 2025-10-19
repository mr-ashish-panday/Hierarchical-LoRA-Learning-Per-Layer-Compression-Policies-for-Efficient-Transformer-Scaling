b  #!/usr/bin/env python3
import os, sys, argparse, yaml, copy, numpy as np, torch, torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

def default_config():
    return {
        'output_dir': './output',
        'meta_lr': 1e-3,
        'rollout_epochs': 10,
        'rollout_samples': 12,
        'temperature_start': 5.0,
        'temperature_end': 3.0,
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
    }

def linear_anneal(ep, total, s, e):
    return s + (e - s) * (ep / max(1, total - 1))

def snapshot_named_modules(model):
    return [(n, m) for n, m in model.named_modules()]

def total_l2_delta(before, after):
    tot = 0.0
    import torch
    for (n1, m1), (n2, m2) in zip(before, after):
        if hasattr(m1, 'weight') and hasattr(m2, 'weight') and m1.weight is not None and m2.weight is not None:
            w1 = m1.weight.data.detach().float().cpu()
            w2 = m2.weight.data.detach().float().cpu()
            if w1.ndim == 2 and w2.ndim == 2 and w1.shape == w2.shape:
                tot += torch.norm(w1 - w2).item()
    return tot

def main():
    cfg = default_config()
    os.makedirs(cfg['output_dir'], exist_ok=True)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Minimal features (uniform dummy) so shapes align; diversity comes from sampling/actions
    # Controller input dims: 3 task features, 2 layer features
    num_layers = 24
    task_feat_dim, layer_feat_dim = 3, 2
    task_feats = torch.zeros(1, task_feat_dim, device=device)
    layer_feats = torch.zeros(1, num_layers, layer_feat_dim, device=device)

    mc = MetaController(task_feat_dim, layer_feat_dim, 64,
                        cfg['ranks'], cfg['bitwidths'], cfg['sparsities'],
                        num_layers=num_layers).to(device)
    opt = optim.Adam(mc.parameters(), lr=cfg['meta_lr'])

    tok = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    base = AutoModelForSequenceClassification.from_pretrained(cfg['model_name'], num_labels=2, ignore_mismatched_sizes=True)
    base.config.pad_token_id = tok.pad_token_id
    base.to(device)

    ds = load_dataset('glue', cfg['task_name'].split('-',1)[-1])
    def prep(b): return tok(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=cfg['max_seq_length'])
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    train_ds, val_ds = data['train'], data['validation']
    args_tr = TrainingArguments(output_dir=os.path.join(cfg['output_dir'],'rollout_dbg'),
                                per_device_eval_batch_size=cfg['eval_batch_size'],
                                per_device_train_batch_size=cfg['train_batch_size'],
                                num_train_epochs=1, learning_rate=cfg['learning_rate'],
                                logging_strategy='no', save_strategy='no',
                                eval_strategy='epoch', disable_tqdm=True, report_to=None)

    running_baseline = 0.0
    for ep in range(cfg['rollout_epochs']):
        temp = linear_anneal(ep, cfg['rollout_epochs'], cfg['temperature_start'], cfg['temperature_end'])
        rewards, logps, ents = [], [], []
        print(f"=== Epoch {ep+1}/{cfg['rollout_epochs']} | Temp={temp:.2f} ===")
        for sidx in range(cfg['rollout_samples']):
            r_p, b_p, s_p = mc(task_feats, layer_feats)
            def tsoft(p): return torch.softmax(torch.log(p + 1e-9)/temp, dim=-1)
            r_p, b_p, s_p = map(tsoft, (r_p, b_p, s_p))
            from torch.distributions import Categorical
            logp = torch.tensor(0., device=device)
            r_idx, b_idx, s_idx = [], [], []
            for li in range(num_layers):
                rd = Categorical(r_p[0,li]); bd = Categorical(b_p[0,li]); sd = Categorical(s_p[0,li])
                ri, bi, si = rd.sample(), bd.sample(), sd.sample()
                logp = logp + rd.log_prob(ri) + bd.log_prob(bi) + sd.log_prob(si)
                r_idx.append(int(ri)); b_idx.append(int(bi)); s_idx.append(int(si))
            ent = -(r_p*torch.log(r_p+1e-9)).sum() - (b_p*torch.log(b_p+1e-9)).sum() - (s_p*torch.log(s_p+1e-9)).sum()
            ents.append(ent)

            if sidx < 3:
                print(f"  Sample {sidx+1}: ranks={r_idx[:6]}, bits={b_idx[:6]}, sparsity={s_idx[:6]}")

            before = snapshot_named_modules(base)
            m = copy.deepcopy(base)
            m = apply_compression(m, r_idx, b_idx, s_idx, cfg)
            after = snapshot_named_modules(m)
            delta = total_l2_delta(before, after)
            if sidx < 3:
                print(f"    Compression L2 delta: {delta:.4f}")

            with torch.no_grad():
                tr = Trainer(model=m, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok)
                ev = tr.evaluate()
            reward = -float(ev['eval_loss'])
            rewards.append(reward)
            del m; torch.cuda.empty_cache()

            logps.append(logp)

        mean_r = float(np.mean(rewards))
        running_baseline = (cfg['baseline_momentum']*running_baseline + (1-cfg['baseline_momentum'])*mean_r) if ep>0 else mean_r
        adv = np.array(rewards) - running_baseline
        adv = (adv - adv.mean())/(adv.std()+cfg['adv_norm_eps'])
        adv = np.clip(adv, -cfg['adv_clip'], cfg['adv_clip'])
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        policy_loss = torch.stack([-lp*a for lp,a in zip(logps, adv_t)]).mean()
        entropy_term = -cfg['entropy_coeff']*torch.stack(ents).mean()
        loss = policy_loss + entropy_term
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
        opt.step()
        print(f"MeanReward={mean_r:.4f} | Baseline={running_baseline:.4f}")

        if (ep+1) % 5 == 0:
            torch.save(mc.state_dict(), f"{cfg['output_dir']}/phase3e_epoch{ep+1}.pt")

    torch.save(mc.state_dict(), os.path.join(cfg['output_dir'], 'final_meta_controller_phase3e.pt'))
    print('Phase3E debug complete.')

if __name__ == '__main__':
    import torch, numpy as np
    main()
