import os, sys, copy, torch, random
sys.path.append('src')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Try the project's apply_compression first
from analysis.utils import apply_compression

def snapshot_named_modules(model):
    return [(n, m) for n, m in model.named_modules()]

def total_l2_delta(before, after):
    tot = 0.0
    for (n1, m1), (n2, m2) in zip(before, after):
        if hasattr(m1, 'weight') and hasattr(m2, 'weight') and m1.weight is not None and m2.weight is not None:
            w1 = m1.weight.data.detach().float().cpu()
            w2 = m2.weight.data.detach().float().cpu()
            if w1.ndim == 2 and w2.ndim == 2 and w1.shape == w2.shape:
                tot += torch.norm(w1 - w2).item()
    return tot

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'gpt2-medium'
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    base.config.pad_token_id = tok.pad_token_id
    base.to(device)

    # Build random actions consistent with Phase 3 configs
    cfg = {
        'ranks':[1,2,4,8,16],
        'bitwidths':[2,4,8,16,32],
        'sparsities':[0.0,0.2,0.5,0.7,0.9]
    }
    # Count linear layers to size action vectors
    linear_layers = [m for m in base.modules() if isinstance(m, torch.nn.Linear)]
    L = len(linear_layers)
    import random
    r_idx = [random.randrange(len(cfg['ranks'])) for _ in range(L)]
    b_idx = [random.randrange(len(cfg['bitwidths'])) for _ in range(L)]
    s_idx = [random.randrange(len(cfg['sparsities'])) for _ in range(L)]
    print(f"Layers={L}, sample actions head: ranks={r_idx[:6]}, bits={b_idx[:6]}, sparsity={s_idx[:6]}")

    # Snapshot before/after and measure L2 delta
    before = snapshot_named_modules(base)
    m = copy.deepcopy(base)
    m = apply_compression(m, r_idx, b_idx, s_idx, cfg)
    after = snapshot_named_modules(m)
    delta = total_l2_delta(before, after)
    print(f"Compression total L2 delta across linear weights: {delta:.6f}")

    # Optional: quick eval to see if loss changes at all for sanity
    ds = load_dataset('glue', 'mrpc')
    def prep(b): return tok(b['sentence1'], b['sentence2'], padding='max_length', truncation=True, max_length=128)
    data = ds.map(prep, batched=True).rename_column('label','labels')
    data.set_format('torch', columns=['input_ids','attention_mask','labels'])
    train_ds, val_ds = data['train'], data['validation']
    args = TrainingArguments(output_dir='./output/tmp_eval', per_device_eval_batch_size=8,
                             logging_strategy='no', save_strategy='no', eval_strategy='epoch',
                             disable_tqdm=True, report_to=None)
    with torch.no_grad():
        from transformers import Trainer
        base_tr = Trainer(model=base, args=args, eval_dataset=val_ds, tokenizer=tok)
        m_tr = Trainer(model=m, args=args, eval_dataset=val_ds, tokenizer=tok)
        base_loss = base_tr.evaluate()['eval_loss']
        m_loss = m_tr.evaluate()['eval_loss']
    print(f"Base eval_loss={base_loss:.6f} vs Compressed eval_loss={m_loss:.6f}")

if __name__ == "__main__":
    main()
