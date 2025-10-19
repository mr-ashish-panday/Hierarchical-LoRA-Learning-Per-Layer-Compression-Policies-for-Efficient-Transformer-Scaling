#!/usr/bin/env python3
"""
debug_meta_controller.py

Creates diagnostics for your MetaController and apply_compression pipeline:
- attempts robust imports (searches repo if necessary)
- prints shapes / mean / std / min / max of controller logits
- samples a policy and computes cost
- optionally applies compression and checks parameter counts
- runs a tiny gradient probe to ensure controller gets non-zero grads
- prints contents of output/baselines/final_pareto.csv if present

Run: python3 debug_meta_controller.py
"""
import os
import sys
import copy
import glob
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

# --- Config ---
CFG = {'ranks':[1,2,4,8,16],'bitwidths':[2,4,8,16,32],'sparsities':[0.0,0.2,0.5,0.7,0.9]}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")

# --- try to add src-like paths to sys.path ---
def add_candidate_paths():
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, 'src'),
        os.path.join(cwd, '..', 'src'),
        os.path.join(cwd, '..', '..', 'src'),
        os.path.join(cwd),
        os.path.join(cwd, '..'),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            print(f"[INFO] Added to sys.path: {p}")
            return
    # fallback
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        print(f"[WARN] No src dir found; added CWD to sys.path: {cwd}")

add_candidate_paths()

# --- robust importer helpers ---
def try_import(module_name, attr=None):
    try:
        mod = __import__(module_name, fromlist=['*'])
        if attr:
            return getattr(mod, attr)
        return mod
    except Exception as e:
        return None

def load_class_from_file(filepath, class_name):
    try:
        spec = importlib.util.spec_from_file_location("dyn_mod_"+os.path.basename(filepath).replace(".","_"), filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except Exception as e:
        return None

# --- try to import MetaController ---
MetaController = try_import('models.meta_controller', 'MetaController')
if MetaController is None:
    # search for a file that contains class MetaController
    found = None
    for root, dirs, files in os.walk(os.getcwd()):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        txt = fh.read()
                    if 'class MetaController' in txt:
                        found = path
                        break
                except Exception:
                    continue
        if found:
            break
    if found:
        MetaController = load_class_from_file(found, 'MetaController')
        if MetaController:
            print(f"[INFO] Loaded MetaController from {found}")
    if MetaController is None:
        print("[ERROR] Could not find MetaController class automatically. Please ensure it's in src/models or run from project root.")
        # exit early but continue trying other diagnostics
        # sys.exit(1)

# --- try to import apply_compression and LowRankLinear ---
apply_compression = None
LowRankLinear = None
try:
    ac_mod = __import__('analysis.apply_compression_simple', fromlist=['*'])
    apply_compression = getattr(ac_mod, 'apply_compression', None)
    LowRankLinear = getattr(ac_mod, 'LowRankLinear', None)
    if apply_compression:
        print("[INFO] Imported apply_compression from analysis.apply_compression_simple")
except Exception as e:
    # attempt search by file content
    found_ac = None
    for root, dirs, files in os.walk(os.getcwd()):
        for f in files:
            if f == 'apply_compression_simple.py' or 'apply_compression' in f:
                try:
                    with open(os.path.join(root,f), 'r', encoding='utf-8') as fh:
                        txt = fh.read()
                    if 'def apply_compression' in txt or 'class LowRankLinear' in txt:
                        found_ac = os.path.join(root,f)
                        break
                except Exception:
                    continue
        if found_ac:
            break
    if found_ac:
        try:
            spec = importlib.util.spec_from_file_location("ac_dyn", found_ac)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            apply_compression = getattr(mod, 'apply_compression', None)
            LowRankLinear = getattr(mod, 'LowRankLinear', None)
            print(f"[INFO] Loaded apply_compression from {found_ac}")
        except Exception as e2:
            print("[WARN] failed to load apply_compression dynamically:", e2)
    else:
        print("[WARN] apply_compression_simple not found automatically.")

# --- load tokenizer & base model (to compute param counts) ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("[INFO] Loading base model (may take a while if not cached)...")
try:
    base_model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
except Exception as e:
    print("[ERROR] Could not load base model:", e)
    base_model = None

# --- determine NL (number of linear-ish layers) ---
def count_linears(m):
    if m is None:
        return 0
    types = (nn.Linear,)
    if LowRankLinear is not None:
        types = (nn.Linear, LowRankLinear)
    return sum(1 for _ in m.modules() if isinstance(_, types))
NL = count_linears(base_model)
print(f"[INFO] NL (count of linear/lowrank modules) = {NL}")

# --- find controller checkpoint file ---
possible_ckpts = [
    'output/baselines/ctrl_budget_32.pt',
    'output/baselines/ctrl_budget_16.pt',
    'output/baselines/ctrl_budget_8.pt',
    'output/baselines/ctrl_budget_2.pt',
    'output/final_meta_controller_phase3e_corrected.pt'
]
ckpt = None
for p in possible_ckpts:
    if os.path.exists(p):
        ckpt = p
        break
# fallback: any .pt in output or output/baselines
if ckpt is None:
    for d in ['output/baselines', 'output']:
        if os.path.isdir(d):
            files = glob.glob(os.path.join(d, '*.pt'))
            if files:
                ckpt = files[0]; break
if ckpt is None:
    print("[WARN] No controller checkpoint found in default locations. Please provide a .pt file in output/baselines or output/")
else:
    print(f"[INFO] Using controller checkpoint: {ckpt}")

# --- instantiate controller if possible ---
C = None
if MetaController is not None:
    try:
        C = MetaController(6, 2, 64, CFG['ranks'], CFG['bitwidths'], CFG['sparsities'], num_layers=NL).to(DEVICE)
        print("[INFO] MetaController instantiated.")
        if ckpt:
            try:
                C.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                print("[INFO] Controller state_dict loaded.")
            except Exception as e:
                print("[WARN] Failed to load checkpoint into MetaController:", e)
    except Exception as e:
        print("[ERROR] Failed to instantiate MetaController:", e)
else:
    print("[ERROR] MetaController not available; cannot run controller diagnostics.")

# --- helper to normalize logits to (NL, K) ---
def normalize_logits(logits, NL):
    if logits is None:
        raise RuntimeError("Logits is None")
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    t = logits
    if t.dim() == 1:
        return t.unsqueeze(0).repeat(NL, 1)
    if t.dim() == 2:
        # (NL, K) or (1, K)
        if t.size(0) == 1:
            return t.repeat(NL, 1)
        if t.size(0) == NL:
            return t
        # fallback: trim/pad
        if t.size(0) > NL:
            return t[:NL, :]
        else:
            return t.repeat(int(np.ceil(NL / t.size(0))), 1)[:NL, :]
    if t.dim() == 3:
        # (1, NL, K) or (B, NL, K)
        if t.size(0) == 1:
            return t.squeeze(0)
        return t[0]
    raise RuntimeError(f"Unexpected logits dim: {t.dim()}")

# --- diagnostics: logits stats, sampling, cost, compression check, grad probe ---
def run_diagnostics():
    if C is None:
        print("[ERROR] Controller instance missing; aborting diagnostics.")
        return
    tf = torch.zeros(1, 6, device=DEVICE)
    lf = torch.zeros(1, max(NL,1), 2, device=DEVICE)

    # forward (no grad) for stats
    C.eval()
    with torch.no_grad():
        try:
            r_log, b_log, s_log = C(tf, lf)
        except Exception as e:
            print("[ERROR] Calling controller failed:", e)
            return

    try:
        r_log = normalize_logits(r_log, NL)
        b_log = normalize_logits(b_log, NL)
        s_log = normalize_logits(s_log, NL)
    except Exception as e:
        print("[ERROR] Normalizing logits failed:", e)
        return

    # print stats
    def print_stats(name, t):
        print(f"[LOGIT] {name} shape={tuple(t.shape)} mean={t.mean().item():.6f} std={t.std().item():.6f} min={t.min().item():.6f} max={t.max().item():.6f}")
    print_stats('r_log', r_log)
    print_stats('b_log', b_log)
    print_stats('s_log', s_log)

    # sample a policy
    r_cat = Categorical(logits=r_log)
    b_cat = Categorical(logits=b_log)
    s_cat = Categorical(logits=s_log)
    r_s = r_cat.sample()  # (NL,)
    b_s = b_cat.sample()
    s_s = s_cat.sample()
    r_idx = [int(x.item()) for x in r_s]
    b_idx = [int(x.item()) for x in b_s]
    s_idx = [int(x.item()) for x in s_s]
    print(f"[SAMPLE] r_idx (first 8): {r_idx[:8]} ... total {len(r_idx)}")
    print(f"[SAMPLE] b_idx (first 8): {b_idx[:8]} ... total {len(b_idx)}")
    print(f"[SAMPLE] s_idx (first 8): {s_idx[:8]} ... total {len(s_idx)}")

    # compute a simple cost
    cost_list = [CFG['ranks'][ri] * CFG['bitwidths'][bi] * (1.0 - CFG['sparsities'][si]) for ri,bi,si in zip(r_idx, b_idx, s_idx)]
    cost = float(np.mean(cost_list)) if cost_list else 0.0
    print(f"[COST] mean cost per layer = {cost:.6f}")

    # check apply_compression effect if available
    if apply_compression is not None and base_model is not None:
        try:
            orig_params = sum(p.numel() for p in base_model.parameters())
            comp = apply_compression(copy.deepcopy(base_model).cpu(), r_idx, b_idx, s_idx, CFG)
            comp_params = sum(p.numel() for p in comp.parameters())
            print(f"[COMP] orig_params={orig_params:,}, comp_params={comp_params:,}, ratio={comp_params/orig_params:.6f}")
        except Exception as e:
            print("[WARN] apply_compression call failed:", e)
    else:
        print("[INFO] apply_compression not available or base_model missing; skipping compression param check.")

    # gradient probe: sample and compute backward on log_prob (fake advantage=+1)
    try:
        C.train()
        # fresh forward (with grad)
        r_log2, b_log2, s_log2 = C(tf, lf)
        r_log2 = normalize_logits(r_log2, NL)
        b_log2 = normalize_logits(b_log2, NL)
        s_log2 = normalize_logits(s_log2, NL)
        r_cat2 = Categorical(logits=r_log2)
        b_cat2 = Categorical(logits=b_log2)
        s_cat2 = Categorical(logits=s_log2)
        r_sample = r_cat2.sample()
        b_sample = b_cat2.sample()
        s_sample = s_cat2.sample()
        logp_sum = r_cat2.log_prob(r_sample).sum() + b_cat2.log_prob(b_sample).sum() + s_cat2.log_prob(s_sample).sum()
        loss = -1.0 * logp_sum  # pretend advantage=+1 to produce gradient
        # zero grads, backward
        for p in C.parameters():
            if p.grad is not None:
                p.grad.detach_(); p.grad.zero_()
        loss.backward()
        grads = [p.grad for p in C.parameters() if p.requires_grad]
        total = len(grads)
        zero = sum(1 for g in grads if g is None or torch.allclose(g, torch.zeros_like(g)))
        print(f"[GRAD PROBE] zero_grad_params = {zero}/{total} (should be < total)")
    except Exception as e:
        print("[WARN] Gradient probe failed:", e)

# --- run diagnostics ---
run_diagnostics()

# --- print final_pareto.csv if exists ---
csv_path = 'output/baselines/final_pareto.csv'
if os.path.exists(csv_path):
    print(f"[INFO] Showing {csv_path}:")
    try:
        with open(csv_path, 'r', encoding='utf-8') as fh:
            txt = fh.read()
        print(txt)
    except Exception as e:
        print("[WARN] Could not read CSV:", e)
else:
    print(f"[INFO] {csv_path} not found; nothing to display.")

print("[DONE] debug_meta_controller.py finished.")
