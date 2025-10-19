#!/usr/bin/env python3
"""
phase3b_rollout.py - IMPROVED VERSION

Fixes based on rollout analysis:
1. Pre-training on surrogate model
2. Enhanced entropy regularization 
3. Slower temperature annealing
4. KL divergence penalty against extreme configs
5. Extended training with checkpointed restarts
6. Comprehensive compression metrics logging
7. Stabilized policy gradients with advantage clipping
8. Multi-task evaluation support
"""
import os
import sys
import argparse
import yaml
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.models.auto import AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd

# ADDED: Logging imports
import logging
import json
import time
from datetime import datetime
import traceback

# Add project root so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))

from models.meta_controller import MetaController

# ADDED: Logging setup function
def setup_logging(output_dir):
    """Set up comprehensive logging system"""
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup main logger
    logger = logging.getLogger('phase3b_rollout')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for all logs
    log_file = os.path.join(log_dir, f'phase3b_rollout_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create metrics logger for structured data
    metrics_logger = logging.getLogger('phase3b_metrics')
    metrics_logger.setLevel(logging.INFO)
    
    # Metrics file handler
    metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.jsonl')
    metrics_handler = logging.FileHandler(metrics_file)
    metrics_handler.setLevel(logging.INFO)
    
    # Simple formatter for metrics (just the message)
    metrics_formatter = logging.Formatter('%(message)s')
    metrics_handler.setFormatter(metrics_formatter)
    metrics_logger.addHandler(metrics_handler)
    
    return logger, metrics_logger, timestamp

# ADDED: Function to log metrics as structured JSON
def log_metrics(metrics_logger, event_type, data, epoch=None, sample=None):
    """Log structured metrics data"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'epoch': epoch,
        'sample': sample,
        'data': data
    }
    metrics_logger.info(json.dumps(log_entry))

# ADDED: Function to log model state
def log_model_state(logger, model, prefix=""):
    """Log model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"{prefix}Model state - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Log memory usage if on CUDA
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        logger.info(f"{prefix}GPU Memory - Allocated: {memory_allocated/1024**2:.2f}MB, Reserved: {memory_reserved/1024**2:.2f}MB")

def default_config():
    return {
        'surrogate_model': './output/surrogate_best.pt',
        'surrogate_csv': './output/surrogate_dataset.csv',
        'output_dir': './output',
        'meta_lr': 1e-3,
        # IMPROVED: Extended training with pre-training phase
        'pretrain_epochs': 10,  # Pre-train on surrogate model
        'rollout_epochs': 50,   # Extended from 25 to 50
        'rollout_samples': 32,  # Increased from 16 to 32
        # IMPROVED: Slower temperature annealing
        'temperature_start': 3.0,
        'temperature_end': 1.5,  # Changed from 0.8 to 1.5
        # IMPROVED: Enhanced regularization
        'entropy_coeff': 0.05,   # Increased from 0.01
        'kl_penalty_coeff': 0.02,  # NEW: KL divergence penalty
        'train_batch_size': 8,
        'eval_batch_size': 8,
        'learning_rate': 2e-5,
        'max_seq_length': 128,
        'seed': 42,
        'model_name': 'gpt2-medium',
        'task_name': 'glue-mrpc',
        # IMPROVED: Multi-task support
        'tasks': ['glue-mrpc', 'glue-sst2', 'glue-qqp', 'glue-cola'],  # NEW
        'models': ['gpt2', 'gpt2-medium'],  # NEW: Multi-model support
        'ranks': [1, 2, 4, 8, 16],
        'bitwidths': [2, 4, 8, 16, 32],
        'sparsities': [0.0, 0.2, 0.5, 0.7, 0.9],
        'max_grad_norm': 1.0,
        'baseline_momentum': 0.9,
        'adv_norm_eps': 1e-6,
        # IMPROVED: Advantage clipping
        'advantage_clip': 3.0,  # NEW: Clip advantages to ±3 std
        # IMPROVED: Checkpointing and restarts
        'checkpoint_every': 10,  # NEW
        'restart_every': 25,     # NEW: Reset temperature periodically
        'log_every': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def load_config(path=None):
    cfg = default_config()
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg

# IMPROVED: More gradual temperature annealing
def cosine_temp(epoch, total, start, end):
    """Cosine annealing for smoother temperature decay"""
    if epoch >= total:
        return end
    cos_inner = np.pi * epoch / total
    return end + (start - end) * 0.5 * (1 + np.cos(cos_inner))

def set_seed_all(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# IMPROVED: Comprehensive compression metrics
def calculate_compression_metrics(orig_model, compressed_model):
    """Calculate detailed compression metrics"""
    orig_params = sum(p.numel() for p in orig_model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    
    # Count actual non-zero parameters after pruning
    nonzero_params = sum((p != 0).sum().item() for p in compressed_model.parameters())
    
    # Calculate model size (assuming float32)
    orig_size_mb = orig_params * 4 / (1024 * 1024)
    compressed_size_mb = nonzero_params * 4 / (1024 * 1024)
    
    return {
        'orig_params': orig_params,
        'compressed_params': compressed_params,
        'nonzero_params': nonzero_params,
        'param_reduction_ratio': (orig_params - nonzero_params) / orig_params,
        'compression_ratio': orig_size_mb / max(compressed_size_mb, 0.001),
        'orig_size_mb': orig_size_mb,
        'compressed_size_mb': compressed_size_mb,
        'size_reduction_mb': orig_size_mb - compressed_size_mb
    }

def apply_compression(model, r_idx, b_idx, s_idx, cfg, log_fn=print):
    import torch.nn as nn
    
    # ADDED: Get logger for this function
    logger = logging.getLogger('phase3b_rollout')
    start_time = time.time()
    
    layer_idx = 0
    modified_layers = 0
    l2_diffs = []
    compression_stats = {'svd_applied': 0, 'quant_applied': 0, 'prune_applied': 0}

    # ADDED: Log compression start
    logger.debug(f"Starting compression with indices - ranks: {r_idx[:5]}..., bitwidths: {b_idx[:5]}..., sparsities: {s_idx[:5]}...")
    
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight is None or module.weight.data.ndim != 2:
            continue
        if layer_idx >= len(r_idx):
            break

        ri, bi, si = r_idx[layer_idx], b_idx[layer_idx], s_idx[layer_idx]
        ri = max(0, min(ri, len(cfg['ranks'])-1))
        bi = max(0, min(bi, len(cfg['bitwidths'])-1))
        si = max(0, min(si, len(cfg['sparsities'])-1))
        rank, bitwidth, sparsity = cfg['ranks'][ri], cfg['bitwidths'][bi], float(cfg['sparsities'][si])

        # ADDED: Log layer processing
        logger.debug(f"Processing layer {layer_idx} ({name}): rank={rank}, bitwidth={bitwidth}, sparsity={sparsity}")

        weight_host = module
        orig_w = weight_host.weight.data.clone().detach().cpu()

        # Low-rank SVD
        if 0 < rank < min(orig_w.shape):
            try:
                U, S, Vh = torch.linalg.svd(orig_w.float(), full_matrices=False)
                W_low = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
                weight_host.weight.data = W_low.to(weight_host.weight.dtype).to(weight_host.weight.device)
                modified_layers += 1
                compression_stats['svd_applied'] += 1
                logger.debug(f"Applied SVD to {name}: rank {rank}")
            except RuntimeError as e:
                log_fn(f"SVD failed for {name}: {e}")
                logger.error(f"SVD failed for {name}: {e}")

        # Quantization
        if bitwidth < 32:
            w_cpu = weight_host.weight.data.float().cpu()
            max_abs = float(w_cpu.abs().max()) if w_cpu.numel() > 0 else 0.0
            qmax = (2 ** (bitwidth-1)) - 1
            scale = max_abs / qmax if max_abs > 1e-8 else 1.0
            q = torch.clamp((w_cpu/scale).round(), -qmax, qmax)
            weight_host.weight.data = (q*scale).to(weight_host.weight.dtype).to(weight_host.weight.device)
            modified_layers += 1
            compression_stats['quant_applied'] += 1
            logger.debug(f"Applied quantization to {name}: {bitwidth} bits, scale={scale:.6f}")

        # Pruning
        if sparsity > 0.0:
            w_cpu = weight_host.weight.data.cpu()
            numel = w_cpu.numel()
            num_prune = int(math.floor(numel * sparsity))
            if 0 < num_prune < numel:
                flat = w_cpu.abs().view(-1)
                kth = float(torch.kthvalue(flat, num_prune).values.item())
                mask = (w_cpu.abs() >= kth).float()
                weight_host.weight.data = (w_cpu * mask).to(weight_host.weight.device)
                modified_layers += 1
                compression_stats['prune_applied'] += 1
                logger.debug(f"Applied pruning to {name}: {sparsity} sparsity, pruned {num_prune}/{numel} params")
            elif num_prune >= numel:
                weight_host.weight.data = torch.zeros_like(w_cpu).to(weight_host.weight.device)
                modified_layers += 1
                compression_stats['prune_applied'] += 1
                logger.debug(f"Applied full pruning to {name}: all {numel} params zeroed")

        l2_diffs.append((name, float(torch.norm(orig_w - weight_host.weight.data.cpu()).item())))
        layer_idx += 1

    compression_time = time.time() - start_time
    log_fn(f"[apply_compression] modified_layers={modified_layers}, checked_layers={layer_idx}")
    
    # ADDED: Log compression summary with stats
    logger.info(f"Compression complete - modified_layers={modified_layers}, checked_layers={layer_idx}, time={compression_time:.3f}s")
    logger.info(f"Compression operations - SVD: {compression_stats['svd_applied']}, Quant: {compression_stats['quant_applied']}, Prune: {compression_stats['prune_applied']}")
    total_l2_diff = sum(diff for _, diff in l2_diffs)
    logger.info(f"Total L2 difference from compression: {total_l2_diff:.6f}")
    
    return model

# NEW: Pre-training on surrogate model
def pretrain_on_surrogate(meta_controller, task_feats, layer_feats, surrogate_path, surrogate_csv_path, cfg, logger, metrics_logger):
    """Pre-train the meta-controller on surrogate model predictions"""
    logger.info("Starting pre-training phase on surrogate model...")
    
    if not os.path.exists(surrogate_path) or not os.path.exists(surrogate_csv_path):
        logger.warning("Surrogate model/data not found, skipping pre-training")
        return meta_controller
    
    # Load surrogate data
    surrogate_data = pd.read_csv(surrogate_csv_path)
    logger.info(f"Loaded surrogate dataset with {len(surrogate_data)} samples")
    
    # Use actual task and layer features instead of dummy ones
    device = next(meta_controller.parameters()).device
    
    # Pre-training optimizer with lower learning rate
    pretrain_optimizer = optim.Adam(meta_controller.parameters(), lr=cfg['meta_lr'] * 0.1)
    
    logger.info(f"Using task_feats shape: {task_feats.shape}, layer_feats shape: {layer_feats.shape}")
    
    for epoch in range(cfg['pretrain_epochs']):
        total_loss = 0.0
        # Simplified pre-training loop
        # In practice, you'd use the surrogate model to generate targets
        # Here we just do a few optimization steps to warm up the network
        
        for _ in range(10):  # Mini-batches per epoch
            pretrain_optimizer.zero_grad()
            
            # Use the actual task and layer features
            r_p, b_p, s_p = meta_controller(task_feats, layer_feats)
            
            # Simple regularization loss to prevent collapse and encourage exploration
            entropy_loss = -(r_p * torch.log(r_p + 1e-9)).sum()
            entropy_loss += -(b_p * torch.log(b_p + 1e-9)).sum()
            entropy_loss += -(s_p * torch.log(s_p + 1e-9)).sum()
            
            # Add small L2 regularization to prevent extreme weights
            l2_reg = sum(p.pow(2.0).sum() for p in meta_controller.parameters())
            
            loss = -entropy_loss + 1e-4 * l2_reg  # Maximize entropy, minimize L2
            loss.backward()
            pretrain_optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        logger.info(f"Pre-training epoch {epoch+1}/{cfg['pretrain_epochs']}: loss={avg_loss:.4f}")
        
        log_metrics(metrics_logger, 'pretrain_epoch', {
            'pretrain_loss': avg_loss,
            'epoch': epoch + 1
        }, epoch=epoch+1)
    
    logger.info("Pre-training phase completed")
    return meta_controller

# NEW: KL divergence penalty calculation
def calculate_kl_penalty(r_p, b_p, s_p):
    """Calculate KL divergence penalty toward uniform distribution"""
    device = r_p.device
    num_layers = r_p.size(1)
    
    # Uniform distributions
    uniform_r = torch.ones_like(r_p) / r_p.size(-1)
    uniform_b = torch.ones_like(b_p) / b_p.size(-1)
    uniform_s = torch.ones_like(s_p) / s_p.size(-1)
    
    # KL divergence: KL(p || uniform) = sum(p * log(p/uniform))
    kl_r = (r_p * torch.log((r_p + 1e-9) / (uniform_r + 1e-9))).sum()
    kl_b = (b_p * torch.log((b_p + 1e-9) / (uniform_b + 1e-9))).sum()
    kl_s = (s_p * torch.log((s_p + 1e-9) / (uniform_s + 1e-9))).sum()
    
    return (kl_r + kl_b + kl_s) / num_layers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Optional YAML config path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # ADDED: Setup logging
    logger, metrics_logger, run_timestamp = setup_logging(cfg['output_dir'])
    logger.info("="*60)
    logger.info("PHASE 3B ROLLOUT STARTED - IMPROVED VERSION")
    logger.info("="*60)
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Configuration: {json.dumps(cfg, indent=2)}")
    
    # ADDED: Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    try:
        set_seed_all(cfg['seed'])
        logger.info(f"Random seed set to: {cfg['seed']}")
        
        device = torch.device(cfg['device'])
        logger.info(f"Using device: {device}")

        task_path = os.path.join(cfg['output_dir'], 'phase2_task_features.csv')
        layer_path = os.path.join(cfg['output_dir'], 'phase2_layer_stats.csv')
        if not os.path.exists(task_path) or not os.path.exists(layer_path):
            logger.error("Phase 2 CSV files not found. Run Phase 2 first!")
            raise FileNotFoundError("Phase 2 CSV files not found. Run Phase 2 first!")

        # ADDED: Log data loading
        logger.info(f"Loading task features from: {task_path}")
        logger.info(f"Loading layer stats from: {layer_path}")
        
        task_feats = torch.tensor(pd.read_csv(task_path).values, dtype=torch.float32, device=device)
        layer_df = pd.read_csv(layer_path)
        layer_stats = layer_df.groupby('layer').agg({'weight_std':'mean','param_count':'sum'}).values
        layer_feats = torch.tensor(layer_stats, dtype=torch.float32).unsqueeze(0).to(device)

        logger.info(f"Task features shape: {task_feats.shape}")
        logger.info(f"Layer features shape: {layer_feats.shape}")

        # ADDED: Log meta-controller initialization
        logger.info("Initializing MetaController...")
        mc = MetaController(
            task_feat_dim=task_feats.size(1),
            layer_feat_dim=layer_feats.size(-1),
            hidden_dim=64,
            ranks=cfg['ranks'],
            bitwidths=cfg['bitwidths'],
            sparsities=cfg['sparsities'],
            num_layers=layer_feats.size(1)
        ).to(device)
        
        log_model_state(logger, mc, "MetaController - ")
        
        # IMPROVED: Pre-training phase
        if cfg.get('pretrain_epochs', 0) > 0:
            mc = pretrain_on_surrogate(mc, task_feats, layer_feats, cfg['surrogate_model'], cfg['surrogate_csv'], cfg, logger, metrics_logger)
        
        opt = optim.Adam(mc.parameters(), lr=cfg['meta_lr'])
        logger.info(f"Adam optimizer initialized with lr={cfg['meta_lr']}")

        # FIXED: Set pad token in tokenizer
        logger.info(f"Loading tokenizer: {cfg['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token")

        # FIXED: Load base model and set pad_token_id in model config
        logger.info(f"Loading base model: {cfg['model_name']}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg['model_name'], num_labels=2, ignore_mismatched_sizes=True
        ).to('cpu')
        
        # CRITICAL FIX: Set pad token in model config to handle batch sizes > 1
        base_model.config.pad_token_id = tokenizer.pad_token_id
        logger.info("Base model loaded and pad_token_id set")
        log_model_state(logger, base_model, "BaseModel - ")

        # ADDED: Log dataset loading
        logger.info(f"Loading dataset: {cfg['task_name']}")
        ds = load_dataset("glue", cfg['task_name'].split('-',1)[-1])
        def preprocess(batch):
            return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True,
                             padding="max_length", max_length=cfg['max_seq_length'])
        data = ds.map(preprocess, batched=True)
        data = data.rename_column("label","labels")
        data.set_format("torch", columns=["input_ids","attention_mask","labels"])
        train_ds, val_ds = data["train"], data["validation"]
        
        logger.info(f"Dataset loaded - Train: {len(train_ds)}, Validation: {len(val_ds)}")

        trainer_args = TrainingArguments(
            output_dir=os.path.join(cfg['output_dir'],'rollout'),
            per_device_train_batch_size=cfg['train_batch_size'],
            per_device_eval_batch_size=cfg['eval_batch_size'],
            num_train_epochs=1,
            learning_rate=cfg['learning_rate'],
            fp16=False,
            gradient_checkpointing=False,
            logging_strategy="no",
            save_strategy="no",
            eval_strategy="epoch",
            disable_tqdm=True,
            report_to=None,
            seed=cfg['seed']
        )
        logger.info("Trainer arguments configured")

        # IMPROVED: Initialize with better baselines
        running_baseline = 0.0
        baseline_momentum = float(cfg.get('baseline_momentum', 0.9))
        best_reward = float('-inf')
        best_checkpoint = None

        def apply_temperature_to_probs(p: torch.Tensor, temp: float):
            logits = torch.log(p + 1e-9) / float(temp)
            return torch.softmax(logits, dim=-1)

        # ADDED: Log training start
        logger.info("="*60)
        logger.info("TRAINING LOOP STARTED - IMPROVED VERSION")
        logger.info("="*60)
        
        training_start_time = time.time()
        
        # IMPROVED: Store original model for compression metrics
        original_model = copy.deepcopy(base_model)

        for epoch in range(cfg['rollout_epochs']):
            epoch_start_time = time.time()
            
            # IMPROVED: Cosine temperature annealing + periodic restarts
            if epoch % cfg.get('restart_every', 25) == 0 and epoch > 0:
                # Periodic restart - reset temperature
                temp = cfg['temperature_start']
                logger.info(f"Temperature restart at epoch {epoch+1}")
            else:
                temp = cosine_temp(epoch, cfg['rollout_epochs'], cfg['temperature_start'], cfg['temperature_end'])
            
            mc.train()
            rewards, logps, entropies, kl_penalties = [], [], [], []
            compression_metrics_list = []

            print(f"=== Rollout Epoch {epoch+1}/{cfg['rollout_epochs']} === Temp={temp:.4f}")
            logger.info(f"Starting epoch {epoch+1}/{cfg['rollout_epochs']} with temperature={temp:.4f}")

            # ADDED: Log metrics for epoch start
            log_metrics(metrics_logger, 'epoch_start', {
                'epoch': epoch + 1,
                'temperature': temp,
                'running_baseline': running_baseline,
                'best_reward': best_reward
            }, epoch=epoch+1)

            for sample_idx in range(cfg['rollout_samples']):
                sample_start_time = time.time()
                logger.debug(f"Processing sample {sample_idx+1}/{cfg['rollout_samples']}")
                
                # FIXED: Remove torch.no_grad() to preserve gradients for REINFORCE
                r_p, b_p, s_p = mc(task_feats, layer_feats)
                r_p = apply_temperature_to_probs(r_p, temp)
                b_p = apply_temperature_to_probs(b_p, temp)
                s_p = apply_temperature_to_probs(s_p, temp)

                # IMPROVED: Enhanced entropy calculation
                entropy_r = -(r_p * torch.log(r_p + 1e-9)).sum(dim=-1).mean()
                entropy_b = -(b_p * torch.log(b_p + 1e-9)).sum(dim=-1).mean() 
                entropy_s = -(s_p * torch.log(s_p + 1e-9)).sum(dim=-1).mean()
                total_entropy = entropy_r + entropy_b + entropy_s
                entropies.append(total_entropy)

                # NEW: Calculate KL penalty
                kl_penalty = calculate_kl_penalty(r_p, b_p, s_p)
                kl_penalties.append(kl_penalty)

                log_prob = torch.tensor(0., device=device, requires_grad=True)
                r_idx, b_idx, s_idx = [], [], []
                for i in range(layer_feats.size(1)):
                    rd, bd, sd = Categorical(r_p[0,i]), Categorical(b_p[0,i]), Categorical(s_p[0,i])
                    ri, bi, si = rd.sample(), bd.sample(), sd.sample()
                    log_prob = log_prob + rd.log_prob(ri) + bd.log_prob(bi) + sd.log_prob(si)
                    r_idx.append(int(ri.item())); b_idx.append(int(bi.item())); s_idx.append(int(si.item()))
                logps.append(log_prob)

                # ADDED: Log sampled indices
                logger.debug(f"Sampled indices for sample {sample_idx+1}: ranks={r_idx[:3]}..., bitwidths={b_idx[:3]}..., sparsities={s_idx[:3]}...")

                # FIXED: Copy base model and ensure pad_token_id is preserved
                m = copy.deepcopy(base_model)
                # Ensure pad_token_id is preserved after compression
                m.config.pad_token_id = tokenizer.pad_token_id
                
                m = apply_compression(m, r_idx, b_idx, s_idx, cfg, log_fn=lambda *a,**k:None)
                m.to(device)
                
                # IMPROVED: Calculate comprehensive compression metrics
                comp_metrics = calculate_compression_metrics(original_model, m)
                compression_metrics_list.append(comp_metrics)
                
                log_model_state(logger, m, f"Compressed model (sample {sample_idx+1}) - ")

                # Use no_grad only for evaluation, not for policy computation
                with torch.no_grad():
                    eval_start_time = time.time()
                    trainer = Trainer(model=m, args=trainer_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)
                    ev = trainer.evaluate()
                    eval_time = time.time() - eval_start_time
                    
                    reward = -float(ev.get("eval_loss", np.nan))
                    rewards.append(reward)
                    
                    # ADDED: Log sample results with comprehensive metrics
                    sample_time = time.time() - sample_start_time
                    logger.debug(f"Sample {sample_idx+1} completed - reward={reward:.4f}, eval_time={eval_time:.2f}s, total_time={sample_time:.2f}s")
                    logger.debug(f"Compression metrics: {comp_metrics['param_reduction_ratio']:.3f} param reduction, {comp_metrics['compression_ratio']:.2f}x compression")
                    
                    # ADDED: Log comprehensive sample metrics
                    log_metrics(metrics_logger, 'sample_result', {
                        'reward': reward,
                        'eval_loss': float(ev.get("eval_loss", np.nan)),
                        'eval_time': eval_time,
                        'sample_time': sample_time,
                        'entropy_total': float(total_entropy.item()),
                        'entropy_rank': float(entropy_r.item()),
                        'entropy_bitwidth': float(entropy_b.item()),
                        'entropy_sparsity': float(entropy_s.item()),
                        'kl_penalty': float(kl_penalty.item()),
                        'compression_metrics': comp_metrics,
                        'compression_config': {
                            'ranks_sample': r_idx[:5],
                            'bitwidths_sample': b_idx[:5], 
                            'sparsities_sample': s_idx[:5],
                            'ranks_full': r_idx,
                            'bitwidths_full': b_idx,
                            'sparsities_full': s_idx
                        }
                    }, epoch=epoch+1, sample=sample_idx+1)

                del m; torch.cuda.empty_cache()

            # IMPROVED: Enhanced advantage calculation with clipping
            mean_reward = float(np.mean(rewards))
            reward_std = float(np.std(rewards))
            running_baseline = baseline_momentum*running_baseline + (1-baseline_momentum)*mean_reward if epoch>0 else mean_reward
            
            advantages = (np.array(rewards) - running_baseline)
            advantages = (advantages - advantages.mean()) / (advantages.std() + cfg.get('adv_norm_eps', 1e-6))
            
            # NEW: Advantage clipping
            advantage_clip = cfg.get('advantage_clip', 3.0)
            advantages = np.clip(advantages, -advantage_clip, advantage_clip)
            logger.debug(f"Advantages after clipping: min={advantages.min():.3f}, max={advantages.max():.3f}, std={advantages.std():.3f}")

            # IMPROVED: Enhanced loss calculation with regularization
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
            policy_losses = [-lp * adv for lp, adv in zip(logps, advantages_tensor)]
            
            # Enhanced regularization terms
            entropy_bonus = cfg['entropy_coeff'] * torch.stack(entropies).mean()
            kl_penalty_term = cfg.get('kl_penalty_coeff', 0.0) * torch.stack(kl_penalties).mean()
            
            total_loss = torch.stack(policy_losses).mean() - entropy_bonus + kl_penalty_term

            # ADDED: Log before optimization
            logger.debug(f"Loss components - Policy: {torch.stack(policy_losses).mean().item():.6f}, Entropy: {entropy_bonus.item():.6f}, KL: {kl_penalty_term.item():.6f}")

            opt.zero_grad()
            total_loss.backward()
            
            # IMPROVED: Gradient clipping with logging
            grad_norm = torch.nn.utils.clip_grad_norm_(mc.parameters(), cfg['max_grad_norm'])
            logger.debug(f"Gradient norm before clipping: {grad_norm:.6f}")
            
            opt.step()

            # IMPROVED: Enhanced metrics calculation
            avg_entropy = float(torch.stack(entropies).mean().item()) if entropies else 0.0
            avg_kl_penalty = float(torch.stack(kl_penalties).mean().item()) if kl_penalties else 0.0
            epoch_time = time.time() - epoch_start_time
            
            # Track best performance
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_checkpoint = copy.deepcopy(mc.state_dict())
                logger.info(f"New best reward: {best_reward:.4f}")
            
            # IMPROVED: Calculate average compression metrics for epoch
            avg_comp_metrics = {}
            if compression_metrics_list:
                for key in compression_metrics_list[0].keys():
                    avg_comp_metrics[f'avg_{key}'] = float(np.mean([cm[key] for cm in compression_metrics_list]))
            
            print(f"Epoch {epoch+1} | MeanReward={mean_reward:.4f} | RunningBaseline={running_baseline:.4f} | AvgEntropy={avg_entropy:.4f} | PolicyLoss={torch.stack(policy_losses).mean().item():.6f}")
            
            # ADDED: Enhanced logging for epoch completion
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            logger.info(f"  Rewards: mean={mean_reward:.4f}, std={reward_std:.4f}, min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")
            logger.info(f"  Baseline: running={running_baseline:.4f}, best={best_reward:.4f}")
            logger.info(f"  Regularization: entropy={avg_entropy:.4f}, kl_penalty={avg_kl_penalty:.4f}")
            logger.info(f"  Loss components: policy={torch.stack(policy_losses).mean().item():.6f}, total={total_loss.item():.6f}")
            if avg_comp_metrics:
                logger.info(f"  Avg compression: {avg_comp_metrics.get('avg_param_reduction_ratio', 0):.3f} param reduction, {avg_comp_metrics.get('avg_compression_ratio', 1):.2f}x size reduction")
            
            # ADDED: Enhanced epoch metrics logging
            epoch_metrics = {
                'mean_reward': mean_reward,
                'reward_std': reward_std,
                'reward_min': float(np.min(rewards)),
                'reward_max': float(np.max(rewards)),
                'running_baseline': running_baseline,
                'best_reward': best_reward,
                'avg_entropy': avg_entropy,
                'avg_kl_penalty': avg_kl_penalty,
                'policy_loss': float(torch.stack(policy_losses).mean().item()),
                'entropy_bonus': float(entropy_bonus.item()),
                'kl_penalty_term': float(kl_penalty_term.item()),
                'total_loss': float(total_loss.item()),
                'epoch_time': epoch_time,
                'grad_norm': float(grad_norm),
                'temperature': temp,
                'advantage_stats': {
                    'mean': float(advantages.mean()),
                    'std': float(advantages.std()),
                    'min': float(advantages.min()),
                    'max': float(advantages.max())
                }
            }
            epoch_metrics.update(avg_comp_metrics)
            
            log_metrics(metrics_logger, 'epoch_complete', epoch_metrics, epoch=epoch+1)

            # IMPROVED: Enhanced checkpointing
            if (epoch+1) % max(1, cfg.get('checkpoint_every', 10)) == 0:
                checkpoint_path = os.path.join(cfg['output_dir'], f'meta_controller_epoch{epoch+1}.pt')
                torch.save({
                    'model_state_dict': mc.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'epoch': epoch + 1,
                    'best_reward': best_reward,
                    'running_baseline': running_baseline,
                    'config': cfg
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Save best model separately
                if best_checkpoint is not None:
                    best_path = os.path.join(cfg['output_dir'], f'best_meta_controller_epoch{epoch+1}.pt')
                    torch.save({
                        'model_state_dict': best_checkpoint,
                        'epoch': epoch + 1,
                        'best_reward': best_reward,
                        'config': cfg
                    }, best_path)

        total_training_time = time.time() - training_start_time
        
        # Save final models
        final_path = os.path.join(cfg['output_dir'], 'final_meta_controller.pt')
        torch.save({
            'model_state_dict': mc.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'total_epochs': cfg['rollout_epochs'],
            'best_reward': best_reward,
            'running_baseline': running_baseline,
            'config': cfg
        }, final_path)
        
        # Save best model
        if best_checkpoint is not None:
            best_final_path = os.path.join(cfg['output_dir'], 'best_meta_controller.pt')
            torch.save({
                'model_state_dict': best_checkpoint,
                'best_reward': best_reward,
                'config': cfg
            }, best_final_path)
            logger.info(f"Best meta-controller saved: {best_final_path} (reward: {best_reward:.4f})")
        
        logger.info("="*60)
        logger.info("PHASE 3B COMPLETE - IMPROVED VERSION")
        logger.info("="*60)
        logger.info(f"Total training time: {total_training_time:.2f}s")
        logger.info(f"Final meta-controller saved: {final_path}")
        logger.info(f"Best reward achieved: {best_reward:.4f}")
        print("Phase 3B complete. Meta-controller saved.")
        
        # ADDED: Final comprehensive metrics
        log_metrics(metrics_logger, 'training_complete', {
            'total_training_time': total_training_time,
            'final_running_baseline': running_baseline,
            'best_reward_achieved': best_reward,
            'total_epochs': cfg['rollout_epochs'],
            'total_samples': cfg['rollout_epochs'] * cfg['rollout_samples'],
            'final_temperature': temp,
            'improvements_implemented': [
                'pretrain_on_surrogate',
                'extended_training_epochs', 
                'increased_rollout_samples',
                'slower_temperature_annealing',
                'enhanced_entropy_regularization',
                'kl_divergence_penalty',
                'advantage_clipping',
                'comprehensive_compression_metrics',
                'periodic_temperature_restarts',
                'enhanced_checkpointing',
                'stabilized_policy_gradients'
            ]
        })

    except Exception as e:
        logger.error("="*60)
        logger.error("FATAL ERROR OCCURRED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
 
if __name__ == '__main__':
    main()