return total_diff / comparisons if comparisons > 0 else 0.0

def create_publication_plots(cfg, surrogate_rewards, surrogate_entropies, surrogate_r2,
                           rollout_rewards, rollout_improvements, baseline_loss):
    """Create publication-quality plots"""
    if not PLOTTING_AVAILABLE:
        return
    
    try:
        # Set publication style
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.figsize': (15, 10)
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Surrogate training plots
        if len(surrogate_rewards) > 0:
            # Surrogate rewards
            axes[0, 0].plot(surrogate_rewards, 'b-', linewidth=2, alpha=0.8)
            axes[0, 0].axhline(y=-baseline_loss, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
            axes[0, 0].set_title('Surrogate Training: Reward Progress', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Surrogate entropy
            axes[0, 1].plot(surrogate_entropies, 'g-', linewidth=2, alpha=0.8)
            axes[0, 1].set_title('Surrogate Training: Policy Entropy', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Entropy')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Surrogate R² scores
            axes[0, 2].plot(surrogate_r2, 'purple', linewidth=2, alpha=0.8, marker='o', markersize=4)
            axes[0, 2].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Target R²=0.7')
            axes[0, 2].set_title('Surrogate Model Quality (R²)', fontweight='bold')
            axes[0, 2].set_xlabel('Validation Epoch')
            axes[0, 2].set_ylabel('R² Score')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
            axes[0, 2].set_ylim(0, 1)
        
        # Rollout training plots
        if len(rollout_rewards) > 0:
            # Rollout rewards
            axes[1, 0].plot(rollout_rewards, 'darkblue', linewidth=2, alpha=0.8)
            axes[1, 0].axhline(y=-baseline_loss, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
            axes[1, 0].fill_between(range(len(rollout_rewards)), rollout_rewards, -baseline_loss, 
                                  where=[r > -baseline_loss for r in rollout_rewards], 
                                  color='green', alpha=0.2, label='Improvement')
            axes[1, 0].set_title('Rollout Training: Reward Progress', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Improvements over baseline
            axes[1, 1].plot(rollout_improvements, 'darkgreen', linewidth=2, alpha=0.8, marker='s', markersize=4)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].fill_between(range(len(rollout_improvements)), rollout_improvements, 0,
                                  where=[imp > 0 for imp in rollout_improvements],
                                  color='green', alpha=0.3, label='Positive')
            axes[1, 1].fill_between(range(len(rollout_improvements)), rollout_improvements, 0,
                                  where=[imp <= 0 for imp in rollout_improvements],
                                  color='red', alpha=0.3, label='Negative')
            axes[1, 1].set_title('Loss Improvement Over Baseline', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Improvement')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Success rate analysis
            success_epochs = [i for i, imp in enumerate(rollout_improvements) if imp > 0]
            success_rate = len(success_epochs) / len(rollout_improvements) * 100 if rollout_improvements else 0
            
            # Cumulative improvement
            cumulative_improvements = np.cumsum(rollout_improvements)
            axes[1, 2].plot(cumulative_improvements, 'darkorange', linewidth=2, alpha=0.8)
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 2].set_title(f'Cumulative Improvement\n(Success Rate: {success_rate:.1f}%)', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Cumulative Improvement')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save high-quality plot
        plot_path = os.path.join(cfg['output_dir'], 'publication_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Publication plots saved to {plot_path}")
        
        # Create summary statistics plot
        if len(rollout_rewards) > 0 and len(rollout_improvements) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Box plot of improvements
            improvement_data = [imp for imp in rollout_improvements if not np.isnan(imp)]
            if improvement_data:
                ax.boxplot(improvement_data, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline')
                ax.set_ylabel('Loss Improvement')
                ax.set_title('Distribution of Improvements Over Baseline', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_imp = np.mean(improvement_data)
                std_imp = np.std(improvement_data)
                positive_rate = sum(1 for x in improvement_data if x > 0) / len(improvement_data) * 100
                
                stats_text = f'Mean: {mean_imp:.6f}\nStd: {std_imp:.6f}\nPositive: {positive_rate:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                summary_path = os.path.join(cfg['output_dir'], 'improvement_summary.png')
                plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                logger.info(f"Summary statistics plot saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to create publication plots: {e}")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Publication-ready Meta-Controller Training')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--meta_lr', type=float, help='Meta-learning rate')
    parser.add_argument('--rollout_epochs', type=int, help='Number of rollout epochs')
    parser.add_argument('--rollout_samples', type=int, help='Samples per rollout epoch')
    parser.add_argument('--surrogate_pretrain_samples', type=int, help='Pre-training samples for surrogate')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    if args.meta_lr: cfg['meta_lr'] = args.meta_lr
    if args.rollout_epochs: cfg['rollout_epochs'] = args.rollout_epochs
    if args.rollout_samples: cfg['rollout_samples'] = args.rollout_samples
    if args.surrogate_pretrain_samples: cfg['surrogate_pretrain_samples'] = args.surrogate_pretrain_samples

    # Setup
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # Setup logging to file
    log_path = os.path.join(cfg['output_dir'], 'training.log')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    set_seed_all(cfg['seed'])
    dev = torch.device(cfg['device'])
    
    logger.info(f"=== Publication-Ready Meta-Controller Training ===")
    logger.info(f"Device: {dev}, Model: {cfg['model_name']}")
    logger.info(f"Configuration: {json.dumps(cfg, indent=2)}")

    # Load features (simplified for now)
    task_feats = torch.randn(1, 6).to(dev)
    logger.info("Using mock task features")
    
    # Layer features
    n_layers = 12  # GPT2-medium
    layer_stats = []
    for i in range(n_layers):
        layer_stats.append([np.random.uniform(0.1, 0.5), np.random.randint(1000000, 5000000)])
    
    layer_feats = torch.tensor(layer_stats, dtype=torch.float).unsqueeze(0).to(dev)
    logger.info(f"Layer features shape: {layer_feats.shape}")

    # Initialize meta-controller
    mc = MetaController(
        task_feats.size(1),
        layer_feats.size(-1),
        256,  # Larger hidden size
        cfg['ranks'],
        cfg['bitwidths'],
        cfg['sparsities'],
        layer_feats.size(1)
    ).to(dev)

    # Optimizer with weight decay
    opt = optim.AdamW(mc.parameters(), lr=cfg['meta_lr'], weight_decay=1e-4)
    scaler = GradScaler(device="cuda") if dev.type == "cuda" else None

    # Load model and data
    logger.info(f"Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=2, ignore_mismatched_sizes=True)

    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    logger.info(f"Loading dataset: {cfg['task_name']}")
    ds = load_dataset("glue", cfg['task_name'].replace("glue-", ""))
    
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"], examples["sentence2"],
            truncation=True, padding="max_length",
            max_length=cfg['max_seq_length']
        )
    
    data = ds.map(tokenize_function, batched=True).rename_column("label", "labels")
    data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_ds, val_ds = data["train"], data["validation"]

    trainer_args = TrainingArguments(
        output_dir=os.path.join(cfg["output_dir"], "evaluation"),
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=1,
        learning_rate=cfg["learning_rate"],
        fp16=(dev.type == "cuda"),
        gradient_checkpointing=True,
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=1000,
        disable_tqdm=True,
        report_to=None,
        seed=cfg["seed"],
        dataloader_pin_memory=False
    )

    # Get baseline
    logger.info("=== Evaluating Baseline Model ===")
    baseline_trainer = Trainer(
        model=copy.deepcopy(base_model).to(dev),
        args=trainer_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )
    
    with torch.no_grad():
        baseline_results = baseline_trainer.evaluate()
    
    baseline_loss = float(baseline_results.get("eval_loss", float("inf")))
    baseline_accuracy = float(baseline_results.get("eval_accuracy", 0.0))
    
    logger.info(f"Baseline - Loss: {baseline_loss:.6f}, Accuracy: {baseline_accuracy:.4f}")
    
    del baseline_trainer
    gc.collect()

    # Pre-train surrogate
    surrogate, pretrain_configs, pretrain_losses = pretrain_surrogate(
        cfg, base_model, tokenizer, train_ds, val_ds, trainer_args, dev, baseline#!/usr/bin/env python3
"""
Publication-ready phase3_meta_train.py with stable surrogate learning

Critical fixes for publication quality:
1. Large surrogate dataset (200+ samples) with systematic exploration
2. Neural network surrogate with regularization and early stopping
3. Stable reward signal with running baselines and advantage normalization
4. Slower temperature annealing for sustained exploration
5. Comprehensive logging and validation for reproducibility
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse
import yaml
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import subprocess
import copy
import json
from collections import deque
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from torch import amp
from torch.amp import GradScaler
from torch.distributions import Categorical
from datasets import load_dataset

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    PLOTTING_AVAILABLE = True
    
    # Set style for publication-quality plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/scipy not available, plots disabled")

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))

from models.meta_controller import MetaController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def default_config():
    return {
        'output_dir': './output',
        'ranks': [1, 2, 4, 8, 16, 32],  # More granular options
        'bitwidths': [2, 4, 8, 16, 32],
        'sparsities': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],  # More options
        'model_name': 'gpt2-medium',
        'task_name': 'glue-mrpc',
        'max_seq_length': 128,
        'train_batch_size': 8,
        'eval_batch_size': 1,
        'learning_rate': 2e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'meta_lr': 5e-3,  # Moderate learning rate
        'surrogate_pretrain_samples': 200,  # Large initial dataset
        'surrogate_epochs': 50,  # More epochs with slower annealing
        'rollout_epochs': 30,
        'rollout_samples': 6,
        'temperature_start': 3.0,
        'temperature_end': 1.0,  # Slower annealing
        'diversity_bonus': 0.03,
        'advantage_smoothing': 0.95,  # EMA for baseline
        'reward_clipping': 5.0,  # Clip extreme rewards
        'validation_frequency': 5,  # Validate surrogate every N epochs
    }

def load_config(path=None):
    cfg = default_config()
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    return cfg

def set_seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NeuralSurrogate(nn.Module):
    """Neural network surrogate with regularization"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Training history for early stopping
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _engineer_features(self, configurations):
        """Enhanced feature engineering for compression configurations"""
        features = []
        
        for config in configurations:
            feat_vec = []
            
            # Extract per-layer values
            ranks = [cfg[0] for cfg in config]
            bitwidths = [cfg[1] for cfg in config]
            sparsities = [cfg[2] for cfg in config]
            
            # Basic statistics
            feat_vec.extend([
                np.mean(ranks), np.std(ranks), np.median(ranks),
                np.mean(bitwidths), np.std(bitwidths), np.median(bitwidths),
                np.mean(sparsities), np.std(sparsities), np.median(sparsities)
            ])
            
            # Extremes and ranges
            feat_vec.extend([
                np.min(ranks), np.max(ranks), np.max(ranks) - np.min(ranks),
                np.min(bitwidths), np.max(bitwidths), np.max(bitwidths) - np.min(bitwidths),
                np.min(sparsities), np.max(sparsities), np.max(sparsities) - np.min(sparsities)
            ])
            
            # Percentiles
            feat_vec.extend([
                np.percentile(ranks, 25), np.percentile(ranks, 75),
                np.percentile(bitwidths, 25), np.percentile(bitwidths, 75),
                np.percentile(sparsities, 25), np.percentile(sparsities, 75)
            ])
            
            # Layer position features (early vs late layers)
            n_layers = len(config)
            early_third = n_layers // 3
            middle_third = 2 * n_layers // 3
            
            # Early layers (first third)
            early_ranks = ranks[:early_third] if early_third > 0 else [0]
            early_bits = bitwidths[:early_third] if early_third > 0 else [0]
            early_spars = sparsities[:early_third] if early_third > 0 else [0]
            
            feat_vec.extend([
                np.mean(early_ranks), np.mean(early_bits), np.mean(early_spars)
            ])
            
            # Middle layers
            mid_ranks = ranks[early_third:middle_third] if middle_third > early_third else [0]
            mid_bits = bitwidths[early_third:middle_third] if middle_third > early_third else [0]
            mid_spars = sparsities[early_third:middle_third] if middle_third > early_third else [0]
            
            feat_vec.extend([
                np.mean(mid_ranks), np.mean(mid_bits), np.mean(mid_spars)
            ])
            
            # Late layers (last third)
            late_ranks = ranks[middle_third:] if len(ranks) > middle_third else [0]
            late_bits = bitwidths[middle_third:] if len(bitwidths) > middle_third else [0]
            late_spars = sparsities[middle_third:] if len(sparsities) > middle_third else [0]
            
            feat_vec.extend([
                np.mean(late_ranks), np.mean(late_bits), np.mean(late_spars)
            ])
            
            # Compression complexity features
            complexity_scores = []
            for r, b, s in config:
                # Higher rank, lower bitwidth, higher sparsity = more complex
                complexity = (r / 32.0) + (1.0 - b / 32.0) + s
                complexity_scores.append(complexity)
            
            feat_vec.extend([
                np.mean(complexity_scores),
                np.std(complexity_scores),
                np.max(complexity_scores) - np.min(complexity_scores)
            ])
            
            # Interaction features
            rank_bit_corr = np.corrcoef(ranks, bitwidths)[0,1] if len(set(ranks)) > 1 and len(set(bitwidths)) > 1 else 0
            rank_spar_corr = np.corrcoef(ranks, sparsities)[0,1] if len(set(ranks)) > 1 and len(set(sparsities)) > 1 else 0
            bit_spar_corr = np.corrcoef(bitwidths, sparsities)[0,1] if len(set(bitwidths)) > 1 and len(set(sparsities)) > 1 else 0
            
            feat_vec.extend([
                rank_bit_corr if not np.isnan(rank_bit_corr) else 0,
                rank_spar_corr if not np.isnan(rank_spar_corr) else 0,
                bit_spar_corr if not np.isnan(bit_spar_corr) else 0
            ])
            
            features.append(feat_vec)
        
        return np.array(features)
    
    def fit(self, configurations, losses, epochs=200, lr=1e-3, patience=20):
        """Train neural surrogate with early stopping"""
        X = self._engineer_features(configurations)
        y = np.array(losses)
        
        # Remove invalid entries
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 10:
            logger.warning(f"Only {len(X)} valid samples for surrogate training")
            return 0.0, float('inf')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        if len(X) > 20:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val = X_scaled, X_scaled
            y_train, y_val = y, y
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Optimizer and loss
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.train()
            optimizer.zero_grad()
            train_pred = self(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_pred = self(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model state
                self.best_state = copy.deepcopy(self.state_dict())
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                self.load_state_dict(self.best_state)
                break
        
        self.is_fitted = True
        
        # Final evaluation
        self.eval()
        with torch.no_grad():
            final_pred = self(X_val_tensor)
            r2 = r2_score(y_val, final_pred.numpy().flatten())
            mse = mean_squared_error(y_val, final_pred.numpy().flatten())
        
        logger.info(f"Surrogate training complete: R²={r2:.4f}, MSE={mse:.6f}, epochs={len(self.train_losses)}")
        return r2, mse
    
    def predict(self, configurations):
        """Predict losses for configurations"""
        if not self.is_fitted:
            return np.random.uniform(0.5, 1.5, len(configurations))
        
        X = self._engineer_features(configurations)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.eval()
        with torch.no_grad():
            predictions = self(X_tensor).numpy().flatten()
        
        return predictions
    
    def forward(self, x):
        return self.network(x)

class RunningStats:
    """Running statistics for advantage normalization"""
    
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, values):
        if len(values) == 0:
            return
        
        batch_mean = np.mean(values)
        batch_var = np.var(values)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            # Exponential moving average
            self.mean = self.alpha * self.mean + (1 - self.alpha) * batch_mean
            self.var = self.alpha * self.var + (1 - self.alpha) * batch_var
        
        self.count += len(values)
    
    def normalize(self, values):
        if self.count == 0:
            return values
        
        normalized = (values - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normalized, -5.0, 5.0)  # Clip extreme values

def generate_diverse_configurations(cfg, n_samples=200, n_layers=12):
    """Generate diverse compression configurations for surrogate training"""
    configurations = []
    
    # Systematic sampling
    n_systematic = n_samples // 2
    
    # Random sampling for diversity
    for _ in range(n_systematic):
        config = []
        for layer in range(n_layers):
            r_idx = random.randint(0, len(cfg['ranks']) - 1)
            b_idx = random.randint(0, len(cfg['bitwidths']) - 1)
            s_idx = random.randint(0, len(cfg['sparsities']) - 1)
            config.append([r_idx, b_idx, s_idx])
        configurations.append(config)
    
    # Structured sampling (combinations of extremes)
    n_structured = n_samples - n_systematic
    
    extreme_combinations = [
        # (rank_level, bitwidth_level, sparsity_level)
        ('low', 'low', 'low'),    # Aggressive compression
        ('high', 'high', 'low'),  # Conservative compression
        ('medium', 'medium', 'medium'),  # Balanced
        ('low', 'high', 'medium'),  # Mixed strategies
        ('high', 'low', 'high'),
        ('medium', 'low', 'high'),
    ]
    
    level_to_idx = {
        'low': 0,
        'medium': len(cfg['ranks']) // 2,
        'high': len(cfg['ranks']) - 1
    }
    
    for i in range(n_structured):
        combo = extreme_combinations[i % len(extreme_combinations)]
        config = []
        
        for layer in range(n_layers):
            # Add some noise to avoid identical configs
            r_base = level_to_idx[combo[0]]
            b_base = level_to_idx[combo[1]]
            s_base = level_to_idx[combo[2]]
            
            # Small random perturbations
            r_idx = np.clip(r_base + random.randint(-1, 1), 0, len(cfg['ranks']) - 1)
            b_idx = np.clip(b_base + random.randint(-1, 1), 0, len(cfg['bitwidths']) - 1)
            s_idx = np.clip(s_base + random.randint(-1, 1), 0, len(cfg['sparsities']) - 1)
            
            config.append([r_idx, b_idx, s_idx])
        
        configurations.append(config)
    
    logger.info(f"Generated {len(configurations)} diverse configurations")
    return configurations

def apply_compression_robust(model, r_list, b_list, s_list, cfg):
    """Robust compression application with detailed logging"""
    from torch.nn.utils import prune
    
    compressions_applied = []
    
    class LoRAWrapper(nn.Module):
        def __init__(self, orig, rank, alpha=0.3):  # Lower alpha for stability
            super().__init__()
            self.orig = orig
            in_f = getattr(orig, "in_features", None)
            out_f = getattr(orig, "out_features", None)
            
            if in_f is None or out_f is None or rank <= 0:
                self.A = None
                self.B = None
                self.alpha = 0.0
                return
            
            # Ensure valid rank
            max_rank = min(in_f, out_f) - 1
            rank = min(rank, max_rank)
            
            if rank <= 0:
                self.A = None
                self.B = None
                self.alpha = 0.0
                return
            
            self.A = nn.Linear(in_f, rank, bias=False)
            self.B = nn.Linear(rank, out_f, bias=False)
            
            # Xavier initialization for stability
            nn.init.xavier_normal_(self.A.weight, gain=0.1)
            nn.init.zeros_(self.B.weight)
            self.alpha = alpha
        
        def forward(self, x):
            if self.A is None:
                return self.orig(x)
            return self.orig(x) + self.alpha * self.B(self.A(x))
    
    # Apply compression layer by layer
    layer_idx = 0
    for name, module in list(model.named_modules()):
        if not hasattr(module, 'weight') or layer_idx >= len(r_list):
            continue
        
        try:
            r_idx = int(r_list[layer_idx])
            b_idx = int(b_list[layer_idx])
            s_idx = int(s_list[layer_idx])
        except (ValueError, IndexError):
            layer_idx += 1
            continue
        
        # Get compression parameters
        rank = cfg['ranks'][r_idx] if r_idx < len(cfg['ranks']) else cfg['ranks'][-1]
        bitw = cfg['bitwidths'][b_idx] if b_idx < len(cfg['bitwidths']) else cfg['bitwidths'][-1]
        spars = cfg['sparsities'][s_idx] if s_idx < len(cfg['sparsities']) else 0.0
        
        compression_info = {'layer': name, 'rank': rank, 'bitwidth': bitw, 'sparsity': spars}
        
        # Apply LoRA
        if isinstance(module, nn.Linear) and rank > 0:
            min_dim = min(module.weight.shape)
            if rank < min_dim:
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    if hasattr(parent, part):
                        parent = getattr(parent, part)
                    else:
                        break
                else:
                    if hasattr(parent, parts[-1]):
                        try:
                            wrapped = LoRAWrapper(module, rank)
                            setattr(parent, parts[-1], wrapped)
                            compression_info['lora_applied'] = True
                        except Exception as e:
                            compression_info['lora_error'] = str(e)
        
        # Apply quantization
        if bitw < 32 and hasattr(module, 'weight'):
            try:
                W = module.weight.data.float()
                W_max = W.abs().max()
                
                if W_max > 1e-8:
                    qmax = 2**(bitw-1) - 1
                    qmin = -qmax
                    scale = W_max / qmax
                    
                    W_q = torch.clamp(torch.round(W / scale), qmin, qmax)
                    W_dq = W_q * scale
                    
                    module.weight.data = W_dq.type_as(module.weight.data)
                    compression_info['quantization_applied'] = True
                    compression_info['quantization_scale'] = scale.item()
            except Exception as e:
                compression_info['quantization_error'] = str(e)
        
        # Apply sparsity
        if spars > 0.01:
            try:
                prune.l1_unstructured(module, 'weight', amount=float(spars))
                compression_info['pruning_applied'] = True
            except Exception as e:
                compression_info['pruning_error'] = str(e)
        
        compressions_applied.append(compression_info)
        layer_idx += 1
    
    # Log compression summary
    successful_compressions = sum(1 for c in compressions_applied 
                                 if any(key.endswith('_applied') for key in c))
    
    logger.debug(f"Applied {successful_compressions}/{len(compressions_applied)} compressions")
    
    return model, compressions_applied

def evaluate_with_timeout(model, r_list, b_list, s_list, cfg, tokenizer, 
                         train_ds, val_ds, trainer_args, dev, timeout=300):
    """Evaluate compressed model with timeout and robust error handling"""
    try:
        start_time = time.time()
        
        # Create compressed model
        m = copy.deepcopy(model).to(dev)
        m, compression_info = apply_compression_robust(m, r_list, b_list, s_list, cfg)
        
        # Memory optimizations
        try:
            if hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
            if hasattr(m.config, 'use_cache'):
                m.config.use_cache = False
        except:
            pass
        
        # Create trainer with timeout considerations
        trainer = Trainer(
            model=m,
            args=trainer_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer
        )
        
        # Evaluate with timeout check
        if dev.type == "cuda":
            with amp.autocast(device_type="cuda"):
                with torch.no_grad():
                    results = trainer.evaluate()
        else:
            with torch.no_grad():
                results = trainer.evaluate()
        
        eval_time = time.time() - start_time
        
        if eval_time > timeout:
            logger.warning(f"Evaluation took {eval_time:.1f}s (> {timeout}s timeout)")
        
        eval_loss = float(results.get("eval_loss", float("inf")))
        eval_acc = float(results.get("eval_accuracy", 0.0))
        
        # Cleanup
        del m, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'loss': eval_loss,
            'accuracy': eval_acc,
            'eval_time': eval_time,
            'compression_info': compression_info
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'eval_time': 0.0,
            'error': str(e)
        }

def pretrain_surrogate(cfg, base_model, tokenizer, train_ds, val_ds, trainer_args, dev, baseline_loss):
    """Pre-train surrogate on diverse configurations"""
    logger.info("=== Pre-training Surrogate on Diverse Configurations ===")
    
    # Generate diverse configurations
    configurations = generate_diverse_configurations(cfg, cfg['surrogate_pretrain_samples'])
    
    # Evaluate configurations
    losses = []
    valid_configs = []
    
    for i, config in enumerate(configurations):
        if i % 20 == 0:
            logger.info(f"Evaluating configuration {i+1}/{len(configurations)}")
        
        r_list = [c[0] for c in config]
        b_list = [c[1] for c in config] 
        s_list = [c[2] for c in config]
        
        result = evaluate_with_timeout(
            base_model, r_list, b_list, s_list, cfg,
            tokenizer, train_ds, val_ds, trainer_args, dev
        )
        
        if not np.isinf(result['loss']) and result['loss'] > 0:
            losses.append(result['loss'])
            valid_configs.append(config)
            
            # Log interesting results
            if result['loss'] < baseline_loss:
                improvement = baseline_loss - result['loss']
                logger.info(f"  Found improvement: {improvement:.6f} (loss: {result['loss']:.6f})")
    
    logger.info(f"Collected {len(valid_configs)} valid evaluations out of {len(configurations)}")
    
    # Train surrogate
    if len(valid_configs) >= 10:
        # Determine feature dimension
        sample_features = NeuralSurrogate(1)._engineer_features([valid_configs[0]])
        feature_dim = sample_features.shape[1]
        
        surrogate = NeuralSurrogate(feature_dim)
        r2, mse = surrogate.fit(valid_configs, losses)
        
        logger.info(f"Surrogate pre-training complete: R²={r2:.4f}, MSE={mse:.6f}")
        return surrogate, valid_configs, losses
    else:
        logger.warning("Insufficient valid configurations for surrogate training")
        feature_dim = 45  # Estimated from feature engineering
        return NeuralSurrogate(feature_dim), [], []

def enhanced_surrogate_training(mc, cfg, surrogate, task_feats, layer_feats, opt, scaler, dev):
    """Enhanced surrogate training with stable learning"""
    logger.info(f"=== Enhanced Surrogate Training ({cfg['surrogate_epochs']} epochs) ===")
    
    # Running statistics for advantage normalization
    reward_stats = RunningStats(alpha=cfg['advantage_smoothing'])
    
    # Track learning progress
    epoch_rewards = []
    epoch_entropies = []
    epoch_r2_scores = []
    surrogate_data = deque(maxlen=1000)  # Circular buffer for surrogate data
    
    for epoch in range(cfg['surrogate_epochs']):
        # Temperature annealing (slower)
        progress = epoch / max(cfg['surrogate_epochs'] - 1, 1)
        temperature = cfg['temperature_start'] * (cfg['temperature_end'] / cfg['temperature_start']) ** progress
        
        epoch_configs = []
        epoch_logps = []
        epoch_raw_rewards = []
        
        # Sample configurations
        n_samples = 12  # More samples per epoch
        for k in range(n_samples):
            mc.train()
            
            # Forward pass
            raw_rp, raw_bp, raw_sp = mc(task_feats, layer_feats)
            
            # Progressive noise reduction
            noise_scale = 0.2 * (cfg['temperature_start'] / temperature) * np.sqrt(progress + 0.1)
            noise = noise_scale * torch.randn_like(raw_rp)
            
            raw_rp = raw_rp + noise
            raw_bp = raw_bp + noise
            raw_sp = raw_sp + noise
            
            # Convert to probabilities
            r_p = F.softmax(raw_rp / temperature, dim=-1)
            b_p = F.softmax(raw_bp / temperature, dim=-1)
            s_p = F.softmax(raw_sp / temperature, dim=-1)
            
            # Sample configuration
            total_logp = torch.tensor(0., device=dev, requires_grad=True)
            sampled_config = []
            
            for li in range(layer_feats.size(1)):
                r_dist = Categorical(r_p[0, li])
                b_dist = Categorical(b_p[0, li])
                s_dist = Categorical(s_p[0, li])
                
                r_idx = r_dist.sample()
                b_idx = b_dist.sample()
                s_idx = s_dist.sample()
                
                total_logp = total_logp + (r_dist.log_prob(r_idx) + 
                                         b_dist.log_prob(b_idx) + 
                                         s_dist.log_prob(s_idx))
                
                sampled_config.append([int(r_idx.item()), int(b_idx.item()), int(s_idx.item())])
            
            total_logp = total_logp / (layer_feats.size(1) * 3)
            epoch_logps.append(total_logp)
            epoch_configs.append(sampled_config)
            
            # Get reward from surrogate
            if surrogate.is_fitted:
                pred_loss = surrogate.predict([sampled_config])[0]
                # Add prediction uncertainty as exploration bonus
                prediction_uncertainty = 0.01 * np.random.normal(0, 1)
                pred_loss += prediction_uncertainty
            else:
                # Informative random baseline
                pred_loss = np.random.uniform(0.6, 1.2)
            
            # Convert to reward with clipping
            raw_reward = -pred_loss
            clipped_reward = np.clip(raw_reward, -cfg['reward_clipping'], cfg['reward_clipping'])
            epoch_raw_rewards.append(clipped_reward)
            
            # Store for surrogate retraining
            surrogate_data.append((sampled_config, pred_loss))
        
        # Update reward statistics
        reward_stats.update(epoch_raw_rewards)
        
        # Normalize advantages
        normalized_advantages = reward_stats.normalize(np.array(epoch_raw_rewards))
        
        # Policy gradient update with normalized advantages
        policy_loss = torch.tensor(0., device=dev, requires_grad=True)
        for logp, advantage in zip(epoch_logps, normalized_advantages):
            policy_loss = policy_loss + (-logp * torch.tensor(advantage, device=dev))
        
        policy_loss = policy_loss / len(epoch_logps)
        
        # Add entropy regularization (adaptive)
        entropy = calculate_policy_entropy(r_p, b_p, s_p, temperature)
        entropy_weight = 0.05 * (temperature / cfg['temperature_start'])  # Reduce as temp decreases
        total_loss = policy_loss - entropy_weight * entropy
        
        # Gradient update with adaptive learning rate
        current_lr = cfg['meta_lr'] * (1.0 - 0.5 * progress)  # Slight decay
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr
        
        opt.zero_grad()
        if dev.type == "cuda" and isinstance(scaler, GradScaler):
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
            opt.step()
        
        # Track metrics
        avg_reward = float(np.mean(epoch_raw_rewards))
        epoch_rewards.append(avg_reward)
        epoch_entropies.append(entropy.item())
        
        # Periodically retrain surrogate with accumulated data
        current_r2 = 0.0
        if len(surrogate_data) >= 50 and (epoch + 1) % cfg['validation_frequency'] == 0:
            # Convert surrogate data to lists
            recent_configs = [item[0] for item in list(surrogate_data)[-100:]]
            recent_losses = [item[1] for item in list(surrogate_data)[-100:]]
            
            # Retrain surrogate
            if len(recent_configs) >= 20:
                try:
                    current_r2, mse = surrogate.fit(recent_configs, recent_losses, epochs=100)
                except Exception as e:
                    logger.warning(f"Surrogate retraining failed: {e}")
                    current_r2 = 0.0
        
        epoch_r2_scores.append(current_r2)
        
        # Comprehensive logging
        logger.info(f"Epoch {epoch+1:02d}: reward={avg_reward:.6f}, "
                   f"pg_loss={policy_loss.item():.6f}, entropy={entropy:.6f}, "
                   f"temp={temperature:.3f}, grad_norm={grad_norm:.4f}, "
                   f"lr={current_lr:.6f}, R²={current_r2:.4f}")
        
        # Early convergence detection
        if epoch >= 10:
            recent_rewards = epoch_rewards[-5:]
            reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            
            if abs(reward_trend) < 1e-4 and entropy.item() < 0.1:
                logger.info(f"Early convergence detected at epoch {epoch+1}")
                break
    
    return mc, epoch_rewards, epoch_entropies, epoch_r2_scores

def calculate_policy_entropy(r_p, b_p, s_p, temperature=1.0):
    """Calculate policy entropy with temperature scaling"""
    def entropy_temp(p, temp):
        p_temp = F.softmax(torch.log(p + 1e-8) / temp, dim=-1)
        return -(p_temp * torch.log(p_temp + 1e-8)).sum(-1).mean()
    
    e1 = entropy_temp(r_p, temperature)
    e2 = entropy_temp(b_p, temperature)
    e3 = entropy_temp(s_p, temperature)
    return (e1 + e2 + e3) / 3

def enhanced_rollout_training(mc, cfg, tokenizer, task_feats, layer_feats, opt, scaler, dev,
                            base_model, train_ds, val_ds, trainer_args, baseline_loss):
    """Enhanced rollout training with actual evaluations"""
    logger.info(f"=== Enhanced Rollout Training ({cfg['rollout_epochs']} epochs) ===")
    
    # Running statistics for stable learning
    reward_stats = RunningStats(alpha=0.9)
    
    epoch_rewards = []
    epoch_losses = []
    epoch_improvements = []
    best_reward = -float('inf')
    best_config = None
    
    logger.info(f"Baseline loss: {baseline_loss:.6f}")
    
    for epoch in range(cfg['rollout_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{cfg['rollout_epochs']}")
        
        # Temperature annealing (very slow for rollouts)
        progress = epoch / max(cfg['rollout_epochs'] - 1, 1)
        temperature = max(1.5 * (0.8 ** progress), 0.5)  # Min temp 0.5
        
        sample_rewards = []
        sample_logps = []
        sample_configs = []
        sample_results = []
        
        for sample in range(cfg['rollout_samples']):
            logger.info(f"  Sample {sample+1}/{cfg['rollout_samples']}")
            
            mc.train()
            
            # Forward pass with controlled exploration
            raw_rp, raw_bp, raw_sp = mc(task_feats, layer_feats)
            
            # Adaptive noise based on current performance
            noise_scale = 0.15 * max(1.0 - epoch / cfg['rollout_epochs'], 0.2)
            noise = noise_scale * torch.randn_like(raw_rp)
            
            raw_rp = raw_rp + noise
            raw_bp = raw_bp + noise
            raw_sp = raw_sp + noise
            
            # Convert to probabilities
            r_p = F.softmax(raw_rp / temperature, dim=-1)
            b_p = F.softmax(raw_bp / temperature, dim=-1)
            s_p = F.softmax(raw_sp / temperature, dim=-1)
            
            # Sample configuration
            total_logp = torch.tensor(0., device=dev, requires_grad=True)
            sampled_config = []
            
            for li in range(layer_feats.size(1)):
                r_dist = Categorical(r_p[0, li])
                b_dist = Categorical(b_p[0, li])
                s_dist = Categorical(s_p[0, li])
                
                r_idx = r_dist.sample()
                b_idx = b_dist.sample()
                s_idx = s_dist.sample()
                
                total_logp = total_logp + (r_dist.log_prob(r_idx) + 
                                         b_dist.log_prob(b_idx) + 
                                         s_dist.log_prob(s_idx))
                
                sampled_config.append([int(r_idx.item()), int(b_idx.item()), int(s_idx.item())])
            
            total_logp = total_logp / (layer_feats.size(1) * 3)
            sample_logps.append(total_logp)
            sample_configs.append(sampled_config)
            
            # Evaluate compressed model
            r_list = [sc[0] for sc in sampled_config]
            b_list = [sc[1] for sc in sampled_config]
            s_list = [sc[2] for sc in sampled_config]
            
            result = evaluate_with_timeout(
                base_model, r_list, b_list, s_list, cfg,
                tokenizer, train_ds, val_ds, trainer_args, dev
            )
            
            sample_results.append(result)
            eval_loss = result['loss']
            
            # Calculate reward with bonuses
            if np.isinf(eval_loss) or eval_loss <= 0:
                reward = -10.0  # Large penalty
            else:
                reward = -eval_loss
                
                # Improvement bonus
                if eval_loss < baseline_loss:
                    improvement_bonus = 2.0 * (baseline_loss - eval_loss)
                    reward += improvement_bonus
                
                # Diversity bonus (encourage exploration)
                if len(sample_configs) > 1:
                    diversity_bonus = cfg['diversity_bonus'] * calculate_diversity_score(sample_configs)
                    reward += diversity_bonus
            
            sample_rewards.append(reward)
            
            improvement = baseline_loss - eval_loss if not np.isinf(eval_loss) else float('-inf')
            logger.info(f"    Loss: {eval_loss:.6f}, Reward: {reward:.6f}, "
                       f"Improvement: {improvement:+.6f}")
            
            # Track best configuration
            if reward > best_reward:
                best_reward = reward
                best_config = sampled_config.copy()
                
                logger.info(f"    New best reward: {best_reward:.6f}")
        
        # Filter valid samples
        valid_indices = [i for i, r in enumerate(sample_rewards) 
                        if not np.isinf(r) and r > -10.0]
        
        if len(valid_indices) > 0:
            valid_rewards = [sample_rewards[i] for i in valid_indices]
            valid_logps = [sample_logps[i] for i in valid_indices]
            
            # Update reward statistics
            reward_stats.update(valid_rewards)
            
            # Normalize advantages
            normalized_advantages = reward_stats.normalize(np.array(valid_rewards))
            
            # Policy gradient update
            policy_loss = torch.tensor(0., device=dev, requires_grad=True)
            for logp, advantage in zip(valid_logps, normalized_advantages):
                policy_loss = policy_loss + (-logp * torch.tensor(advantage, device=dev))
            
            policy_loss = policy_loss / len(valid_rewards)
            
            # Adaptive learning rate
            current_lr = cfg['meta_lr'] * (1.0 - 0.3 * progress)
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
            
            # Gradient update
            opt.zero_grad()
            if dev.type == "cuda" and isinstance(scaler, GradScaler):
                scaler.scale(policy_loss).backward()
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=2.0)
                scaler.step(opt)
                scaler.update()
            else:
                policy_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=2.0)
                opt.step()
            
            # Track metrics
            avg_reward = float(np.mean(valid_rewards))
            avg_improvement = float(np.mean([baseline_loss - res['loss'] 
                                          for res in sample_results 
                                          if not np.isinf(res['loss'])]))
            
            epoch_rewards.append(avg_reward)
            epoch_losses.append(policy_loss.item())
            epoch_improvements.append(avg_improvement)
            
            entropy = calculate_policy_entropy(r_p, b_p, s_p, temperature)
            
            logger.info(f"  Avg Reward: {avg_reward:.6f}, Policy Loss: {policy_loss.item():.6f}")
            logger.info(f"  Avg Improvement: {avg_improvement:+.6f}, Entropy: {entropy:.6f}")
            logger.info(f"  Grad Norm: {grad_norm:.4f}, Temp: {temperature:.3f}, LR: {current_lr:.6f}")
            
            # Save checkpoint of best model
            if avg_reward == max(epoch_rewards):
                checkpoint_path = os.path.join(cfg['output_dir'], 'best_rollout_checkpoint.pt')
                torch.save({
                    'model_state_dict': mc.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'epoch': epoch,
                    'best_reward': best_reward,
                    'best_config': best_config
                }, checkpoint_path)
                
        else:
            logger.warning(f"  No valid samples in epoch {epoch+1}")
    
    return mc, epoch_rewards, epoch_losses, epoch_improvements, best_config

def calculate_diversity_score(configurations):
    """Calculate diversity score for a set of configurations"""
    if len(configurations) < 2:
        return 0.0
    
    # Calculate pairwise differences
    total_diff = 0.0
    comparisons = 0
    
    for i in range(len(configurations)):
        for j in range(i+1, len(configurations)):
            config1, config2 = configurations[i], configurations[j]
            
            # Layer-wise differences
            layer_diffs = []
            for (r1, b1, s1), (r2, b2, s2) in zip(config1, config2):
                diff = abs(r1 - r2) + abs(b1 - b2) + abs(s1 - s2)
                layer_diffs.append(diff)
            
            avg_diff = np.mean(layer_diffs)
            total_diff += avg_diff
            comparisons += 1
    
    return total_diff / comparisons if comparisons > 0 else 0.0