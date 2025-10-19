#!/usr/bin/env python3
"""
phase3a_surrogate.py

Phase 3A: Surrogate Training for Meta-Controller.
Loads Phase 1 sensitivity data, trains a robust surrogate model (MLP),
and saves the surrogate and training dataset for Phase 3B.
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))

from analysis.utils import load_phase1_data, set_seed_all

class SurrogateNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def default_config():
    return {
        'phase1_csv': './output/phase1_sensitivity.csv',
        'surrogate_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'output_dir': './output',
        'seed': 42,
    }

def load_config(path=None):
    cfg = default_config()
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, help='Optional YAML config path')
    args = p.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    set_seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Phase 1 data
    feats, losses = load_phase1_data(cfg['phase1_csv'])
    X = torch.tensor(feats, dtype=torch.float32, device=device)
    y = torch.tensor(losses, dtype=torch.float32, device=device)

    # Shuffle and train/val split
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build surrogate model
    model = SurrogateNet(input_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, cfg['surrogate_epochs']+1):
        model.train()
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        total_loss = 0.0
        for i in range(0, len(X_train), cfg['batch_size']):
            xb = X_train[i:i+cfg['batch_size']]
            yb = y_train[i:i+cfg['batch_size']]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        avg_train = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_loss = loss_fn(pred_val, y_val).item()

        print(f"Epoch {epoch:02d}: Train MSE={avg_train:.4f}, Val MSE={val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], 'surrogate_best.pt'))

    # Save full dataset for Phase 3B
    df = pd.DataFrame(feats, columns=['rank','bitwidth','sparsity'])
    df['loss'] = losses
    df.to_csv(os.path.join(cfg['output_dir'], 'surrogate_dataset.csv'), index=False)
    print("Surrogate training complete. Model and dataset saved.")

if __name__ == '__main__':
    main()
