#!/usr/bin/env python3
"""
1. Debug apply_compression parameter reduction.
2. If it works, train controllers for budgets.
"""
import os, sys, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import csv

# Project path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(proj_root, 'src'))
from analysis.apply_compression_simple import apply_compression
from models.meta_controller import MetaController

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

CFG = {
    'ranks':[1,2,4,8,16],
    'bitwidths':[2,4,8,16,32],
    'sparsities':[0.0,0.2,0.5,0.7,0.9]
}

def count_linears(m):
    return len([l for l in m.modules() if isinstance(l, nn.Linear)])

def test_compression():
    print("Testing apply_compression for parameter reduction...")
    model = AutoModelForSequenceClassification.from_pretrained('gpt2-medium', num_labels=2).to(DEVICE)
    L = count_linears(model)
    policies = [
        ([0]*L, [0]*L, [0]*L),
        ([4]*L, [4]*L, [0]*L),
    ]
    for i,(r,b,s) in enumerate(policies):
        m = copy.deepcopy(model)
        comp = apply_compression(m, r, b, s, CFG)
        orig = sum(p.numel() for p in model.parameters())
        new  = sum(p.numel() for p in comp.parameters())
        print(f"Policy {i}: orig={orig}, comp={new}, ratio={new/orig:.4f}")
    print("If ratio==1.0 for policy 1, fix apply_compression before proceeding.")
    return

def main():
    test_compression()
    # If compression works, add training code here.
    # Omitted for brevity; first ensure compression reduces params.

if __name__=="__main__":
    main()
