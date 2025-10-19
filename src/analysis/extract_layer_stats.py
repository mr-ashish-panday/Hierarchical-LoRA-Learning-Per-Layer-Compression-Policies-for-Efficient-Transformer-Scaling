import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))

def load_config(path=None):
    if path is None:
        # Look for config in project root
        path = os.path.join(project_root, 'src', 'config', 'default.yaml')
    if not os.path.exists(path):
        # Fallback: create basic config
        return {
            'output_dir': './output',
            'ranks': [1, 2, 4, 8, 16],
            'bitwidths': [2, 4, 8],
            'sparsities': [0.0, 0.5, 0.75, 0.9]
        }
    return yaml.safe_load(open(path))

def extract_layer_statistics(model):
    """Extract real layer statistics from the model"""
    stats = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.data
            layer_stats = {
                'layer_name': name,
                'weight_mean': float(weight.mean()),
                'weight_std': float(weight.std()),
                'weight_min': float(weight.min()),
                'weight_max': float(weight.max()),
                'param_count': int(weight.numel())
            }
            stats.append(layer_stats)
    
    return pd.DataFrame(stats)

def main():
    cfg = load_config()
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    print("Loading model for layer analysis...")
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    
    print("Extracting layer statistics...")
    layer_stats = extract_layer_statistics(model)
    
    # Add layer index for easier grouping
    layer_stats['layer'] = range(len(layer_stats))
    
    output_path = os.path.join(cfg['output_dir'], 'phase2_layer_stats.csv')
    layer_stats.to_csv(output_path, index=False)
    print(f"Saved real layer statistics to {output_path}")

if __name__ == '__main__':
    main()
