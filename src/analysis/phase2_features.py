import os
import yaml
import torch
import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.cuda import max_memory_allocated, reset_peak_memory_stats

def load_config(path="src/config/default.yaml"):
    return yaml.safe_load(open(path))

def setup_tokenizer(model_name):
    """Setup tokenizer with proper padding token handling"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle missing padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

def get_text_from_sample(sample, task_name):
    """Extract text fields based on task type"""
    if task_name in ["glue-cola", "glue-sst2"]:
        # Single sentence tasks
        return sample.get("sentence", ""), None
    elif task_name in ["glue-mrpc", "glue-qqp", "glue-mnli", "glue-qnli", "glue-rte", "glue-wnli"]:
        # Sentence pair tasks
        return sample.get("sentence1", ""), sample.get("sentence2", "")
    else:
        # Default fallback
        return sample.get("sentence1", sample.get("sentence", "")), sample.get("sentence2", None)

def extract_task_features(task_name, tokenizer, dataset, cfg):
    """Extract dataset and task-specific features"""
    print(f"Extracting task features for {task_name}...")
    
    # Dataset size
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"]) if "validation" in dataset else 0
    
    # Sequence length stats on train split (sample first 1000 for efficiency)
    sample_size = min(1000, train_size)
    lengths = []
    
    for i, example in enumerate(dataset["train"]):
        if i >= sample_size:
            break
            
        text1, text2 = get_text_from_sample(example, task_name)
        
        if text2:
            # Sentence pair
            tokens = tokenizer(text1, text2, truncation=True, padding=False)["input_ids"]
        else:
            # Single sentence
            tokens = tokenizer(text1, truncation=True, padding=False)["input_ids"]
        
        lengths.append(len(tokens))
    
    avg_seq_len = float(np.mean(lengths)) if lengths else 0.0
    max_seq_len = float(np.max(lengths)) if lengths else 0.0
    min_seq_len = float(np.min(lengths)) if lengths else 0.0
    
    # Vocab diversity: unique tokens / total tokens (on sample)
    all_tokens = []
    for i, example in enumerate(dataset["train"]):
        if i >= sample_size:
            break
            
        text1, text2 = get_text_from_sample(example, task_name)
        
        if text2:
            tokens = tokenizer(text1, text2, truncation=True, padding=False)["input_ids"]
        else:
            tokens = tokenizer(text1, truncation=True, padding=False)["input_ids"]
        
        all_tokens.extend(tokens)
    
    vocab_diversity = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    
    return {
        "train_size": train_size, 
        "val_size": val_size,
        "avg_seq_len": avg_seq_len,
        "max_seq_len": max_seq_len,
        "min_seq_len": min_seq_len,
        "vocab_diversity": vocab_diversity
    }

def extract_layer_stats(model, cfg):
    """Extract statistics from model layers"""
    print("Extracting layer statistics...")
    
    stats = []
    for name, param in model.named_parameters():
        if "layer" in name and param.requires_grad:
            arr = param.data.cpu().numpy()
            stats.append({
                "layer": name,
                "weight_std": float(arr.std()),
                "weight_mean": float(arr.mean()),
                "weight_abs_mean": float(np.abs(arr).mean()),
                "param_count": arr.size
            })
    
    return stats

def measure_hardware_metrics(model, inputs, cfg):
    """Measure hardware performance metrics"""
    print("Measuring hardware metrics...")
    
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        reset_peak_memory_stats(device)
    
    # Measure latency
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    latency = time.time() - start
    
    # Memory usage
    if device.type == "cuda":
        peak_mem = max_memory_allocated(device) / (1024**2)  # MB
    else:
        peak_mem = 0.0  # CPU memory tracking is more complex
    
    # Energy: placeholder (requires external tooling like nvidia-ml-py)
    energy = None
    
    return {
        "latency_s": latency, 
        "peak_vram_mb": peak_mem, 
        "energy_j": energy,
        "device": str(device)
    }

def main():
    try:
        cfg = load_config()
        print(f"Loaded config: {cfg}")
        
        os.makedirs(cfg["output_dir"], exist_ok=True)
        
        # Setup tokenizer with padding token handling
        tokenizer = setup_tokenizer(cfg["model_name"])
        print(f"Tokenizer setup complete. Pad token: {tokenizer.pad_token}")
        
        # Load dataset
        task_name = cfg["task_name"].replace("glue-", "")
        print(f"Loading dataset: glue/{task_name}")
        ds = load_dataset("glue", task_name)
        
        # Extract task features
        task_feats = extract_task_features(cfg["task_name"], tokenizer, ds, cfg)
        
        # Prepare sample input for hardware measurement
        sample = ds["validation"][0] if "validation" in ds else ds["train"][0]
        text1, text2 = get_text_from_sample(sample, cfg["task_name"])
        
        inputs = tokenizer(
            text1, 
            text2,
            truncation=True, 
            padding="max_length", 
            max_length=cfg.get("max_seq_length", 512),
            return_tensors="pt"
        )
        
        print(f"Sample input shape: {inputs['input_ids'].shape}")
        
        # Load model
        print("Loading model...")
        num_labels = len(set(ds["train"]["label"]))
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        )
        
        # Resize token embeddings if we added new tokens
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # Extract features
        layer_stats = extract_layer_stats(model, cfg)
        hw_metrics = measure_hardware_metrics(model, inputs, cfg)
        
        # Save results
        output_dir = cfg["output_dir"]
        
        # Save task features
        task_df = pd.DataFrame([task_feats])
        task_path = os.path.join(output_dir, "phase2_task_features.csv")
        task_df.to_csv(task_path, index=False)
        print(f"Task features saved to: {task_path}")
        
        # Save layer stats
        if layer_stats:
            layer_df = pd.DataFrame(layer_stats)
            layer_path = os.path.join(output_dir, "phase2_layer_stats.csv")
            layer_df.to_csv(layer_path, index=False)
            print(f"Layer stats saved to: {layer_path}")
        
        # Save hardware metrics
        hw_df = pd.DataFrame([hw_metrics])
        hw_path = os.path.join(output_dir, "phase2_hardware_metrics.csv")
        hw_df.to_csv(hw_path, index=False)
        print(f"Hardware metrics saved to: {hw_path}")
        
        print(f"\nPhase 2 feature extraction completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Task: {cfg['task_name']}")
        print(f"- Model: {cfg['model_name']}")
        print(f"- Train samples: {task_feats['train_size']}")
        print(f"- Validation samples: {task_feats['val_size']}")
        print(f"- Avg sequence length: {task_feats['avg_seq_len']:.2f}")
        print(f"- Hardware device: {hw_metrics['device']}")
        print(f"- Inference latency: {hw_metrics['latency_s']:.4f}s")
        
    except Exception as e:
        print(f"Error during phase 2 feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())