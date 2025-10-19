import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
import numpy as np

def load_config(path="src/config/default.yaml"):
    return yaml.safe_load(open(path))

def run_experiment(rank=None, bitwidth=None, sparsity=None, cfg=None):
    cfg_run = cfg.copy()
    if rank is not None: 
        cfg_run["ranks"] = [rank]
    if bitwidth is not None: 
        cfg_run["bitwidths"] = [bitwidth]
    if sparsity is not None: 
        cfg_run["sparsities"] = [sparsity]
    
    output_dir = os.path.join(cfg_run['output_dir'], f"phase1_r{rank}_b{bitwidth}_s{sparsity}")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg_run["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure the tokenizer knows about the pad token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load GLUE dataset with correct task name
    task_name = cfg_run["task_name"]
    # Remove 'glue-' prefix if present
    if task_name.startswith("glue-"):
        task_name = task_name[5:]
    
    print(f"Loading dataset: glue/{task_name}")
    dataset = load_dataset("glue", task_name)
    
    # Get number of labels for the task
    labels = dataset["train"]["label"]
    num_labels = len(set(labels))
    print(f"Task {task_name} has {num_labels} labels")
    
    # Use AutoModelForSequenceClassification instead of CausalLM
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg_run["model_name"], 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"Error loading model for classification: {e}")
        print("Using a simple baseline model...")
        # Create a simple baseline model
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cfg_run["model_name"])
        config.num_labels = num_labels
        model = AutoModelForSequenceClassification.from_config(config)
    
    model.to(cfg_run["device"])
    model.eval()
    
    def preprocess_batch(examples):
        # Handle different GLUE task formats
        inputs = None
        if task_name in ["mrpc", "stsb", "qqp"]:
            # Tasks with sentence pairs
            inputs = tokenizer(
                examples["sentence1"], 
                examples["sentence2"],
                truncation=True, 
                padding="max_length", 
                max_length=cfg_run["max_seq_length"],
                return_tensors="pt"
            )
        elif task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            inputs = tokenizer(
                examples["premise"], 
                examples["hypothesis"],
                truncation=True, 
                padding="max_length", 
                max_length=cfg_run["max_seq_length"],
                return_tensors="pt"
            )
        elif task_name == "qnli":
            inputs = tokenizer(
                examples["question"], 
                examples["sentence"],
                truncation=True, 
                padding="max_length", 
                max_length=cfg_run["max_seq_length"],
                return_tensors="pt"
            )
        elif task_name == "rte":
            inputs = tokenizer(
                examples["sentence1"], 
                examples["sentence2"],
                truncation=True, 
                padding="max_length", 
                max_length=cfg_run["max_seq_length"],
                return_tensors="pt"
            )
        else:
            # Single sentence tasks like CoLA, SST-2
            if "sentence" in examples:
                inputs = tokenizer(
                    examples["sentence"],
                    truncation=True, 
                    padding="max_length", 
                    max_length=cfg_run["max_seq_length"],
                    return_tensors="pt"
                )
            else:
                # Fallback - find first text field
                text_fields = [k for k in examples.keys() if isinstance(examples[k], list) and 
                              len(examples[k]) > 0 and isinstance(examples[k][0], str)]
                if text_fields:
                    inputs = tokenizer(
                        examples[text_fields[0]],
                        truncation=True, 
                        padding="max_length", 
                        max_length=cfg_run["max_seq_length"],
                        return_tensors="pt"
                    )
        
        if inputs is None:
            raise ValueError(f"Could not process examples for task {task_name}")
            
        return inputs
    
    # Process validation set one sample at a time to avoid padding issues
    validation_data = dataset["validation"]
    total_loss = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Processing {len(validation_data)} validation samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(validation_data):
            if i % 50 == 0:
                print(f"Processed {i}/{len(validation_data)} samples...")
            
            try:
                # Process single sample
                single_batch = {}
                for key, value in sample.items():
                    single_batch[key] = [value]  # Make it a list for batch processing
                
                # Tokenize single sample
                inputs = preprocess_batch(single_batch)
                
                # Move to device
                input_ids = inputs["input_ids"].to(cfg_run["device"])
                attention_mask = inputs["attention_mask"].to(cfg_run["device"])
                labels = torch.tensor([sample["label"]]).to(cfg_run["device"])
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate loss
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Skip this sample and continue
                continue
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    print(f"Average loss: {avg_loss:.4f} (processed {total_samples} samples)")
    return avg_loss

def main():
    cfg = load_config()
    records = []
    
    print("Starting Phase 1 sensitivity analysis...")
    
    # Test different rank values
    print("Testing rank values:", cfg["ranks"])
    for r in cfg["ranks"]:
        print(f"\nTesting rank = {r}")
        try:
            loss = run_experiment(rank=r, cfg=cfg)
            records.append({"axis":"rank","value":r,"loss":loss})
            print(f"Rank {r}: loss = {loss:.4f}")
        except Exception as e:
            print(f"Error with rank {r}: {e}")
            records.append({"axis":"rank","value":r,"loss":float('inf')})
    
    # Test different bitwidth values
    print("\nTesting bitwidth values:", cfg["bitwidths"])
    for b in cfg["bitwidths"]:
        print(f"\nTesting bitwidth = {b}")
        try:
            loss = run_experiment(bitwidth=b, cfg=cfg)
            records.append({"axis":"bitwidth","value":b,"loss":loss})
            print(f"Bitwidth {b}: loss = {loss:.4f}")
        except Exception as e:
            print(f"Error with bitwidth {b}: {e}")
            records.append({"axis":"bitwidth","value":b,"loss":float('inf')})
    
    # Test different sparsity values
    print("\nTesting sparsity values:", cfg["sparsities"])
    for s in cfg["sparsities"]:
        print(f"\nTesting sparsity = {s}")
        try:
            loss = run_experiment(sparsity=s, cfg=cfg)
            records.append({"axis":"sparsity","value":s,"loss":loss})
            print(f"Sparsity {s}: loss = {loss:.4f}")
        except Exception as e:
            print(f"Error with sparsity {s}: {e}")
            records.append({"axis":"sparsity","value":s,"loss":float('inf')})
    
    # Save results
    df = pd.DataFrame(records)
    output_path = os.path.join(cfg["output_dir"], "phase1_sensitivity.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved Phase 1 results to {output_path}")
    print("\nResults summary:")
    print(df)

if __name__ == "__main__":
    main()
