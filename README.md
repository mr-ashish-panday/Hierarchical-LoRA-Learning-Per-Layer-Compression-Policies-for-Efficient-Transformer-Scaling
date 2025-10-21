# Hierarchical Sparsity–Quantization–Rank LoRA
## Project Overview
Joint meta-learning framework for layer-wise sparsity, quantization, and rank optimization.
## Structure
- src/: source code modules  
- data/: datasets and preprocessing  
- notebooks/: exploratory analysis  
- output/: logs, checkpoints, and results
## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Git LFS (Large File Storage)

This repository uses Git LFS to manage large model files efficiently. Git LFS is enabled to track large files such as model checkpoints and binary files.

### Installation

If you haven't already, install Git LFS:

```bash
# On Ubuntu/Debian
sudo apt-get install git-lfs

# On macOS
brew install git-lfs

# On Windows
# Download from https://git-lfs.github.com/
```

### Initialize Git LFS

```bash
# Initialize Git LFS in your repository
git lfs install

# Track large file types
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "*.bin"
git lfs track "*.pth"
git lfs track "*.safetensors"
git lfs track "*.h5"
git lfs track "*.pkl"

# Add the .gitattributes file
git add .gitattributes
git commit -m "Configure Git LFS for large model files"
```

### Usage Example

```bash
# Add a large model file
git add models/checkpoint.pt
git commit -m "Add model checkpoint"
git push

# Clone repository with LFS files
git clone https://github.com/mr-ashish-panday/Hierarchical-LoRA-Learning-Per-Layer-Compression-Policies-for-Efficient-Transformer-Scaling.git
cd Hierarchical-LoRA-Learning-Per-Layer-Compression-Policies-for-Efficient-Transformer-Scaling
git lfs pull
```

### Tracked File Types

The following large file types are automatically tracked by Git LFS:
- `*.pt` - PyTorch model files
- `*.ckpt` - Checkpoint files
- `*.bin` - Binary model files
- `*.pth` - PyTorch model files
- `*.safetensors` - SafeTensors format
- `*.h5` - HDF5 files
- `*.pkl` - Pickle files

# Hierarchical-LoRA-Learning-Per-Layer-Compression-Policies-for-Efficient-Transformer-Scaling
