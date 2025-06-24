# Setup and Installation Guide

This guide provides detailed instructions for setting up the GPT-Based Drone Data Leakage Classification system on various platforms.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for installation, 5GB for full datasets
- **Internet**: Required for downloading models and dependencies

### Recommended Requirements

- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 16GB or more
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (optional, for faster training)
- **Storage**: SSD with 10GB+ free space

### Software Dependencies

- Git (for cloning repository)
- Python package manager (pip)
- Virtual environment tool (venv, conda, or virtualenv)

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/username/drone-gpt-classification.git
cd drone-gpt-classification
```

#### Step 2: Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; import transformers; print('Installation successful!')"
```

### Method 2: Conda Installation

```bash
# Create conda environment
conda create -n drone-gpt python=3.11
conda activate drone-gpt

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Method 3: Docker Installation

```bash
# Build Docker image
docker build -t drone-gpt-classification .

# Run container
docker run -it --gpus all -v $(pwd):/workspace drone-gpt-classification

# For CPU-only systems:
docker run -it -v $(pwd):/workspace drone-gpt-classification
```

### Method 4: Development Installation

```bash
# Clone repository
git clone https://github.com/username/drone-gpt-classification.git
cd drone-gpt-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Platform-Specific Instructions

### Windows

#### Prerequisites

1. **Install Python 3.11+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked during installation

2. **Install Git**
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default installation options

3. **Install Visual Studio Build Tools** (if needed)
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Required for some Python packages with C extensions

#### Installation Steps

```cmd
# Open Command Prompt or PowerShell as Administrator
# Clone repository
git clone https://github.com/username/drone-gpt-classification.git
cd drone-gpt-classification

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows-Specific Issues

- **Long Path Names**: Enable long path support in Windows 10+
- **Antivirus**: Add project directory to antivirus exclusions
- **PowerShell Execution Policy**: May need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS

#### Prerequisites

1. **Install Homebrew**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.11+**
   ```bash
   brew install python@3.11
   ```

3. **Install Git**
   ```bash
   brew install git
   ```

#### Installation Steps

```bash
# Clone repository
git clone https://github.com/username/drone-gpt-classification.git
cd drone-gpt-classification

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS-Specific Issues

- **Xcode Command Line Tools**: Install with `xcode-select --install`
- **M1/M2 Macs**: Some packages may need ARM64-specific versions
- **Permissions**: May need to use `sudo` for system-wide installations

### Linux (Ubuntu/Debian)

#### Prerequisites

```bash
# Update package list
sudo apt update

# Install Python 3.11 and pip
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip

# Install Git
sudo apt install git

# Install build essentials (for compiling packages)
sudo apt install build-essential
```

#### Installation Steps

```bash
# Clone repository
git clone https://github.com/username/drone-gpt-classification.git
cd drone-gpt-classification

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux-Specific Issues

- **CUDA Support**: Install NVIDIA drivers and CUDA toolkit for GPU support
- **Memory**: Ensure sufficient swap space for large model loading
- **Permissions**: Avoid using `sudo` with pip in virtual environments

### CentOS/RHEL/Fedora

```bash
# Install Python 3.11 (may need EPEL repository)
sudo dnf install python3.11 python3.11-pip python3.11-devel

# Install Git and build tools
sudo dnf install git gcc gcc-c++ make

# Follow standard installation steps
```

## Verification

### Basic Verification

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Test Python imports
python -c "
import torch
import transformers
import numpy as np
import pandas as pd
import sklearn
print('All core dependencies imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Model Testing

```bash
# Run basic model test
python scripts/test_gpt_model.py

# Expected output:
# Data loading successful!
# Model initialization successful!
# Text representation test successful!
# Classification test successful!
# All tests passed!
```

### Data Processing Test

```bash
# Test data loading and preprocessing
python -c "
from utils.data_utils import load_drone_data
X, y = load_drone_data('data/dataset_64.csv')
print(f'Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features')
print(f'Class distribution: {np.bincount(y)}')
"
```

### Full Pipeline Test

```bash
# Run complete analysis pipeline
python scripts/data_analysis.py

# Check generated outputs
ls figures/  # Should contain visualization files
ls outputs/  # Should contain analysis results
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when importing packages

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. CUDA Issues

**Problem**: PyTorch not detecting GPU

**Solutions**:
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Memory Issues

**Problem**: Out of memory errors during training

**Solutions**:
- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision training
- Close other applications

```python
# In config.yaml, reduce batch_size
training:
  batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 2  # Compensate with accumulation
```

#### 4. Slow Performance

**Problem**: Training/inference is very slow

**Solutions**:
- Enable GPU if available
- Use smaller model variants
- Optimize data loading
- Enable mixed precision

```bash
# Check if using GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

#### 5. Package Conflicts

**Problem**: Conflicting package versions

**Solutions**:
```bash
# Create fresh environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use conda for better dependency resolution
conda env create -f environment.yml
```

### Platform-Specific Troubleshooting

#### Windows

- **Long path issues**: Enable long paths in Windows settings
- **Antivirus interference**: Add exclusions for Python and project directories
- **PowerShell restrictions**: Adjust execution policy

#### macOS

- **M1/M2 compatibility**: Use ARM64-compatible packages
- **Xcode tools**: Ensure command line tools are installed
- **Homebrew issues**: Update Homebrew and packages

#### Linux

- **Permission errors**: Avoid using sudo with pip in virtual environments
- **Missing libraries**: Install development packages (python3-dev, build-essential)
- **CUDA setup**: Properly configure NVIDIA drivers and CUDA toolkit

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: Search GitHub issues for similar problems
2. **Create new issue**: Use the bug report template
3. **Provide details**: Include error messages, system info, and steps to reproduce
4. **Contact maintainers**: Email for urgent issues

## Advanced Configuration

### GPU Configuration

For systems with multiple GPUs:

```python
# Set specific GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only

# Or in Python code
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

### Memory Optimization

For systems with limited memory:

```yaml
# config.yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 2
  dataloader_num_workers: 2
  pin_memory: false

model:
  max_length: 128  # Reduce from 256
```

### Performance Tuning

```yaml
# config.yaml
training:
  mixed_precision: true
  compile_model: true  # PyTorch 2.0+
  dataloader_num_workers: 4
  pin_memory: true
```

### Custom Data Paths

```yaml
# config.yaml
paths:
  data_dir: "/custom/path/to/data"
  model_dir: "/custom/path/to/models"
  output_dir: "/custom/path/to/outputs"
```

### Environment Variables

```bash
# Set environment variables for configuration
export DRONE_DATA_PATH="/path/to/data"
export DRONE_MODEL_PATH="/path/to/models"
export CUDA_VISIBLE_DEVICES="0,1"  # Use specific GPUs
```

## Next Steps

After successful installation:

1. **Read the documentation**: Check README.md and docs/
2. **Run examples**: Try the quick start examples
3. **Explore notebooks**: Look at example notebooks (if available)
4. **Customize configuration**: Modify config.yaml for your needs
5. **Train your model**: Follow the training guide

For detailed usage instructions, see the main [README.md](README.md) file.

