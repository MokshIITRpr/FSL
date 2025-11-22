# Tesla P100 GPU Setup Guide

This guide explains how to set up your environment to train on Tesla P100 GPUs.

## Problem

Tesla P100 GPUs have compute capability sm_60 (Pascal architecture), which is **not supported by PyTorch 2.0+**. PyTorch 2.0 and later only support compute capabilities sm_70 and above.

## Solution

Install PyTorch 1.13.1 with CUDA 11.7, which supports Tesla P100 GPUs.

## Quick Setup (Recommended)

Run the automated setup script:

```bash
bash setup_p100_environment.sh
```

This script will:
1. Create a new conda environment named `pytorch_p100` with Python 3.11
2. Install PyTorch 1.13.1 with CUDA 11.7
3. Install all required dependencies
4. Test GPU compatibility

## Manual Setup

If you prefer to set up manually:

### Step 1: Create Conda Environment

```bash
conda create -n pytorch_p100 python=3.11
conda activate pytorch_p100
```

### Step 2: Install PyTorch 1.13.1 with CUDA 11.7

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Step 3: Install Dependencies

```bash
# Install other dependencies (skip torch and torchvision)
pip install numpy pillow opencv-python scikit-image scikit-learn scipy matplotlib seaborn tensorboard tqdm pyyaml requests gdown kaggle albumentations
```

Or install from requirements.txt (but skip torch/torchvision):

```bash
grep -v "^torch" requirements.txt | grep -v "^pytorch-lightning" | pip install -r /dev/stdin
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

You should see:
- PyTorch version: 1.13.1+cu117
- CUDA available: True
- GPU: Tesla P100-PCIE-12GB (or similar)

## Using the Environment

### Activate the Environment

```bash
conda activate pytorch_p100
```

### Run Training

```bash
python train_ssl.py --data_root /path/to/data --dataset tuberlin --device cuda
```

The training script will now:
- ✅ Detect your Tesla P100 GPUs
- ✅ Verify compatibility
- ✅ Train on GPU

## Troubleshooting

### Error: "CUDA is not available"

- Make sure you've activated the correct conda environment: `conda activate pytorch_p100`
- Verify CUDA drivers are installed: `nvidia-smi`
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Error: "no kernel image is available for execution on the device"

This means PyTorch 2.0+ is still installed. Make sure you're using PyTorch 1.13.1:
```bash
python -c "import torch; print(torch.__version__)"
```

If it shows 2.x, reinstall PyTorch 1.13.1:
```bash
pip uninstall torch torchvision
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Multiple GPU Training

If you have multiple Tesla P100 GPUs, you can use them by setting the device to `cuda`. The script will automatically use all available GPUs. For multi-GPU training, you may need to use `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`.

## Notes

- **Python Version**: PyTorch 1.13.1 requires Python 3.8-3.11. Python 3.13 is not supported.
- **CUDA Version**: PyTorch 1.13.1+cu117 works with CUDA 11.7. Your system CUDA (12.2) is fine - PyTorch comes with its own CUDA runtime.
- **Performance**: Tesla P100 GPUs are older but still capable. Expect good performance for most deep learning tasks.

## Alternative: Use CPU (Not Recommended)

If you cannot install PyTorch 1.13.1, you can train on CPU (much slower):

```bash
python train_ssl.py --data_root /path/to/data --dataset tuberlin --device cpu --force_cpu
```

However, training on CPU will be **significantly slower** than GPU training.



