# Fix for torchvision Installation Error

## Problem
The setup script fails because `torchvision==0.14.1+cu117` doesn't exist in the PyTorch wheel repository.

## Solution

The setup script has been updated to:
1. First try installing `torchvision==0.14.1` from regular PyPI (works with PyTorch 1.13.1+cu117)
2. If that fails, fallback to `torchvision==0.15.0+cu117` which is available

## Manual Installation (if script still fails)

If the automated script still has issues, you can manually install:

```bash
# Activate the environment
conda activate pytorch_p100

# Install PyTorch
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install torchvision (try one of these)
pip install torchvision==0.14.1
# OR
pip install torchvision==0.15.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Clean Up Partial Installation

If the environment was partially created, remove it first:

```bash
conda env remove -n pytorch_p100
```

Then run the setup script again:

```bash
bash setup_p100_environment.sh
```

## Verify Installation

After installation, verify everything works:

```bash
conda activate pytorch_p100
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

You should see:
- PyTorch: 1.13.1+cu117
- CUDA: True
- GPU: Tesla P100-PCIE-12GB (or similar)



