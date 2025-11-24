# Using pytorch_p100 Environment for Training

## ✅ Good News!

The `pytorch_p100` environment already has **PyTorch 2.0.0+cu117** installed, which **WORKS with Tesla P100 GPUs**! 

Despite the warnings, PyTorch 2.0.0+cu117 (CUDA 11.7 build) includes support for compute capability sm_60.

## Quick Start

### 1. Activate the Environment

```bash
conda activate pytorch_p100
```

### 2. Install Missing Dependencies

```bash
# Fix NumPy compatibility
pip install "numpy<2.0"

# Install required packages
pip install -r requirements.txt
```

Or install essential packages:

```bash
pip install numpy pillow opencv-python scikit-image scikit-learn scipy matplotlib seaborn tensorboard tqdm pyyaml requests gdown kaggle albumentations
```

### 3. Verify GPU Compatibility

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

You should see:
- PyTorch: 2.0.0+cu117
- CUDA: True
- GPU: Tesla P100-PCIE-12GB

### 4. Run Training

```bash
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --epochs 10 \
    --batch_size 32 \
    --device cuda \
    --checkpoint_dir checkpoints/ssl
```

## Current Environment Status

- ✅ **PyTorch**: 2.0.0+cu117 (works with Tesla P100)
- ✅ **torchvision**: 0.15.0+cu117
- ✅ **NumPy**: 1.26.4 (compatible)
- ⚠️ **Other dependencies**: Need to be installed

## Important Notes

1. **Always use `conda activate pytorch_p100`** before running training scripts
2. **Do NOT use the base environment** - it has PyTorch 2.9.0+cu128 which doesn't support Tesla P100
3. The warnings about sm_60 are misleading - PyTorch 2.0.0+cu117 actually works!
4. Make sure NumPy < 2.0 to avoid compatibility issues

## Troubleshooting

### Error: "ModuleNotFoundError"

Install missing dependencies:
```bash
conda activate pytorch_p100
pip install -r requirements.txt
```

### Error: "CUDA not available"

Make sure you're in the pytorch_p100 environment:
```bash
conda activate pytorch_p100
python -c "import torch; print(torch.cuda.is_available())"
```

### Error: "NumPy compatibility warning"

Downgrade NumPy:
```bash
pip install "numpy<2.0"
```

## Summary

**You're all set!** Just:
1. `conda activate pytorch_p100`
2. Install dependencies if needed
3. Run your training scripts

The environment is ready to use with your Tesla P100 GPUs! 🚀



