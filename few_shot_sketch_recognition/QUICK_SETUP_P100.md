# Quick Setup for Tesla P100 GPU Training

## 🚀 Quick Start

To train on your Tesla P100 GPUs, run this command:

```bash
bash setup_p100_environment.sh
```

This will:
1. Create a conda environment `pytorch_p100` with Python 3.11
2. Install PyTorch 1.13.1 with CUDA 11.7 (supports Tesla P100)
3. Install all required dependencies
4. Test GPU compatibility

## 📋 After Setup

### Activate the environment:
```bash
conda activate pytorch_p100
```

### Run training:
```bash
python train_ssl.py --data_root /path/to/data --dataset tuberlin --device cuda
```

or

```bash
python train_few_shot.py --data_root /path/to/data --dataset tuberlin --device cuda
```

## ✅ What Changed

The training scripts now:
- ✅ **Check GPU compatibility** before starting training
- ✅ **Fail with clear instructions** if PyTorch is incompatible
- ✅ **Show GPU information** when compatible
- ✅ **Prevent crashes** by detecting issues early

## 🔧 If Setup Fails

1. **Check conda is installed**: `conda --version`
2. **Check CUDA drivers**: `nvidia-smi`
3. **Manually install PyTorch 1.13.1**:
   ```bash
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

## 📚 More Details

See `P100_SETUP.md` for detailed information and troubleshooting.



