# Environment Fixes for pytorch_p100

## Issues Fixed

### 1. NumPy Compatibility Issue
**Problem**: NumPy 2.x is incompatible with PyTorch 2.0.0+cu117 and torchvision 0.15.0+cu117
**Solution**: Downgrade to NumPy < 2.0

### 2. OpenCV Compatibility Issue  
**Problem**: OpenCV 4.12+ requires NumPy >= 2.0, but we need NumPy < 2.0 for PyTorch
**Solution**: Use opencv-python-headless < 4.10 which works with NumPy 1.x

### 3. DataLoader Multiprocessing Issue
**Problem**: NumPy incompatibility causes errors in DataLoader worker processes
**Solution**: Set num_workers=0 by default (single-process data loading)

## Quick Fix

Run the fix script:

```bash
bash fix_environment.sh
```

Or manually:

```bash
conda activate pytorch_p100

# Fix NumPy
pip uninstall numpy -y
pip install "numpy<2.0"

# Fix OpenCV
pip uninstall opencv-python opencv-python-headless -y
pip install "opencv-python-headless<4.10"
```

## Verified Working Versions

- ✅ **PyTorch**: 2.0.0+cu117
- ✅ **torchvision**: 0.15.0+cu117
- ✅ **NumPy**: 1.26.4
- ✅ **OpenCV**: 4.9.0 (headless)
- ✅ **Python**: 3.11

## Training Script Changes

- Default `num_workers=0` to avoid multiprocessing issues
- Can override with `--num_workers N` if needed
- GPU compatibility check updated to recognize PyTorch 2.0.0+cu117

## Testing

After fixing, verify the environment:

```bash
conda activate pytorch_p100
python -c "
import numpy
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms

print(f'NumPy: {numpy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test image transforms
img = Image.new('RGB', (224, 224))
transform = transforms.ToTensor()
tensor = transform(img)
print(f'✅ Transforms work! Tensor shape: {tensor.shape}')
"
```

## Running Training

```bash
conda activate pytorch_p100
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --epochs 10 \
    --batch_size 32 \
    --device cuda \
    --num_workers 0 \
    --checkpoint_dir checkpoints/ssl
```

## Notes

- **num_workers=0**: Single-process data loading (slower but more compatible)
- **num_workers > 0**: Multi-process data loading (faster but may have issues with NumPy)
- If you get NumPy errors with num_workers > 0, use `--num_workers 0`

## Summary

All compatibility issues have been fixed:
1. ✅ NumPy downgraded to 1.26.4
2. ✅ OpenCV downgraded to 4.9.0 (headless)
3. ✅ Default num_workers set to 0
4. ✅ GPU compatibility check updated
5. ✅ All imports verified working

Your environment is now ready for training! 🚀



