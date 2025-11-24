#!/bin/bash
# Quick fix script for pytorch_p100 environment
# Fixes NumPy and OpenCV compatibility issues

set -u

ENV_NAME="pytorch_p100"

echo "=========================================="
echo "Fixing pytorch_p100 Environment"
echo "=========================================="
echo ""

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME} || {
    echo "Error: Environment ${ENV_NAME} not found."
    echo "Please run: bash setup_p100_environment.sh"
    exit 1
}

echo "Step 1: Fixing NumPy compatibility..."
pip uninstall numpy -y
pip install "numpy<2.0" --no-cache-dir

echo ""
echo "Step 2: Fixing OpenCV compatibility..."
pip uninstall opencv-python opencv-python-headless -y
pip install "opencv-python-headless<4.10" --no-cache-dir

echo ""
echo "Step 3: Verifying installation..."
python -c "
import numpy
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms

print('=' * 60)
print('Environment Check')
print('=' * 60)
print(f'NumPy: {numpy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test NumPy with transforms (like in DataLoader)
img = Image.new('RGB', (224, 224), color='red')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
tensor = transform(img)
print(f'✅ Image transforms work! Tensor shape: {tensor.shape}')

print('=' * 60)
print('✅ All checks passed!')
print('=' * 60)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Environment fixed successfully!"
    echo "=========================================="
    echo ""
    echo "You can now run training:"
    echo "  conda activate ${ENV_NAME}"
    echo "  python train_ssl.py --data_root <path> --dataset tuberlin --device cuda"
else
    echo ""
    echo "=========================================="
    echo "❌ Fix failed"
    echo "=========================================="
    exit 1
fi



