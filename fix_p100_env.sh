#!/bin/bash
# Quick fix script for pytorch_p100 environment
# Fixes NumPy compatibility and ensures correct PyTorch version

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

echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not installed"

echo ""
echo "Fixing NumPy compatibility (downgrading to NumPy < 2.0)..."
pip install "numpy<2.0" --upgrade

echo ""
echo "Verifying PyTorch and GPU compatibility..."
python -c "
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    
    # Test GPU compatibility
    print()
    print('Testing GPU with BatchNorm...')
    try:
        test_model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ).cuda()
        test_input = torch.randn(1, 1, 32, 32).cuda()
        _ = test_model(test_input)
        print('✅ GPU compatibility test PASSED!')
        print('✅ Your Tesla P100 GPUs are ready to use!')
        del test_model, test_input
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'❌ GPU test failed: {e}')
        print('You may need to install PyTorch 1.13.1+cu117')
        exit(1)
else:
    print('⚠️  CUDA is not available')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Environment is ready!"
    echo "=========================================="
    echo ""
    echo "You can now run training:"
    echo "  conda activate ${ENV_NAME}"
    echo "  python train_ssl.py --data_root <path> --dataset tuberlin --device cuda"
else
    echo ""
    echo "=========================================="
    echo "❌ Environment fix failed"
    echo "=========================================="
    exit 1
fi



