#!/usr/bin/env python
"""
GPU Compatibility Checker for Few-Shot Sketch Recognition

This script checks if your GPU is compatible with the installed PyTorch version
and provides instructions for fixing compatibility issues.
"""

import torch
import sys


def check_gpu_compatibility():
    """Check GPU compatibility and provide recommendations."""
    print("=" * 80)
    print("GPU Compatibility Check")
    print("=" * 80)
    print()
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print()
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA is not available. Training will use CPU (much slower).")
        print()
        return
    
    # Check each GPU
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    print()
    
    incompatible_gpus = []
    compatible_gpus = []
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
        
        # Check if it's a known incompatible GPU
        p100_indicators = ['p100', 'tesla p100', 'gp100']
        is_p100 = any(indicator in gpu_name.lower() for indicator in p100_indicators)
        
        if is_p100 and torch.__version__.startswith('2.'):
            print(f"  ❌ INCOMPATIBLE: Tesla P100 (compute capability sm_60) is not supported by PyTorch 2.0+")
            incompatible_gpus.append((i, gpu_name))
        else:
            # Try to test if it actually works
            try:
                torch.cuda.set_device(i)
                test_tensor = torch.zeros(1).to(f'cuda:{i}')
                # Try a simple operation that requires a kernel
                test_model = torch.nn.BatchNorm2d(4).to(f'cuda:{i}')
                test_input = torch.randn(1, 4, 4, 4).to(f'cuda:{i}')
                _ = test_model(test_input)
                print(f"  ✅ COMPATIBLE")
                compatible_gpus.append((i, gpu_name))
                del test_tensor, test_model, test_input
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ❌ INCOMPATIBLE: {str(e)[:100]}")
                incompatible_gpus.append((i, gpu_name))
        print()
    
    # Provide recommendations
    if incompatible_gpus:
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print()
        print("Your GPUs are not compatible with the current PyTorch installation.")
        print()
        print("Option 1: Use CPU mode (slower but works)")
        print("  Add --force_cpu flag when running training scripts")
        print()
        print("Option 2: Install compatible PyTorch version (requires Python 3.10 or 3.11)")
        print("  1. Create a new conda environment with Python 3.11:")
        print("     conda create -n pytorch_p100 python=3.11")
        print("     conda activate pytorch_p100")
        print()
        print("  2. Install PyTorch 1.13.1 with CUDA 11.7 (supports sm_60):")
        print("     pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \\")
        print("         --extra-index-url https://download.pytorch.org/whl/cu117")
        print()
        print("  3. Verify installation:")
        print("     python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\"")
        print()
        print("Option 3: Use a different machine with newer GPUs (V100, A100, etc.)")
        print()
    else:
        print("=" * 80)
        print("✅ All GPUs are compatible!")
        print("=" * 80)
        print()
    
    print("=" * 80)


if __name__ == '__main__':
    check_gpu_compatibility()



