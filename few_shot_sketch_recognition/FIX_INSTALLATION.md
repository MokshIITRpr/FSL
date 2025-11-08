# Installation Error Fix 🔧

## What Went Wrong?

The `learn2learn` package failed to install because:
- It requires Cython to build from source
- It had a file conflict during installation
- **Good news**: It's not actually needed! All few-shot learning is custom-implemented.

## Quick Fix (3 Steps)

### Step 1: Delete the old virtual environment
```bash
cd /Users/psychaosz/Desktop/AI/few_shot_sketch_recognition
rm -rf venv
```

### Step 2: Create a fresh virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install updated requirements (fixed)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

That's it! Should work now. ✅

## What Changed?

I removed two problematic optional dependencies:
- ❌ `learn2learn` - Not needed (we have custom implementations)
- ❌ `wandb` - Optional logging tool (you can add it later if needed)

All core functionality remains intact!

## Verify Installation

```bash
# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch {torch.__version__} installed!')"

# Check if other packages work
python -c "import numpy, PIL, sklearn, matplotlib; print('All core packages OK!')"
```

## Alternative: Install Without Virtual Environment

If you prefer (not recommended but works):
```bash
cd /Users/psychaosz/Desktop/AI/few_shot_sketch_recognition
pip install -r requirements.txt
```

## Still Having Issues?

### Issue: "No module named 'cv2'"
**Fix**: OpenCV installation issue, try:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Issue: PyTorch installation taking forever
**Fix**: Install PyTorch first, then other packages:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### Issue: "Permission denied"
**Fix**: Never use `sudo` with virtual environments. Just:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## What You Can Do Now

Once installed, test with:
```bash
# Activate venv
source venv/bin/activate

# Test the framework
python main.py --help
python train_ssl.py --help

# Should see help messages - means it's working!
```

## Optional Packages (Install Later If Needed)

```bash
# For advanced experiment tracking with Weights & Biases
pip install wandb

# For Jupyter notebook support
pip install jupyter ipykernel

# For interactive plots
pip install plotly
```

---

**TL;DR**: 
1. Delete `venv` folder
2. Create new venv: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`

✅ Done!

