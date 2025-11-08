# Setup Guide - Virtual Environment

## Why Use a Virtual Environment? 🤔

A virtual environment is like a **isolated sandbox** for your Python project:

✅ **Keeps packages separate** - Won't affect other Python projects  
✅ **No conflicts** - Different projects can use different package versions  
✅ **Easy cleanup** - Just delete the `venv` folder to remove everything  
✅ **Portable** - Easy to share exact dependencies via `requirements.txt`  
✅ **Safe** - Doesn't modify your system Python installation  

## Quick Setup (Easy Way) 🚀

Just run the setup script:

```bash
cd /Users/psychaosz/Desktop/AI/few_shot_sketch_recognition
bash setup.sh
```

That's it! Everything is set up automatically.

## Manual Setup (Step by Step) 📝

### Step 1: Create Virtual Environment

```bash
cd /Users/psychaosz/Desktop/AI/few_shot_sketch_recognition
python3 -m venv venv
```

This creates a folder called `venv` with an isolated Python installation.

### Step 2: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

**You'll see `(venv)` in your prompt** - this means it's active!

```bash
(venv) user@computer:~/few_shot_sketch_recognition$ 
      ↑
   This means virtual environment is active
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, torchvision, and all other dependencies **only inside the virtual environment**.

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed!')"
```

Should output something like: `PyTorch 2.0.0 installed!`

## Daily Usage 💻

### Every time you open a new terminal:

```bash
# Navigate to project
cd /Users/psychaosz/Desktop/AI/few_shot_sketch_recognition

# Activate virtual environment
source venv/bin/activate

# Now you can run scripts
python main.py --help
python train_ssl.py --help
```

### When you're done:

```bash
deactivate
```

## Common Questions ❓

### Q: Do I need to create the venv every time?
**A:** No! Create it once. After that, just activate it.

### Q: What if I forget to activate the venv?
**A:** The script will fail because packages won't be found. Just activate it and try again.

### Q: Can I delete the venv folder?
**A:** Yes! If something goes wrong, just:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q: How much space does venv take?
**A:** About 1-2GB (includes PyTorch, which is large)

### Q: Can I use conda instead?
**A:** Yes! See below.

## Alternative: Using Conda 🐍

If you have Anaconda or Miniconda:

```bash
# Create conda environment
conda create -n sketch_recognition python=3.9

# Activate it
conda activate sketch_recognition

# Install dependencies
pip install -r requirements.txt

# Run project
python main.py --help

# Deactivate when done
conda deactivate
```

## Troubleshooting 🔧

### "command not found: python3"
**Solution:** Use `python` instead:
```bash
python -m venv venv
```

### "No module named 'torch'"
**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Permission denied"
**Solution:** Don't use `sudo`. Virtual environments don't need admin rights.

### Packages not installing
**Solution:** Upgrade pip first:
```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Reference Card 📇

```bash
# SETUP (once)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# DAILY USE
source venv/bin/activate  # Activate venv
python main.py --help      # Run scripts
deactivate                 # Exit venv

# CLEANUP
rm -rf venv                # Delete venv
# Then recreate if needed
```

## What Gets Installed? 📦

The `requirements.txt` installs:

- **PyTorch** - Deep learning framework
- **torchvision** - Image processing for PyTorch
- **NumPy** - Numerical computing
- **Pillow** - Image loading
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Plotting
- **tqdm** - Progress bars
- And more (see `requirements.txt`)

**Total size**: ~1.5-2GB

## Still Confused? 🤷

Think of it this way:

- **Without venv**: Installing packages affects your entire system
- **With venv**: Installing packages only affects this project folder

It's like having a separate "mini Python installation" just for this project!

---

**TL;DR**: Run `bash setup.sh` and you're good to go! 🎉

