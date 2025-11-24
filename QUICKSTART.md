# Quick Start Guide

This guide will help you get started with the Few-Shot Sketch Recognition Framework in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- GPU with CUDA support (optional but recommended)
- ~5GB disk space for datasets

## Step-by-Step Guide

### 1. Install Dependencies (1 minute)

**Recommended: Use a Virtual Environment** to keep dependencies isolated:

```bash
cd few_shot_sketch_recognition

# Create virtual environment (one-time setup)
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR on Windows: venv\Scripts\activate

# Install requirements in the virtual environment
pip install -r requirements.txt
```

**Alternative: Install directly** (not recommended, but works):
```bash
cd few_shot_sketch_recognition
pip install -r requirements.txt
```

**Note**: Always remember to activate the virtual environment before running scripts:
```bash
source venv/bin/activate  # Run this each time you open a new terminal
```

### 2. Download Dataset (5-10 minutes)

Download the TU-Berlin sketch dataset:

```bash
python main.py download --dataset tuberlin --output_dir data/tuberlin
```

Or for a quicker test with QuickDraw (smaller files):

```bash
python main.py download --dataset quickdraw --output_dir data/quickdraw --n_categories 20
```

### 3. Quick Test Run (2 minutes)

Test with minimal training to ensure everything works:

```bash
# Quick SSL pretraining (10 epochs)
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --epochs 10 \
    --batch_size 32 \
    --checkpoint_dir checkpoints/test_ssl

# Quick few-shot training (5 epochs)
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/test_ssl/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --epochs 5 \
    --n_train_episodes 100 \
    --checkpoint_dir checkpoints/test_fs
# Evaluate
python evaluate.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --checkpoint checkpoints/test_fs/best_model.pth \
    --encoder resnet50  \
    --few_shot_model prototypical \
    --n_way 5 \
    --n_shot 5 \
    --n_episodes 50 \
    --visualize
```

### 4. Full Training Pipeline (30-60 minutes on GPU)

Once everything works, run the full pipeline:

```bash
python main.py pipeline \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --few_shot_model prototypical \
    --ssl_epochs 100 \
    --fs_epochs 50 \
    --n_way 5 \
    --n_shot 5
```

## Expected Results

After full training, you should see:

- **SSL Training**: Loss decreasing from ~8.0 to ~2.0
- **Few-Shot Training**: Accuracy increasing to ~60-70%
- **Test Accuracy**: 5-way 5-shot accuracy around 60-68%

## What's Next?

1. **Try Different Configurations**:
   - Different encoders: `--encoder resnet18`
   - Different SSL methods: `--ssl_method byol`
   - Different few-shot models: `--few_shot_model matching`

2. **Experiment with Few-Shot Settings**:
   - 1-shot: `--n_shot 1` (harder)
   - 10-way: `--n_way 10` (harder)
   - 10-shot: `--n_shot 10` (easier)

3. **Visualize Results**:
   - Check `results/` for episode visualizations
   - Review training logs in `logs/`

4. **Read Full Documentation**:
   - See README.md for detailed explanations
   - Check configuration files in `configs/`

## Troubleshooting

### "Out of Memory" Error

Reduce batch size and image size:
```bash
--batch_size 32 --image_size 128
```

### "Dataset Not Found" Error

Make sure you downloaded the dataset first:
```bash
python main.py download --dataset tuberlin --output_dir data/tuberlin
```

### Slow Training

- Use GPU if available (automatic if CUDA is available)
- Reduce workers: `--num_workers 2`
- Train on fewer episodes: `--n_train_episodes 500`

## Need Help?

- Check the full README.md
- Review example configs in `configs/`
- Look at the code comments - they're detailed!

Happy experimenting! 🚀

