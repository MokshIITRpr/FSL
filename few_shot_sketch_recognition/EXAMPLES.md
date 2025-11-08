# Usage Examples

This document provides practical examples for different use cases of the Few-Shot Sketch Recognition Framework.

## Table of Contents
1. [Quick Test Run](#quick-test-run)
2. [Training Examples](#training-examples)
3. [Evaluation Examples](#evaluation-examples)
4. [Experiment Configurations](#experiment-configurations)
5. [Common Workflows](#common-workflows)

---

## Quick Test Run

### Minimal Example (5 minutes)

Test the framework with minimal training to ensure everything works:

```bash
# Step 1: Download a small subset
python main.py download --dataset quickdraw --output_dir data/quickdraw --n_categories 10

# Step 2: Quick SSL training (10 epochs)
python train_ssl.py \
    --dataset quickdraw \
    --data_root data/quickdraw \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --epochs 10 \
    --batch_size 64 \
    --checkpoint_dir checkpoints/test

# Step 3: Quick few-shot training (5 epochs)
python train_few_shot.py \
    --dataset quickdraw \
    --data_root data/quickdraw \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/test/best_model.pth \
    --n_way 3 \
    --n_shot 3 \
    --n_train_episodes 50 \
    --epochs 5 \
    --checkpoint_dir checkpoints/test_fs

# Step 4: Evaluate
python evaluate.py \
    --dataset quickdraw \
    --data_root data/quickdraw \
    --checkpoint checkpoints/test_fs/best_model.pth \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --n_way 3 \
    --n_shot 3 \
    --n_episodes 20 \
    --visualize
```

---

## Training Examples

### Example 1: SimCLR on TU-Berlin (Full Training)

```bash
# Download dataset (one-time)
python main.py download --dataset tuberlin --output_dir data/tuberlin

# Train SimCLR with custom CNN encoder
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --embedding_dim 512 \
    --projection_dim 128 \
    --temperature 0.5 \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --checkpoint_dir checkpoints/ssl/simclr_tuberlin

# Expected output:
# Epoch 1 - Average Loss: 7.234
# Epoch 50 - Average Loss: 4.123
# Epoch 100 - Average Loss: 2.567
# Epoch 200 - Average Loss: 1.890
# Best model saved to: checkpoints/ssl/simclr_tuberlin/best_model.pth
```

### Example 2: BYOL on QuickDraw

```bash
# Download QuickDraw with 50 categories
python main.py download --dataset quickdraw --output_dir data/quickdraw --n_categories 50

# Train BYOL with ResNet18 encoder
python train_ssl.py \
    --dataset quickdraw \
    --data_root data/quickdraw \
    --encoder resnet18 \
    --ssl_method byol \
    --embedding_dim 512 \
    --projection_dim 256 \
    --ema_decay 0.996 \
    --batch_size 256 \
    --epochs 200 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/ssl/byol_quickdraw

# BYOL typically has more stable loss curves
```

### Example 3: Prototypical Networks (5-way 5-shot)

```bash
# Train using pretrained SimCLR encoder
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/simclr_tuberlin/best_model.pth \
    --distance_metric euclidean \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --n_train_episodes 1000 \
    --n_val_episodes 200 \
    --epochs 100 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/few_shot/proto_5w5s

# Expected progression:
# Epoch 1 [Train] - Loss: 1.523, Accuracy: 0.385
# Epoch 1 [Val] - Loss: 1.402, Accuracy: 0.421
# Epoch 50 [Train] - Loss: 0.234, Accuracy: 0.634
# Epoch 50 [Val] - Loss: 0.312, Accuracy: 0.598
# Epoch 100 [Train] - Loss: 0.145, Accuracy: 0.678
# Epoch 100 [Val] - Loss: 0.298, Accuracy: 0.612
```

### Example 4: Matching Networks with Attention

```bash
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model matching \
    --pretrained_encoder checkpoints/ssl/simclr_tuberlin/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --epochs 100 \
    --checkpoint_dir checkpoints/few_shot/matching_5w5s
```

### Example 5: Training from Scratch (No SSL)

```bash
# Train few-shot model without SSL pretraining
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --n_way 5 \
    --n_shot 5 \
    --epochs 150 \
    --checkpoint_dir checkpoints/few_shot/proto_scratch

# Note: Will likely achieve ~10-15% lower accuracy than with SSL
```

---

## Evaluation Examples

### Example 1: Standard Evaluation (5-way 5-shot)

```bash
python evaluate.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --checkpoint checkpoints/few_shot/proto_5w5s/best_model.pth \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --n_episodes 600 \
    --visualize \
    --output_dir results/5way5shot

# Expected output:
# ============================================================
# EVALUATION RESULTS
# ============================================================
# Configuration: 5-way 5-shot
# Model: prototypical
# Episodes evaluated: 600
# ------------------------------------------------------------
# Mean Accuracy: 0.6543 (65.43%)
# Std Accuracy: 0.0234
# 95% Confidence Interval: ±0.0187
# ============================================================
```

### Example 2: Challenging Scenario (5-way 1-shot)

```bash
python evaluate.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --checkpoint checkpoints/few_shot/proto_5w5s/best_model.pth \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --n_way 5 \
    --n_shot 1 \
    --n_query 15 \
    --n_episodes 600 \
    --output_dir results/5way1shot

# Expected: ~40-50% accuracy (much harder with only 1 example)
```

### Example 3: More Classes (10-way 5-shot)

```bash
python evaluate.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --checkpoint checkpoints/few_shot/proto_5w5s/best_model.pth \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --n_way 10 \
    --n_shot 5 \
    --n_query 10 \
    --n_episodes 600 \
    --output_dir results/10way5shot

# Expected: ~50-60% accuracy (harder with more classes)
```

---

## Experiment Configurations

### Configuration 1: Low-Resource Setup

For systems with limited GPU memory:

```bash
# SSL Training
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --batch_size 32 \
    --image_size 128 \
    --epochs 150 \
    --checkpoint_dir checkpoints/ssl/low_resource

# Few-Shot Training
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/low_resource/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_train_episodes 500 \
    --image_size 128 \
    --checkpoint_dir checkpoints/few_shot/low_resource
```

### Configuration 2: High-Performance Setup

For systems with powerful GPUs:

```bash
# SSL Training with larger model
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder resnet50 \
    --ssl_method byol \
    --batch_size 256 \
    --image_size 224 \
    --epochs 300 \
    --embedding_dim 2048 \
    --checkpoint_dir checkpoints/ssl/high_performance

# Few-Shot Training
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder resnet50 \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/high_performance/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_train_episodes 2000 \
    --embedding_dim 2048 \
    --epochs 150 \
    --checkpoint_dir checkpoints/few_shot/high_performance
```

### Configuration 3: Fast Experimentation

Quick iterations for hyperparameter tuning:

```bash
# Download small dataset
python main.py download --dataset quickdraw --output_dir data/quickdraw --n_categories 20

# Quick SSL
python train_ssl.py \
    --dataset quickdraw \
    --data_root data/quickdraw \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --batch_size 128 \
    --epochs 50 \
    --checkpoint_dir checkpoints/ssl/quick

# Quick few-shot
python train_few_shot.py \
    --dataset quickdraw \
    --data_root data/quickdraw \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/quick/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_train_episodes 500 \
    --epochs 30 \
    --checkpoint_dir checkpoints/few_shot/quick
```

---

## Common Workflows

### Workflow 1: Complete Pipeline

```bash
# 1. Download data
python main.py download --dataset tuberlin --output_dir data/tuberlin

# 2. Run full pipeline (SSL + Few-shot + Evaluation)
python main.py pipeline \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --few_shot_model prototypical \
    --ssl_epochs 200 \
    --fs_epochs 100 \
    --n_way 5 \
    --n_shot 5

# Results will be in results/
```

### Workflow 2: Compare SSL Methods

```bash
# Train with SimCLR
python train_ssl.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --ssl_method simclr --epochs 200 \
    --checkpoint_dir checkpoints/ssl/simclr

# Train with BYOL
python train_ssl.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --ssl_method byol --epochs 200 \
    --checkpoint_dir checkpoints/ssl/byol

# Train few-shot with SimCLR encoder
python train_few_shot.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/simclr/best_model.pth \
    --epochs 100 --checkpoint_dir checkpoints/fs/with_simclr

# Train few-shot with BYOL encoder
python train_few_shot.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/byol/best_model.pth \
    --epochs 100 --checkpoint_dir checkpoints/fs/with_byol

# Evaluate both
python evaluate.py --dataset tuberlin --data_root data/tuberlin \
    --checkpoint checkpoints/fs/with_simclr/best_model.pth \
    --encoder sketch_cnn --few_shot_model prototypical \
    --n_way 5 --n_shot 5 --n_episodes 600 \
    --output_dir results/simclr

python evaluate.py --dataset tuberlin --data_root data/tuberlin \
    --checkpoint checkpoints/fs/with_byol/best_model.pth \
    --encoder sketch_cnn --few_shot_model prototypical \
    --n_way 5 --n_shot 5 --n_episodes 600 \
    --output_dir results/byol
```

### Workflow 3: Compare Few-Shot Algorithms

```bash
# Train Prototypical Networks
python train_few_shot.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/best_model.pth \
    --epochs 100 --checkpoint_dir checkpoints/fs/prototypical

# Train Matching Networks
python train_few_shot.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --few_shot_model matching \
    --pretrained_encoder checkpoints/ssl/best_model.pth \
    --epochs 100 --checkpoint_dir checkpoints/fs/matching

# Train Relation Networks
python train_few_shot.py --dataset tuberlin --data_root data/tuberlin \
    --encoder sketch_cnn --few_shot_model relation \
    --pretrained_encoder checkpoints/ssl/best_model.pth \
    --epochs 100 --checkpoint_dir checkpoints/fs/relation

# Evaluate all three
for model in prototypical matching relation; do
    python evaluate.py --dataset tuberlin --data_root data/tuberlin \
        --checkpoint checkpoints/fs/$model/best_model.pth \
        --encoder sketch_cnn --few_shot_model $model \
        --n_way 5 --n_shot 5 --n_episodes 600 \
        --output_dir results/$model
done
```

### Workflow 4: Hyperparameter Search

```bash
# Try different temperatures for SimCLR
for temp in 0.3 0.5 0.7; do
    python train_ssl.py --dataset tuberlin --data_root data/tuberlin \
        --encoder sketch_cnn --ssl_method simclr \
        --temperature $temp --epochs 100 \
        --checkpoint_dir checkpoints/ssl/simclr_temp_$temp
done

# Try different n-shot values
for shot in 1 5 10; do
    python train_few_shot.py --dataset tuberlin --data_root data/tuberlin \
        --encoder sketch_cnn --few_shot_model prototypical \
        --pretrained_encoder checkpoints/ssl/best_model.pth \
        --n_way 5 --n_shot $shot --epochs 100 \
        --checkpoint_dir checkpoints/fs/shot_$shot
    
    python evaluate.py --dataset tuberlin --data_root data/tuberlin \
        --checkpoint checkpoints/fs/shot_$shot/best_model.pth \
        --encoder sketch_cnn --few_shot_model prototypical \
        --n_way 5 --n_shot $shot --n_episodes 600 \
        --output_dir results/shot_$shot
done
```

---

## Tips and Best Practices

### For Better Results

1. **Always use SSL pretraining** - typically adds 10-20% accuracy
2. **Use larger batch sizes for SSL** - better negative samples (128-256)
3. **Train SSL longer** - 200+ epochs recommended
4. **Use cosine annealing** - automatically applied in training scripts
5. **Try different encoders** - ResNet50 may work better but is slower

### For Faster Experiments

1. **Use QuickDraw** - faster to download and load than TU-Berlin
2. **Reduce episodes** - 500 train episodes still work reasonably well
3. **Use smaller images** - `--image_size 128` is 4× faster
4. **Fewer categories** - Start with 20 categories for quick tests

### For Debugging

1. **Use `--visualize` flag** - see what model is predicting
2. **Check logs** - detailed training info in `logs/`
3. **Start small** - Test with 10 epochs first
4. **Monitor GPU usage** - Reduce batch size if OOM errors

---

## Expected Training Times (on RTX 3090)

| Task | Configuration | Time |
|------|--------------|------|
| SSL (SimCLR) | 200 epochs, batch=128 | ~2 hours |
| SSL (BYOL) | 200 epochs, batch=256 | ~3 hours |
| Few-Shot | 100 epochs, 1000 episodes/epoch | ~45 minutes |
| Evaluation | 600 episodes | ~5 minutes |
| Full Pipeline | SSL + Few-Shot + Eval | ~3 hours |

*Times vary based on hardware and dataset size*

---

## Next Steps

After running these examples:
1. Experiment with different configurations
2. Try your own sketch datasets
3. Modify augmentation strategies
4. Implement new few-shot algorithms
5. Share your results!

For more details, see the full documentation in README.md and FRAMEWORK_DESIGN.md.

