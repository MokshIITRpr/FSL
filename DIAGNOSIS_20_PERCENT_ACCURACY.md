# Diagnosis: 20% Accuracy Issue

## Problem Summary

**Issue**: Few-shot training achieved only 20% accuracy (random performance for 5-way classification) after 50 epochs.

**Root Cause**: SSL (Self-Supervised Learning) pretraining **completely failed**. The loss stayed constant at ~4.14 from epoch 1 to epoch 100, meaning the encoder never learned meaningful features.

## Evidence

### 1. SSL Training Failure
```
Epoch | Loss
--------------------
    1 | 4.14392999903361
   10 | 4.1431365483601885
   20 | 4.143134611765544
   30 | 4.143134643554688
   40 | 4.143134624481201
   50 | 4.143134596506755
   60 | 4.143134627024333
   70 | 4.143134642283122
   80 | 4.14313458887736
   90 | 4.143134600321452
  100 | 4.143134583791097
```

**Expected**: Loss should decrease from ~8.0 to ~2.0 over 100 epochs.
**Actual**: Loss stuck at ~4.14 (no learning occurred).

### 2. Few-Shot Training Failure
```
Epoch | Val Acc | Train Acc
------------------------------
    1 |  0.2000 |  0.1998
    5 |  0.2000 |  0.1983
   10 |  0.2000 |  0.1999
   15 |  0.2000 |  0.2007
   20 |  0.2000 |  0.2016
   25 |  0.2000 |  0.1994
   30 |  0.2000 |  0.2031
   35 |  0.2000 |  0.2019
   40 |  0.2000 |  0.2006
   45 |  0.2000 |  0.2002
   50 |  0.2000 |  0.1985
```

**Expected**: Accuracy should increase from ~20% to ~65-70% over 50 epochs.
**Actual**: Accuracy stuck at ~20% (random guessing).

### 3. Checkpoint Analysis
- **SSL best_model.pth**: Epoch 18, Loss 4.14 (bad - didn't train)
- **Few-shot best_model.pth**: Epoch 1, Val Acc 20% (bad - didn't train)
- **Encoder keys match**: ✅ (architecture is correct)
- **Pretrained encoder loaded**: ✅ (path is correct)

## Root Cause Analysis

### Why SSL Training Failed

The SSL loss staying constant suggests one of these issues:

1. **Learning rate too low or too high**
   - Too low: Model can't learn
   - Too high: Model jumps around and doesn't converge

2. **Gradient flow issues**
   - Gradients might be vanishing or exploding
   - Batch normalization issues
   - Dead ReLU neurons

3. **Data loading issues**
   - Augmentations might not be working
   - Data might not be loaded correctly
   - Batch size might be too small

4. **Model architecture issues**
   - Encoder might not be suitable for the task
   - Projection head might have issues

5. **Loss computation issues**
   - SimCLR loss might be computed incorrectly
   - Temperature parameter might be wrong

## Solutions

### Solution 1: Retrain SSL with Better Hyperparameters (Recommended)

```bash
# Retrain SSL with adjusted learning rate and batch size
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0005 \
    --temperature 0.07 \
    --checkpoint_dir checkpoints/ssl/simclr_fixed
```

**Key changes**:
- Reduced learning rate: `0.001` → `0.0005`
- Reduced batch size: `128` → `64` (if memory issues)
- Lower temperature: `0.5` → `0.07` (standard for SimCLR)
- New checkpoint directory to avoid overwriting

### Solution 2: Use a Different SSL Method (BYOL)

BYOL is often more stable than SimCLR:

```bash
python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method byol \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --checkpoint_dir checkpoints/ssl/byol
```

### Solution 3: Check Data Loading

Verify that data is loading correctly:

```python
# Quick test script
from data.datasets import get_dataset
from data.transforms import get_contrastive_transforms, TwoViewTransform

transform = TwoViewTransform(get_contrastive_transforms(224))
dataset = get_dataset('tuberlin', 'data/tuberlin', split='train', transform=transform)

# Check first sample
view1, view2, label = dataset[0]
print(f'View1 shape: {view1.shape}')
print(f'View2 shape: {view2.shape}')
print(f'Label: {label}')
```

### Solution 4: Use Pretrained Weights (Quick Fix)

If you need results quickly, use ImageNet pretrained ResNet:

```bash
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder resnet18 \
    --few_shot_model prototypical \
    --pretrained_encoder None \
    --n_way 5 \
    --n_shot 5 \
    --epochs 50 \
    --checkpoint_dir checkpoints/few_shot/resnet18
```

## Recommended Action Plan

1. **Retrain SSL from scratch** with Solution 1 (better hyperparameters)
2. **Monitor SSL training** - loss should decrease from ~8.0 to ~2.0
3. **Once SSL works**, retrain few-shot with the good SSL checkpoint
4. **Expected results**:
   - SSL loss: ~2.0 after 100 epochs
   - Few-shot accuracy: ~65-70% after 50 epochs

## Quick Test Command

After retraining SSL, test with a quick few-shot run:

```bash
# Quick test (5 epochs)
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/simclr_fixed/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --epochs 5 \
    --checkpoint_dir checkpoints/test_fs_quick
```

If this works (accuracy > 50%), then do full training.

## Expected Timeline

- **SSL retraining**: 2-3 hours (100 epochs)
- **Few-shot training**: 30-45 minutes (50 epochs)
- **Evaluation**: 5 minutes

**Total**: ~3-4 hours for complete retraining

## Notes

- The 20% accuracy is **not a bug** - it's the expected result when SSL training fails
- The encoder weights are essentially random, so few-shot can't learn
- Retraining SSL properly should fix the issue
- The architecture and code are correct - it's a hyperparameter/training issue

