# Fix: Checkpoint Path Error

## Problem
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/test_ssl/best_model.pth'
```

## Solution

The checkpoint file is in `checkpoints/ssl/best_model.pth`, not `checkpoints/test_ssl/best_model.pth`.

### Correct Command

```bash
python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --epochs 5 \
    --n_train_episodes 100 \
    --checkpoint_dir checkpoints/test_fs
```

### Available Checkpoints

Check what checkpoints are available:

```bash
ls -lh checkpoints/ssl/*.pth
```

You should see:
- `checkpoints/ssl/best_model.pth` - Best model (recommended)
- `checkpoints/ssl/checkpoint_epoch_10.pth` - Latest epoch
- `checkpoints/ssl/checkpoint_epoch_*.pth` - Other epochs

### Use Best Model

The `best_model.pth` is the recommended checkpoint to use as it has the lowest loss.

### Alternative: Use Latest Epoch

If you prefer the latest epoch:

```bash
--pretrained_encoder checkpoints/ssl/checkpoint_epoch_10.pth
```

## Quick Fix

Just change the path from:
```
--pretrained_encoder checkpoints/test_ssl/best_model.pth
```

To:
```
--pretrained_encoder checkpoints/ssl/best_model.pth
```



