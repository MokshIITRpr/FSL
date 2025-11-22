# Memory Error Fix - ResNet50 + BYOL

## Problem
ResNet50 + BYOL runs out of memory on Tesla P100 (12GB) GPUs because:
1. **ResNet50 is large**: ~25M parameters
2. **BYOL uses 2 networks**: Online + Target = 2x memory
3. **Batch size too large**: 128 batch size is too much

## Solution Applied

### Automatic Batch Size Reduction
The training script now automatically reduces batch size:
- **ResNet50 + BYOL**: Batch size reduced to **12** (from 64/128)
- **ResNet50 + SimCLR**: Batch size reduced to **32**
- **ResNet18 + BYOL**: Batch size reduced to **32**

### Memory Optimizations Added
1. ✅ Automatic batch size reduction for large models
2. ✅ Periodic GPU cache clearing (every 5 batches)
3. ✅ Tensor cleanup after each batch
4. ✅ Disabled pin_memory to save GPU memory

## Updated Command

The script will automatically handle memory optimization. Just run:

```bash
conda activate pytorch_p100 && python main.py pipeline \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder resnet50 \
    --ssl_method byol \
    --few_shot_model prototypical \
    --batch_size 64 \
    --ssl_epochs 150 \
    --fs_epochs 80 \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15
```

**Note**: Batch size will be automatically reduced to 12 for ResNet50 + BYOL.

## Alternative: Use ResNet18 (Faster, Less Memory)

If ResNet50 still gives OOM errors, use ResNet18:

```bash
conda activate pytorch_p100 && python main.py pipeline \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder resnet18 \
    --ssl_method byol \
    --few_shot_model prototypical \
    --batch_size 64 \
    --ssl_epochs 150 \
    --fs_epochs 80 \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15
```

**ResNet18 Benefits**:
- ✅ Less memory usage (batch size 32 instead of 12)
- ✅ Faster training (~2x faster)
- ✅ Still excellent accuracy (~69-71% vs ~72-75%)
- ✅ Fits comfortably in 12GB GPU

## Expected Performance

| Model | Batch Size | Memory | Accuracy | Time |
|-------|------------|--------|----------|------|
| ResNet50 + BYOL | 12 | High | ~72-75% | ~8-10 hours |
| ResNet18 + BYOL | 32 | Medium | ~69-71% | ~4-5 hours |

## If Still Getting OOM

1. **Reduce batch size further**: Try 8 or 4
2. **Use ResNet18**: Much more memory efficient
3. **Use SimCLR instead of BYOL**: Only 1 network (less memory)
4. **Reduce image size**: Try 128x128 instead of 224x224

## Recommendation

For 8-hour training with 12GB GPUs:
- **Best accuracy**: ResNet50 + BYOL (batch_size=12, auto-adjusted)
- **Best balance**: ResNet18 + BYOL (batch_size=32, faster, still great accuracy)


