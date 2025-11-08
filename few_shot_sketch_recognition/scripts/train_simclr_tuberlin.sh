#!/bin/bash
# Train SimCLR on TU-Berlin dataset
#
# This script pretrains a sketch encoder using SimCLR (contrastive learning)
# on the TU-Berlin sketch dataset.

set -e  # Exit on error

echo "=========================================="
echo "Training SimCLR on TU-Berlin Dataset"
echo "=========================================="

# Configuration
DATASET="tuberlin"
DATA_ROOT="data/tuberlin"
ENCODER="sketch_cnn"
SSL_METHOD="simclr"
BATCH_SIZE=128
EPOCHS=200
LR=0.001
TEMPERATURE=0.5
CHECKPOINT_DIR="checkpoints/ssl/simclr_tuberlin"

# Check if dataset exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Dataset not found at $DATA_ROOT"
    echo "Please download the dataset first:"
    echo "  python main.py download --dataset tuberlin --output_dir $DATA_ROOT"
    exit 1
fi

# Run training
python train_ssl.py \
    --dataset $DATASET \
    --data_root $DATA_ROOT \
    --encoder $ENCODER \
    --ssl_method $SSL_METHOD \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --temperature $TEMPERATURE \
    --checkpoint_dir $CHECKPOINT_DIR

echo "=========================================="
echo "Training Complete!"
echo "Best model saved to: $CHECKPOINT_DIR/best_model.pth"
echo "=========================================="

