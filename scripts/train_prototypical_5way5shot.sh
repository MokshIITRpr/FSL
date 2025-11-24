#!/bin/bash
# Train Prototypical Network (5-way 5-shot) on TU-Berlin
#
# This script trains a Prototypical Network using a pretrained encoder
# from SimCLR pretraining.

set -e  # Exit on error

echo "=========================================="
echo "Training Prototypical Network (5-way 5-shot)"
echo "=========================================="

# Configuration
DATASET="tuberlin"
DATA_ROOT="data/tuberlin"
ENCODER="sketch_cnn"
FEW_SHOT_MODEL="prototypical"
PRETRAINED_ENCODER="checkpoints/ssl/simclr_tuberlin/best_model.pth"
N_WAY=5
N_SHOT=5
N_QUERY=15
EPOCHS=100
LR=0.001
CHECKPOINT_DIR="checkpoints/few_shot/prototypical_5way5shot"

# Check if dataset exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Dataset not found at $DATA_ROOT"
    exit 1
fi

# Check if pretrained encoder exists
if [ ! -f "$PRETRAINED_ENCODER" ]; then
    echo "Warning: Pretrained encoder not found at $PRETRAINED_ENCODER"
    echo "Training from scratch (without SSL pretraining)"
    PRETRAINED_ENCODER=""
fi

# Run training
python train_few_shot.py \
    --dataset $DATASET \
    --data_root $DATA_ROOT \
    --encoder $ENCODER \
    --few_shot_model $FEW_SHOT_MODEL \
    ${PRETRAINED_ENCODER:+--pretrained_encoder $PRETRAINED_ENCODER} \
    --n_way $N_WAY \
    --n_shot $N_SHOT \
    --n_query $N_QUERY \
    --epochs $EPOCHS \
    --lr $LR \
    --checkpoint_dir $CHECKPOINT_DIR

echo "=========================================="
echo "Training Complete!"
echo "Best model saved to: $CHECKPOINT_DIR/best_model.pth"
echo "=========================================="

