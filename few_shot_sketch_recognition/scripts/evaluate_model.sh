#!/bin/bash
# Evaluate trained few-shot model
#
# This script evaluates a trained model on test set with unseen classes

set -e  # Exit on error

echo "=========================================="
echo "Evaluating Few-Shot Model"
echo "=========================================="

# Configuration
DATASET="tuberlin"
DATA_ROOT="data/tuberlin"
ENCODER="sketch_cnn"
FEW_SHOT_MODEL="prototypical"
CHECKPOINT="checkpoints/few_shot/prototypical_5way5shot/best_model.pth"
N_WAY=5
N_SHOT=5
N_QUERY=15
N_EPISODES=600
OUTPUT_DIR="results"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please train a model first or specify correct checkpoint path"
    exit 1
fi

# Run evaluation
python evaluate.py \
    --dataset $DATASET \
    --data_root $DATA_ROOT \
    --encoder $ENCODER \
    --few_shot_model $FEW_SHOT_MODEL \
    --checkpoint $CHECKPOINT \
    --n_way $N_WAY \
    --n_shot $N_SHOT \
    --n_query $N_QUERY \
    --n_episodes $N_EPISODES \
    --visualize \
    --output_dir $OUTPUT_DIR

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

