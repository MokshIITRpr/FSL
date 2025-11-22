#!/bin/bash
# Complete SSL + Few-Shot Retraining Pipeline
# This script runs everything in one go and can be run in tmux/screen

set -e  # Exit on error

echo "============================================================"
echo "Full Retraining Pipeline - SSL + Few-Shot"
echo "============================================================"
echo ""
echo "This script will:"
echo "1. Train SSL with correct hyperparameters (100 epochs, ~2-3 hours)"
echo "2. Verify SSL training succeeded"
echo "3. Train few-shot model (50 epochs, ~30-45 minutes)"
echo "4. Evaluate final model"
echo ""
echo "Total time: ~3-4 hours"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Create checkpoint directories
SSL_DIR="checkpoints/ssl/simclr_fixed"
FS_DIR="checkpoints/few_shot/prototypical_fixed"
mkdir -p $SSL_DIR
mkdir -p $FS_DIR

# ============================================================
# STEP 1: SSL Training
# ============================================================
echo ""
echo "============================================================"
echo "STEP 1/4: SSL Training (100 epochs)"
echo "============================================================"
echo ""
echo "Training with correct hyperparameters:"
echo "  Batch size: 128 (was 32)"
echo "  Temperature: 0.07 (was 0.5)"
echo "  Learning rate: 0.001"
echo ""

python train_ssl.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --ssl_method simclr \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --temperature 0.07 \
    --weight_decay 1e-4 \
    --checkpoint_dir $SSL_DIR \
    --device cuda

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ SSL training failed! Check the logs above."
    exit 1
fi

# ============================================================
# STEP 2: Verify SSL Training
# ============================================================
echo ""
echo "============================================================"
echo "STEP 2/4: Verifying SSL Training"
echo "============================================================"
echo ""

python verify_ssl_training.py $SSL_DIR

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ SSL training verification FAILED!"
    echo "   Loss did not decrease as expected."
    echo "   Please check hyperparameters and logs."
    echo ""
    echo "   Training stopped. Not proceeding to few-shot training."
    exit 1
fi

# ============================================================
# STEP 3: Few-Shot Training
# ============================================================
echo ""
echo "============================================================"
echo "STEP 3/4: Few-Shot Training (50 epochs)"
echo "============================================================"
echo ""
echo "Using pretrained encoder: $SSL_DIR/best_model.pth"
echo ""

python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder $SSL_DIR/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --epochs 50 \
    --lr 0.001 \
    --checkpoint_dir d$FS_DIR \
    --device cuda

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Few-shot training failed! Check the logs above."
    exit 1
fi

# ============================================================
# STEP 4: Evaluation
# ============================================================
echo ""
echo "============================================================"
echo "STEP 4/4: Final Evaluation"
echo "============================================================"
echo ""

python evaluate.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --checkpoint $FS_DIR/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --n_episodes 600 \
    --visualize \
    --output_dir results_fixed

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Evaluation failed! Check the logs above."
    exit 1
fi

# ============================================================
# Final Summary
# ============================================================
echo ""
echo "============================================================"
echo "✅ COMPLETE! All steps finished successfully!"
echo "============================================================"
echo ""
echo "Results:"
echo "  SSL checkpoint: $SSL_DIR/best_model.pth"
echo "  Few-shot checkpoint: $FS_DIR/best_model.pth"
echo "  Evaluation results: results_fixed/results_5way_5shot.txt"
echo "  Visualizations: results_fixed/episode_*.png"
echo ""
echo "Check results_fixed/results_5way_5shot.txt for final accuracy."
echo "Expected: 65-70% (much better than previous 20%!)"
echo ""
echo "Training logs saved in logs/ directory"
echo ""

