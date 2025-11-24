#!/bin/bash
# Continue from SSL - Run Few-Shot Training + Evaluation
# SSL training already completed successfully!

set -e  # Exit on error

# Activate pytorch_p100 environment for P100 GPU compatibility
echo "Activating pytorch_p100 environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch_p100

echo "============================================================"
echo "Continuing Training - Few-Shot + Evaluation"
echo "============================================================"
echo ""
echo "SSL training completed successfully!"
echo "  Best Loss: 5.0826 (much better than 4.14!)"
echo "  Checkpoint: checkpoints/ssl/simclr_fixed/best_model.pth"
echo ""

# Create checkpoint directory
FS_DIR="checkpoints/few_shot/prototypical_fixed"
mkdir -p $FS_DIR

# ============================================================
# STEP 1: Few-Shot Training
# ============================================================
echo ""
echo "============================================================"
echo "STEP 1/2: Few-Shot Training (50 epochs)"
echo "============================================================"
echo ""
echo "Using pretrained encoder: checkpoints/ssl/simclr_fixed/best_model.pth"
echo ""

python train_few_shot.py \
    --dataset tuberlin \
    --data_root data/tuberlin \
    --encoder sketch_cnn \
    --few_shot_model prototypical \
    --pretrained_encoder checkpoints/ssl/simclr_fixed/best_model.pth \
    --n_way 5 \
    --n_shot 5 \
    --epochs 50 \
    --lr 0.001 \
    --checkpoint_dir $FS_DIR \
    --device cuda

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Few-shot training failed! Check the logs above."
    exit 1
fi

# ============================================================
# STEP 2: Evaluation
# ============================================================
echo ""
echo "============================================================"
echo "STEP 2/2: Final Evaluation"
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
echo "  SSL checkpoint: checkpoints/ssl/simclr_fixed/best_model.pth"
echo "  Few-shot checkpoint: $FS_DIR/best_model.pth"
echo "  Evaluation results: results_fixed/results_5way_5shot.txt"
echo "  Visualizations: results_fixed/episode_*.png"
echo ""
echo "Check results_fixed/results_5way_5shot.txt for final accuracy."
echo "Expected: 65-70% (much better than previous 20%!)"
echo ""

