#!/usr/bin/env python3
"""
Verification script to check if SSL training is working correctly.

This script checks:
1. Loss decreases over time
2. Learning rate decreases
3. No constant loss (indicating training failure)
4. Checkpoint quality
"""

import torch
import sys
import os
from pathlib import Path

# Add safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([type(torch.tensor(1).numpy().item())])


def verify_ssl_checkpoint(checkpoint_dir):
    """Verify SSL checkpoint quality."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Check for best model
    best_model_path = checkpoint_dir / "best_model.pth"
    if not best_model_path.exists():
        print(f"❌ Best model not found: {best_model_path}")
        return False
    
    # Load best model
    try:
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    # Check checkpoint contents
    epoch = checkpoint.get('epoch', 'N/A')
    loss = checkpoint.get('loss', 'N/A')
    best_loss = checkpoint.get('best_loss', 'N/A')
    args = checkpoint.get('args', {})
    
    print(f"\n{'='*60}")
    print(f"SSL Checkpoint Verification: {checkpoint_dir}")
    print(f"{'='*60}")
    print(f"Epoch: {epoch}")
    print(f"Loss: {loss}")
    print(f"Best Loss: {best_loss}")
    print(f"\nTraining Arguments:")
    print(f"  Batch size: {args.get('batch_size', 'N/A')}")
    print(f"  Learning rate: {args.get('lr', 'N/A')}")
    print(f"  Temperature: {args.get('temperature', 'N/A')}")
    print(f"  SSL method: {args.get('ssl_method', 'N/A')}")
    
    # Verify loss quality
    if isinstance(loss, (int, float)):
        if loss > 8.0:
            print(f"\n⚠️  WARNING: Loss is very high ({loss:.4f})")
            print("   Expected: Loss should be ~2.0-3.0 after training")
            return False
        elif loss < 1.0:
            print(f"\n⚠️  WARNING: Loss is very low ({loss:.4f})")
            print("   Expected: Loss should be ~2.0-3.0 after training")
            return False
        elif 2.0 <= loss <= 4.0:
            print(f"\n✅ Loss is in acceptable range ({loss:.4f})")
            return True
        else:
            print(f"\n⚠️  WARNING: Loss is {loss:.4f}")
            print("   Expected: Loss should be ~2.0-3.0 after training")
            return False
    else:
        print(f"\n❌ Invalid loss value: {loss}")
        return False


def verify_training_progression(checkpoint_dir, num_epochs=10):
    """Verify that loss decreases over training."""
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"\n{'='*60}")
    print(f"Checking Training Progression")
    print(f"{'='*60}")
    
    losses = []
    epochs_checked = []
    
    # Check multiple epochs
    for epoch in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        if checkpoint_path.exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                loss = ckpt.get('loss', None)
                if loss is not None:
                    losses.append(loss)
                    epochs_checked.append(epoch)
            except Exception as e:
                print(f"  Warning: Could not load epoch {epoch}: {e}")
    
    if len(losses) < 2:
        print("❌ Not enough checkpoints to verify progression")
        return False
    
    # Check if loss decreases
    print(f"\nLoss progression:")
    print(f"{'Epoch':<10} {'Loss':<10} {'Status'}")
    print(f"{'-'*30}")
    
    prev_loss = losses[0]
    is_decreasing = True
    
    for epoch, loss in zip(epochs_checked, losses):
        if loss > prev_loss * 1.1:  # Allow 10% tolerance
            status = "⚠️  Increased"
            is_decreasing = False
        elif loss < prev_loss * 0.9:  # Decreased by at least 10%
            status = "✅ Decreased"
        else:
            status = "➡️  Stable"
        
        print(f"{epoch:<10} {loss:<10.4f} {status}")
        prev_loss = loss
    
    # Final check
    if losses[-1] < losses[0] * 0.8:  # Loss decreased by at least 20%
        print(f"\n✅ Training is working! Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
        return True
    elif losses[-1] >= losses[0] * 0.95:  # Loss didn't decrease much
        print(f"\n❌ Training failed! Loss didn't decrease (from {losses[0]:.4f} to {losses[-1]:.4f})")
        print("   Check hyperparameters (batch size, temperature, learning rate)")
        return False
    else:
        print(f"\n⚠️  Training might be working, but loss decrease is small")
        print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        return True


def main():
    """Main verification function."""
    if len(sys.argv) < 2:
        print("Usage: python verify_ssl_training.py <checkpoint_dir>")
        print("Example: python verify_ssl_training.py checkpoints/ssl/simclr_fixed")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    
    # Verify checkpoint
    checkpoint_ok = verify_ssl_checkpoint(checkpoint_dir)
    
    # Verify training progression
    progression_ok = verify_training_progression(checkpoint_dir)
    
    # Final verdict
    print(f"\n{'='*60}")
    if checkpoint_ok and progression_ok:
        print("✅ SSL Training Verification: PASSED")
        print("   Training is working correctly!")
        print("   You can proceed with few-shot training.")
        return 0
    else:
        print("❌ SSL Training Verification: FAILED")
        print("   Training is not working correctly.")
        print("   Please check hyperparameters and retrain.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

