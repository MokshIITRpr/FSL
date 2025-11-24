"""
Checkpoint saving and loading utilities.
"""

import torch
from pathlib import Path


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth', is_best=False):
    """
    Save training checkpoint.
    
    Args:
        state (dict): State dictionary with model, optimizer, etc.
        checkpoint_dir (str): Directory to save checkpoint
        filename (str): Checkpoint filename
        is_best (bool): Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device (str): Device to map checkpoint to
        
    Returns:
        dict: Checkpoint state (epoch, metrics, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint

