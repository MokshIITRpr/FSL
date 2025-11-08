"""
Training script for self-supervised learning (SimCLR/BYOL).

This script pretrains the encoder using contrastive learning on sketch data,
creating a robust embedding space that generalizes to unseen classes.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.backbone import get_encoder
from models.contrastive import get_ssl_model
from data.datasets import get_dataset, ContrastiveDataset
from data.transforms import get_contrastive_transforms, TwoViewTransform
from utils.metrics import AverageMeter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Self-Supervised Learning for Sketch Recognition')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='tuberlin',
                       choices=['tuberlin', 'quickdraw'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='sketch_cnn',
                       choices=['sketch_cnn', 'resnet18', 'resnet34', 'resnet50'],
                       help='Encoder architecture')
    parser.add_argument('--ssl_method', type=str, default='simclr',
                       choices=['simclr', 'byol'],
                       help='Self-supervised learning method')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Projection dimension for contrastive learning')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for SimCLR')
    parser.add_argument('--ema_decay', type=float, default=0.996,
                       help='EMA decay for BYOL')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/ssl',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, device, ssl_method, epoch, logger):
    """
    Train for one epoch.
    
    Args:
        model: SSL model (SimCLR or BYOL)
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        ssl_method: SSL method name
        epoch: Current epoch number
        logger: Logger instance
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    loss_meter = AverageMeter('loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (view1, view2, _) in enumerate(pbar):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        if ssl_method == 'simclr':
            z1, z2 = model(view1, view2)
            loss = model.compute_loss(z1, z2)
        elif ssl_method == 'byol':
            (pred1, pred2), (proj1, proj2) = model(view1, view2)
            loss = model.compute_loss((pred1, pred2), (proj1, proj2))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target network for BYOL
        if ssl_method == 'byol':
            model.update_target_network()
        
        # Update metrics
        loss_meter.update(loss.item(), view1.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    logger.info(f'Epoch {epoch} - Average Loss: {loss_meter.avg:.4f}')
    
    return loss_meter.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(log_dir='logs/ssl', name='ssl_training')
    logger.info(f'Arguments: {args}')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create data transforms
    transform = TwoViewTransform(
        get_contrastive_transforms(image_size=args.image_size)
    )
    
    # Load dataset
    logger.info(f'Loading {args.dataset} dataset from {args.data_root}')
    base_dataset = get_dataset(
        args.dataset,
        args.data_root,
        split='train',
        transform=None  # Will be applied by ContrastiveDataset
    )
    
    # Wrap in contrastive dataset
    train_dataset = ContrastiveDataset(base_dataset, transform)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f'Training samples: {len(train_dataset)}')
    
    # Create encoder
    logger.info(f'Creating {args.encoder} encoder')
    encoder = get_encoder(
        args.encoder,
        input_channels=1,
        embedding_dim=args.embedding_dim
    )
    
    # Create SSL model
    logger.info(f'Creating {args.ssl_method} model')
    if args.ssl_method == 'simclr':
        model = get_ssl_model(
            'simclr',
            encoder=encoder,
            projection_dim=args.projection_dim,
            temperature=args.temperature
        )
    elif args.ssl_method == 'byol':
        model = get_ssl_model(
            'byol',
            encoder=encoder,
            projection_dim=args.projection_dim,
            moving_average_decay=args.ema_decay
        )
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Training loop
    logger.info('Starting training...')
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train for one epoch
        avg_loss = train_epoch(
            model, train_loader, optimizer, device,
            args.ssl_method, epoch, logger
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Learning rate: {current_lr:.6f}')
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        if epoch % 10 == 0 or is_best:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'args': vars(args)
            }
            
            save_checkpoint(
                state,
                args.checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
                is_best=is_best
            )
    
    logger.info('Training complete!')
    logger.info(f'Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()

