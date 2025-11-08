"""
Training script for few-shot learning models.

This script trains few-shot learning models (Prototypical Networks, Matching Networks)
on sketch data, either from scratch or using pretrained SSL encoders.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbone import get_encoder
from models.few_shot import get_few_shot_model
from data.datasets import get_dataset
from data.transforms import get_sketch_transforms
from data.samplers import EpisodeSampler
from utils.metrics import compute_accuracy, compute_few_shot_metrics, AverageMeter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Few-Shot Learning for Sketch Recognition')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='tuberlin',
                       choices=['tuberlin', 'quickdraw'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--train_classes', type=int, default=200,
                       help='Number of classes for training (rest for testing)')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='sketch_cnn',
                       choices=['sketch_cnn', 'resnet18', 'resnet34', 'resnet50'],
                       help='Encoder architecture')
    parser.add_argument('--few_shot_model', type=str, default='prototypical',
                       choices=['prototypical', 'matching', 'relation'],
                       help='Few-shot learning model')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--distance_metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine'],
                       help='Distance metric for Prototypical Networks')
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                       help='Path to pretrained encoder checkpoint (from SSL)')
    
    # Few-shot configuration
    parser.add_argument('--n_way', type=int, default=5,
                       help='Number of classes per episode')
    parser.add_argument('--n_shot', type=int, default=5,
                       help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15,
                       help='Number of query samples per class')
    parser.add_argument('--n_train_episodes', type=int, default=1000,
                       help='Number of training episodes per epoch')
    parser.add_argument('--n_val_episodes', type=int, default=200,
                       help='Number of validation episodes')
    parser.add_argument('--n_test_episodes', type=int, default=600,
                       help='Number of test episodes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/few_shot',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    return parser.parse_args()


def train_epoch(model, train_sampler, device, optimizer, epoch, logger):
    """
    Train for one epoch.
    
    Args:
        model: Few-shot learning model
        train_sampler: Episode sampler for training
        device: Device to train on
        optimizer: Optimizer
        epoch: Current epoch number
        logger: Logger instance
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    loss_meter = AverageMeter('loss')
    accuracy_meter = AverageMeter('accuracy')
    
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(train_sampler, desc=f'Epoch {epoch} [Train]', 
                total=len(train_sampler))
    
    for episode in pbar:
        # Move data to device
        support_images = episode['support_images'].to(device)
        support_labels = episode['support_labels'].to(device)
        query_images = episode['query_images'].to(device)
        query_labels = episode['query_labels'].to(device)
        n_way = episode['n_way']
        n_shot = episode['n_shot']
        
        # Forward pass
        logits = model(support_images, support_labels, query_images, n_way, n_shot)
        
        # Compute loss
        loss = criterion(logits, query_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = compute_accuracy(predictions, query_labels)
        
        # Update metrics
        loss_meter.update(loss.item())
        accuracy_meter.update(accuracy)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{accuracy_meter.avg:.4f}'
        })
    
    logger.info(f'Epoch {epoch} [Train] - Loss: {loss_meter.avg:.4f}, '
                f'Accuracy: {accuracy_meter.avg:.4f}')
    
    return loss_meter.avg, accuracy_meter.avg


def validate(model, val_sampler, device, epoch, logger):
    """
    Validate the model.
    
    Args:
        model: Few-shot learning model
        val_sampler: Episode sampler for validation
        device: Device to run on
        epoch: Current epoch number
        logger: Logger instance
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    loss_meter = AverageMeter('loss')
    accuracy_meter = AverageMeter('accuracy')
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(val_sampler, desc=f'Epoch {epoch} [Val]',
                   total=len(val_sampler))
        
        for episode in pbar:
            # Move data to device
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)
            n_way = episode['n_way']
            n_shot = episode['n_shot']
            
            # Forward pass
            logits = model(support_images, support_labels, query_images, n_way, n_shot)
            
            # Compute loss and accuracy
            loss = criterion(logits, query_labels)
            predictions = torch.argmax(logits, dim=1)
            accuracy = compute_accuracy(predictions, query_labels)
            
            # Update metrics
            loss_meter.update(loss.item())
            accuracy_meter.update(accuracy)
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{accuracy_meter.avg:.4f}'
            })
    
    logger.info(f'Epoch {epoch} [Val] - Loss: {loss_meter.avg:.4f}, '
                f'Accuracy: {accuracy_meter.avg:.4f}')
    
    return loss_meter.avg, accuracy_meter.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(log_dir='logs/few_shot', name='few_shot_training')
    logger.info(f'Arguments: {args}')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load datasets
    logger.info(f'Loading {args.dataset} dataset from {args.data_root}')
    
    train_transform = get_sketch_transforms('train', args.image_size, 'medium')
    val_transform = get_sketch_transforms('val', args.image_size)
    
    train_dataset = get_dataset(
        args.dataset,
        args.data_root,
        split='train',
        transform=train_transform,
        train_classes=args.train_classes
    )
    
    val_dataset = get_dataset(
        args.dataset,
        args.data_root,
        split='val',
        transform=val_transform,
        train_classes=args.train_classes
    )
    
    test_dataset = get_dataset(
        args.dataset,
        args.data_root,
        split='test',
        transform=val_transform,
        train_classes=args.train_classes
    )
    
    logger.info(f'Train classes: {len(train_dataset.classes)}')
    logger.info(f'Test classes: {len(test_dataset.classes)}')
    
    # Create episode samplers
    train_sampler = EpisodeSampler(
        train_dataset,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        n_episodes=args.n_train_episodes
    )
    
    val_sampler = EpisodeSampler(
        val_dataset,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        n_episodes=args.n_val_episodes
    )
    
    test_sampler = EpisodeSampler(
        test_dataset,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        n_episodes=args.n_test_episodes
    )
    
    # Create encoder
    logger.info(f'Creating {args.encoder} encoder')
    encoder = get_encoder(
        args.encoder,
        input_channels=1,
        embedding_dim=args.embedding_dim
    )
    
    # Load pretrained encoder if specified
    if args.pretrained_encoder:
        logger.info(f'Loading pretrained encoder from {args.pretrained_encoder}')
        checkpoint = torch.load(args.pretrained_encoder, map_location=device)
        
        # Extract encoder weights from SSL model
        encoder_state = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.') or key.startswith('online_encoder.'):
                new_key = key.replace('encoder.', '').replace('online_encoder.', '')
                encoder_state[new_key] = value
        
        encoder.load_state_dict(encoder_state, strict=False)
        logger.info('Pretrained encoder loaded successfully')
    
    # Create few-shot model
    logger.info(f'Creating {args.few_shot_model} model')
    model = get_few_shot_model(
        args.few_shot_model,
        encoder=encoder,
        distance_metric=args.distance_metric
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_sampler, device, optimizer, epoch, logger
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_sampler, device, epoch, logger
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        if epoch % 5 == 0 or is_best:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }
            
            save_checkpoint(
                state,
                args.checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
                is_best=is_best
            )
    
    # Final evaluation on test set
    logger.info('Evaluating on test set...')
    test_loss, test_acc = validate(model, test_sampler, device, args.epochs, logger)
    logger.info(f'Test Accuracy: {test_acc:.4f}')
    
    logger.info('Training complete!')
    logger.info(f'Best validation accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()

