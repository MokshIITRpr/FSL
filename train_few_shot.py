"""
Training script for few-shot learning models.

This script trains few-shot learning models (Prototypical Networks, Matching networks)
on sketch data, either from scratch or using pretrained SSL encoders.
"""

import os
import sys
import time
import math
import random
import argparse
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from models.backbone import get_encoder
from models.few_shot import get_few_shot_model
from data.datasets import get_dataset
from data.transforms import get_sketch_transforms
from data.samplers import EpisodeSampler
from utils.metrics import compute_accuracy, compute_few_shot_metrics, AverageMeter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def check_gpu_compatibility(device='cuda'):
    """
    Check if CUDA device is compatible with current PyTorch installation.
    
    Returns:
        tuple: (is_compatible, device_name, fallback_device)
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return True, 'cpu', 'cpu'
    
    try:
        # Get GPU name
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        
        # Check if it's a Tesla P100 or other sm_60 GPU
        # These GPUs have compute capability 6.0 which is not supported by PyTorch 2.0+
        p100_indicators = ['p100', 'tesla p100', 'gp100']
        is_p100 = any(indicator in gpu_name.lower() for indicator in p100_indicators)
        
        if is_p100:
            # Check PyTorch version compatibility
            # PyTorch 1.13.x and earlier support sm_60 (Pascal architecture)
            # PyTorch 2.0.0+cu117 also supports sm_60 (CUDA 11.7 build includes sm_60)
            # PyTorch 2.0+ with CUDA 12.x does NOT support sm_60
            torch_version = torch.__version__
            if torch_version.startswith('2.0.0+cu117') or torch_version.startswith('2.0+cu117'):
                # PyTorch 2.0.0+cu117 works with Tesla P100
                pass
            elif torch_version.startswith('2.') and '+cu11' in torch_version:
                # PyTorch 2.x with CUDA 11.x might work, test it
                pass
            elif torch_version.startswith('2.') and ('+cu12' in torch_version or '+cu128' in torch_version):
                # PyTorch 2.x with CUDA 12.x does NOT support sm_60
                return False, gpu_name, 'cpu'
            elif torch_version.startswith('1.13') or torch_version.startswith('1.12') or torch_version.startswith('1.11'):
                # PyTorch 1.11-1.13 support sm_60
                pass
            else:
                # Other versions, proceed to test to be safe
                pass
        
        # For other GPUs, try a simple test to verify compatibility
        # We test with BatchNorm since that's where the error typically occurs
        try:
            test_model = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU()
            ).to('cuda')
            
            test_input = torch.randn(1, 1, 32, 32).to('cuda')
            
            # Suppress warnings during test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = test_model(test_input)
            
            # Clean up
            del test_model, test_input
            torch.cuda.empty_cache()
            
            return True, gpu_name, 'cuda'
            
        except (RuntimeError, torch.cuda.DeviceError) as e:
            # This is the actual compatibility test failure
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['no kernel image', 'cudart', 'not compatible', 'cuda capability']):
                # Clean up if needed
                torch.cuda.empty_cache()
                return False, gpu_name, 'cpu'
            else:
                # Unknown error during test, assume compatible but log it
                torch.cuda.empty_cache()
                # Re-raise to see what the actual error is
                raise
        
    except Exception as e:
        # Fallback: if we can't determine compatibility, assume incompatible
        # to avoid crashes during training
        error_msg = str(e).lower()
        if 'cuda' in error_msg or 'gpu' in error_msg:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'Unknown'
            return False, gpu_name, 'cpu'
        else:
            # Unexpected error, re-raise
            raise


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
    parser.add_argument('--logit_scale', type=float, default=10.0,
                       help='Scale/temperature for logits (useful with cosine metric)')
    parser.add_argument('--feature_norm', action='store_true',
                       help='L2-normalize embeddings before computing logits')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights during training')
    
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
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 = single process, recommended for NumPy compatibility)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU mode even if CUDA is available')
    parser.add_argument('--fp16', action='store_true',
                       help='Enable mixed precision training (FP16)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/few_shot',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                       choices=['weak', 'medium', 'strong'],
                       help='Training augmentation strength for sketches')
    parser.add_argument('--results_dir', type=str, default='results_fixed',
                       help='Directory to save evaluation results')
    parser.add_argument('--eval_shots', type=str, default='1,2,3,4,5',
                       help='Comma-separated list of shot counts to evaluate at test time')
    
    return parser.parse_args()


def train_epoch(model, train_sampler, device, optimizer, criterion, scaler, max_grad_norm, epoch, logger, accumulation_steps=4):
    """
    Train for one epoch with mixed precision and gradient accumulation.
    
    Args:
        model: Few-shot learning model
        train_sampler: Episode sampler for training
        device: Device to train on
        optimizer: Optimizer
        criterion: Loss function
        scaler: Gradient scaler for mixed precision
        max_grad_norm: Maximum gradient norm for clipping
        epoch: Current epoch number
        logger: Logger instance
        accumulation_steps: Number of batches to accumulate gradients over
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    loss_meter = AverageMeter('loss')
    accuracy_meter = AverageMeter('accuracy')
    
    optimizer.zero_grad()
    
    # Create progress bar
    pbar = tqdm(train_sampler, desc=f'Epoch {epoch} [Train]', 
                total=len(train_sampler), ncols=100)
    
    for batch_idx, episode in enumerate(pbar):
        # Get data from episode
        support_images = episode['support_images'].to(device, non_blocking=True)
        support_labels = episode['support_labels'].to(device, non_blocking=True)
        query_images = episode['query_images'].to(device, non_blocking=True)
        query_labels = episode['query_labels'].to(device, non_blocking=True)
        n_way = episode['n_way']
        n_shot = episode['n_shot']
        # Move data to device
        support_images = support_images.to(device, non_blocking=True)
        support_labels = support_labels.to(device, non_blocking=True)
        query_images = query_images.to(device, non_blocking=True)
        query_labels = query_labels.to(device, non_blocking=True)
        
        # Get n_way and n_shot from the first sample in the batch
        n_way = len(torch.unique(support_labels))
        n_shot = len(support_labels) // n_way
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Forward pass
            logits = model(support_images, support_labels, query_images, n_way, n_shot)
            
            # Compute loss with gradient accumulation
            loss = criterion(logits, query_labels) / accumulation_steps
        
        # Backward pass with gradient accumulation
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_sampler):
            # Gradient clipping
            if max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step with mixed precision
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Step the scheduler (OneCycleLR updates per batch)
            if hasattr(optimizer, 'scheduler') and optimizer.scheduler is not None:
                optimizer.scheduler.step()
                
            optimizer.zero_grad()
        
        # Update metrics (unscale the loss for logging)
        loss_scaled = loss.item() * accumulation_steps
        predictions = torch.argmax(logits.detach(), dim=1)
        accuracy = compute_accuracy(predictions, query_labels)
        
        batch_size = query_labels.size(0)
        loss_meter.update(loss_scaled, batch_size)
        accuracy_meter.update(accuracy, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}', 
            'acc': f'{accuracy_meter.avg:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Log detailed metrics at the end of the epoch
    logger.info(f'Epoch {epoch} [Train] - Loss: {loss_meter.avg:.4f}, '
                f'Accuracy: {accuracy_meter.avg:.2f}%, '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    # Log to TensorBoard if available
    if hasattr(logger, 'add_scalar'):
        logger.add_scalar('train/loss', loss_meter.avg, epoch)
        logger.add_scalar('train/acc', accuracy_meter.avg, epoch)
        logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    return loss_meter.avg, accuracy_meter.avg


def validate(model, val_sampler, device, criterion, epoch, logger, scaler=None):
    """
    Validate the model.
    
    Args:
        model: Few-shot learning model
        val_sampler: Episode sampler for validation
        device: Device to run on
        criterion: Loss function
        epoch: Current epoch number
        logger: Logger instance
        scaler: Gradient scaler for mixed precision (optional)
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    loss_meter = AverageMeter('loss')
    accuracy_meter = AverageMeter('accuracy')
    
    with torch.no_grad():
        # Create progress bar
        pbar = tqdm(val_sampler, desc=f'Epoch {epoch} [Val]',
                   total=len(val_sampler), ncols=100)
        
        for batch_idx, episode in enumerate(pbar):
            # Get data from episode
            support_images = episode['support_images'].to(device, non_blocking=True)
            support_labels = episode['support_labels'].to(device, non_blocking=True)
            query_images = episode['query_images'].to(device, non_blocking=True)
            query_labels = episode['query_labels'].to(device, non_blocking=True)
            n_way = episode['n_way']
            n_shot = episode['n_shot']
            # Move data to device
            support_images = support_images.to(device, non_blocking=True)
            support_labels = support_labels.to(device, non_blocking=True)
            query_images = query_images.to(device, non_blocking=True)
            query_labels = query_labels.to(device, non_blocking=True)
            
            # Get n_way and n_shot from the first sample in the batch
            n_way = len(torch.unique(support_labels))
            n_shot = len(support_labels) // n_way
            
            # Mixed precision evaluation
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # Forward pass
                logits = model(support_images, support_labels, query_images, n_way, n_shot)
                
                # Compute loss
                loss = criterion(logits, query_labels)
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = compute_accuracy(predictions, query_labels)
            
            # Update metrics
            batch_size = query_labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(accuracy, batch_size)
            
            # Update progress bar every 10% of validation
            if (batch_idx + 1) % max(1, len(val_sampler) // 10) == 0 or (batch_idx + 1) == len(val_sampler):
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{accuracy_meter.avg:.2f}%'
                })
    
    # Log detailed validation metrics
    logger.info(f'Epoch {epoch} [Val] - Loss: {loss_meter.avg:.4f}, '
                f'Accuracy: {accuracy_meter.avg:.2f}%')
    
    # Log to TensorBoard if available
    if hasattr(logger, 'add_scalar'):
        logger.add_scalar('val/loss', loss_meter.avg, epoch)
        logger.add_scalar('val/acc', accuracy_meter.avg, epoch)
    
    return loss_meter.avg, accuracy_meter.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(log_dir='logs/few_shot', name='few_shot_training')
    logger.info(f'Arguments: {args}')
    
    # Force GPU usage if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Always use first GPU
        num_gpus = torch.cuda.device_count()
        logger.info('=' * 80)
        logger.info(f'✅ USING GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'   PyTorch version: {torch.__version__}')
        logger.info(f'   CUDA version: {torch.version.cuda}')
        logger.info(f'   Number of GPUs: {num_gpus}')
        for i in range(num_gpus):
            logger.info(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
        logger.info('=' * 80)
    else:
        device = torch.device('cpu')
        logger.warning('⚠️  CUDA is not available. Using CPU mode. Training will be much slower.')
    
    logger.info(f'Training device: {device}')
    
    # Load datasets
    logger.info(f'Loading {args.dataset} dataset from {args.data_root}')
    
    train_transform = get_sketch_transforms('train', args.image_size, args.augmentation_strength)
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
        import os
        checkpoint_path = args.pretrained_encoder
        if not os.path.exists(checkpoint_path):
            logger.error(f'Checkpoint file not found: {checkpoint_path}')
            logger.error('Please check the path and try again.')
            logger.info('Available checkpoints:')
            # Check common checkpoint directories
            for check_dir in ['checkpoints/ssl', 'checkpoints/test_ssl']:
                if os.path.exists(check_dir):
                    files = [f for f in os.listdir(check_dir) if f.endswith('.pth')]
                    if files:
                        logger.info(f'  {check_dir}: {files[:5]}')  # Show first 5
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        
        logger.info(f'Loading pretrained encoder from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract encoder weights from SSL model
        encoder_state = {}
        model_state = checkpoint.get('model_state_dict', checkpoint)  # Handle both formats
        for key, value in model_state.items():
            if key.startswith('encoder.') or key.startswith('online_encoder.'):
                new_key = key.replace('encoder.', '').replace('online_encoder.', '')
                encoder_state[new_key] = value
        
        if not encoder_state:
            logger.warning('No encoder weights found in checkpoint. Trying to load entire state dict...')
            # Try loading all weights if no encoder prefix found
            encoder.load_state_dict(model_state, strict=False)
        else:
            encoder.load_state_dict(encoder_state, strict=False)
        logger.info('Pretrained encoder loaded successfully')
    
    # Create few-shot model
    logger.info(f'Creating {args.few_shot_model} model')
    model = get_few_shot_model(
        args.few_shot_model,
        encoder=encoder,
        distance_metric=args.distance_metric,
        logit_scale=args.logit_scale,
        feature_norm=args.feature_norm
    )
    model = model.to(device)

    # Optionally freeze encoder parameters
    if args.freeze_encoder and hasattr(model, 'encoder'):
        for p in model.encoder.parameters():
            p.requires_grad = False
        logger.info('Encoder parameters are frozen (feature extractor mode).')
    
    # Optimizer with weight decay and gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduling with warmup, constant phase, and cosine decay
    total_steps = args.epochs * len(train_sampler)
    warmup_steps = int(0.1 * total_steps)  # 10% of training for warmup
    constant_steps = int(0.3 * total_steps)  # 30% of training at constant LR
    
    def lr_lambda(current_step):
        # Linear warmup for the first 10% of training
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Constant learning rate for the next 30% of training
        elif current_step < warmup_steps + constant_steps:
            return 1.0
        
        # Cosine decay for the remaining 60% of training
        else:
            progress = float(current_step - warmup_steps - constant_steps) / \
                      float(max(1, total_steps - warmup_steps - constant_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # Use OneCycleLR for more stable training
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # Same as warmup_steps
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,  # Initial LR = max_lr/25
        final_div_factor=10000.0  # Min LR = initial_lr/10000
    )
    
    # Gradient clipping with higher value for more stable training
    max_grad_norm = 2.0
    
    # Label smoothing with less aggressive smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
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
        model.train()
        train_loss, train_acc = train_epoch(
            model, train_sampler, device, optimizer, criterion, scaler, 
            max_grad_norm, epoch, logger
        )
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = validate(
                model, val_sampler, device, criterion, epoch, logger
            )
        
        # Log learning rate and other metrics at the end of each epoch
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch} Summary:')
        logger.info(f'  Learning Rate: {current_lr:.6e}')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Log to TensorBoard if available
        if hasattr(logger, 'add_scalar'):
            logger.add_scalar('train/loss', train_loss, epoch)
            logger.add_scalar('train/acc', train_acc, epoch)
            logger.add_scalar('val/loss', val_loss, epoch)
            logger.add_scalar('val/acc', val_acc, epoch)
            logger.add_scalar('lr', current_lr, epoch)
            
        # Update learning rate scheduler (OneCycleLR updates per batch, not per epoch)
        # So we don't call scheduler.step() here as it's handled in train_epoch
        
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
    
    # Final evaluation on test set (+ multi-shot sweep)
    logger.info('Evaluating on test set...')
    # Always evaluate at training shot as baseline
    base_test_loss, base_test_acc = validate(model, test_sampler, device, args.epochs, logger)
    logger.info(f'Test Accuracy (n_shot={args.n_shot}): {base_test_acc:.4f}')

    # Prepare results directory and metadata
    results_root = Path(args.results_dir)
    exp_name = f"{args.dataset}_{args.few_shot_model}_{args.encoder}_{args.distance_metric}_w{args.n_way}_q{args.n_query}"
    exp_dir = results_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Parse eval shots list
    try:
        eval_shots = [int(s.strip()) for s in args.eval_shots.split(',') if s.strip()]
    except Exception:
        eval_shots = [1, 2, 3, 4, 5]

    # Evaluate across requested shots
    shot_results = []
    for shot in eval_shots:
        logger.info(f'Evaluating {args.n_way}-way {shot}-shot ...')
        eval_sampler = EpisodeSampler(
            test_dataset,
            n_way=args.n_way,
            n_shot=shot,
            n_query=args.n_query,
            n_episodes=args.n_test_episodes
        )
        metrics = compute_few_shot_metrics(
            model, eval_sampler, device, n_episodes=args.n_test_episodes
        )
        shot_result = {
            'n_way': args.n_way,
            'n_shot': shot,
            'n_query': args.n_query,
            'mean_accuracy': float(metrics['mean_accuracy']),
            'std_accuracy': float(metrics['std_accuracy']),
            'confidence_interval': float(metrics['confidence_interval']),
            'episodes': int(args.n_test_episodes)
        }
        shot_results.append(shot_result)

        # Save per-shot JSON
        out_file = exp_dir / f"results_{args.dataset}_{args.few_shot_model}_{args.encoder}_{args.distance_metric}_w{args.n_way}_s{shot}_q{args.n_query}_ep{args.n_test_episodes}.json"
        with open(out_file, 'w') as f:
            json.dump({
                'args': vars(args),
                'best_val_acc': float(best_val_acc),
                'result': shot_result
            }, f, indent=2)
        logger.info(f'Saved results to {out_file}')

    # Save consolidated summary
    summary_path = exp_dir / f"summary_{args.dataset}_{args.few_shot_model}_{args.encoder}_{args.distance_metric}_w{args.n_way}_q{args.n_query}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'best_val_acc': float(best_val_acc),
            'base_test_acc_at_train_shot': float(base_test_acc),
            'shots': shot_results
        }, f, indent=2)
    logger.info(f'Saved summary to {summary_path}')

    logger.info('Training complete!')
    logger.info(f'Best validation accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()

