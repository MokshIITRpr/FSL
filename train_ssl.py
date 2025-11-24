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
import warnings

from models.backbone import get_encoder
from models.contrastive import get_ssl_model
from data.datasets import get_dataset, ContrastiveDataset
from data.transforms import get_contrastive_transforms, TwoViewTransform
from utils.metrics import AverageMeter
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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps (simulates larger batch)')
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
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 = single process, recommended for NumPy compatibility)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/ssl',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU mode even if CUDA is available')
    
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, device, ssl_method, epoch, logger, gradient_accumulation_steps=1):
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
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    loss_meter = AverageMeter('loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    optimizer.zero_grad()  # Zero gradients at start
    
    for batch_idx, (view1, view2, _) in enumerate(pbar):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        batch_size_batch = view1.size(0)
        if ssl_method == 'simclr':
            z1, z2 = model(view1, view2)
            loss = model.compute_loss(z1, z2)
        elif ssl_method == 'byol':
            (pred1, pred2), (proj1, proj2) = model(view1, view2)
            loss = model.compute_loss((pred1, pred2), (proj1, proj2))
        
        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Update weights every N steps (gradient accumulation)
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Update target network for BYOL (only after optimizer step)
            if ssl_method == 'byol':
                model.update_target_network()
        
        # Update metrics (scale loss back for logging)
        loss_meter.update(loss.item() * gradient_accumulation_steps, batch_size_batch)
        
        # Clear cache periodically to free memory (less frequently with gradient accumulation)
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # Delete tensors to free memory
        del view1, view2, loss
        if ssl_method == 'simclr':
            del z1, z2
        elif ssl_method == 'byol':
            del pred1, pred2, proj1, proj2
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    # Final optimizer step if there are remaining gradients
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        if ssl_method == 'byol':
            model.update_target_network()
    
    # Final cleanup
    torch.cuda.empty_cache()
    logger.info(f'Epoch {epoch} - Average Loss: {loss_meter.avg:.4f}')
    
    return loss_meter.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(log_dir='logs/ssl', name='ssl_training')
    logger.info(f'Arguments: {args}')
    
    # Set device with compatibility check (GPU-only mode)
    if args.force_cpu:
        device = torch.device('cpu')
        logger.warning('⚠️  CPU mode forced. Training will be slower.')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            logger.error('=' * 80)
            logger.error('ERROR: CUDA is not available!')
            logger.error('Cannot train on GPU. Please check your CUDA installation.')
            logger.error('=' * 80)
            raise RuntimeError('CUDA is not available. Cannot train on GPU.')
        
        is_compatible, gpu_name, fallback_device = check_gpu_compatibility('cuda')
        
        if not is_compatible:
            logger.error('=' * 80)
            logger.error(f'❌ GPU INCOMPATIBILITY ERROR')
            logger.error(f'GPU: {gpu_name} is not compatible with PyTorch {torch.__version__}')
            logger.error(f'PyTorch {torch.__version__} does not support compute capability sm_60 (Tesla P100)')
            logger.error('')
            logger.error('🔧 SOLUTION: Install compatible PyTorch version')
            logger.error('')
            logger.error('Run the setup script to create a compatible environment:')
            logger.error('   bash setup_p100_environment.sh')
            logger.error('')
            logger.error('Or manually:')
            logger.error('   1. Create conda environment with Python 3.11:')
            logger.error('      conda create -n pytorch_p100 python=3.11')
            logger.error('      conda activate pytorch_p100')
            logger.error('')
            logger.error('   2. Install PyTorch 1.13.1 with CUDA 11.7:')
            logger.error('      pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \\')
            logger.error('          --extra-index-url https://download.pytorch.org/whl/cu117')
            logger.error('')
            logger.error('   3. Install other dependencies from requirements.txt')
            logger.error('')
            logger.error('   4. Re-run this training script')
            logger.error('=' * 80)
            raise RuntimeError(f'GPU {gpu_name} is not compatible with PyTorch {torch.__version__}. '
                             f'Please install PyTorch 1.13.1+cu117 to use Tesla P100 GPUs.')
        else:
            device = torch.device('cuda')
            num_gpus = torch.cuda.device_count()
            logger.info('=' * 80)
            logger.info(f'✅ GPU COMPATIBLE: Using {gpu_name}')
            logger.info(f'   PyTorch version: {torch.__version__}')
            logger.info(f'   CUDA version: {torch.version.cuda}')
            logger.info(f'   Number of GPUs: {num_gpus}')
            for i in range(num_gpus):
                logger.info(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
            logger.info('=' * 80)
    else:
        device = torch.device('cpu')
        logger.warning('⚠️  Using CPU mode. Training will be much slower.')
        logger.warning('   To use GPU, set --device cuda')
    
    logger.info(f'Training device: {device}')
    
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
    # Use num_workers=0 by default to avoid NumPy multiprocessing issues
    # User can override with --num_workers flag if needed
    num_workers = args.num_workers
    if num_workers > 0:
        logger.info(f'Using {num_workers} data loading workers')
        logger.warning('If you encounter NumPy errors, try setting --num_workers 0')
    else:
        logger.info('Using single-process data loading (num_workers=0)')
    
    # Adjust batch size for large models to avoid OOM
    # ResNet50 + BYOL uses ~2x memory (online + target networks)
    batch_size = args.batch_size
    original_batch_size = batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    if 'resnet50' in args.encoder.lower() and args.ssl_method == 'byol':
        # ResNet50 + BYOL: Use smaller batch size with gradient accumulation
        # ResNet50 (~25M params) + BYOL (2x networks) needs very small batch on 12GB GPU
        if batch_size > 20:
            batch_size = 20
            # Use gradient accumulation to simulate larger batch (20 * 3 = 60 effective)
            # This reduces iterations from 600 to 200 per epoch = 3x faster!
            if gradient_accumulation_steps == 1:
                gradient_accumulation_steps = 3  # Simulate batch size 60
            logger.warning('=' * 80)
            logger.warning(f'⚠️  MEMORY OPTIMIZATION: ResNet50 + BYOL detected')
            logger.warning(f'   Reducing batch size from {original_batch_size} to {batch_size} to avoid OOM')
            logger.warning(f'   Using gradient accumulation: {gradient_accumulation_steps} steps')
            logger.warning(f'   Effective batch size: {batch_size * gradient_accumulation_steps} = 60')
            logger.warning(f'   ⚡ Training will be 3x faster (200 iters/epoch vs 600 iters/epoch)')
            logger.warning(f'   (BYOL uses 2 networks, ResNet50 is large = high memory usage)')
            logger.warning(f'   💡 TIP: For even faster training, use --encoder resnet18 (2-3x faster)')
            logger.warning('=' * 80)
    elif 'resnet50' in args.encoder.lower():
        # ResNet50 + SimCLR: Can use larger batch (only 1 network)
        if batch_size > 32:
            batch_size = 32
            logger.warning(f'ResNet50 detected: Reducing batch size from {original_batch_size} to {batch_size} to avoid OOM')
    elif 'resnet18' in args.encoder.lower() and args.ssl_method == 'byol':
        # ResNet18 + BYOL: Medium batch size (faster!)
        if batch_size > 32:
            batch_size = 32
            logger.warning(f'ResNet18 + BYOL detected: Reducing batch size from {original_batch_size} to {batch_size} to avoid OOM')
        
    logger.info(f'Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}')
    logger.info(f'Effective batch size: {batch_size * gradient_accumulation_steps}')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to save GPU memory
        drop_last=True
    )
    
    logger.info(f'Using batch size: {batch_size}')
    
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
            args.ssl_method, epoch, logger,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1)
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

