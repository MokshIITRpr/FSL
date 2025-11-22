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
    
    # Set device with compatibility check (GPU-only mode)
    if args.device == 'cuda':
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

