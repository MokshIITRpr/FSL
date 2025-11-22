"""
Main entry point for the Few-Shot Sketch Recognition Framework.

This script provides a unified interface for:
1. Downloading datasets
2. Training self-supervised models
3. Training few-shot models
4. Evaluating models
5. Running experiments with different configurations
"""

import argparse
import os
import sys


def download_datasets(args):
    """Download sketch datasets."""
    from data.download import download_tuberlin, download_quickdraw
    
    print("=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)
    
    if args.dataset == 'tuberlin' or args.dataset == 'all':
        print("\nDownloading TU-Berlin dataset...")
        download_tuberlin(args.output_dir)
    
    if args.dataset == 'quickdraw' or args.dataset == 'all':
        print("\nDownloading QuickDraw dataset...")
        download_quickdraw(
            os.path.join(args.output_dir, 'quickdraw'),
            max_categories=args.n_categories
        )
    
    print("\nDataset download complete!")


def train_ssl(args):
    """Train self-supervised learning model."""
    print("=" * 60)
    print("SELF-SUPERVISED LEARNING TRAINING")
    print("=" * 60)
    
    # Import and run SSL training
    from train_ssl import main as train_ssl_main
    train_ssl_main()


def train_few_shot(args):
    """Train few-shot learning model."""
    print("=" * 60)
    print("FEW-SHOT LEARNING TRAINING")
    print("=" * 60)
    
    # Import and run few-shot training
    from train_few_shot import main as train_few_shot_main
    train_few_shot_main()


def evaluate_model(args):
    """Evaluate trained model."""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Import and run evaluation
    from evaluate import main as evaluate_main
    evaluate_main()


def run_full_pipeline(args):
    """
    Run the full training and evaluation pipeline.
    
    Steps:
    1. Train SSL model (if not provided)
    2. Train few-shot model using SSL pretrained encoder
    3. Evaluate on test set
    """
    print("=" * 60)
    print("RUNNING FULL PIPELINE")
    print("=" * 60)
    
    # Check if datasets exist
    if not os.path.exists(args.data_root):
        print(f"\nError: Dataset not found at {args.data_root}")
        print("Please download the dataset first using:")
        print(f"  python main.py download --dataset {args.dataset} --output_dir {args.data_root}")
        sys.exit(1)
    
    # Step 1: Train SSL model if not provided
    if args.pretrained_encoder is None:
        print("\n" + "=" * 60)
        print("STEP 1: Training Self-Supervised Model")
        print("=" * 60)
        
        ssl_checkpoint_dir = f'checkpoints/ssl/{args.ssl_method}'
        os.makedirs(ssl_checkpoint_dir, exist_ok=True)
        
        # Build SSL training command
        ssl_cmd = [
            'python', 'train_ssl.py',
            '--dataset', args.dataset,
            '--data_root', args.data_root,
            '--encoder', args.encoder,
            '--ssl_method', args.ssl_method,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.ssl_epochs),
            '--lr', str(args.ssl_lr),
            '--checkpoint_dir', ssl_checkpoint_dir
        ]
        
        print(f"\nRunning command: {' '.join(ssl_cmd)}")
        os.system(' '.join(ssl_cmd))
        
        # Use best model
        args.pretrained_encoder = f'{ssl_checkpoint_dir}/best_model.pth'
    
    print(f"\nUsing pretrained encoder: {args.pretrained_encoder}")
    
    # Step 2: Train few-shot model
    print("\n" + "=" * 60)
    print("STEP 2: Training Few-Shot Model")
    print("=" * 60)
    
    fs_checkpoint_dir = f'checkpoints/few_shot/{args.few_shot_model}'
    os.makedirs(fs_checkpoint_dir, exist_ok=True)
    
    fs_cmd = [
        'python', 'train_few_shot.py',
        '--dataset', args.dataset,
        '--data_root', args.data_root,
        '--encoder', args.encoder,
        '--few_shot_model', args.few_shot_model,
        '--pretrained_encoder', args.pretrained_encoder,
        '--n_way', str(args.n_way),
        '--n_shot', str(args.n_shot),
        '--n_query', str(args.n_query),
        '--epochs', str(args.fs_epochs),
        '--lr', str(args.fs_lr),
        '--checkpoint_dir', fs_checkpoint_dir
    ]
    
    print(f"\nRunning command: {' '.join(fs_cmd)}")
    os.system(' '.join(fs_cmd))
    
    # Step 3: Evaluate
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating Model")
    print("=" * 60)
    
    eval_cmd = [
        'python', 'evaluate.py',
        '--dataset', args.dataset,
        '--data_root', args.data_root,
        '--encoder', args.encoder,
        '--few_shot_model', args.few_shot_model,
        '--checkpoint', f'{fs_checkpoint_dir}/best_model.pth',
        '--n_way', str(args.n_way),
        '--n_shot', str(args.n_shot),
        '--n_query', str(args.n_query),
        '--n_episodes', '600',
        '--visualize',
        '--output_dir', 'results'
    ]
    
    print(f"\nRunning command: {' '.join(eval_cmd)}")
    os.system(' '.join(eval_cmd))
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Few-Shot Sketch Recognition Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download TU-Berlin dataset
  python main.py download --dataset tuberlin --output_dir data/tuberlin
  
  # Run full pipeline
  python main.py pipeline --dataset tuberlin --data_root data/tuberlin
  
  # Train SSL model only
  python main.py train-ssl --dataset tuberlin --data_root data/tuberlin
  
  # Train few-shot model with pretrained encoder
  python main.py train-fs --dataset tuberlin --data_root data/tuberlin \\
      --pretrained_encoder checkpoints/ssl/best_model.pth
  
  # Evaluate model
  python main.py evaluate --dataset tuberlin --data_root data/tuberlin \\
      --checkpoint checkpoints/few_shot/best_model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument('--dataset', type=str, required=True,
                                choices=['tuberlin', 'quickdraw', 'all'],
                                help='Dataset to download')
    download_parser.add_argument('--output_dir', type=str, default='data',
                                help='Output directory')
    download_parser.add_argument('--n_categories', type=int, default=50,
                                help='Number of QuickDraw categories')
    
    # Pipeline command (run everything)
    pipeline_parser = subparsers.add_parser('pipeline', 
                                           help='Run full training pipeline')
    pipeline_parser.add_argument('--dataset', type=str, default='tuberlin')
    pipeline_parser.add_argument('--data_root', type=str, required=True)
    pipeline_parser.add_argument('--encoder', type=str, default='sketch_cnn')
    pipeline_parser.add_argument('--ssl_method', type=str, default='simclr')
    pipeline_parser.add_argument('--few_shot_model', type=str, default='prototypical')
    pipeline_parser.add_argument('--pretrained_encoder', type=str, default=None)
    pipeline_parser.add_argument('--batch_size', type=int, default=32,
                                help='Batch size (use 32 for ResNet50, 64-128 for smaller models)')
    pipeline_parser.add_argument('--ssl_epochs', type=int, default=100)
    pipeline_parser.add_argument('--ssl_lr', type=float, default=0.001)
    pipeline_parser.add_argument('--fs_epochs', type=int, default=50)
    pipeline_parser.add_argument('--fs_lr', type=float, default=0.001)
    pipeline_parser.add_argument('--n_way', type=int, default=5)
    pipeline_parser.add_argument('--n_shot', type=int, default=5)
    pipeline_parser.add_argument('--n_query', type=int, default=15)
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'download':
        download_datasets(args)
    elif args.command == 'train-ssl':
        train_ssl(args)
    elif args.command == 'train-fs':
        train_few_shot(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'pipeline':
        run_full_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

