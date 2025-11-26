"""
Evaluation script for few-shot sketch recognition.

This script evaluates trained models on test sets with unseen classes,
comparing different methods (SSL pretrained vs from scratch, different few-shot algorithms).
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbone import get_encoder
from models.few_shot import get_few_shot_model
from models.supervised import SupervisedBaseline
from data.datasets import get_dataset
from data.transforms import get_sketch_transforms
from data.samplers import EpisodeSampler
from utils.metrics import compute_few_shot_metrics, compute_accuracy
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.visualization import visualize_few_shot_episode, plot_embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Few-Shot Sketch Recognition')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='tuberlin',
                       choices=['tuberlin', 'quickdraw'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--train_classes', type=int, default=200,
                       help='Number of classes used for training')
    parser.add_argument('--quickdraw_categories', type=str, default=None,
                       help='Comma separated list of QuickDraw categories (default: all)')
    parser.add_argument('--quickdraw_max_samples', type=int, default=10000,
                       help='Max samples per QuickDraw category per split')
    parser.add_argument('--quickdraw_class_split', type=str, default=None,
                       help='Comma separated class counts for train,val,test (e.g., "70,18,21")')
    parser.add_argument('--quickdraw_split_seed', type=int, default=42,
                       help='Random seed for QuickDraw class split shuffling')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--encoder', type=str, default='sketch_cnn',
                       help='Encoder architecture')
    parser.add_argument('--few_shot_model', type=str, default='prototypical',
                       help='Few-shot learning model type')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension')
    
    # Few-shot configuration
    parser.add_argument('--n_way', type=int, default=5,
                       help='Number of classes per episode')
    parser.add_argument('--n_shot', type=int, default=5,
                       help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15,
                       help='Number of query samples per class')
    parser.add_argument('--n_episodes', type=int, default=600,
                       help='Number of test episodes')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize example episodes')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    return parser.parse_args()


def evaluate_few_shot(model, test_sampler, device, n_episodes, visualize=False, 
                     output_dir='results'):
    """
    Evaluate few-shot model on test episodes.
    
    Args:
        model: Few-shot learning model
        test_sampler: Episode sampler
        device: Device to run on
        n_episodes: Number of episodes to evaluate
        visualize: Whether to visualize example episodes
        output_dir: Directory to save visualizations
        
    Returns:
        dict: Evaluation results
    """
    model.eval()
    
    accuracies = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_sampler, desc='Evaluating', total=min(n_episodes, len(test_sampler)))
        
        for i, episode in enumerate(pbar):
            if i >= n_episodes:
                break
            
            # Move data to device
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)
            n_way = episode['n_way']
            n_shot = episode['n_shot']
            
            # Get predictions
            predictions = model.predict(
                support_images, support_labels, query_images, n_way, n_shot
            )
            
            # Compute accuracy
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
            
            # Visualize first few episodes
            if visualize and i < 3:
                os.makedirs(output_dir, exist_ok=True)
                visualize_few_shot_episode(
                    support_images, support_labels,
                    query_images, query_labels,
                    predictions, n_way, n_shot,
                    save_path=f'{output_dir}/episode_{i+1}.png'
                )
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{np.mean(accuracies):.4f}'})
    
    accuracies = np.array(accuracies)
    
    # Compute statistics
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    confidence_interval = 1.96 * std_acc / np.sqrt(len(accuracies))
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'confidence_interval': confidence_interval,
        'accuracies': accuracies,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(log_dir='logs/eval', name='evaluation')
    logger.info(f'Arguments: {args}')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load test dataset
    logger.info(f'Loading test dataset from {args.data_root}')
    test_transform = get_sketch_transforms('test', args.image_size)
    
    dataset_kwargs = {}
    if args.dataset.lower() == 'tuberlin':
        dataset_kwargs['train_classes'] = args.train_classes
    elif args.dataset.lower() == 'quickdraw':
        if args.quickdraw_categories:
            dataset_kwargs['categories'] = [
                cat.strip() for cat in args.quickdraw_categories.split(',')
                if cat.strip()
            ]
        dataset_kwargs['max_samples_per_category'] = args.quickdraw_max_samples
        if args.quickdraw_class_split:
            try:
                split_counts = [
                    int(part.strip()) for part in args.quickdraw_class_split.split(',')
                    if part.strip()
                ]
                if len(split_counts) != 3:
                    raise ValueError
            except ValueError:
                raise ValueError('--quickdraw_class_split must be formatted as train,val,test (e.g., "70,18,21")')
            dataset_kwargs['class_split'] = tuple(split_counts)
            dataset_kwargs['split_seed'] = args.quickdraw_split_seed
        else:
            dataset_kwargs['class_split'] = None
            dataset_kwargs['split_seed'] = args.quickdraw_split_seed
    test_dataset = get_dataset(
        args.dataset,
        args.data_root,
        split='test',
        transform=test_transform,
        **dataset_kwargs
    )
    
    logger.info(f'Test classes: {len(test_dataset.classes)} (unseen during training)')
    logger.info(f'Test samples: {len(test_dataset)}')
    
    # Create episode sampler
    test_sampler = EpisodeSampler(
        test_dataset,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        n_episodes=args.n_episodes
    )
    
    # Create encoder
    logger.info(f'Creating {args.encoder} encoder')
    encoder = get_encoder(
        args.encoder,
        input_channels=1,
        embedding_dim=args.embedding_dim
    )
    
    # Create few-shot model
    logger.info(f'Creating {args.few_shot_model} model')
    model = get_few_shot_model(
        args.few_shot_model,
        encoder=encoder
    )
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    
    logger.info(f'Checkpoint info:')
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_acc' in checkpoint:
        logger.info(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")
    
    # Evaluate
    logger.info(f'\nEvaluating {args.n_way}-way {args.n_shot}-shot classification...')
    logger.info(f'Number of episodes: {args.n_episodes}')
    
    results = evaluate_few_shot(
        model, test_sampler, device, args.n_episodes,
        visualize=args.visualize, output_dir=args.output_dir
    )
    
    # Print results
    logger.info('\n' + '='*60)
    logger.info('EVALUATION RESULTS')
    logger.info('='*60)
    logger.info(f'Configuration: {args.n_way}-way {args.n_shot}-shot')
    logger.info(f'Model: {args.few_shot_model}')
    logger.info(f'Encoder: {args.encoder}')
    logger.info(f'Episodes evaluated: {len(results["accuracies"])}')
    logger.info('-'*60)
    logger.info(f'Mean Accuracy: {results["mean_accuracy"]:.4f} ({results["mean_accuracy"]*100:.2f}%)')
    logger.info(f'Std Accuracy: {results["std_accuracy"]:.4f}')
    logger.info(f'95% Confidence Interval: ±{results["confidence_interval"]:.4f}')
    logger.info('='*60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = f'{args.output_dir}/results_{args.n_way}way_{args.n_shot}shot.txt'
    
    with open(results_file, 'w') as f:
        f.write('='*60 + '\n')
        f.write('EVALUATION RESULTS\n')
        f.write('='*60 + '\n')
        f.write(f'Configuration: {args.n_way}-way {args.n_shot}-shot\n')
        f.write(f'Model: {args.few_shot_model}\n')
        f.write(f'Encoder: {args.encoder}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'Episodes evaluated: {len(results["accuracies"])}\n')
        f.write('-'*60 + '\n')
        f.write(f'Mean Accuracy: {results["mean_accuracy"]:.4f} ({results["mean_accuracy"]*100:.2f}%)\n')
        f.write(f'Std Accuracy: {results["std_accuracy"]:.4f}\n')
        f.write(f'95% Confidence Interval: ±{results["confidence_interval"]:.4f}\n')
        f.write('='*60 + '\n')
    
    logger.info(f'\nResults saved to {results_file}')
    
    if args.visualize:
        logger.info(f'Visualizations saved to {args.output_dir}')


if __name__ == '__main__':
    main()

