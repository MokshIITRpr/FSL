import torch
import torchvision.transforms
import logging
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from models.backbone import get_encoder
from models.few_shot import get_few_shot_model
from data.datasets import get_dataset
from data.samplers import EpisodeSampler
from utils.metrics import AverageMeter

def setup_logger():
    """Set up the logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('model_testing')

def test_model(checkpoint_path, dataset_name, data_root, n_way=5, n_shot=5, n_query=15, n_episodes=600, device='cuda:0'):
    """Test a trained model on the test set."""
    logger = setup_logger()
    
    # Set up device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load checkpoint
    logger.info(f'Loading model from {checkpoint_path}')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.warning(f'Loading with weights_only=True failed: {e}')
        logger.warning('Trying with weights_only=False (use only if you trust the source of this model)')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    logger.info('Initializing model...')
    encoder = get_encoder(
        checkpoint['args']['encoder'],
        input_channels=1,
        embedding_dim=checkpoint['args']['embedding_dim']
    )
    
    model = get_few_shot_model(
        checkpoint['args']['few_shot_model'],
        encoder=encoder,
        distance_metric=checkpoint['args']['distance_metric'],
        logit_scale=checkpoint['args'].get('logit_scale', 1.0),
        feature_norm=checkpoint['args'].get('feature_norm', False)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Set up test dataset and sampler with proper transforms
    logger.info(f'Loading {dataset_name} test dataset...')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    test_dataset = get_dataset(
        dataset_name,
        data_root,
        split='test',
        transform=transform,
        train_classes=checkpoint['args'].get('train_classes')
    )
    
    test_sampler = EpisodeSampler(
        test_dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_episodes=n_episodes
    )
    
    # Test loop
    logger.info(f'Running {n_episodes} test episodes...')
    accuracy_meter = AverageMeter('accuracy')
    
    with torch.no_grad():
        pbar = tqdm(test_sampler, desc='Testing', ncols=100, total=n_episodes)
        for episode in pbar:
            # Get data from episode
            support_images = episode['support_images'].to(device) if torch.is_tensor(episode['support_images']) else \
                           torch.stack([img if torch.is_tensor(img) else torch.from_numpy(np.array(img)).float().div(255) 
                                      for img in episode['support_images']]).to(device)
            
            support_labels = episode['support_labels'].to(device) if torch.is_tensor(episode['support_labels']) else \
                           torch.tensor(episode['support_labels'], device=device)
            
            query_images = episode['query_images'].to(device) if torch.is_tensor(episode['query_images']) else \
                         torch.stack([img if torch.is_tensor(img) else torch.from_numpy(np.array(img)).float().div(255) 
                                    for img in episode['query_images']]).to(device)
            
            query_labels = episode['query_labels'].to(device) if torch.is_tensor(episode['query_labels']) else \
                         torch.tensor(episode['query_labels'], device=device)
            
            # Ensure proper shape for grayscale images (add channel dimension if needed)
            if len(support_images.shape) == 3:
                support_images = support_images.unsqueeze(1)
            if len(query_images.shape) == 3:
                query_images = query_images.unsqueeze(1)
            
            # Forward pass
            logits = model(support_images, support_labels, query_images, n_way, n_shot)
            
            # Compute accuracy
            _, preds = torch.max(logits, 1)
            correct = (preds == query_labels).sum().item()
            accuracy = 100.0 * correct / len(query_labels)
            
            # Update metrics
            accuracy_meter.update(accuracy, len(query_labels))
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{accuracy_meter.avg:.2f}%'})
    
    # Print final results
    logger.info('\n' + '=' * 50)
    logger.info(f'Test Results - {n_way}-way {n_shot}-shot:')
    logger.info(f'  Accuracy: {accuracy_meter.avg:.2f}%')
    logger.info('=' * 50 + '\n')
    
    return accuracy_meter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a few-shot learning model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--dataset', type=str, default='tuberlin',
                      help='Dataset name (default: tuberlin)')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory of the dataset')
    parser.add_argument('--n_way', type=int, default=5,
                      help='Number of classes per episode (default: 5)')
    parser.add_argument('--n_shot', type=int, default=5,
                      help='Number of support examples per class (default: 5)')
    parser.add_argument('--n_query', type=int, default=15,
                      help='Number of query examples per class (default: 15)')
    parser.add_argument('--n_episodes', type=int, default=600,
                      help='Number of test episodes (default: 600)')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use for testing (default: cuda:0)')
    
    args = parser.parse_args()
    
    test_model(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        data_root=args.data_root,
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset', type=str, default='tuberlin',
                        help='Dataset name (default: tuberlin)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--n_way', type=int, default=5,
                        help='Number of classes per episode (default: 5)')
    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of support examples per class (default: 5)')
    parser.add_argument('--n_query', type=int, default=15,
                        help='Number of query examples per class (default: 15)')
    parser.add_argument('--n_episodes', type=int, default=600,
                        help='Number of test episodes (default: 600)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for testing (default: cuda:0)')
    
    args = parser.parse_args()
    
    test_model(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        data_root=args.data_root,
