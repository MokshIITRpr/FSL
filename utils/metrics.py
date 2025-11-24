"""
Evaluation metrics for sketch recognition.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy.
    
    Args:
        predictions (torch.Tensor or np.ndarray): Predicted labels
        targets (torch.Tensor or np.ndarray): Ground truth labels
        
    Returns:
        float: Accuracy (0-1)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    correct = (predictions == targets).sum()
    total = len(targets)
    
    return correct / total


def compute_confusion_matrix(predictions, targets, class_names=None):
    """
    Compute confusion matrix.
    
    Args:
        predictions (torch.Tensor or np.ndarray): Predicted labels
        targets (torch.Tensor or np.ndarray): Ground truth labels
        class_names (list): List of class names
        
    Returns:
        np.ndarray: Confusion matrix
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    cm = confusion_matrix(targets, predictions)
    
    return cm


def compute_few_shot_metrics(model, test_loader, device, n_episodes=100):
    """
    Compute few-shot learning metrics over multiple episodes.
    
    Args:
        model: Few-shot learning model
        test_loader: Data loader for test episodes
        device: Device to run on
        n_episodes (int): Number of episodes to evaluate
        
    Returns:
        dict: Dictionary with metrics (mean_accuracy, std_accuracy, etc.)
    """
    model.eval()
    
    accuracies = []
    
    with torch.no_grad():
        for i, episode in enumerate(test_loader):
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
            
            # Compute accuracy for this episode
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    
    # Compute confidence interval (95%)
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    confidence_interval = 1.96 * std_acc / np.sqrt(len(accuracies))
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'confidence_interval': confidence_interval,
        'accuracies': accuracies
    }


def compute_top_k_accuracy(logits, targets, k=5):
    """
    Compute top-k accuracy.
    
    Args:
        logits (torch.Tensor): Model logits (batch_size, num_classes)
        targets (torch.Tensor): Ground truth labels (batch_size,)
        k (int): Top-k value
        
    Returns:
        float: Top-k accuracy
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct_k.mul_(1.0 / batch_size)
        
    return accuracy.item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name='metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


def evaluate_classification(model, data_loader, device):
    """
    Evaluate a classification model.
    
    Args:
        model: Classification model
        data_loader: Data loader
        device: Device to run on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    accuracy_meter = AverageMeter('accuracy')
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Compute accuracy
            accuracy = compute_accuracy(predictions, targets)
            accuracy_meter.update(accuracy, images.size(0))
            
            # Store predictions
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    return {
        'accuracy': accuracy_meter.avg,
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy()
    }

