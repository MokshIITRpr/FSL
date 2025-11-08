"""
Visualization utilities for analysis and debugging.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_embeddings(embeddings, labels, method='tsne', title='Embedding Visualization',
                   save_path=None, class_names=None):
    """
    Visualize embeddings using dimensionality reduction.
    
    Args:
        embeddings (np.ndarray or torch.Tensor): Embeddings (N, D)
        labels (np.ndarray or torch.Tensor): Labels (N,)
        method (str): 'tsne' or 'pca'
        title (str): Plot title
        save_path (str): Path to save figure (optional)
        class_names (list): List of class names
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names else f'Class {label}'
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[i]], label=label_name, alpha=0.6, s=50)
    
    plt.title(title)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(history, save_path=None):
    """
    Plot training curves (loss, accuracy, etc.).
    
    Args:
        history (dict): Training history with keys like 'train_loss', 'val_loss', etc.
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_few_shot_episode(support_images, support_labels, query_images, 
                               query_labels, predictions=None, n_way=5, n_shot=5,
                               save_path=None):
    """
    Visualize a few-shot learning episode.
    
    Args:
        support_images (torch.Tensor): Support images (n_way*n_shot, C, H, W)
        support_labels (torch.Tensor): Support labels
        query_images (torch.Tensor): Query images
        query_labels (torch.Tensor): Query labels
        predictions (torch.Tensor): Predicted labels (optional)
        n_way (int): Number of classes
        n_shot (int): Number of support samples per class
        save_path (str): Path to save figure
    """
    # Convert to numpy
    if isinstance(support_images, torch.Tensor):
        support_images = support_images.cpu().numpy()
    if isinstance(query_images, torch.Tensor):
        query_images = query_images.cpu().numpy()
    if isinstance(support_labels, torch.Tensor):
        support_labels = support_labels.cpu().numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.cpu().numpy()
    if predictions is not None and isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Create figure
    n_query_display = min(5, len(query_images))
    fig, axes = plt.subplots(n_way + 1, max(n_shot, n_query_display), 
                            figsize=(15, 3 * (n_way + 1)))
    
    if n_way == 1:
        axes = axes.reshape(-1, max(n_shot, n_query_display))
    
    # Plot support set
    for class_idx in range(n_way):
        for shot_idx in range(n_shot):
            img_idx = class_idx * n_shot + shot_idx
            img = support_images[img_idx]
            
            # Handle different image formats
            if img.shape[0] == 1:  # (1, H, W)
                img = img[0]
            elif img.shape[0] == 3:  # (3, H, W)
                img = np.transpose(img, (1, 2, 0))
            
            axes[class_idx, shot_idx].imshow(img, cmap='gray')
            axes[class_idx, shot_idx].axis('off')
            
            if shot_idx == 0:
                axes[class_idx, shot_idx].set_ylabel(f'Class {class_idx}', 
                                                     fontsize=12, fontweight='bold')
            
            if class_idx == 0:
                axes[class_idx, shot_idx].set_title(f'Support {shot_idx + 1}')
        
        # Clear extra subplots in support rows
        for extra_idx in range(n_shot, max(n_shot, n_query_display)):
            axes[class_idx, extra_idx].axis('off')
    
    # Plot query samples
    for query_idx in range(n_query_display):
        img = query_images[query_idx]
        
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        axes[n_way, query_idx].imshow(img, cmap='gray')
        axes[n_way, query_idx].axis('off')
        
        # Add prediction label
        true_label = query_labels[query_idx]
        if predictions is not None:
            pred_label = predictions[query_idx]
            color = 'green' if pred_label == true_label else 'red'
            axes[n_way, query_idx].set_title(
                f'True: {true_label}\nPred: {pred_label}',
                color=color, fontweight='bold'
            )
        else:
            axes[n_way, query_idx].set_title(f'True: {true_label}')
        
        if query_idx == 0:
            axes[n_way, query_idx].set_ylabel('Query', fontsize=12, fontweight='bold')
    
    # Clear extra subplots in query row
    for extra_idx in range(n_query_display, max(n_shot, n_query_display)):
        axes[n_way, extra_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save figure
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

