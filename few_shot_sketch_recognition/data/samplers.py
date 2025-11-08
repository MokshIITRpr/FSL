"""
Samplers for few-shot learning episodes.

Implements episodic sampling strategies where each batch contains:
- Support set: N-way K-shot (N classes, K examples per class)
- Query set: Examples from the same N classes to classify
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from collections import defaultdict


class FewShotSampler(Sampler):
    """
    Sampler for few-shot learning episodes.
    
    Creates episodes with N-way K-shot support sets and Q query samples per class.
    
    Args:
        labels (list): List of labels for all samples in dataset
        n_way (int): Number of classes per episode
        n_shot (int): Number of support samples per class
        n_query (int): Number of query samples per class
        n_episodes (int): Number of episodes to sample
    """
    
    def __init__(self, labels, n_way=5, n_shot=5, n_query=15, n_episodes=100):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
        # Ensure each class has enough samples
        for cls in self.classes:
            n_samples = len(self.class_to_indices[cls])
            if n_samples < n_shot + n_query:
                raise ValueError(
                    f"Class {cls} has only {n_samples} samples, "
                    f"but need {n_shot + n_query} (n_shot + n_query)"
                )
    
    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        """
        Generate few-shot episodes.
        
        Yields:
            list: Indices for one episode (support + query samples)
        """
        for _ in range(self.n_episodes):
            # Randomly select N classes
            episode_classes = np.random.choice(
                self.classes, 
                size=self.n_way, 
                replace=False
            )
            
            support_indices = []
            query_indices = []
            
            for cls in episode_classes:
                # Get all indices for this class
                cls_indices = self.class_to_indices[cls].copy()
                np.random.shuffle(cls_indices)
                
                # Select support and query samples
                support = cls_indices[:self.n_shot]
                query = cls_indices[self.n_shot:self.n_shot + self.n_query]
                
                support_indices.extend(support)
                query_indices.extend(query)
            
            # Yield indices for this episode
            yield support_indices + query_indices


class EpisodeSampler:
    """
    Episode generator for few-shot learning.
    
    More flexible than FewShotSampler - generates complete episodes
    with support and query sets organized by class.
    
    Args:
        dataset: PyTorch dataset
        n_way (int): Number of classes per episode
        n_shot (int): Number of support samples per class
        n_query (int): Number of query samples per class
        n_episodes (int): Number of episodes
    """
    
    def __init__(self, dataset, n_way=5, n_shot=5, n_query=15, n_episodes=100):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Get all labels
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
    
    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        """
        Generate episodes.
        
        Yields:
            dict: Episode with keys:
                - 'support_images': Tensor (n_way * n_shot, C, H, W)
                - 'support_labels': Tensor (n_way * n_shot,)
                - 'query_images': Tensor (n_way * n_query, C, H, W)
                - 'query_labels': Tensor (n_way * n_query,)
        """
        for _ in range(self.n_episodes):
            # Randomly select N classes
            episode_classes = np.random.choice(
                self.classes,
                size=self.n_way,
                replace=False
            )
            
            support_images = []
            support_labels = []
            query_images = []
            query_labels = []
            
            for new_label, original_class in enumerate(episode_classes):
                # Get indices for this class
                cls_indices = self.class_to_indices[original_class].copy()
                np.random.shuffle(cls_indices)
                
                # Sample support and query
                support_idx = cls_indices[:self.n_shot]
                query_idx = cls_indices[self.n_shot:self.n_shot + self.n_query]
                
                # Load support samples
                for idx in support_idx:
                    img, _ = self.dataset[idx]
                    support_images.append(img)
                    support_labels.append(new_label)  # Use episode-specific label
                
                # Load query samples
                for idx in query_idx:
                    img, _ = self.dataset[idx]
                    query_images.append(img)
                    query_labels.append(new_label)
            
            # Stack into tensors
            support_images = torch.stack(support_images)
            support_labels = torch.tensor(support_labels)
            query_images = torch.stack(query_images)
            query_labels = torch.tensor(query_labels)
            
            yield {
                'support_images': support_images,
                'support_labels': support_labels,
                'query_images': query_images,
                'query_labels': query_labels,
                'n_way': self.n_way,
                'n_shot': self.n_shot
            }


def collate_episode(batch):
    """
    Custom collate function for episodic batching.
    
    Args:
        batch (list): List of episodes
        
    Returns:
        dict: Batched episode data
    """
    # Batch is already a list of episodes
    # Just return the first episode (batch size = 1 for episodic training)
    return batch[0]

