"""
Few-shot learning algorithms for sketch recognition.

Implements:
- Prototypical Networks: Classify based on distance to class prototypes
- Matching Networks: Classify using attention over support set
- Relation Networks: Learn a comparison metric

These models enable recognition of new sketch classes with only a few examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-shot Learning.
    
    Classifies query samples based on their distance to class prototypes,
    where each prototype is the mean embedding of support samples from that class.
    
    Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
    
    Args:
        encoder (nn.Module): Backbone encoder to extract features
        distance_metric (str): Distance metric ('euclidean' or 'cosine')
    """
    
    def __init__(self, encoder, distance_metric='euclidean', logit_scale=1.0, feature_norm=False):
        super(PrototypicalNetwork, self).__init__()
        
        self.encoder = encoder
        self.distance_metric = distance_metric
        # Scale applied to logits. For cosine, behaves like temperature (higher -> sharper)
        self.logit_scale = logit_scale
        # If True, L2-normalize embeddings before computing distances/similarities
        self.feature_norm = feature_norm
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """
        Compute class prototypes from support set.
        
        Args:
            support_embeddings (torch.Tensor): Support embeddings (n_support, embedding_dim)
            support_labels (torch.Tensor): Support labels (n_support,)
            n_way (int): Number of classes
            
        Returns:
            torch.Tensor: Class prototypes (n_way, embedding_dim)
        """
        prototypes = []
        
        for class_idx in range(n_way):
            # Get embeddings for this class
            class_mask = (support_labels == class_idx)
            class_embeddings = support_embeddings[class_mask]
            
            # Compute prototype as mean of class embeddings
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_way, embedding_dim)
        return prototypes
    
    def compute_distances(self, query_embeddings, prototypes):
        """
        Compute distances between queries and prototypes.
        
        Args:
            query_embeddings (torch.Tensor): Query embeddings (n_query, embedding_dim)
            prototypes (torch.Tensor): Class prototypes (n_way, embedding_dim)
            
        Returns:
            torch.Tensor: Distance matrix (n_query, n_way)
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||q - p||
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity (kept for compatibility)
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, support_images, support_labels, query_images, n_way, n_shot):
        """
        Forward pass for few-shot classification.
        
        Args:
            support_images (torch.Tensor): Support images (n_way*n_shot, C, H, W)
            support_labels (torch.Tensor): Support labels (n_way*n_shot,)
            query_images (torch.Tensor): Query images (n_query, C, H, W)
            n_way (int): Number of classes
            n_shot (int): Number of examples per class
            
        Returns:
            torch.Tensor: Logits for query images (n_query, n_way)
        """
        # Encode support and query images
        support_embeddings = self.encoder(support_images)  # (n_way*n_shot, embedding_dim)
        query_embeddings = self.encoder(query_images)      # (n_query, embedding_dim)
        
        # Optional feature normalization
        if self.feature_norm:
            support_embeddings = F.normalize(support_embeddings, dim=1)
            query_embeddings = F.normalize(query_embeddings, dim=1)
        
        # Compute class prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)
        
        # Compute logits
        if self.distance_metric == 'cosine':
            # Use scaled cosine similarities as logits
            # Ensure normalized features for cosine if not already
            q = F.normalize(query_embeddings, dim=1)
            p = F.normalize(prototypes, dim=1)
            similarities = torch.mm(q, p.t())  # (n_query, n_way)
            logits = similarities * self.logit_scale
        else:
            # Negative Euclidean distance as logits (optionally scaled)
            distances = self.compute_distances(query_embeddings, prototypes)
            logits = -distances * self.logit_scale
        
        return logits
    
    def predict(self, support_images, support_labels, query_images, n_way, n_shot):
        """
        Predict classes for query images.
        
        Args:
            support_images (torch.Tensor): Support images
            support_labels (torch.Tensor): Support labels
            query_images (torch.Tensor): Query images
            n_way (int): Number of classes
            n_shot (int): Number of examples per class
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        logits = self.forward(support_images, support_labels, query_images, n_way, n_shot)
        predictions = torch.argmax(logits, dim=1)
        return predictions


class AttentionModule(nn.Module):
    """
    Attention mechanism for Matching Networks.
    
    Computes attention weights between query and support samples.
    """
    
    def __init__(self, embedding_dim):
        super(AttentionModule, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1)
        )
    
    def forward(self, query, support):
        """
        Compute attention weights.
        
        Args:
            query (torch.Tensor): Query embedding (embedding_dim,)
            support (torch.Tensor): Support embeddings (n_support, embedding_dim)
            
        Returns:
            torch.Tensor: Attention weights (n_support,)
        """
        n_support = support.shape[0]
        
        # Repeat query for each support sample
        query_repeated = query.unsqueeze(0).repeat(n_support, 1)
        
        # Concatenate query and support
        combined = torch.cat([query_repeated, support], dim=1)
        
        # Compute attention scores
        scores = self.attention(combined).squeeze(1)
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=0)
        
        return weights


class MatchingNetwork(nn.Module):
    """
    Matching Networks for One Shot Learning.
    
    Classifies query samples using attention mechanism over support set,
    allowing the model to adapt to new classes by attending to relevant examples.
    
    Reference: Vinyals et al., "Matching Networks for One Shot Learning", NeurIPS 2016
    
    Args:
        encoder (nn.Module): Backbone encoder
        use_attention (bool): Whether to use attention (True) or simple nearest neighbor (False)
    """
    
    def __init__(self, encoder, use_attention=True):
        super(MatchingNetwork, self).__init__()
        
        self.encoder = encoder
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = AttentionModule(encoder.embedding_dim)
    
    def forward(self, support_images, support_labels, query_images, n_way, n_shot):
        """
        Forward pass for few-shot classification.
        
        Args:
            support_images (torch.Tensor): Support images (n_way*n_shot, C, H, W)
            support_labels (torch.Tensor): Support labels (n_way*n_shot,)
            query_images (torch.Tensor): Query images (n_query, C, H, W)
            n_way (int): Number of classes
            n_shot (int): Number of examples per class
            
        Returns:
            torch.Tensor: Logits for query images (n_query, n_way)
        """
        # Encode support and query images
        support_embeddings = self.encoder(support_images)  # (n_support, embedding_dim)
        query_embeddings = self.encoder(query_images)      # (n_query, embedding_dim)
        
        n_query = query_embeddings.shape[0]
        n_support = support_embeddings.shape[0]
        
        # Initialize logits
        logits = torch.zeros(n_query, n_way, device=query_images.device)
        
        # Process each query
        for q_idx in range(n_query):
            query = query_embeddings[q_idx]
            
            if self.use_attention:
                # Compute attention weights over all support samples
                attention_weights = self.attention(query, support_embeddings)
                
                # Compute weighted sum of one-hot labels
                support_labels_onehot = F.one_hot(support_labels, num_classes=n_way).float()
                logits[q_idx] = torch.mm(
                    attention_weights.unsqueeze(0),
                    support_labels_onehot
                ).squeeze(0)
            else:
                # Simple cosine similarity-based classification
                query_norm = F.normalize(query.unsqueeze(0), dim=1)
                support_norm = F.normalize(support_embeddings, dim=1)
                
                # Compute similarities
                similarities = torch.mm(query_norm, support_norm.t()).squeeze(0)
                
                # Average similarities for each class
                for class_idx in range(n_way):
                    class_mask = (support_labels == class_idx)
                    class_similarities = similarities[class_mask]
                    logits[q_idx, class_idx] = class_similarities.mean()
        
        return logits
    
    def predict(self, support_images, support_labels, query_images, n_way, n_shot):
        """
        Predict classes for query images.
        
        Args:
            support_images (torch.Tensor): Support images
            support_labels (torch.Tensor): Support labels
            query_images (torch.Tensor): Query images
            n_way (int): Number of classes
            n_shot (int): Number of examples per class
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        logits = self.forward(support_images, support_labels, query_images, n_way, n_shot)
        predictions = torch.argmax(logits, dim=1)
        return predictions


class RelationNetwork(nn.Module):
    """
    Relation Networks for few-shot learning.
    
    Learns to compare query samples with support samples using a learned
    relation module (neural network).
    
    Reference: Sung et al., "Learning to Compare: Relation Network 
               for Few-Shot Learning", CVPR 2018
    
    Args:
        encoder (nn.Module): Feature encoder
        relation_dim (int): Hidden dimension for relation module
    """
    
    def __init__(self, encoder, relation_dim=64):
        super(RelationNetwork, self).__init__()
        
        self.encoder = encoder
        
        # Relation module: learns similarity metric
        # Input: concatenated embeddings (2 * embedding_dim)
        # Output: similarity score
        self.relation_module = nn.Sequential(
            nn.Linear(encoder.embedding_dim * 2, relation_dim * 2),
            nn.BatchNorm1d(relation_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim * 2, relation_dim),
            nn.BatchNorm1d(relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, support_images, support_labels, query_images, n_way, n_shot):
        """
        Forward pass for few-shot classification.
        
        Args:
            support_images (torch.Tensor): Support images
            support_labels (torch.Tensor): Support labels
            query_images (torch.Tensor): Query images
            n_way (int): Number of classes
            n_shot (int): Number of examples per class
            
        Returns:
            torch.Tensor: Relation scores (n_query, n_way)
        """
        # Encode support and query images
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)
        
        n_query = query_embeddings.shape[0]
        
        # Compute prototypes for each class
        prototypes = []
        for class_idx in range(n_way):
            class_mask = (support_labels == class_idx)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # (n_way, embedding_dim)
        
        # Compute relation scores
        relation_scores = torch.zeros(n_query, n_way, device=query_images.device)
        
        for q_idx in range(n_query):
            query = query_embeddings[q_idx].unsqueeze(0).repeat(n_way, 1)
            
            # Concatenate query with each prototype
            combined = torch.cat([query, prototypes], dim=1)
            
            # Compute relation scores
            scores = self.relation_module(combined).squeeze(1)
            relation_scores[q_idx] = scores
        
        return relation_scores
    
    def predict(self, support_images, support_labels, query_images, n_way, n_shot):
        """Predict classes for query images."""
        scores = self.forward(support_images, support_labels, query_images, n_way, n_shot)
        predictions = torch.argmax(scores, dim=1)
        return predictions


def get_few_shot_model(model_type='prototypical', encoder=None, **kwargs):
    """
    Factory function to create few-shot learning model.
    
    Args:
        model_type (str): Type of few-shot model ('prototypical', 'matching', 'relation')
        encoder (nn.Module): Backbone encoder
        **kwargs: Additional arguments
        
    Returns:
        nn.Module: Few-shot learning model
    """
    if model_type == 'prototypical':
        allowed = {'distance_metric', 'logit_scale', 'feature_norm'}
        args = {k: v for k, v in kwargs.items() if k in allowed}
        return PrototypicalNetwork(encoder, **args)
    elif model_type == 'matching':
        allowed = {'use_attention'}
        args = {k: v for k, v in kwargs.items() if k in allowed}
        return MatchingNetwork(encoder, **args)
    elif model_type == 'relation':
        allowed = {'relation_dim'}
        args = {k: v for k, v in kwargs.items() if k in allowed}
        return RelationNetwork(encoder, **args)
    else:
        raise ValueError(f"Unknown few-shot model type: {model_type}")

