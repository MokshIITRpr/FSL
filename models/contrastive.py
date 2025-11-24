"""
Self-supervised contrastive learning methods for sketch representation learning.

Implements:
- SimCLR (Simple Framework for Contrastive Learning of Visual Representations)
- BYOL (Bootstrap Your Own Latent)

These methods learn sketch representations without labels by contrasting
different augmented views of the same sketch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Projects embeddings to a space where contrastive loss is computed.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output projection dimension
    """
    
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
    
    Learns representations by maximizing agreement between differently augmented
    views of the same sketch using a contrastive loss (NT-Xent).
    
    Reference: Chen et al., "A Simple Framework for Contrastive Learning 
               of Visual Representations", ICML 2020
    
    Args:
        encoder (nn.Module): Backbone encoder network
        projection_dim (int): Dimension of projection head output
        temperature (float): Temperature parameter for NT-Xent loss
    """
    
    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head: maps embeddings to contrastive space
        self.projection_head = ProjectionHead(
            input_dim=encoder.embedding_dim,
            hidden_dim=2048,
            output_dim=projection_dim
        )
    
    def forward(self, x1, x2):
        """
        Forward pass with two augmented views.
        
        Args:
            x1 (torch.Tensor): First augmented view (batch_size, C, H, W)
            x2 (torch.Tensor): Second augmented view (batch_size, C, H, W)
            
        Returns:
            tuple: (z1, z2) - Projected representations for both views
        """
        # Encode both views
        h1 = self.encoder(x1)  # (B, embedding_dim)
        h2 = self.encoder(x2)  # (B, embedding_dim)
        
        # Project to contrastive space
        z1 = self.projection_head(h1)  # (B, projection_dim)
        z2 = self.projection_head(h2)  # (B, projection_dim)
        
        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def compute_loss(self, z1, z2):
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        
        Args:
            z1 (torch.Tensor): Projections from first view (B, projection_dim)
            z2 (torch.Tensor): Projections from second view (B, projection_dim)
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        batch_size = z1.shape[0]
        
        # Concatenate projections: [z1; z2]
        z = torch.cat([z1, z2], dim=0)  # (2B, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
        
        # Create mask for positive pairs
        # Positive pairs: (i, i+B) and (i+B, i)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # Remove self-similarity (diagonal)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Create positive pair indices
        # For each i in first half, positive is i+B
        # For each i in second half, positive is i-B
        positive_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])
        
        # Extract positive similarities
        positive_sim = sim_matrix[torch.arange(2 * batch_size, device=z.device), 
                                 positive_indices].unsqueeze(1)
        
        # Compute log-sum-exp for all similarities (denominator of softmax)
        # Use logsumexp for numerical stability
        denominator = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        
        # NT-Xent loss: -log(exp(sim_positive) / sum(exp(sim_all)))
        loss = -torch.mean(positive_sim - denominator)
        
        return loss
    
    def get_embedding(self, x):
        """
        Get embedding for inference (without projection head).
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedding vector
        """
        return self.encoder(x)


class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning.
    
    Learns representations using two networks (online and target) without 
    negative pairs. The online network is trained to predict the target network's
    representation, while the target network is updated via exponential moving average.
    
    Reference: Grill et al., "Bootstrap your own latent: A new approach to 
               self-supervised Learning", NeurIPS 2020
    
    Args:
        encoder (nn.Module): Backbone encoder network
        projection_dim (int): Dimension of projection head output
        hidden_dim (int): Hidden dimension in projection/prediction heads
        moving_average_decay (float): EMA decay rate for target network
    """
    
    def __init__(self, encoder, projection_dim=256, hidden_dim=4096, 
                 moving_average_decay=0.996):
        super(BYOL, self).__init__()
        
        self.moving_average_decay = moving_average_decay
        
        # Online network (trainable)
        self.online_encoder = encoder
        self.online_projector = self._build_projector(
            encoder.embedding_dim, hidden_dim, projection_dim
        )
        # Predictor: maps online projections to target space
        self.predictor = self._build_predictor(projection_dim, hidden_dim)
        
        # Target network (updated via EMA)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target network (no gradient computation)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _build_projector(self, input_dim, hidden_dim, output_dim):
        """Build MLP projector."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _build_predictor(self, input_dim, hidden_dim):
        """Build MLP predictor."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )
    
    @torch.no_grad()
    def update_target_network(self):
        """
        Update target network parameters using exponential moving average.
        
        θ_target = τ * θ_target + (1 - τ) * θ_online
        """
        # Update target encoder
        for online_param, target_param in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_param.data = (
                self.moving_average_decay * target_param.data +
                (1 - self.moving_average_decay) * online_param.data
            )
        
        # Update target projector
        for online_param, target_param in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_param.data = (
                self.moving_average_decay * target_param.data +
                (1 - self.moving_average_decay) * online_param.data
            )
    
    def forward(self, x1, x2):
        """
        Forward pass with two augmented views.
        
        Args:
            x1 (torch.Tensor): First augmented view
            x2 (torch.Tensor): Second augmented view
            
        Returns:
            tuple: ((online_pred1, online_pred2), (target_proj1, target_proj2))
        """
        # Online network forward pass
        online_embed1 = self.online_encoder(x1)
        online_embed2 = self.online_encoder(x2)
        
        online_proj1 = self.online_projector(online_embed1)
        online_proj2 = self.online_projector(online_embed2)
        
        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)
        
        # Target network forward pass (no gradient)
        with torch.no_grad():
            target_embed1 = self.target_encoder(x1)
            target_embed2 = self.target_encoder(x2)
            
            target_proj1 = self.target_projector(target_embed1)
            target_proj2 = self.target_projector(target_embed2)
        
        return (online_pred1, online_pred2), (target_proj1, target_proj2)
    
    def compute_loss(self, online_preds, target_projs):
        """
        Compute BYOL loss (mean squared error in normalized space).
        
        Loss is symmetric: L = ||pred1 - proj2||² + ||pred2 - proj1||²
        
        Args:
            online_preds (tuple): (pred1, pred2) from online network
            target_projs (tuple): (proj1, proj2) from target network
            
        Returns:
            torch.Tensor: BYOL loss value
        """
        pred1, pred2 = online_preds
        proj1, proj2 = target_projs
        
        # Normalize predictions and projections
        pred1 = F.normalize(pred1, dim=1)
        pred2 = F.normalize(pred2, dim=1)
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        
        # Compute symmetric loss
        # pred1 should match proj2 (cross-view prediction)
        # pred2 should match proj1 (cross-view prediction)
        loss1 = 2 - 2 * (pred1 * proj2.detach()).sum(dim=1).mean()
        loss2 = 2 - 2 * (pred2 * proj1.detach()).sum(dim=1).mean()
        
        return (loss1 + loss2) / 2
    
    def get_embedding(self, x):
        """
        Get embedding for inference using online encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedding vector
        """
        return self.online_encoder(x)


def get_ssl_model(method='simclr', encoder=None, **kwargs):
    """
    Factory function to create self-supervised learning model.
    
    Args:
        method (str): SSL method ('simclr' or 'byol')
        encoder (nn.Module): Backbone encoder
        **kwargs: Additional arguments for the SSL model
        
    Returns:
        nn.Module: SSL model instance
    """
    if method == 'simclr':
        return SimCLR(encoder, **kwargs)
    elif method == 'byol':
        return BYOL(encoder, **kwargs)
    else:
        raise ValueError(f"Unknown SSL method: {method}")

