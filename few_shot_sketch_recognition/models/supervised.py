"""
Supervised baseline model for comparison.

A standard supervised classification model that serves as a baseline
to compare against few-shot learning approaches.
"""

import torch
import torch.nn as nn


class SupervisedBaseline(nn.Module):
    """
    Supervised classification baseline.
    
    Standard supervised learning model with encoder + classifier.
    Used as a baseline to compare against few-shot methods.
    
    Args:
        encoder (nn.Module): Backbone encoder
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
    """
    
    def __init__(self, encoder, num_classes, dropout=0.5):
        super(SupervisedBaseline, self).__init__()
        
        self.encoder = encoder
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.embedding_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images (batch_size, C, H, W)
            
        Returns:
            torch.Tensor: Class logits (batch_size, num_classes)
        """
        # Extract features
        features = self.encoder(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_embedding(self, x):
        """
        Get embedding vector (useful for transfer learning).
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Embedding vectors
        """
        return self.encoder(x)
    
    def predict(self, x):
        """
        Predict class labels.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions

