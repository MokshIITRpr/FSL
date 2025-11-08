"""
Backbone encoders for sketch feature extraction.

This module implements various encoder architectures optimized for sketch recognition:
- Custom CNN encoder tailored for sketch features
- ResNet-based encoder with modifications for sketch domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SketchEncoder(nn.Module):
    """
    Custom CNN encoder designed specifically for sketch recognition.
    
    Features:
    - Multiple convolutional blocks with batch normalization
    - Residual connections for better gradient flow
    - Adaptive pooling for flexible input sizes
    - Dropout for regularization
    
    Args:
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        embedding_dim (int): Dimension of output embedding vector
        dropout (float): Dropout probability
    """
    
    def __init__(self, input_channels=1, embedding_dim=512, dropout=0.3):
        super(SketchEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Convolutional Block 1: Extract basic sketch features
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Convolutional Block 2: Learn stroke patterns
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 3: Learn complex shapes
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 4: Learn abstract representations
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head for embedding
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Embedding vector of shape (batch_size, embedding_dim)
        """
        # Pass through convolutional blocks
        x = self.conv1(x)  # (B, 64, H/4, W/4)
        x = self.conv2(x)  # (B, 128, H/8, W/8)
        x = self.conv3(x)  # (B, 256, H/16, W/16)
        x = self.conv4(x)  # (B, 512, H/32, W/32)
        
        # Global pooling
        x = self.global_pool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        
        # Project to embedding space
        x = self.projection(x)  # (B, embedding_dim)
        
        return x
    
    def get_features(self, x):
        """
        Extract features before the projection layer.
        Useful for visualization and analysis.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature vector before projection
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder adapted for sketch recognition.
    
    Uses pretrained ResNet as backbone but modifies the first layer
    to accept grayscale sketch images and adds custom projection head.
    
    Args:
        resnet_version (str): ResNet version ('resnet18', 'resnet34', 'resnet50')
        embedding_dim (int): Dimension of output embedding
        pretrained (bool): Whether to use ImageNet pretrained weights
        input_channels (int): Number of input channels
    """
    
    def __init__(self, resnet_version='resnet18', embedding_dim=512, 
                 pretrained=True, input_channels=1):
        super(ResNetEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pretrained ResNet
        if resnet_version == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif resnet_version == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif resnet_version == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        # Modify first conv layer for grayscale input if needed
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                     stride=2, padding=3, bias=False)
        
        # Extract all layers except the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom projection head for embedding space
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through ResNet encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Embedding vector of shape (batch_size, embedding_dim)
        """
        # Extract features using ResNet backbone
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        # Project to embedding space
        x = self.projection(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract features before the projection layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature vector before projection
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def get_encoder(encoder_type='sketch_cnn', **kwargs):
    """
    Factory function to get encoder by type.
    
    Args:
        encoder_type (str): Type of encoder ('sketch_cnn', 'resnet18', 'resnet34', 'resnet50')
        **kwargs: Additional arguments for encoder initialization
        
    Returns:
        nn.Module: Encoder instance
    """
    if encoder_type == 'sketch_cnn':
        return SketchEncoder(**kwargs)
    elif encoder_type in ['resnet18', 'resnet34', 'resnet50']:
        return ResNetEncoder(resnet_version=encoder_type, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

