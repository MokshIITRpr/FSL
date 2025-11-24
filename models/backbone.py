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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEBlock(out_channels) if use_se else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            out = self.se(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SketchEncoder(nn.Module):
    """
    Enhanced CNN encoder with residual connections and attention for sketch recognition.
    
    Features:
    - Deep residual network with squeeze-and-excitation blocks
    - Skip connections for better gradient flow
    - Adaptive pooling for flexible input sizes
    - Dropout and stochastic depth for regularization
    
    Args:
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        embedding_dim (int): Dimension of output embedding vector
        dropout (float): Dropout probability
        width_multiplier (float): Width multiplier for the network (default: 1.0)
    """
    
    def __init__(self, input_channels=1, embedding_dim=512, dropout=0.3, width_multiplier=1.0):
        super(SketchEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.in_channels = int(64 * width_multiplier)
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(int(64 * width_multiplier), 2)
        self.layer2 = self._make_layer(int(128 * width_multiplier), 2, stride=2)
        self.layer3 = self._make_layer(int(256 * width_multiplier), 2, stride=2)
        self.layer4 = self._make_layer(int(512 * width_multiplier), 2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head with more capacity
        self.projection = nn.Sequential(
            nn.Linear(int(512 * width_multiplier), int(512 * width_multiplier)),
            nn.BatchNorm1d(int(512 * width_multiplier)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(512 * width_multiplier), embedding_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [ResBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
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

