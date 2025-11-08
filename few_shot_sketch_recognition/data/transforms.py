"""
Data augmentation and preprocessing transforms for sketches.

Implements sketch-specific augmentations that preserve the structural
properties of hand-drawn sketches while providing variation for
contrastive learning.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import random


class RandomStrokeDropout:
    """
    Randomly drop out portions of sketch strokes.
    
    Simulates partial sketch recognition scenarios and provides
    augmentation for contrastive learning.
    
    Args:
        dropout_prob (float): Probability of dropout for each pixel
    """
    
    def __init__(self, dropout_prob=0.1):
        self.dropout_prob = dropout_prob
    
    def __call__(self, img):
        """
        Apply random stroke dropout to sketch.
        
        Args:
            img (PIL.Image): Input sketch image
            
        Returns:
            PIL.Image: Augmented sketch
        """
        img_array = np.array(img)
        mask = np.random.rand(*img_array.shape) > self.dropout_prob
        img_array = img_array * mask
        return Image.fromarray(img_array.astype(np.uint8))


class RandomDeformation:
    """
    Apply random elastic deformation to sketch.
    
    Simulates natural variation in hand-drawn sketches.
    
    Args:
        alpha (float): Deformation strength
        sigma (float): Gaussian filter sigma
    """
    
    def __init__(self, alpha=10, sigma=3):
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, img):
        """Apply random deformation."""
        # Apply slight rotation and shear
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=255)
        return img


class RandomStrokeThickness:
    """
    Randomly vary stroke thickness.
    
    Simulates variation in drawing pressure and pen width.
    
    Args:
        min_thickness (int): Minimum dilation/erosion
        max_thickness (int): Maximum dilation/erosion
    """
    
    def __init__(self, min_thickness=-1, max_thickness=2):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
    
    def __call__(self, img):
        """Apply random stroke thickness variation."""
        thickness = random.randint(self.min_thickness, self.max_thickness)
        
        if thickness > 0:
            # Dilate (thicken strokes)
            for _ in range(thickness):
                img = img.filter(ImageFilter.MaxFilter(3))
        elif thickness < 0:
            # Erode (thin strokes)
            for _ in range(abs(thickness)):
                img = img.filter(ImageFilter.MinFilter(3))
        
        return img


def get_sketch_transforms(mode='train', image_size=224, augmentation_strength='strong'):
    """
    Get appropriate transforms for sketch data.
    
    Args:
        mode (str): 'train', 'val', or 'test'
        image_size (int): Target image size
        augmentation_strength (str): 'weak', 'medium', or 'strong'
        
    Returns:
        transforms.Compose: Composed transforms
    """
    if mode == 'train':
        if augmentation_strength == 'weak':
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        elif augmentation_strength == 'medium':
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                RandomStrokeThickness(min_thickness=-1, max_thickness=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        else:  # strong
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                RandomStrokeThickness(min_thickness=-1, max_thickness=2),
                RandomStrokeDropout(dropout_prob=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]
    else:
        # Validation and test transforms (no augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    
    return transforms.Compose(transform_list)


def get_contrastive_transforms(image_size=224):
    """
    Get augmentation transforms for contrastive learning.
    
    Creates two different augmented views of the same sketch for
    SimCLR or BYOL training.
    
    Args:
        image_size (int): Target image size
        
    Returns:
        transforms.Compose: Contrastive augmentation pipeline
    """
    # Strong augmentation for contrastive learning
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        RandomStrokeThickness(min_thickness=-1, max_thickness=2),
        RandomStrokeDropout(dropout_prob=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


class TwoViewTransform:
    """
    Generate two different augmented views of the same image.
    
    Used for contrastive learning (SimCLR, BYOL).
    
    Args:
        transform (callable): Transform to apply
    """
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        """
        Generate two views.
        
        Args:
            x: Input image
            
        Returns:
            tuple: (view1, view2)
        """
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2

