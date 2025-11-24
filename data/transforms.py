"""
Data augmentation and preprocessing transforms for sketches.

Implements sketch-specific augmentations that preserve the structural
properties of hand-drawn sketches while providing variation for
contrastive learning.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops
import random
import math
import cv2


class RandomStrokeDropout:
    """
    Randomly drop out portions of sketch strokes with more natural patterns.
    
    Simulates partial sketch recognition scenarios and provides
    augmentation for contrastive learning.
    
    Args:
        dropout_prob (float): Base probability of dropout for each pixel
        max_patches (int): Maximum number of patches to drop
        min_patch_size (int): Minimum size of each dropout patch
        max_patch_size (int): Maximum size of each dropout patch
    """
    
    def __init__(self, dropout_prob=0.1, max_patches=5, min_patch_size=10, max_patch_size=30):
        self.dropout_prob = dropout_prob
        self.max_patches = max_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
    
    def __call__(self, img):
        """
        Apply random stroke dropout to sketch using patch-based dropout.
        
        Args:
            img (PIL.Image): Input sketch image
            
        Returns:
            PIL.Image: Augmented sketch
        """
        img_array = np.array(img)
        
        # Apply random pixel dropout
        if random.random() < self.dropout_prob:
            mask = np.ones_like(img_array, dtype=bool)
            
            # Add random patch dropouts
            for _ in range(random.randint(1, self.max_patches)):
                h, w = img_array.shape
                patch_size = random.randint(self.min_patch_size, self.max_patch_size)
                y = random.randint(0, h - patch_size)
                x = random.randint(0, w - patch_size)
                mask[y:y+patch_size, x:x+patch_size] = 0
            
            img_array = img_array * mask
            
        return Image.fromarray(img_array.astype(np.uint8))


class RandomElasticDeformation:
    """
    Apply elastic deformation to sketch using random displacement fields.
    
    Simulates natural variation in hand-drawn sketches.
    
    Args:
        alpha (float): Deformation strength
        sigma (float): Gaussian filter sigma
        alpha_affine (float): Affine transformation strength
    """
    
    def __init__(self, alpha=20, sigma=5, alpha_affine=0.03):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
    
    def __call__(self, img):
        """Apply elastic deformation."""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Random affine
        center_square = np.float32((h, w)) // 2
        square_size = min(h, w) // 3
        
        pts1 = np.float32([
            center_square + square_size,
            [center_square[0]+square_size, center_square[1]-square_size],
            center_square - square_size
        ])
        
        pts2 = pts1 + np.random.uniform(
            -self.alpha_affine * square_size,
            self.alpha_affine * square_size,
            size=pts1.shape
        ).astype(np.float32)
        
        M = cv2.getAffineTransform(pts1, pts2)
        img_affine = cv2.warpAffine(img_np, M, (w, h), borderValue=255)
        
        # Random elastic deformation
        shape = img_affine.shape
        shape_size = shape[:2]
        
        # Random displacement fields
        dx = cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1).astype(np.float32),
            ksize=None,
            sigmaX=self.sigma,
            sigmaY=self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1).astype(np.float32),
            ksize=None,
            sigmaX=self.sigma,
            sigmaY=self.sigma
        ) * self.alpha
        
        # Create grid and apply deformation
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        
        # Ensure coordinates are within bounds
        map_x = np.clip(map_x, 0, shape[1] - 1)
        map_y = np.clip(map_y, 0, shape[0] - 1)
        
        # Apply the deformation
        img_deformed = cv2.remap(
            img_affine,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )
        
        return Image.fromarray(img_deformed.astype(np.uint8))


class RandomStrokeThickness:
    """
    Randomly vary stroke thickness with more natural variations.
    
    Simulates variation in drawing pressure and pen width using
    more sophisticated morphological operations.
    
    Args:
        min_thickness (int): Minimum thickness adjustment (-3 to 3)
        max_thickness (int): Maximum thickness adjustment (-3 to 3)
        prob (float): Probability of applying thickness variation
    """
    
    def __init__(self, min_thickness=-2, max_thickness=2, prob=0.8):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.prob = prob
    
    def __call__(self, img):
        """Apply random stroke thickness variation."""
        if random.random() > self.prob:
            return img
            
        img_array = np.array(img)
        thickness = random.randint(self.min_thickness, self.max_thickness)
        kernel_size = 3
        
        if thickness > 0:
            # Dilate (thicken strokes)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            for _ in range(abs(thickness)):
                img_array = cv2.dilate(img_array, kernel, iterations=1)
        elif thickness < 0:
            # Erode (thin strokes)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            for _ in range(abs(thickness)):
                img_array = cv2.erode(img_array, kernel, iterations=1)
        
        return Image.fromarray(img_array.astype(np.uint8))


class RandomInvert:
    """Randomly invert the image with given probability."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.invert(img)
        return img

class RandomJitter:
    """Randomly adjust brightness, contrast and sharpness of the image."""
    def __init__(self, brightness=0.4, contrast=0.4, sharpness=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
    
    def __call__(self, img):
        if self.brightness > 0:
            enhancer = ImageEnhance.Brightness(img)
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = enhancer.enhance(max(0.1, factor))
            
        if self.contrast > 0:
            enhancer = ImageEnhance.Contrast(img)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = enhancer.enhance(max(0.1, factor))
            
        if self.sharpness > 0:
            enhancer = ImageEnhance.Sharpness(img)
            factor = 1.0 + random.uniform(-self.sharpness, self.sharpness)
            img = enhancer.enhance(max(0.1, factor))
            
        return img

def get_sketch_transforms(mode='train', image_size=224, augmentation_strength='strong'):
    """
    Get appropriate transforms for sketch data with advanced augmentations.
    
    Args:
        mode (str): 'train', 'val', or 'test'
        image_size (int): Target image size
        augmentation_strength (str): 'weak', 'medium', or 'strong'
        
    Returns:
        transforms.Compose: Composed transforms
    """
    # Common transforms for all modes
    common_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    
    if mode == 'train':
        if augmentation_strength == 'weak':
            transform_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                RandomInvert(p=0.2),
                RandomJitter(brightness=0.2, contrast=0.2, sharpness=0.1)
            ] + common_transforms
                
        elif augmentation_strength == 'medium':
            transform_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                RandomStrokeThickness(min_thickness=-1, max_thickness=1, prob=0.7),
                RandomJitter(brightness=0.3, contrast=0.3, sharpness=0.2),
                RandomInvert(p=0.3)
            ] + common_transforms
            
        else:  # strong
            transform_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(25),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10
                ),
                RandomElasticDeformation(alpha=20, sigma=5, alpha_affine=0.03),
                RandomStrokeThickness(min_thickness=-2, max_thickness=2, prob=0.8),
                RandomStrokeDropout(dropout_prob=0.1, max_patches=5, min_patch_size=5, max_patch_size=20),
                RandomJitter(brightness=0.4, contrast=0.4, sharpness=0.3),
                RandomInvert(p=0.4),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2)
            ] + common_transforms
    else:
        # Validation and test transforms (minimal augmentation)
        transform_list = common_transforms
    
    return transforms.Compose(transform_list)


def get_contrastive_transforms(image_size=224):
    """
    Get strong augmentation transforms for contrastive learning.
    
    Creates two different augmented views of the same sketch for
    SimCLR or BYOL training with more aggressive augmentations.
    
    Args:
        image_size (int): Target image size
        
    Returns:
        transforms.Compose: Contrastive augmentation pipeline
    """
    # Strong augmentation for contrastive learning
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=15
        ),
        RandomElasticDeformation(alpha=25, sigma=6, alpha_affine=0.04),
        RandomStrokeThickness(min_thickness=-2, max_thickness=3, prob=0.9),
        RandomStrokeDropout(dropout_prob=0.15, max_patches=8, min_patch_size=5, max_patch_size=25),
        RandomInvert(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.RandomApply([transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2)
        )], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.2,
            hue=0.1
        )], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.RandomErasing(
            p=0.5, 
            scale=(0.02, 0.2), 
            ratio=(0.3, 3.3), 
            value=1.0  # White for sketches
        )], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    
    return transforms.Compose(transform_list)


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

