"""
Dataset classes for sketch recognition.

Implements data loaders for:
- TU-Berlin sketch dataset
- Google QuickDraw dataset
- Generic sketch datasets
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import random


class SketchDataset(Dataset):
    """
    Generic sketch dataset class.
    
    Loads sketches from a directory structure where each subdirectory
    represents a class.
    
    Directory structure:
        root/
            class1/
                sketch1.png
                sketch2.png
            class2/
                sketch1.png
                sketch2.png
    
    Args:
        root_dir (str): Root directory containing class subdirectories
        transform (callable): Transform to apply to images
        classes (list): List of class names to include (None for all)
        max_samples_per_class (int): Maximum samples per class (None for all)
    """
    
    def __init__(self, root_dir, transform=None, classes=None, 
                 max_samples_per_class=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all class directories
        if classes is None:
            class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
            self.classes = [d.name for d in class_dirs]
        else:
            self.classes = classes
            class_dirs = [self.root_dir / c for c in classes]
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_dir in class_dirs:
            if not class_dir.exists():
                continue
            
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(class_dir.glob(ext))
            
            # Limit samples if specified
            if max_samples_per_class is not None:
                image_paths = image_paths[:max_samples_per_class]
            
            # Add to samples list
            for img_path in image_paths:
                self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name from index."""
        return self.classes[idx]


class TUBerlinDataset(SketchDataset):
    """
    TU-Berlin sketch dataset loader.
    
    The TU-Berlin dataset contains 20,000 sketches across 250 object categories.
    Each category has 80 sketches drawn by different participants.
    
    Dataset reference: "How Do Humans Sketch Objects?" (Eitz et al., SIGGRAPH 2012)
    
    Args:
        root_dir (str): Path to TU-Berlin dataset root
        split (str): 'train', 'val', or 'test'
        transform (callable): Transform to apply
        train_classes (int): Number of classes for training (rest for testing)
    """
    
    def __init__(self, root_dir, split='train', transform=None, train_classes=200):
        self.split = split
        self.train_classes = train_classes
        
        # Get all classes
        root_path = Path(root_dir)
        all_classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
        
        # Split into train/test classes
        if split == 'train':
            classes = all_classes[:train_classes]
            max_samples = 60  # Use 60 samples per class for training
        elif split == 'val':
            classes = all_classes[:train_classes]
            max_samples = 10  # Use 10 samples per class for validation
        else:  # test
            classes = all_classes[train_classes:]  # Unseen classes for testing
            max_samples = None  # Use all samples for testing
        
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            classes=classes,
            max_samples_per_class=max_samples
        )


class QuickDrawDataset(Dataset):
    """
    Google QuickDraw dataset loader.
    
    QuickDraw is a large-scale dataset with 50M drawings across 345 categories.
    We use the simplified bitmap version (28x28 grayscale images).
    
    Dataset reference: https://github.com/googlecreativelab/quickdraw-dataset
    
    Args:
        root_dir (str): Path to QuickDraw numpy files
        categories (list): List of categories to load
        split (str): 'train', 'val', or 'test'
        transform (callable): Transform to apply
        max_samples_per_category (int): Maximum samples per category
    """
    
    def __init__(self, root_dir, categories=None, split='train', 
                 transform=None, max_samples_per_category=10000,
                 class_split=None, split_seed=42):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_samples_per_category = max_samples_per_category
        self.class_split = class_split
        self.split_seed = split_seed
        
        # Detect storage format (npy files or directories with bitmaps)
        self._npy_files = list(self.root_dir.glob('*.npy'))
        self.use_numpy_files = len(self._npy_files) > 0

        if categories is None:
            if self.use_numpy_files:
                categories = [f.stem for f in self._npy_files]
            else:
                categories = [
                    d.name for d in self.root_dir.iterdir()
                    if d.is_dir()
                ]

        categories = sorted(categories)
        if self.class_split:
            self.split_categories = self._create_class_split(categories)
            categories = self.split_categories.get(self.split, [])
        else:
            self.split_categories = None
            if self.use_numpy_files:
                categories = self._filter_missing_numpy(categories)
            else:
                categories = self._filter_missing_dirs(categories)

        self.categories = categories
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.classes = self.categories  # Align attribute name with other datasets
        
        if self.use_numpy_files:
            self._load_numpy_data()
        else:
            self._load_image_paths()

    def _filter_missing_numpy(self, categories):
        available = {f.stem for f in self._npy_files}
        missing = [cat for cat in categories if cat not in available]
        if missing:
            print(f"Warning: Missing QuickDraw npy files for: {missing}")
        return [cat for cat in categories if cat in available]

    def _filter_missing_dirs(self, categories):
        available_dirs = {
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        }
        missing = [cat for cat in categories if cat not in available_dirs]
        if missing:
            print(f"Warning: Missing QuickDraw directories for: {missing}")
        return [cat for cat in categories if cat in available_dirs]

    def _create_class_split(self, categories):
        split_map = {'train': [], 'val': [], 'test': []}
        train_c, val_c, test_c = self.class_split
        total_needed = train_c + val_c + test_c
        if total_needed > len(categories):
            raise ValueError(
                f"Requested class split ({total_needed}) exceeds available categories ({len(categories)})"
            )
        shuffled = categories.copy()
        random.Random(self.split_seed).shuffle(shuffled)
        idx = 0
        split_map['train'] = shuffled[idx:idx + train_c]
        idx += train_c
        split_map['val'] = shuffled[idx:idx + val_c]
        idx += val_c
        split_map['test'] = shuffled[idx:idx + test_c]
        return split_map
    
    def _init_from_numpy(self, categories):
        """Configure categories when numpy files are available."""
        if categories is None:
            categories = [f.stem for f in self._npy_files]
        missing = [
            cat for cat in categories
            if not (self.root_dir / f"{cat}.npy").exists()
        ]
        if missing:
            print(f"Warning: Missing QuickDraw npy files for: {missing}")
        return sorted([cat for cat in categories if (self.root_dir / f"{cat}.npy").exists()])
    
    def _init_from_image_dirs(self, categories):
        """Configure categories when data is stored in class directories."""
        available_dirs = {
            d.name: d for d in self.root_dir.iterdir() if d.is_dir()
        }
        if categories is None:
            categories = list(available_dirs.keys())
        missing = [cat for cat in categories if cat not in available_dirs]
        if missing:
            print(f"Warning: Missing QuickDraw directories for: {missing}")
        return sorted([cat for cat in categories if cat in available_dirs])
    
    def _split_indices(self, n_samples):
        """Return start/end indices for the requested split."""
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        if self.split == 'train':
            start_idx, end_idx = 0, min(train_end, self.max_samples_per_category)
        elif self.split == 'val':
            start_idx = train_end
            end_idx = min(val_end, start_idx + self.max_samples_per_category)
        else:  # test
            start_idx = val_end
            end_idx = min(n_samples, start_idx + self.max_samples_per_category)
        
        return start_idx, end_idx
    
    def _load_numpy_data(self):
        """Load QuickDraw data stored as numpy arrays."""
        self.data = []
        self.labels = []
        
        for category in self.categories:
            npy_path = self.root_dir / f"{category}.npy"
            if not npy_path.exists():
                continue
            
            category_data = np.load(npy_path)
            start_idx, end_idx = self._split_indices(len(category_data))
            category_data = category_data[start_idx:end_idx]
            
            self.data.append(category_data)
            self.labels.extend([self.class_to_idx[category]] * len(category_data))
        
        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
        else:
            self.data = np.array([])
        self.labels = np.array(self.labels)
    
    def _load_image_paths(self):
        """Load QuickDraw data stored as directories of bitmap images."""
        self.samples = []
        
        for category in self.categories:
            class_dir = self.root_dir / category
            if not class_dir.exists():
                continue
            
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                image_paths.extend(class_dir.glob(ext))
            image_paths = sorted(image_paths)
            
            start_idx, end_idx = self._split_indices(len(image_paths))
            image_paths = image_paths[start_idx:end_idx]
            
            class_idx = self.class_to_idx[category]
            for img_path in image_paths:
                self.samples.append((str(img_path), class_idx))
        
        self.labels = np.array([label for _, label in self.samples])
    
    def __len__(self):
        if self.use_numpy_files:
        return len(self.data)
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image, label)
        """
        if self.use_numpy_files:
        img_array = self.data[idx]
        label = self.labels[idx]
        image = Image.fromarray(img_array.astype(np.uint8), mode='L')
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('L')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """Get class name from index."""
        return self.categories[idx]


class ContrastiveDataset(Dataset):
    """
    Wrapper dataset for contrastive learning.
    
    Returns two augmented views of each sample for SimCLR or BYOL training.
    
    Args:
        base_dataset (Dataset): Base dataset to wrap
        transform (callable): Transform that returns two views
    """
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get two augmented views of a sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (view1, view2, label)
        """
        # Get original image and label
        image, label = self.base_dataset[idx]
        
        # If image is already a tensor, convert back to PIL for augmentation
        if isinstance(image, torch.Tensor):
            # Denormalize and convert to PIL
            image = image.squeeze().numpy()
            image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='L')
        
        # Generate two views
        view1, view2 = self.transform(image)
        
        return view1, view2, label


def get_dataset(dataset_name, root_dir, split='train', transform=None, **kwargs):
    """
    Factory function to get dataset by name.
    
    Args:
        dataset_name (str): Name of dataset ('tuberlin', 'quickdraw', 'custom')
        root_dir (str): Root directory of dataset
        split (str): Data split
        transform (callable): Transform to apply
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dataset: Dataset instance
    """
    if dataset_name.lower() == 'tuberlin':
        return TUBerlinDataset(root_dir, split=split, transform=transform, **kwargs)
    elif dataset_name.lower() == 'quickdraw':
        return QuickDrawDataset(root_dir, split=split, transform=transform, **kwargs)
    elif dataset_name.lower() == 'custom':
        return SketchDataset(root_dir, transform=transform, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

