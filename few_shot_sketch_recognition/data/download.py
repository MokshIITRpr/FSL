"""
Dataset download utilities.

Provides functions to download and prepare TU-Berlin and QuickDraw datasets.
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    Download file from URL with progress bar.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save file
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def download_tuberlin(root_dir, extract=True):
    """
    Download TU-Berlin sketch dataset.
    
    The dataset is available at:
    http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
    
    Args:
        root_dir (str): Directory to download and extract to
        extract (bool): Whether to extract after downloading
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset URL
    url = "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    zip_path = root_path / "sketches_png.zip"
    
    print(f"Downloading TU-Berlin dataset to {zip_path}...")
    print("Note: This is a large file (~1.5GB) and may take some time.")
    
    try:
        download_url(url, str(zip_path))
        print("Download complete!")
        
        if extract:
            print(f"Extracting to {root_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_path)
            print("Extraction complete!")
            
            # Remove zip file to save space
            zip_path.unlink()
            print("Cleaned up zip file.")
            
    except Exception as e:
        print(f"Error downloading TU-Berlin dataset: {e}")
        print("\nPlease download manually from:")
        print("http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip")
        print(f"And extract to: {root_path}")


def download_quickdraw(root_dir, categories=None, max_categories=50):
    """
    Download Google QuickDraw dataset (simplified numpy format).
    
    Downloads .npy files from Google Cloud Storage.
    Each file contains ~100k-200k grayscale 28x28 images for one category.
    
    Args:
        root_dir (str): Directory to save .npy files
        categories (list): List of category names to download (None for popular ones)
        max_categories (int): Maximum number of categories if categories is None
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Base URL for QuickDraw dataset
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    
    # Default popular categories if none specified
    if categories is None:
        categories = [
            'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
            'animal migration', 'ant', 'anvil', 'apple', 'arm',
            'asparagus', 'axe', 'backpack', 'banana', 'bandage',
            'barn', 'baseball', 'baseball bat', 'basket', 'basketball',
            'bat', 'bathtub', 'beach', 'bear', 'beard',
            'bed', 'bee', 'belt', 'bench', 'bicycle',
            'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry',
            'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet',
            'brain', 'bread', 'bridge', 'broccoli', 'broom',
            'bucket', 'bulldozer', 'bus', 'bush', 'butterfly'
        ][:max_categories]
    
    print(f"Downloading {len(categories)} categories from QuickDraw dataset...")
    print(f"Saving to: {root_path}")
    
    for category in categories:
        # Format category name for URL (spaces to %20)
        url_category = category.replace(' ', '%20')
        url = f"{base_url}{url_category}.npy"
        
        output_path = root_path / f"{category}.npy"
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"Skipping {category} (already exists)")
            continue
        
        try:
            print(f"Downloading {category}...")
            download_url(url, str(output_path))
        except Exception as e:
            print(f"Error downloading {category}: {e}")
            continue
    
    print("Download complete!")


def prepare_quickdraw_subset(source_dir, output_dir, n_samples_per_class=5000):
    """
    Create a smaller subset of QuickDraw for faster experimentation.
    
    Args:
        source_dir (str): Directory with full .npy files
        output_dir (str): Directory to save subset
        n_samples_per_class (int): Number of samples per category
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(source_path.glob('*.npy'))
    
    print(f"Creating QuickDraw subset with {n_samples_per_class} samples per class...")
    
    for npy_file in tqdm(npy_files):
        # Load full data
        import numpy as np
        data = np.load(npy_file)
        
        # Sample subset
        if len(data) > n_samples_per_class:
            indices = np.random.choice(len(data), n_samples_per_class, replace=False)
            data = data[indices]
        
        # Save subset
        output_file = output_path / npy_file.name
        np.save(output_file, data)
    
    print(f"Subset saved to: {output_path}")


if __name__ == "__main__":
    """
    Example usage for downloading datasets.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download sketch datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['tuberlin', 'quickdraw'],
                       help='Dataset to download')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--n_categories', type=int, default=50,
                       help='Number of QuickDraw categories to download')
    
    args = parser.parse_args()
    
    if args.dataset == 'tuberlin':
        download_tuberlin(args.output_dir)
    elif args.dataset == 'quickdraw':
        download_quickdraw(args.output_dir, max_categories=args.n_categories)

