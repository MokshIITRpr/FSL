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


def download_tuberlin(root_dir, extract=True, use_kaggle=True):
    """
    Download TU-Berlin sketch dataset from Kaggle or original source.
    
    This function attempts to download the TU-Berlin dataset. By default, it tries
    Kaggle first (requires API credentials), then falls back to the original URL.
    
    Note:
        To use Kaggle API, you need to:
        1. Install kaggle package: pip install kaggle
        2. Get your API credentials from https://www.kaggle.com/settings
        3. Place kaggle.json in ~/.kaggle/ directory
        4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
    
    Args:
        root_dir (str): Directory to download and extract to
        extract (bool): Whether to extract after downloading
        use_kaggle (bool): If True, try Kaggle first; if False, try original URL first
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Kaggle dataset identifier
    kaggle_dataset = "rishikashili/tuberlin"
    
    # Original URL (fallback)
    original_url = "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    
    print(f"Downloading TU-Berlin dataset to {root_path}...")
    print("Note: This is a large file (~1.5GB) and may take some time.")
    
    # Strategy 1: Try Kaggle first if use_kaggle is True
    if use_kaggle:
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            print("\n[Method 1] Attempting to download from Kaggle...")
            print(f"Dataset: https://www.kaggle.com/datasets/{kaggle_dataset}")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Create a temporary directory for download
            temp_download_dir = root_path / ".kaggle_download"
            temp_download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            print(f"Downloading dataset: {kaggle_dataset}")
            api.dataset_download_files(
                kaggle_dataset,
                path=str(temp_download_dir),
                unzip=True,  # Always unzip from Kaggle
                quiet=False
            )
            
            # Move files from temp directory to root_path
            # Kaggle typically creates a subdirectory with the dataset name
            downloaded_dirs = [d for d in temp_download_dir.iterdir() if d.is_dir()]
            downloaded_files = [f for f in temp_download_dir.iterdir() if f.is_file()]
            
            # Move directories and files to root_path
            for item in downloaded_dirs:
                # Check if directory already exists in root_path
                dest = root_path / item.name
                if dest.exists():
                    # Merge contents
                    for src_file in item.rglob("*"):
                        if src_file.is_file():
                            rel_path = src_file.relative_to(item)
                            dest_file = dest / rel_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(src_file), str(dest_file))
                else:
                    shutil.move(str(item), str(dest))
            
            for item in downloaded_files:
                shutil.move(str(item), str(root_path / item.name))
            
            # Clean up temp directory
            if temp_download_dir.exists():
                shutil.rmtree(temp_download_dir)
            
            # Clean up any remaining zip files
            zip_files = list(root_path.glob("*.zip"))
            for zip_file in zip_files:
                zip_file.unlink()
                print(f"Cleaned up {zip_file.name}")
            
            print(f"\n✓ Dataset downloaded successfully from Kaggle to: {root_path}")
            return
            
        except ImportError:
            print("\n⚠ Kaggle package not installed. Trying alternative method...")
            # Continue to fallback method
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠ Failed to download from Kaggle: {error_msg}")
            
            # Check for authentication issues
            if "403" in error_msg or "401" in error_msg or "authentication" in error_msg.lower():
                print("Authentication error - trying fallback method...")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print("Dataset not found on Kaggle - trying fallback method...")
            else:
                print("Unexpected error - trying fallback method...")
    
    # Strategy 2: Try original URL as fallback
    zip_path = root_path / "sketches_png.zip"
    print(f"\n[Method 2] Attempting to download from original source...")
    print(f"URL: {original_url}")
    
    try:
        download_url(original_url, str(zip_path))
        print("Download complete!")
        
        if extract:
            print(f"Extracting to {root_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_path)
            print("Extraction complete!")
            
            # Remove zip file to save space
            zip_path.unlink()
            print("Cleaned up zip file.")
        
        print(f"\n✓ Dataset downloaded successfully from original source to: {root_path}")
        return
        
    except Exception as e:
        print(f"\n✗ Failed to download from original source: {e}")
        
        # If zip file was partially downloaded, clean it up
        if zip_path.exists():
            zip_path.unlink()
    
    # If both methods failed, provide manual instructions
    print("\n" + "="*60)
    print("ERROR: All download methods failed!")
    print("="*60)
    print("\nPlease download manually using one of these options:")
    print("\nOption 1: Kaggle (Recommended)")
    print(f"  1. Visit: https://www.kaggle.com/datasets/{kaggle_dataset}")
    print("  2. Click 'Download' button (requires Kaggle account)")
    print("  3. Extract the zip file to:", root_path)
    print("\nOption 2: Set up Kaggle API")
    print("  1. Install: pip install kaggle")
    print("  2. Get API token from: https://www.kaggle.com/settings")
    print("  3. Place kaggle.json in ~/.kaggle/")
    print("  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    print("  5. Run this script again")
    print("\nOption 3: Original source")
    print(f"  URL: {original_url}")
    print(f"  Extract to: {root_path}")
    print("="*60)
    raise Exception("Failed to download TU-Berlin dataset from all available sources")


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

