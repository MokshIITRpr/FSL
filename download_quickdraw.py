import os
import requests
import numpy as np
from tqdm import tqdm
import urllib.request
import json
from pathlib import Path

def download_quickdraw(categories, output_dir, max_items_per_category=1000):
    """Download QuickDraw dataset for specified categories.
    
    Args:
        categories (list): List of category names to download
        output_dir (str): Directory to save the dataset
        max_items_per_category (int): Maximum number of images per category to download
    """
    # Updated base URL for QuickDraw dataset
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for category in tqdm(categories, desc="Downloading categories"):
        try:
            # Format the URL and output path
            category_formatted = category.lower().replace(' ', '%20').replace('(', '').replace(')', '').replace(' ', '')
            category_url = f"{base_url}/{category_formatted}.npy"
            output_path = os.path.join(output_dir, category)
            os.makedirs(output_path, exist_ok=True)
            
            # Download the numpy array with better error handling
            print(f"\nDownloading {category}...")
            try:
                urllib.request.urlretrieve(category_url, f"{category}.npy")
            except urllib.error.HTTPError as e:
                print(f"  Could not download {category}: {e}")
                continue
            
            # Load the numpy array
            images = np.load(f"{category}.npy")
            
            # Save each image
            for i, img in enumerate(images[:max_items_per_category]):
                # Reshape the array to 28x28 image
                img = img.reshape(28, 28)
                # Save as PNG
                from PIL import Image
                img = Image.fromarray(img.astype('uint8') * 255)
                img.save(os.path.join(output_path, f"{category}_{i:05d}.png"))
                
            # Clean up the temporary file
            os.remove(f"{category}.npy")
            
            print(f"Downloaded {len(images[:max_items_per_category])} images of {category}")
            
        except Exception as e:
            print(f"Error downloading {category}: {e}")

def get_quickdraw_categories():
    """Get a list of available QuickDraw categories"""
    try:
        response = requests.get("https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt")
        response.raise_for_status()
        return [line.strip() for line in response.text.split('\n') if line.strip()]
    except Exception as e:
        print(f"Error fetching QuickDraw categories: {e}")
        return []

def map_tuberlin_to_quickdraw(tuberlin_categories):
    """Map TU-Berlin categories to QuickDraw categories"""
    quickdraw_categories = []
    
    # Common mappings between TU-Berlin and QuickDraw
    mapping = {
        'airplane': 'airplane',
        'alarm clock': 'alarm clock',
        'angel': 'angel',
        'ant': 'ant',
        'apple': 'apple',
        'arm': 'arm',
        'armchair': 'armchair',
        'axe': 'axe',
        'backpack': 'backpack',
        'banana': 'banana',
        'barn': 'barn',
        'baseball bat': 'baseball bat',
        'basket': 'basket',
        'bathtub': 'bathtub',
        'bear (animal)': 'bear',
        'bed': 'bed',
        'bee': 'bee',
        'beer-mug': 'beer',
        'bell': 'bell',
        'bench': 'bench',
        'bicycle': 'bicycle',
        'binoculars': 'binoculars',
        'book': 'book',
        'bookshelf': 'bookshelf',
        'bottle opener': 'bottle',
        'bowl': 'bowl',
        'bridge': 'bridge',
        'bus': 'bus',
        'butterfly': 'butterfly',
        'cactus': 'cactus',
        'cake': 'cake',
        'calculator': 'calculator',
        'camel': 'camel',
        'camera': 'camera',
        'candle': 'candle',
        'car (sedan)': 'car',
        'castle': 'castle',
        'cat': 'cat',
        'cell phone': 'cell phone',
        'chair': 'chair',
        'church': 'church',
        'cloud': 'cloud',
        'computer monitor': 'computer',
        'computer-mouse': 'mouse',
        'couch': 'couch',
        'cow': 'cow',
        'crab': 'crab',
        'crocodile': 'crocodile',
        'crown': 'crown',
        'cup': 'cup',
        'diamond': 'diamond',
        'dog': 'dog',
        'dolphin': 'dolphin',
        'donut': 'donut',
        'door': 'door',
        'dragon': 'dragon',
        'duck': 'duck',
        'elephant': 'elephant',
        'envelope': 'envelope',
        'eye': 'eye',
        'eyeglasses': 'eyeglasses',
        'face': 'face',
        'fan': 'fan',
        'feather': 'feather',
        'fire hydrant': 'fire hydrant',
        'fish': 'fish',
        'flashlight': 'flashlight',
        'flower with stem': 'flower',
        'flying bird': 'bird',
        'foot': 'foot',
        'fork': 'fork',
        'frog': 'frog',
        'giraffe': 'giraffe',
        'guitar': 'guitar',
        'hamburger': 'hamburger',
        'hammer': 'hammer',
        'hand': 'hand',
        'hat': 'hat',
        'head': 'head',
        'hedgehog': 'hedgehog',
        'helicopter': 'helicopter',
        'helmet': 'helmet',
        'horse': 'horse',
        'hot air balloon': 'hot air balloon',
        'house': 'house',
        'ice-cream-cone': 'ice cream',
        'key': 'key',
        'keyboard': 'keyboard',
        'knife': 'knife',
        'ladder': 'ladder',
        'laptop': 'laptop',
        'leaf': 'leaf',
        'lightbulb': 'light bulb',
        'lion': 'lion',
        'lobster': 'lobster',
        'mailbox': 'mailbox',
        'microphone': 'microphone',
        'monkey': 'monkey',
        'moon': 'moon',
        'mosquito': 'mosquito',
        'motorbike': 'motorcycle',
        'mountain': 'mountain',
        'mouse (animal)': 'mouse',
        'mushroom': 'mushroom',
        'nose': 'nose',
        'octopus': 'octopus',
        'owl': 'owl',
        'palm tree': 'palm tree',
        'panda': 'panda',
        'paper clip': 'paper clip',
        'parachute': 'parachute',
        'penguin': 'penguin',
        'piano': 'piano',
        'pickup truck': 'pickup truck',
        'pig': 'pig',
        'pineapple': 'pineapple',
        'pizza': 'pizza',
        'potted plant': 'potted plant',
        'rabbit': 'rabbit',
        'radio': 'radio',
        'rainbow': 'rainbow',
        'rifle': 'rifle',
        'sailboat': 'sailboat',
        'santa claus': 'santa claus',
        'satellite': 'satellite',
        'scissors': 'scissors',
        'scorpion': 'scorpion',
        'screwdriver': 'screwdriver',
        'shark': 'shark',
        'sheep': 'sheep',
        'ship': 'ship',
        'shoe': 'shoe',
        'skateboard': 'skateboard',
        'skull': 'skull',
        'skyscraper': 'skyscraper',
        'snail': 'snail',
        'snake': 'snake',
        'snowman': 'snowman',
        'sock': 'socks',
        'spider': 'spider',
        'spoon': 'spoon',
        'squirrel': 'squirrel',
        'strawberry': 'strawberry',
        'submarine': 'submarine',
        'suitcase': 'suitcase',
        'sun': 'sun',
        'swan': 'swan',
        'sword': 'sword',
        'table': 'table',
        'teapot': 'teapot',
        'teddy-bear': 'teddy-bear',
        'telephone': 'telephone',
        'television': 'television',
        'tent': 'tent',
        'tiger': 'tiger',
        'toilet': 'toilet',
        'tooth': 'tooth',
        'toothbrush': 'toothbrush',
        'tractor': 'tractor',
        'traffic light': 'traffic light',
        'train': 'train',
        'tree': 'tree',
        'truck': 'truck',
        'umbrella': 'umbrella',
        'vase': 'vase',
        'violin': 'violin',
        'wheel': 'wheel',
        'windmill': 'windmill',
        'wine-bottle': 'wine bottle',
        'zebra': 'zebra'
    }
    
    for category in tuberlin_categories:
        # Try exact match first
        if category in mapping:
            quickdraw_categories.append(mapping[category])
        else:
            # Try case-insensitive match
            matched = False
            for k, v in mapping.items():
                if k.lower() == category.lower():
                    quickdraw_categories.append(v)
                    matched = True
                    break
            if not matched:
                print(f"  No match found for: {category}")
    
    return quickdraw_categories

if __name__ == "__main__":
    # Get the absolute path to the data directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data", "quickdraw")
    
    print("Fetching available QuickDraw categories...")
    all_quickdraw_categories = get_quickdraw_categories()
    
    if not all_quickdraw_categories:
        print("Error: Could not fetch QuickDraw categories. Using default categories...")
        all_quickdraw_categories = [
            'airplane', 'apple', 'banana', 'bicycle', 'bird', 'book', 'car', 'cat', 
            'chair', 'clock', 'dog', 'eye', 'fish', 'flower', 'house', 'moon', 'mountain',
            'pizza', 'shoe', 'smiley face', 'sun', 'tree'
        ]
    
    # Get the list of categories from TU-Berlin
    tuberlin_dir = os.path.join(base_dir, "data", "tuberlin")
    if os.path.exists(tuberlin_dir):
        tuberlin_categories = [d for d in os.listdir(tuberlin_dir) 
                             if os.path.isdir(os.path.join(tuberlin_dir, d))]
        print(f"Found {len(tuberlin_categories)} categories in TU-Berlin dataset")
        
        # Map TU-Berlin categories to QuickDraw categories
        categories = map_tuberlin_to_quickdraw(tuberlin_categories)
        print(f"Mapped to {len(categories)} QuickDraw categories")
    else:
        print("TU-Berlin directory not found. Using default categories...")
        categories = all_quickdraw_categories[:20] 
    
    print("\nStarting QuickDraw download...")
    print(f"Categories to download: {', '.join(categories[:10])}{'...' if len(categories) > 10 else ''}")
    print(f"Saving to: {output_dir}")
    download_quickdraw(categories, output_dir, max_items_per_category=50)
    
    print("\nQuickDraw dataset download complete!")
