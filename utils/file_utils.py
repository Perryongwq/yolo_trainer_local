import os
import shutil
import glob
import random

def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception:
            return False
    return True

def get_file_extension(file_path):
    """
    Get the extension of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase) including dot
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def is_image_file(file_path):
    """
    Check if a file is an image based on its extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an image, False otherwise
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    return get_file_extension(file_path) in image_extensions

def is_video_file(file_path):
    """
    Check if a file is a video based on its extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a video, False otherwise
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return get_file_extension(file_path) in video_extensions

def list_image_files(directory, recursive=False):
    """
    List all image files in a directory
    
    Args:
        directory: Directory path
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    if not os.path.isdir(directory):
        return []
    
    if recursive:
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']:
            pattern = os.path.join(directory, '**', f'*.{ext}')
            image_files.extend(glob.glob(pattern, recursive=True))
        return sorted(image_files)
    else:
        return sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if is_image_file(os.path.join(directory, f))
        ])

def list_video_files(directory, recursive=False):
    """
    List all video files in a directory
    
    Args:
        directory: Directory path
        recursive: Whether to search recursively
        
    Returns:
        List of video file paths
    """
    if not os.path.isdir(directory):
        return []
    
    if recursive:
        video_files = []
        for ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']:
            pattern = os.path.join(directory, '**', f'*.{ext}')
            video_files.extend(glob.glob(pattern, recursive=True))
        return sorted(video_files)
    else:
        return sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if is_video_file(os.path.join(directory, f))
        ])

def get_filename_without_extension(file_path):
    """
    Get the filename without extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, copy_files=True):
    """
    Split a dataset into train, val, and test sets
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for the split dataset
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
        copy_files: Whether to copy files (True) or move them (False)
        
    Returns:
        Dictionary with counts of files in each split
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total_ratio <= 1.001:  # Allow for floating point error
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Ensure output directories exist
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        ensure_dir(directory)
        # Create separate directories for images and labels
        ensure_dir(os.path.join(directory, "images"))
        ensure_dir(os.path.join(directory, "labels"))
    
    # List all image files
    image_files = list_image_files(input_dir)
    random.shuffle(image_files)
    
    # Calculate split sizes
    num_files = len(image_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train+num_val]
    test_files = image_files[num_train+num_val:]
    
    # Define a function to copy or move files
    def transfer_file(src, dst):
        if copy_files:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
    
    # Process each split
    splits = {
        "train": (train_files, train_dir),
        "val": (val_files, val_dir),
        "test": (test_files, test_dir)
    }
    
    file_counts = {}
    
    for split_name, (files, split_dir) in splits.items():
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        
        count = 0
        for img_file in files:
            # Transfer image
            dst_img = os.path.join(images_dir, os.path.basename(img_file))
            transfer_file(img_file, dst_img)
            
            # Check for corresponding label file
            filename = get_filename_without_extension(img_file)
            label_file = os.path.join(os.path.dirname(img_file), filename + ".txt")
            
            if os.path.exists(label_file):
                dst_label = os.path.join(labels_dir, filename + ".txt")
                transfer_file(label_file, dst_label)
            
            count += 1
        
        file_counts[split_name] = count
    
    return file_counts

def find_image_label_pairs(images_dir, labels_dir=None):
    """
    Find pairs of images and labels
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels (defaults to same directory)
        
    Returns:
        Dictionary mapping image paths to label paths
    """
    if labels_dir is None:
        labels_dir = images_dir
    
    image_files = list_image_files(images_dir)
    pairs = {}
    
    for img_file in image_files:
        filename = get_filename_without_extension(img_file)
        label_file = os.path.join(labels_dir, filename + ".txt")
        
        if os.path.exists(label_file):
            pairs[img_file] = label_file
    
    return pairs

def create_yolo_dataset_structure(output_dir):
    """
    Create a standard YOLO dataset directory structure
    
    Args:
        output_dir: Output directory
        
    Returns:
        Dictionary with paths to created directories
    """
    dirs = {
        "root": output_dir,
        "train": {
            "images": os.path.join(output_dir, "train", "images"),
            "labels": os.path.join(output_dir, "train", "labels")
        },
        "val": {
            "images": os.path.join(output_dir, "val", "images"),
            "labels": os.path.join(output_dir, "val", "labels")
        },
        "test": {
            "images": os.path.join(output_dir, "test", "images"),
            "labels": os.path.join(output_dir, "test", "labels")
        }
    }
    
    # Create directories
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            ensure_dir(dirs[split][subdir])
    
    return dirs
