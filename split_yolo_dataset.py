#!/usr/bin/env python3
"""
YOLO Dataset Splitter
Splits PNG images and LabelMe JSON annotations into train/val sets (80-20),
converts LabelMe format to YOLO format, and organizes files accordingly.
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict


def get_image_dimensions_from_json(json_path):
    """Extract image dimensions from LabelMe JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('imageWidth', None), data.get('imageHeight', None)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None, None


def polygon_to_bbox(points):
    """
    Convert polygon/linestrip points to bounding box.
    Returns: (x_min, y_min, x_max, y_max)
    """
    if not points or len(points) == 0:
        return None
    
    # Handle nested list structure: [[x1, y1], [x2, y2], ...]
    if isinstance(points[0], list) and len(points[0]) == 2:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
    else:
        return None
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    return (x_min, y_min, x_max, y_max)


def normalize_coordinates(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO normalized format.
    Returns: (center_x, center_y, width, height) all normalized to 0-1
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center and dimensions
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clamp to [0, 1] range
    center_x_norm = max(0.0, min(1.0, center_x_norm))
    center_y_norm = max(0.0, min(1.0, center_y_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return (center_x_norm, center_y_norm, width_norm, height_norm)


def get_all_labels(json_files):
    """Extract all unique labels from JSON files to create class mapping."""
    labels = set()
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                if label:
                    labels.add(label)
        except Exception as e:
            print(f"Warning: Could not read {json_path}: {e}")
    
    # Sort labels for consistent class IDs
    sorted_labels = sorted(list(labels))
    label_to_class = {label: idx for idx, label in enumerate(sorted_labels)}
    
    print(f"Found {len(sorted_labels)} unique classes:")
    for label, class_id in label_to_class.items():
        print(f"  {class_id}: {label}")
    
    return label_to_class


def convert_labelme_to_yolo(json_path, label_to_class):
    """
    Convert LabelMe JSON to YOLO format.
    Returns: list of YOLO format lines (class_id center_x center_y width height)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')
        
        if img_width is None or img_height is None:
            print(f"Warning: {json_path} missing image dimensions")
            return []
        
        yolo_lines = []
        
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', '')
            
            if not label or not points:
                continue
            
            # Get class ID
            if label not in label_to_class:
                print(f"Warning: Unknown label '{label}' in {json_path}, skipping")
                continue
            
            class_id = label_to_class[label]
            
            # Convert to bounding box
            bbox = polygon_to_bbox(points)
            if bbox is None:
                print(f"Warning: Could not convert shape to bbox in {json_path}")
                continue
            
            # Normalize coordinates
            center_x, center_y, width, height = normalize_coordinates(
                bbox, img_width, img_height
            )
            
            # Format: class_id center_x center_y width height
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        return yolo_lines
    
    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        return []


def split_dataset(source_dir='.', train_ratio=0.8, random_seed=42):
    """
    Main function to split dataset into train/val sets.
    
    Args:
        source_dir: Directory containing PNG and JSON files
        train_ratio: Ratio for training set (default 0.8 = 80%)
        random_seed: Random seed for reproducibility
    """
    source_path = Path(source_dir)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Find all PNG files
    png_files = list(source_path.glob('*.png'))
    json_files = list(source_path.glob('*.json'))
    
    print(f"Found {len(png_files)} PNG files and {len(json_files)} JSON files")
    
    # Create sets for matching
    png_basenames = {f.stem for f in png_files}
    json_basenames = {f.stem for f in json_files}
    
    # Find pairs (PNG with JSON) and unlabeled images
    labeled_pairs = []
    unlabeled_images = []
    
    for png_file in png_files:
        if png_file.stem in json_basenames:
            json_file = source_path / f"{png_file.stem}.json"
            labeled_pairs.append((png_file, json_file))
        else:
            unlabeled_images.append(png_file)
    
    print(f"Found {len(labeled_pairs)} labeled image pairs")
    print(f"Found {len(unlabeled_images)} unlabeled images")
    
    # Get all unique labels and create class mapping
    json_files_for_labels = [pair[1] for pair in labeled_pairs]
    label_to_class = get_all_labels(json_files_for_labels)
    
    # Save class mapping to file
    classes_file = source_path / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        for label, class_id in sorted(label_to_class.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    print(f"\nSaved class mapping to {classes_file}")
    
    # Shuffle labeled pairs for random split
    random.shuffle(labeled_pairs)
    
    # Split into train and val
    split_idx = int(len(labeled_pairs) * train_ratio)
    train_pairs = labeled_pairs[:split_idx]
    val_pairs = labeled_pairs[split_idx:]
    
    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # Create directory structure
    dirs_to_create = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels',
        'unlabel'
    ]
    
    for dir_name in dirs_to_create:
        dir_path = source_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Process training set
    print("\nProcessing training set...")
    for png_file, json_file in train_pairs:
        # Copy image
        dest_image = source_path / 'train' / 'images' / png_file.name
        shutil.copy2(png_file, dest_image)
        
        # Convert and save label
        yolo_lines = convert_labelme_to_yolo(json_file, label_to_class)
        label_file = source_path / 'train' / 'labels' / f"{png_file.stem}.txt"
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines) + '\n')
    
    # Process validation set
    print("Processing validation set...")
    for png_file, json_file in val_pairs:
        # Copy image
        dest_image = source_path / 'val' / 'images' / png_file.name
        shutil.copy2(png_file, dest_image)
        
        # Convert and save label
        yolo_lines = convert_labelme_to_yolo(json_file, label_to_class)
        label_file = source_path / 'val' / 'labels' / f"{png_file.stem}.txt"
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines) + '\n')
    
    # Copy unlabeled images
    print("Copying unlabeled images...")
    for png_file in unlabeled_images:
        dest_image = source_path / 'unlabel' / png_file.name
        shutil.copy2(png_file, dest_image)
    
    print(f"\n✓ Completed! Copied {len(unlabeled_images)} unlabeled images to unlabel/")
    print(f"✓ Training set: {len(train_pairs)} images and labels")
    print(f"✓ Validation set: {len(val_pairs)} images and labels")


if __name__ == '__main__':
    import sys
    
    # Get source directory from command line or use current directory
    source_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print("=" * 60)
    print("YOLO Dataset Splitter")
    print("=" * 60)
    print(f"Source directory: {os.path.abspath(source_dir)}\n")
    
    split_dataset(source_dir=source_dir, train_ratio=0.8, random_seed=42)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

