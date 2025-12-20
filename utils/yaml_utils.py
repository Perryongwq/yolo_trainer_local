import os
import yaml
import copy

def load_yaml(filepath):
    """
    Load and parse a YAML file
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        Parsed YAML content or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load YAML file: {str(e)}")
        return None

def save_yaml(filepath, content):
    """
    Save content to a YAML file
    
    Args:
        filepath: Path to save the YAML file
        content: Content to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Failed to save YAML file: {str(e)}")
        return False

def validate_yolo_yaml(content):
    """
    Validate YOLO YAML content
    
    Args:
        content: YAML content to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Required fields
    required_fields = ['path', 'train', 'val', 'names']
    
    # Check required fields
    for field in required_fields:
        if field not in content:
            return False, f"Missing required field: {field}"
    
    # Check names field
    if not isinstance(content['names'], (list, dict)):
        return False, "Names field must be a list or dictionary"
    
    # Check paths
    for field in ['train', 'val', 'test']:
        if field in content and not isinstance(content[field], str):
            return False, f"{field} path must be a string"
    
    return True, ""

def create_yolo_yaml(output_path, dataset_path, class_names, format_version=None):
    """
    Create a new YOLO YAML file
    
    Args:
        output_path: Path to save the YAML file
        dataset_path: Path to the dataset
        class_names: List of class names
        format_version: Optional format version (e.g., 11 for YOLOv11)
        
    Returns:
        True if successful, False otherwise
    """
    # Create YAML content
    content = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': class_names
    }
    
    # Add format version if specified
    if format_version:
        content['format_version'] = format_version
        
        # Add YOLOv11 specific settings
        if format_version == 11:
            content['advanced_augmentation'] = True
    
    # Save YAML file
    return save_yaml(output_path, content)

def merge_yaml(base_content, update_content):
    """
    Merge two YAML contents
    
    Args:
        base_content: Base YAML content
        update_content: Content to update with
        
    Returns:
        Merged YAML content
    """
    # Make a deep copy of the base content
    merged = copy.deepcopy(base_content)
    
    # Update with new content
    for key, value in update_content.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge dictionaries
            merged[key] = merge_yaml(merged[key], value)
        else:
            # Replace or add value
            merged[key] = value
    
    return merged

def normalize_paths(content, base_dir):
    """
    Normalize paths in YAML content relative to a base directory
    
    Args:
        content: YAML content
        base_dir: Base directory
        
    Returns:
        YAML content with normalized paths
    """
    # Make a deep copy of the content
    normalized = copy.deepcopy(content)
    
    # Normalize path
    if 'path' in normalized and not os.path.isabs(normalized['path']):
        normalized['path'] = os.path.normpath(os.path.join(base_dir, normalized['path']))
    
    return normalized
