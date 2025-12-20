import os
import yaml
from utils.event import Event

class DatasetManager:
    """
    Manages dataset operations including YAML file loading, saving, and dataset structure.
    Centralizes dataset-related functionality.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the dataset manager
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        
        # Dataset content from YAML
        self.dataset_content = None
        
        # Events
        self.on_dataset_loaded = Event()
        self.on_dataset_saved = Event()
        self.on_dataset_error = Event()
    
    def load_yaml(self, yaml_path):
        """
        Load a YAML dataset file
        
        Args:
            yaml_path: Path to the YAML file
            
        Returns:
            Dataset content dictionary or None if failed
        """
        try:
            # Try to find the file in various possible locations
            resolved_path = self._resolve_yaml_path(yaml_path)
            if resolved_path:
                yaml_path = resolved_path
            
            # Load YAML file
            with open(yaml_path, 'r') as f:
                self.dataset_content = yaml.safe_load(f)
            
            # Update config
            self.config_manager.set('yaml_path', yaml_path)
            if 'path' in self.dataset_content:
                self.config_manager.set('dataset_path', self.dataset_content['path'])
            
            # Trigger event
            self.on_dataset_loaded(yaml_path, self.dataset_content)
            
            return self.dataset_content
            
        except Exception as e:
            error_msg = f"Failed to load YAML file: {str(e)}"
            self.on_dataset_error(error_msg, e)
            return None
    
    def save_yaml(self, yaml_path, content=None):
        """
        Save dataset content to a YAML file
        
        Args:
            yaml_path: Path to save the YAML file
            content: Optional content to save (uses current dataset_content if None)
            
        Returns:
            Success status (True/False)
        """
        try:
            # Use provided content or current dataset_content
            content_to_save = content if content is not None else self.dataset_content
            
            if content_to_save is None:
                error_msg = "No dataset content to save"
                self.on_dataset_error(error_msg, None)
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
            
            # Save YAML file
            with open(yaml_path, 'w') as f:
                yaml.dump(content_to_save, f, default_flow_style=False, sort_keys=False)
            
            # Update current dataset content if provided
            if content is not None:
                self.dataset_content = content
            
            # Update config
            self.config_manager.set('yaml_path', yaml_path)
            if 'path' in content_to_save:
                self.config_manager.set('dataset_path', content_to_save['path'])
            
            # Trigger event
            self.on_dataset_saved(yaml_path, content_to_save)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to save YAML file: {str(e)}"
            self.on_dataset_error(error_msg, e)
            return False
    
    def create_default_yaml(self, yaml_path, dataset_path=None, classes=None):
        """
        Create a default YAML dataset file
        
        Args:
            yaml_path: Path to save the YAML file
            dataset_path: Optional dataset path (defaults to ./datasets/custom)
            classes: Optional list of classes (defaults to ["class0"])
            
        Returns:
            Success status (True/False)
        """
        # Default values
        dataset_path = dataset_path or "./datasets/custom"
        classes = classes or ["class0"]
        
        # Create content
        content = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': classes
        }
        
        # Save to file
        return self.save_yaml(yaml_path, content)
    
    def get_yaml_path(self):
        """
        Get current YAML file path
        
        Returns:
            YAML file path or empty string if not set
        """
        return self.config_manager.get('yaml_path', '')
    
    def set_yaml_path(self, yaml_path):
        """
        Set YAML file path in config
        
        Args:
            yaml_path: Path to the YAML file
        """
        self.config_manager.set('yaml_path', yaml_path)
    
    def get_dataset_content(self):
        """
        Get current dataset content
        
        Returns:
            Dataset content dictionary or None if not loaded
        """
        return self.dataset_content
    
    def get_classes(self):
        """
        Get class names from dataset
        
        Returns:
            List of class names or empty list if not loaded
        """
        if self.dataset_content and 'names' in self.dataset_content:
            return self.dataset_content['names']
        return []
    
    def validate_dataset_structure(self, dataset_path=None):
        """
        Validate the dataset directory structure
        
        Args:
            dataset_path: Optional dataset path (uses config value if None)
            
        Returns:
            Dictionary with validation results
        """
        # Get dataset path from config if not provided
        if dataset_path is None:
            dataset_path = self.config_manager.get('dataset_path', '')
        
        if not dataset_path or not os.path.isdir(dataset_path):
            return {
                'valid': False,
                'error': 'Dataset directory does not exist',
                'details': {}
            }
        
        # Check expected subdirectories
        expected_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        
        # Optional directories
        optional_dirs = ['images/test', 'labels/test']
        
        # Check directories
        dir_status = {}
        for dir_path in expected_dirs + optional_dirs:
            full_path = os.path.join(dataset_path, dir_path)
            exists = os.path.isdir(full_path)
            is_optional = dir_path in optional_dirs
            
            dir_status[dir_path] = {
                'exists': exists,
                'optional': is_optional,
                'valid': exists or is_optional
            }
        
        # Count images and labels
        counts = {}
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(dataset_path, 'images', split)
            label_dir = os.path.join(dataset_path, 'labels', split)
            
            img_count = len([f for f in os.listdir(img_dir)]) if os.path.isdir(img_dir) else 0
            label_count = len([f for f in os.listdir(label_dir)]) if os.path.isdir(label_dir) else 0
            
            counts[split] = {
                'images': img_count,
                'labels': label_count,
                'match': img_count == label_count
            }
        
        # Overall validation result
        required_valid = all(dir_status[d]['valid'] for d in expected_dirs)
        
        return {
            'valid': required_valid,
            'error': None if required_valid else 'Missing required directories',
            'details': {
                'directories': dir_status,
                'counts': counts
            }
        }
    
    def _resolve_yaml_path(self, yaml_path):
        """
        Try to find the YAML file in various possible locations
        
        Args:
            yaml_path: Original YAML file path
            
        Returns:
            Resolved absolute path or None if not found
        """
        # Check if the path is absolute and exists
        if os.path.isabs(yaml_path) and os.path.isfile(yaml_path):
            return yaml_path
        
        # List of possible locations
        possible_paths = [
            # Exact path provided
            yaml_path,
            # Current working directory
            os.path.join(os.getcwd(), yaml_path),
            # Script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", yaml_path),
            # Default datasets directory
            os.path.join(os.getcwd(), "datasets", yaml_path),
            # Custom datasets directory
            os.path.join(os.getcwd(), "data", yaml_path),
        ]
        
        # Check if any of these paths exist
        for path in possible_paths:
            if os.path.isfile(path):
                return os.path.abspath(path)
        
        return None
