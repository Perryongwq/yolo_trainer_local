import os
import json
from utils.event import Event

class ConfigManager:
    """
    Central configuration management system for the application.
    Manages application settings and provides a clean interface for updates.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the configuration manager
        
        Args:
            config_file: Optional path to a configuration file
        """
        # Default configuration values
        self._config = {
            # Dataset settings
            "dataset_path": "",
            "yaml_path": "",
            
            # Model settings
            "model_path": "yolo11l.pt",
            
            # Training parameters
            "training": {
                "epochs": 100,
                "imgsz": 640,
                "lr0": 0.001,
                "lrf": 0.2,
                "patience": 0,
                "optimizer": "Adam",
                "batch_size": 16,
                "pretrained": True,
                "save_period": 0,
                "device": "auto"
            },
            
            # Evaluation settings
            "evaluation": {
                "confidence_threshold": 0.4,
                "show_labels": True,
                "show_confidence": True,
                "show_measurements": True
            },
            
            # Annotation settings
            "annotation": {
                "confidence_threshold": 0.25,
                "mode": "yolo"  # yolo, sam, or hybrid
            }
        }
        
        # Events
        self.on_config_changed = Event()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def get(self, key, default=None):
        """
        Get a configuration value
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """
        Set a configuration value
        
        Args:
            key: Configuration key to set
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the correct nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Trigger the event
        self.on_config_changed(key, value)
    
    def get_training_params(self):
        """
        Get all training parameters as a dictionary
        
        Returns:
            Dictionary of training parameters
        """
        return self._config.get('training', {}).copy()
    
    def get_evaluation_params(self):
        """
        Get all evaluation parameters as a dictionary
        
        Returns:
            Dictionary of evaluation parameters
        """
        return self._config.get('evaluation', {}).copy()
    
    def get_annotation_params(self):
        """
        Get all annotation parameters as a dictionary
        
        Returns:
            Dictionary of annotation parameters
        """
        return self._config.get('annotation', {}).copy()
    
    def load_config(self, file_path):
        """
        Load configuration from a JSON file
        
        Args:
            file_path: Path to the configuration file
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                
                # Update configuration
                self._config.update(config)
                
                # Trigger event
                self.on_config_changed('all', None)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
    
    def save_config(self, file_path):
        """
        Save current configuration to a JSON file
        
        Args:
            file_path: Path to save the configuration file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
