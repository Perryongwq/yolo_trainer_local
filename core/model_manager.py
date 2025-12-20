import os
import torch
from ultralytics import YOLO
from utils.event import Event

class ModelManager:
    """
    Manages YOLO models, including loading, inference, and export operations.
    Centralizes model-related functionality and provides clean interfaces.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the model manager
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.model = None
        self.model_path = None
        self.class_names = {}
        
        # Events
        self.on_model_loaded = Event()
        self.on_inference_completed = Event()
        self.on_model_error = Event()
        
        # Initialize class colors dictionary
        self.class_colors = {}
    
    def load_model(self, model_path, force_reload=False):
        """
        Load a YOLO model
        
        Args:
            model_path: Path to the model file
            force_reload: Whether to force reload if the model is already loaded
            
        Returns:
            Success status (True/False)
        """
        # Skip if the model is already loaded and not forcing reload
        if self.model is not None and self.model_path == model_path and not force_reload:
            return True
        
        try:
            # Resolve model path if needed
            resolved_path = self.resolve_model_path(model_path)
            if resolved_path:
                model_path = resolved_path
            
            # Load the model
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # Get class names from model
            try:
                self.class_names = self.model.names
            except AttributeError:
                # Default class names if not available in model
                self.class_names = {i: f"Class {i}" for i in range(100)}
            
            # Generate class colors
            self.generate_class_colors()
            
            # Trigger event
            self.on_model_loaded(self.model, self.model_path, self.class_names)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.on_model_error(error_msg, e)
            return False
    
    def generate_class_colors(self):
        """Generate consistent colors for each class"""
        import random
        random.seed(42)  # For reproducible colors
        
        self.class_colors = {}
        for cls_id, class_name in self.class_names.items():
            # Generate random color
            self.class_colors[cls_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
    
    def run_inference(self, image_path, confidence=0.25, **kwargs):
        """
        Run inference on an image
        
        Args:
            image_path: Path to the image
            confidence: Confidence threshold
            **kwargs: Additional arguments for prediction
            
        Returns:
            Detection results or None if failed
        """
        if self.model is None:
            self.on_model_error("No model loaded", None)
            return None
        
        try:
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=confidence,
                **kwargs
            )
            
            # Trigger event
            self.on_inference_completed(results, image_path)
            
            return results
            
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            self.on_model_error(error_msg, e)
            return None
    
    def run_batch_inference(self, image_paths, confidence=0.25, **kwargs):
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths
            confidence: Confidence threshold
            **kwargs: Additional arguments for prediction
            
        Returns:
            List of detection results or empty list if failed
        """
        if self.model is None:
            self.on_model_error("No model loaded", None)
            return []
        
        results = []
        for image_path in image_paths:
            try:
                # Run inference on single image
                result = self.model.predict(
                    source=image_path,
                    conf=confidence,
                    **kwargs
                )
                results.append((image_path, result))
                
            except Exception as e:
                error_msg = f"Inference failed for {image_path}: {str(e)}"
                self.on_model_error(error_msg, e)
        
        return results
    
    def export_model(self, format="onnx", **kwargs):
        """
        Export the model to a different format
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            **kwargs: Additional export arguments
            
        Returns:
            Path to the exported model or None if failed
        """
        if self.model is None:
            self.on_model_error("No model loaded", None)
            return None
        
        try:
            # Export the model
            exported_path = self.model.export(format=format, **kwargs)
            return exported_path
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.on_model_error(error_msg, e)
            return None
    
    def get_class_name(self, class_id):
        """
        Get class name for a class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            Class name or default string if not found
        """
        return self.class_names.get(class_id, f"Class {class_id}")
    
    def get_class_color(self, class_id):
        """
        Get color for a class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            RGB color tuple
        """
        return self.class_colors.get(class_id, (0, 255, 0))
    
    def resolve_model_path(self, model_name):
        """
        Try to find the model file in various possible locations.
        Also handles yolo11/yolov11 naming variants.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Resolved absolute path or None if not found
        """
        # Check for both naming formats
        model_names = [model_name]
        
        # Add alternate model name if applicable
        if "yolov11" in model_name.lower():
            model_names.append(model_name.lower().replace("yolov11", "yolo11"))
        elif "yolo11" in model_name.lower():
            model_names.append(model_name.lower().replace("yolo11", "yolov11"))
        
        # List of possible locations to check for each model name
        for name in model_names:
            possible_paths = [
                # Exact path provided
                name,
                # Current working directory
                os.path.join(os.getcwd(), name),
                # Script directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", name),
                # Script directory parent
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", name),
                # Models subdirectory
                os.path.join(os.getcwd(), "models", name),
                # Weights subdirectory
                os.path.join(os.getcwd(), "weights", name),
            ]
            
            # Check if any of these paths exist
            for path in possible_paths:
                if os.path.exists(path):
                    return os.path.abspath(path)
        
        return None
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                "loaded": False,
                "path": None,
                "classes": 0,
                "has_gpu": torch.cuda.is_available()
            }
        
        return {
            "loaded": True,
            "path": self.model_path,
            "classes": len(self.class_names),
            "class_names": self.class_names,
            "has_gpu": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
