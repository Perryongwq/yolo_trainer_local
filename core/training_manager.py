import os
import sys
import traceback
import torch
import io
import threading
import time
from contextlib import redirect_stdout, redirect_stderr

# CRITICAL: Set matplotlib backend BEFORE importing YOLO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from ultralytics import YOLO
from utils.event import Event

class CustomStream(io.StringIO):
    """Custom stream to capture and redirect output"""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, text):
        super().write(text)
        if text.strip():  # Only process non-empty text
            self.callback(text)
        
    def flush(self):
        super().flush()

class TrainingManager:
    """Manages the training process for YOLO models"""
    
    def __init__(self, config_manager):
        """
        Initialize the training manager
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.stop_event = threading.Event()
        self.training_thread = None
        
        # Events
        self.on_training_started = Event()
        self.on_training_progress = Event()
        self.on_training_completed = Event()
        self.on_training_error = Event()
        
        # Initialize training state
        self.is_training = False
    
    def start_training(self, model_path, dataset_path, params=None):
        """
        Start training process in a separate thread
        
        Args:
            model_path: Path to the model file
            dataset_path: Path to the dataset YAML file
            params: Optional training parameters
        """
        if self.is_training:
            self.on_training_error("Training is already in progress")
            return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Set training flag
        self.is_training = True
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(model_path, dataset_path, params),
            daemon=True
        )
        self.training_thread.start()
        
        return True
    
    def stop_training(self):
        """Stop the current training process"""
        if self.is_training:
            self.stop_event.set()
            return True
        return False
    
    def _run_training(self, model_path, dataset_path, params=None):
        """
        Run the training process
        
        Args:
            model_path: Path to the model file
            dataset_path: Path to the dataset YAML file
            params: Optional training parameters
        """
        try:
            # Ensure matplotlib is using non-interactive backend
            matplotlib.use('Agg')
            plt.ioff()
            
            # Trigger training started event
            self.on_training_started(model_path, dataset_path)
            
            # Resolve model path
            resolved_path = self._resolve_model_path(model_path)
            if resolved_path:
                self.on_training_progress(f"Found model at: {resolved_path}")
                model_path = resolved_path
            else:
                self.on_training_progress(f"Warning: Model file not found in searched paths.")
                self.on_training_progress(f"Searched in: {os.getcwd()}")
                self.on_training_progress(f"Will try to let YOLO handle downloading...")
            
            # Get parameters from config if not provided
            if params is None:
                params = self.config_manager.get_training_params()
            
            # GPU/CUDA checks
            has_cuda = torch.cuda.is_available()
            device = params.get('device', 'auto')
            
            # Check device compatibility and adjust if needed
            if device != 'cpu' and not has_cuda:
                self.on_training_progress(f"WARNING: Device '{device}' requested but CUDA is not available.")
                self.on_training_progress("Switching to CPU. This will make training slower.")
                device = 'cpu'
                params['device'] = 'cpu'
            elif device == 'auto' and not has_cuda:
                self.on_training_progress("Device 'auto' selected but CUDA is not available. Using CPU.")
                device = 'cpu'
                params['device'] = 'cpu'
            elif has_cuda and device.isdigit():
                gpu_id = int(device)
                if gpu_id >= torch.cuda.device_count():
                    self.on_training_progress(f"WARNING: GPU {gpu_id} requested but only {torch.cuda.device_count()} GPUs available.")
                    self.on_training_progress(f"Using GPU 0 instead.")
                    device = '0'
                    params['device'] = '0'
            
            # Initialize YOLO model
            model = YOLO(model_path)
            
            # Prepare training parameters - ONLY INCLUDE SUPPORTED PARAMETERS
            train_args = {
                'data': dataset_path,
                'epochs': params.get('epochs', 100),
                'imgsz': params.get('imgsz', 640),
                'lr0': params.get('lr0', 0.001),
                'lrf': params.get('lrf', 0.2),
                'patience': params.get('patience', 0),
                'optimizer': params.get('optimizer', 'Adam'),
                'verbose': True,  # Ensure verbose output
                'plots': True,    # Keep plots enabled, but they'll be saved to disk
                'pretrained': params.get('pretrained', True),
                'device': params.get('device', 'cpu'),
                'batch': params.get('batch_size', 16),
                'save_period': params.get('save_period', 0)
            }
            
            # Log training parameters
            self.on_training_progress("Starting training with parameters:")
            for key, value in train_args.items():
                self.on_training_progress(f"  {key}: {value}")
            
            # Log device information
            if device == 'cpu':
                self.on_training_progress("Training on CPU")
            elif has_cuda:
                if device == '0' or device == 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    self.on_training_progress(f"Training on GPU: {gpu_name}")
                elif ',' in str(device):
                    self.on_training_progress(f"Training on multiple GPUs: {device}")
                else:
                    self.on_training_progress(f"Training on device: {device}")
            
            # Create a custom stream to capture stdout and stderr
            output_stream = CustomStream(lambda text: self.on_training_progress(text.rstrip()))
            
            # Redirect stdout and stderr to our custom stream to capture all output
            with redirect_stdout(output_stream), redirect_stderr(output_stream):
                # Start training
                self.on_training_progress("--- TRAINING OUTPUT ---")
                results = model.train(**train_args)
            
            self.on_training_progress("--- END OF TRAINING OUTPUT ---")
            
            # Training completed successfully
            self.on_training_completed(True, "Training complete!", results)
            
        except Exception as e:
            error_message = f"Error during training: {str(e)}"
            self.on_training_progress(error_message)
            self.on_training_progress(f"Current working directory: {os.getcwd()}")
            
            # Print detailed debug information for troubleshooting
            self.on_training_progress(f"Exception details: {traceback.format_exc()}")
            
            # Provide specific guidance for common errors
            error_str = str(e).lower()
            if "cuda" in error_str:
                self.on_training_progress("CUDA/GPU ERROR: There seems to be an issue with your GPU setup.")
                self.on_training_progress("Try the following:")
                self.on_training_progress("1. Set device to 'cpu' in Advanced Options")
                self.on_training_progress("2. Check if your GPU drivers are up to date")
                self.on_training_progress("3. Ensure PyTorch is installed with CUDA support")
                self.on_training_progress("4. Run 'nvidia-smi' in command prompt to check GPU status")
            elif "out of memory" in error_str:
                self.on_training_progress("GPU MEMORY ERROR: Your GPU ran out of memory.")
                self.on_training_progress("Try the following:")
                self.on_training_progress("1. Reduce batch size in Advanced Options")
                self.on_training_progress("2. Use a smaller model (e.g., yolov8n instead of yolov8x)")
                self.on_training_progress("3. Reduce image size")
            elif "main thread is not in main loop" in error_str:
                self.on_training_progress("THREADING ERROR: Matplotlib backend conflict.")
                self.on_training_progress("This should be fixed by setting the Agg backend.")
                self.on_training_progress("If the error persists, try disabling plots with 'plots': False")
            
            # Trigger training error event
            self.on_training_error(error_message, e)
            
        finally:
            self.is_training = False
            self.stop_event.set()
            # Ensure any remaining plots are closed safely
            try:
                plt.close('all')
            except:
                pass
    
    def _resolve_model_path(self, model_name):
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
            self.on_training_progress(f"Checking for model name: {name}")
            
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
                self.on_training_progress(f"Checking for model at: {path}")
                if os.path.exists(path):
                    return os.path.abspath(path)
        
        return None
