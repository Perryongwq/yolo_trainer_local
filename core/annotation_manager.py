import os
import cv2
import numpy as np
import threading
import torch
from utils.event import Event

class AnnotationManager:
    """
    Manages the auto-annotation process using YOLO models.
    Centralizes annotation-related functionality.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the annotation manager
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        
        # Annotation state
        self.annotations = {}  # Dictionary to store annotations for each image
        self.current_image_path = None
        self.image_folder = None
        self.image_files = []
        
        # SAM model properties
        self.sam_model = None
        self.sam_predictor = None
        self.sam_device = None
        
        # Check if SAM is available
        self.sam_available = self._check_sam_available()
        
        # Events
        self.on_annotation_started = Event()
        self.on_annotation_progress = Event()
        self.on_annotation_completed = Event()
        self.on_annotation_error = Event()
        self.on_annotations_saved = Event()
    
    def _check_sam_available(self):
        """
        Check if the Segment Anything Model (SAM) is available
        
        Returns:
            True if SAM is available, False otherwise
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
            return True
        except ImportError:
            return False
    
    def load_sam_model(self, model_path):
        """
        Load a SAM model
        
        Args:
            model_path: Path to the SAM model file
            
        Returns:
            Success status (True/False)
        """
        if not self.sam_available:
            self.on_annotation_error("SAM is not installed. Run: pip install segment-anything", None)
            return False
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Detect available device (CUDA/CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Determine model type from filename
            model_type = "vit_h"  # Default to highest quality
            if "vit_b" in model_path.lower():
                model_type = "vit_b"
            elif "vit_l" in model_path.lower():
                model_type = "vit_l"
            
            # Load SAM model
            self.sam_model = sam_model_registry[model_type](checkpoint=model_path)
            # Move model to appropriate device
            self.sam_model.to(device=device)
            self.sam_predictor = SamPredictor(self.sam_model)
            
            # Store device for reference
            self.sam_device = device
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load SAM model: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return False
    
    def set_image_folder(self, folder_path):
        """
        Set the image folder and load image list
        
        Args:
            folder_path: Path to the image folder
            
        Returns:
            List of image files or None if failed
        """
        try:
            self.image_folder = folder_path
            
            # Get all image files in the folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            self.image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) 
                               if f.lower().endswith(image_extensions)]
            
            # Initialize annotations dictionary
            for img_path in self.image_files:
                if img_path not in self.annotations:
                    self.annotations[img_path] = []
                
                # Check if annotation file exists
                label_path = self._get_label_path(img_path)
                if os.path.exists(label_path):
                    self._load_existing_annotations(img_path, label_path)
            
            return self.image_files
            
        except Exception as e:
            error_msg = f"Failed to set image folder: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return None
    
    def _get_label_path(self, img_path, output_folder=None):
        """
        Convert image path to label path
        
        Args:
            img_path: Path to the image file
            output_folder: Optional output folder for labels
            
        Returns:
            Path to the label file
        """
        if output_folder is None:
            # Get from config or use same folder as image
            output_folder = self.config_manager.get('annotation.output_folder', None)
            if output_folder is None:
                output_folder = os.path.dirname(img_path)
        
        # Get filename without extension and add .txt extension
        filename = os.path.basename(img_path)
        base_filename = os.path.splitext(filename)[0]
        return os.path.join(output_folder, f"{base_filename}.txt")
    
    def _load_existing_annotations(self, img_path, label_path):
        """
        Load existing YOLO format annotations
        
        Args:
            img_path: Path to the image file
            label_path: Path to the label file
        """
        try:
            # Get image dimensions
            img = cv2.imread(img_path)
            if img is None:
                self.on_annotation_error(f"Failed to load image: {img_path}", None)
                return
            
            height, width = img.shape[:2]
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            annotations = []
            for line in lines:
                # Parse YOLO format: class_id x_center y_center width height
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Convert normalized coordinates to pixel coordinates
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # Convert to xmin, ymin, xmax, ymax
                    xmin = int(x_center - w/2)
                    ymin = int(y_center - h/2)
                    xmax = int(x_center + w/2)
                    ymax = int(y_center + h/2)
                    
                    confidence = 1.0  # Assume 100% confidence for manual annotations
                    
                    annotations.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
            
            self.annotations[img_path] = annotations
            
        except Exception as e:
            self.on_annotation_error(f"Error loading annotations for {img_path}: {str(e)}", e)
    
    def annotate_image(self, img_path, model, confidence_threshold=0.25, selected_classes=None, mode="yolo"):
        """
        Annotate a single image
        
        Args:
            img_path: Path to the image file
            model: YOLO model instance
            confidence_threshold: Confidence threshold for detections
            selected_classes: Optional list of class IDs to include
            mode: Annotation mode (yolo, sam, or hybrid)
            
        Returns:
            List of annotations or None if failed
        """
        try:
            # Check if the image exists
            if not os.path.isfile(img_path):
                self.on_annotation_error(f"Image file not found: {img_path}", None)
                return None
            
            # Default to all classes if not specified
            if selected_classes is None:
                selected_classes = list(model.names.keys())
            
            # Trigger start event
            self.on_annotation_started(img_path, mode)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                self.on_annotation_error(f"Failed to read image: {img_path}", None)
                return None
            
            # Convert to RGB for model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Initialize annotations list
            annotations = []
            
            if mode in ["yolo", "hybrid"]:
                # Run YOLO detection
                results = model(image_rgb, conf=confidence_threshold)
                
                # Process results
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        class_id = int(class_ids[i])
                        confidence = float(confs[i])
                        bbox = [int(x) for x in boxes[i]]  # xmin, ymin, xmax, ymax
                        
                        if class_id in selected_classes:
                            if mode == "hybrid" and self.sam_predictor:
                                # Use SAM to refine the YOLO bounding box
                                self.sam_predictor.set_image(image_rgb)
                                input_box = np.array(bbox)
                                masks, _, _ = self.sam_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=input_box[None, :],
                                    multimask_output=False,
                                )
                                
                                if masks.shape[0] > 0:
                                    # Take the first mask and convert to bbox
                                    coords = np.argwhere(masks[0])
                                    if coords.shape[0] > 0:
                                        ymin, xmin = coords.min(axis=0)
                                        ymax, xmax = coords.max(axis=0)
                                        bbox = [xmin, ymin, xmax + 1, ymax + 1]
                            
                            annotations.append({
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': bbox
                            })
            
            # Store annotations
            self.annotations[img_path] = annotations
            
            # Trigger completed event
            self.on_annotation_completed(img_path, annotations)
            
            return annotations
            
        except Exception as e:
            error_msg = f"Failed to annotate image: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return None
    
    def annotate_batch(self, model, confidence_threshold=0.25, selected_classes=None, mode="yolo"):
        """
        Annotate all images in the folder
        
        Args:
            model: YOLO model instance
            confidence_threshold: Confidence threshold for detections
            selected_classes: Optional list of class IDs to include
            mode: Annotation mode (yolo, sam, or hybrid)
            
        Returns:
            Dictionary of image paths to annotations or None if failed
        """
        if not self.image_files:
            self.on_annotation_error("No image files loaded", None)
            return None
        
        # Default to all classes if not specified
        if selected_classes is None and hasattr(model, 'names'):
            selected_classes = list(model.names.keys())
        
        # Start batch annotation in a separate thread
        thread = threading.Thread(
            target=self._batch_annotation_thread,
            args=(model, confidence_threshold, selected_classes, mode),
            daemon=True
        )
        thread.start()
        
        return self.annotations
    
    def _batch_annotation_thread(self, model, confidence_threshold, selected_classes, mode):
        """
        Thread function for batch annotation
        
        Args:
            model: YOLO model instance
            confidence_threshold: Confidence threshold for detections
            selected_classes: List of class IDs to include
            mode: Annotation mode (yolo, sam, or hybrid)
        """
        try:
            total_images = len(self.image_files)
            self.on_annotation_started(self.image_folder, mode)
            
            for i, img_path in enumerate(self.image_files):
                # Update progress
                progress = (i + 1) / total_images * 100
                self.on_annotation_progress(progress, f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)}")
                
                # Annotate the image
                self.annotate_image(img_path, model, confidence_threshold, selected_classes, mode)
            
            # Trigger completed event
            self.on_annotation_completed(self.image_folder, self.annotations)
            
        except Exception as e:
            error_msg = f"Batch annotation failed: {str(e)}"
            self.on_annotation_error(error_msg, e)
    
    def save_annotations(self, output_folder=None):
        """
        Save annotations for all images in YOLO format
        
        Args:
            output_folder: Optional output folder for labels
            
        Returns:
            Number of saved annotations or -1 if failed
        """
        try:
            if not self.annotations:
                self.on_annotation_error("No annotations to save", None)
                return 0
            
            # Use provided output folder or get from config
            if output_folder is None:
                output_folder = self.config_manager.get('annotation.output_folder', None)
            
            # Create output folder if it doesn't exist
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
            
            saved_count = 0
            for img_path, annos in self.annotations.items():
                if not annos:  # Skip if no annotations for this image
                    continue
                
                # Get label path
                label_path = self._get_label_path(img_path, output_folder)
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                
                # Get image dimensions
                img = cv2.imread(img_path)
                if img is None:
                    self.on_annotation_error(f"Failed to read image: {img_path}", None)
                    continue
                
                height, width = img.shape[:2]
                
                with open(label_path, 'w') as f:
                    for anno in annos:
                        bbox = anno['bbox']
                        class_id = anno['class_id']
                        
                        # Convert xmin, ymin, xmax, ymax to YOLO format (normalized x_center, y_center, width, height)
                        xmin, ymin, xmax, ymax = bbox
                        
                        # Clamp coordinates to image bounds to prevent issues with out-of-bounds annotations
                        xmin = max(0, min(xmin, width - 1))
                        ymin = max(0, min(ymin, height - 1))
                        xmax = max(0, min(xmax, width))
                        ymax = max(0, min(ymax, height))
                        
                        # Ensure valid box if clamping changed it to invalid (e.g., xmax < xmin)
                        if xmax <= xmin: xmax = xmin + 1
                        if ymax <= ymin: ymax = ymin + 1
                        
                        bbox_width = xmax - xmin
                        bbox_height = ymax - ymin
                        
                        x_center = (xmin + xmax) / 2.0 / width
                        y_center = (ymin + ymax) / 2.0 / height
                        norm_width = bbox_width / width
                        norm_height = bbox_height / height
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
                saved_count += 1
            
            # Trigger event
            self.on_annotations_saved(saved_count, output_folder)
            
            return saved_count
            
        except Exception as e:
            error_msg = f"Failed to save annotations: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return -1
    
    def get_annotations(self, img_path=None):
        """
        Get annotations for a specific image or all images
        
        Args:
            img_path: Optional image path (returns all annotations if None)
            
        Returns:
            Dictionary of annotations or annotations for the specified image
        """
        if img_path:
            return self.annotations.get(img_path, [])
        return self.annotations
    
    def add_annotation(self, img_path, class_id, bbox, confidence=1.0):
        """
        Add a manual annotation
        
        Args:
            img_path: Path to the image file
            class_id: Class ID
            bbox: Bounding box [xmin, ymin, xmax, ymax]
            confidence: Confidence value
            
        Returns:
            Success status (True/False)
        """
        try:
            if img_path not in self.annotations:
                self.annotations[img_path] = []
            
            self.annotations[img_path].append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': bbox
            })
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to add annotation: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return False
    
    def delete_annotation(self, img_path, annotation_index):
        """
        Delete an annotation
        
        Args:
            img_path: Path to the image file
            annotation_index: Index of the annotation to delete
            
        Returns:
            Success status (True/False)
        """
        try:
            if img_path not in self.annotations:
                return False
            
            if annotation_index < 0 or annotation_index >= len(self.annotations[img_path]):
                return False
            
            del self.annotations[img_path][annotation_index]
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete annotation: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return False
    
    def update_annotation(self, img_path, annotation_index, class_id=None, bbox=None, confidence=None):
        """
        Update an annotation
        
        Args:
            img_path: Path to the image file
            annotation_index: Index of the annotation to update
            class_id: Optional new class ID
            bbox: Optional new bounding box
            confidence: Optional new confidence value
            
        Returns:
            Success status (True/False)
        """
        try:
            if img_path not in self.annotations:
                return False
            
            if annotation_index < 0 or annotation_index >= len(self.annotations[img_path]):
                return False
            
            # Update fields if provided
            if class_id is not None:
                self.annotations[img_path][annotation_index]['class_id'] = class_id
            
            if bbox is not None:
                self.annotations[img_path][annotation_index]['bbox'] = bbox
            
            if confidence is not None:
                self.annotations[img_path][annotation_index]['confidence'] = confidence
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to update annotation: {str(e)}"
            self.on_annotation_error(error_msg, e)
            return False
