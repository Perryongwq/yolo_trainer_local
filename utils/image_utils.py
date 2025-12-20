import cv2
import numpy as np
from PIL import Image, ImageTk
import random
import os

def resize_image(image, target_size, keep_aspect_ratio=True):
    """
    Resize an image to the target size
    
    Args:
        image: OpenCV image (numpy array)
        target_size: Tuple of (width, height)
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        # Calculate scale factor
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Center image on canvas
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        # Simply resize to target size
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def load_image(image_path, target_size=None, keep_aspect_ratio=True):
    """
    Load an image from a file and optionally resize it
    
    Args:
        image_path: Path to the image file
        target_size: Optional tuple of (width, height) for resizing
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Loaded image as OpenCV image (numpy array)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if target_size:
        image = resize_image(image, target_size, keep_aspect_ratio)
    
    return image

def save_image(image, output_path, is_bgr=False):
    """
    Save an image to a file
    
    Args:
        image: OpenCV image (numpy array)
        output_path: Path to save the image
        is_bgr: Whether the image is in BGR format (default is RGB)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR if needed
        if not is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save image
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def draw_bounding_boxes(image, boxes, class_ids=None, class_names=None, 
                       confidences=None, colors=None, show_labels=True):
    """
    Draw bounding boxes on an image
    
    Args:
        image: OpenCV image (numpy array)
        boxes: List of bounding boxes in format [xmin, ymin, xmax, ymax]
        class_ids: Optional list of class IDs
        class_names: Optional dictionary mapping class IDs to names
        confidences: Optional list of confidence scores
        colors: Optional dictionary mapping class IDs to colors
        show_labels: Whether to show labels
        
    Returns:
        Image with bounding boxes drawn
    """
    # Make a copy of the image
    result = image.copy()
    
    for i, box in enumerate(boxes):
        # Get class ID and confidence
        class_id = class_ids[i] if class_ids else None
        confidence = confidences[i] if confidences else None
        
        # Get color for this class
        if colors and class_id in colors:
            color = colors[class_id]
        else:
            # Generate a random color if not provided
            random.seed(class_id if class_id is not None else i)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Draw bounding box
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label if needed
        if show_labels:
            # Prepare label text
            label_parts = []
            
            if class_id is not None and class_names and class_id in class_names:
                label_parts.append(class_names[class_id])
            elif class_id is not None:
                label_parts.append(f"Class {class_id}")
            
            if confidence is not None:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw background for text
                cv2.rectangle(result, (xmin, ymin - text_size[1] - 5), (xmin + text_size[0], ymin), color, -1)
                
                # Draw text
                cv2.putText(result, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result

def draw_polygon(image, points, color=(0, 255, 0), thickness=2, closed=True):
    """
    Draw a polygon on an image
    
    Args:
        image: OpenCV image (numpy array)
        points: List of (x, y) points
        color: Color of the polygon
        thickness: Line thickness
        closed: Whether to close the polygon
        
    Returns:
        Image with polygon drawn
    """
    # Make a copy of the image
    result = image.copy()
    
    # Convert points to the format required by polylines
    points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
    # Draw polygon
    cv2.polylines(result, [points_array], closed, color, thickness)
    
    return result

def convert_yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized x_center, y_center, width, height) to bounding box format (xmin, ymin, xmax, ymax)
    
    Args:
        x_center: Normalized x center coordinate
        y_center: Normalized y center coordinate
        width: Normalized width
        height: Normalized height
        img_width: Image width
        img_height: Image height
        
    Returns:
        Tuple of (xmin, ymin, xmax, ymax) in pixel coordinates
    """
    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate bounding box coordinates
    xmin = int(x_center_px - width_px / 2)
    ymin = int(y_center_px - height_px / 2)
    xmax = int(x_center_px + width_px / 2)
    ymax = int(y_center_px + height_px / 2)
    
    # Ensure coordinates are within image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)
    
    return xmin, ymin, xmax, ymax

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Convert bounding box format (xmin, ymin, xmax, ymax) to YOLO format (normalized x_center, y_center, width, height)
    
    Args:
        xmin: Left coordinate
        ymin: Top coordinate
        xmax: Right coordinate
        ymax: Bottom coordinate
        img_width: Image width
        img_height: Image height
        
    Returns:
        Tuple of (x_center, y_center, width, height) in normalized coordinates
    """
    # Ensure coordinates are within image bounds
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))
    
    # Ensure valid box if clamping changed it to invalid (e.g., xmax < xmin)
    if xmax <= xmin: 
        xmax = xmin + 1
    if ymax <= ymin: 
        ymax = ymin + 1
    
    # Calculate width and height
    width_px = xmax - xmin
    height_px = ymax - ymin
    
    # Calculate center coordinates
    x_center_px = xmin + width_px / 2
    y_center_px = ymin + height_px / 2
    
    # Normalize coordinates
    x_center = x_center_px / img_width
    y_center = y_center_px / img_height
    width = width_px / img_width
    height = height_px / img_height
    
    return x_center, y_center, width, height

def overlay_masks(image, masks, colors=None, alpha=0.5):
    """
    Overlay segmentation masks on an image
    
    Args:
        image: OpenCV image (numpy array)
        masks: List of binary masks
        colors: Optional list of colors for each mask
        alpha: Transparency of the masks
        
    Returns:
        Image with masks overlaid
    """
    # Make a copy of the image
    result = image.copy()
    
    for i, mask in enumerate(masks):
        # Generate color if not provided
        if colors and i < len(colors):
            color = colors[i]
        else:
            random.seed(i)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Create color mask
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = color
        
        # Blend with original image
        result = cv2.addWeighted(result, 1, color_mask, alpha, 0)
        
        # Draw contours around the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    
    return result

def to_tkinter_image(image, size=None):
    """
    Convert an OpenCV image to a Tkinter-compatible PhotoImage
    
    Args:
        image: OpenCV image (numpy array)
        size: Optional tuple of (width, height) for resizing
        
    Returns:
        Tkinter-compatible PhotoImage
    """
    # Ensure image is in RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize if needed
    if size:
        image = resize_image(image, size)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Convert to PhotoImage
    return ImageTk.PhotoImage(pil_image)
