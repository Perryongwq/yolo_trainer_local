"""Image viewer widget for PyQt5"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect


class ImageViewer(QLabel):
    """A widget for displaying images with annotations"""
    
    # Signal emitted when a bounding box is drawn
    boxDrawn = pyqtSignal(list)
    
    def __init__(self, parent=None, width=600, height=400, use_matplotlib=False):
        """
        Initialize the image viewer
        
        Args:
            parent: Parent widget
            width: Minimum width
            height: Minimum height
            use_matplotlib: Whether to use matplotlib (not implemented for PyQt5)
        """
        super().__init__(parent)
        
        self.setMinimumSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc;")
        self.setText("No image loaded")
        
        # Image data
        self.original_image = None
        self.display_pixmap = None
        self.annotations = []
        self.selected_annotation_idx = -1
        
        # Display options
        self.show_labels = True
        self.show_confidence = True
        self.show_boxes = True
        
        # Drawing mode
        self.drawing_mode = False
        self.draw_start = None
        self.draw_current = None
        
        # Enable mouse tracking for drawing
        self.setMouseTracking(True)
    
    def load_image(self, image_path):
        """
        Load and display an image
        
        Args:
            image_path: Path to the image file
        """
        # Read image with OpenCV
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            self.setText("Failed to load image")
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Store and display
        self.display_pixmap = QPixmap.fromImage(q_image)
        self._update_display()
    
    def set_annotations(self, annotations):
        """
        Set annotations to display
        
        Args:
            annotations: List of annotation dictionaries with 'bbox', 'class_id', 'confidence'
        """
        self.annotations = annotations
        self._update_display()
    
    def set_display_options(self, show_labels=True, show_confidence=True, show_boxes=True):
        """Set display options for annotations"""
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_boxes = show_boxes
        self._update_display()
    
    def select_annotation(self, index):
        """Highlight a specific annotation"""
        self.selected_annotation_idx = index
        self._update_display()
    
    def enable_drawing_mode(self, enabled):
        """Enable or disable drawing mode"""
        self.drawing_mode = enabled
        if not enabled:
            self.draw_start = None
            self.draw_current = None
            self._update_display()
    
    def _update_display(self):
        """Update the displayed image with annotations"""
        if self.display_pixmap is None:
            return
        
        # Create a copy to draw on
        pixmap = self.display_pixmap.copy()
        
        # Draw annotations if any
        if self.annotations and self.show_boxes:
            painter = QPainter(pixmap)
            
            for idx, anno in enumerate(self.annotations):
                bbox = anno.get('bbox', [])
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                
                # Choose color based on selection
                if idx == self.selected_annotation_idx:
                    pen = QPen(Qt.yellow, 3)
                else:
                    pen = QPen(Qt.green, 2)
                
                painter.setPen(pen)
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                # Draw label and confidence
                if self.show_labels or self.show_confidence:
                    label_parts = []
                    if self.show_labels and 'class_id' in anno:
                        # Try to get class name from model manager (will be set by tab)
                        label_parts.append(f"Class {anno['class_id']}")
                    if self.show_confidence and 'confidence' in anno:
                        label_parts.append(f"{anno['confidence']:.2f}")
                    
                    if label_parts:
                        label_text = " ".join(label_parts)
                        font = QFont()
                        font.setPointSize(10)
                        painter.setFont(font)
                        painter.setPen(Qt.white)
                        painter.drawText(int(x1) + 5, int(y1) + 15, label_text)
            
            painter.end()
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def mousePressEvent(self, event):
        """Handle mouse press for drawing"""
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.draw_start = event.pos()
            self.draw_current = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing"""
        if self.drawing_mode and self.draw_start:
            self.draw_current = event.pos()
            self._update_display()
            
            # Draw temporary rectangle
            if self.pixmap():
                temp_pixmap = self.pixmap().copy()
                painter = QPainter(temp_pixmap)
                pen = QPen(Qt.red, 2, Qt.DashLine)
                painter.setPen(pen)
                
                rect = QRect(self.draw_start, self.draw_current).normalized()
                painter.drawRect(rect)
                painter.end()
                
                self.setPixmap(temp_pixmap)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for drawing"""
        if self.drawing_mode and event.button() == Qt.LeftButton and self.draw_start:
            self.draw_current = event.pos()
            
            # Calculate bbox in original image coordinates
            if self.display_pixmap:
                # Get scaling factor
                pixmap_size = self.pixmap().size()
                widget_size = self.size()
                
                # Calculate actual image position and size
                x_offset = (widget_size.width() - pixmap_size.width()) // 2
                y_offset = (widget_size.height() - pixmap_size.height()) // 2
                
                # Convert widget coordinates to pixmap coordinates
                x1 = max(0, self.draw_start.x() - x_offset)
                y1 = max(0, self.draw_start.y() - y_offset)
                x2 = max(0, self.draw_current.x() - x_offset)
                y2 = max(0, self.draw_current.y() - y_offset)
                
                # Scale to original image size
                scale_x = self.display_pixmap.width() / pixmap_size.width() if pixmap_size.width() > 0 else 1
                scale_y = self.display_pixmap.height() / pixmap_size.height() if pixmap_size.height() > 0 else 1
                
                bbox = [
                    int(min(x1, x2) * scale_x),
                    int(min(y1, y2) * scale_y),
                    int(max(x1, x2) * scale_x),
                    int(max(y1, y2) * scale_y)
                ]
                
                # Emit signal
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid box
                    self.boxDrawn.emit(bbox)
            
            self.draw_start = None
            self.draw_current = None
    
    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        self._update_display()

