"""Auto annotation tab for PyQt5"""
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QCheckBox, QWidget, QVBoxLayout, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class AnnotationTab:
    """Tab for auto-annotating images using trained YOLO models"""
    
    def __init__(self, ui, model_manager, annotation_manager, logger):
        """
        Initialize the annotation tab
        
        Args:
            ui: UI object from generated ui_new_mainwindow.py
            model_manager: ModelManager instance
            annotation_manager: AnnotationManager instance
            logger: Logger instance
        """
        self.ui = ui
        self.model_manager = model_manager
        self.annotation_manager = annotation_manager
        self.logger = logger
        
        # Check SAM availability
        self.sam_available = self.annotation_manager.sam_available
        self.sam_model_loaded = False
        
        # Variables
        self.current_image_index = 0
        self.class_checkboxes = {}  # Dictionary to hold checkboxes for each class
        
        # Connect UI signals to handlers
        self._connect_signals()
        
        # Connect to model manager events
        self.model_manager.on_model_loaded.subscribe(self.update_model_info)
        
        # Connect to annotation manager events
        self.annotation_manager.on_annotation_completed.subscribe(self.on_annotation_completed)
        self.annotation_manager.on_annotation_progress.subscribe(self.on_annotation_progress)
        self.annotation_manager.on_annotations_saved.subscribe(self.on_annotations_saved)
        
        # Initialize UI state
        self._initialize_ui()
    
    def _connect_signals(self):
        """Connect Qt signals to handler methods"""
        # Model browsers
        self.ui.pushButton_yolo_browse.clicked.connect(self._browse_yolo_model)
        self.ui.pushButton_sam_browse.clicked.connect(self._browse_sam_model)
        self.ui.pushButton_load_models.clicked.connect(self.load_models)
        
        # Dataset browsers
        self.ui.pushButton_annotation_folder_browse.clicked.connect(self._browse_annotation_folder)
        self.ui.pushButton_output_folder_browse.clicked.connect(self._browse_output_folder)
        
        # Annotation mode
        self.ui.comboBox_annotation_mode.currentTextChanged.connect(self._on_annotation_mode_changed)
        
        # Confidence slider
        self.ui.slider_annotation_confidence.valueChanged.connect(self._on_confidence_changed)
        
        # Navigation buttons
        self.ui.pushButton_previous.clicked.connect(self.previous_image)
        self.ui.pushButton_next.clicked.connect(self.next_image)
        
        # Annotation buttons
        self.ui.pushButton_annotate_current.clicked.connect(self.annotate_current)
        self.ui.pushButton_annotate_all.clicked.connect(self.annotate_batch)
        self.ui.pushButton_save_annotations.clicked.connect(self.save_annotations)
        
        # Class selection buttons
        self.ui.pushButton_select_all.clicked.connect(self.select_all_classes)
        self.ui.pushButton_deselect_all.clicked.connect(self.deselect_all_classes)
        
        # Annotation details buttons
        self.ui.pushButton_delete_annotation.clicked.connect(self.delete_selected_annotation)
        self.ui.pushButton_edit_annotation.clicked.connect(self.edit_selected_annotation)
        self.ui.pushButton_add_manual_box.clicked.connect(self.add_manual_box)
    
    def _initialize_ui(self):
        """Initialize UI with default values"""
        # Set initial model path if available
        if self.model_manager.model_path:
            self.ui.lineEdit_yolo_model.setText(self.model_manager.model_path)
        
        # Set initial confidence display
        self._on_confidence_changed(self.ui.slider_annotation_confidence.value())
        
        # Display SAM availability status
        if self.sam_available:
            self.logger.info("SAM is available. You can use 'YOLO+SAM Hybrid' mode for better boundaries.")
        else:
            self.logger.info("SAM not installed. Only 'YOLO Only' mode is available. To enable SAM: pip install segment-anything")
    
    def _browse_yolo_model(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select YOLO Model",
            "",
            "PyTorch Models (*.pt);;All files (*.*)"
        )
        
        if file_path:
            self.ui.lineEdit_yolo_model.setText(file_path)
            self.logger.info(f"YOLO model selected: {file_path}")
    
    def _browse_sam_model(self):
        """Browse for SAM model file"""
        # Show help message if SAM is not available
        if not self.sam_available:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("SAM Installation Required")
            msg.setText("Segment Anything Model (SAM) is not installed.")
            msg.setInformativeText(
                "To use SAM for better object boundaries:\n\n"
                "1. Install SAM:\n"
                "   pip install segment-anything\n\n"
                "2. Download a SAM model checkpoint:\n"
                "   • vit_h (~2.4GB) - Best quality\n"
                "   • vit_l (~1.2GB) - Good balance\n"
                "   • vit_b (~375MB) - Fastest\n\n"
                "Download from:\n"
                "https://github.com/facebookresearch/segment-anything#model-checkpoints\n\n"
                "Model filename should contain 'vit_b', 'vit_l', or 'vit_h' to auto-detect type."
            )
            msg.exec_()
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select SAM Model (vit_b, vit_l, or vit_h)",
            "",
            "PyTorch Models (*.pth);;All files (*.*)"
        )
        
        if file_path:
            self.ui.lineEdit_sam_model.setText(file_path)
            self.logger.info(f"SAM model selected: {file_path}")
    
    def _browse_annotation_folder(self):
        """Browse for annotation folder"""
        folder_path = QFileDialog.getExistingDirectory(
            None,
            "Select Image Folder"
        )
        
        if folder_path:
            self.ui.lineEdit_annotation_folder.setText(folder_path)
            
            # Set default output folder
            if not self.ui.lineEdit_output_folder.text():
                output_path = os.path.join(folder_path, "labels")
                self.ui.lineEdit_output_folder.setText(output_path)
            
            # Load image list
            image_files = self.annotation_manager.set_image_folder(folder_path)
            
            if image_files:
                self.current_image_index = 0
                self.load_current_image()
                self.logger.info(f"Loaded {len(image_files)} images from {folder_path}")
            else:
                self.logger.info(f"No images found in {folder_path}")
    
    def _browse_output_folder(self):
        """Browse for output folder"""
        folder_path = QFileDialog.getExistingDirectory(
            None,
            "Select Output Folder"
        )
        
        if folder_path:
            self.ui.lineEdit_output_folder.setText(folder_path)
            self.annotation_manager.config_manager.set('annotation.output_folder', folder_path)
    
    def _on_annotation_mode_changed(self, mode):
        """Handle annotation mode change"""
        if mode == "SAM Only" or mode == "YOLO+SAM Hybrid":
            if not self.sam_available or not self.sam_model_loaded:
                QMessageBox.warning(
                    None,
                    "SAM Not Ready",
                    "SAM is not installed or model not loaded. Please select YOLO Only mode."
                )
                self.ui.comboBox_annotation_mode.setCurrentText("YOLO Only")
    
    def _on_confidence_changed(self, value):
        """Handle confidence slider change"""
        confidence = value / 100.0
        self.ui.label_annotation_confidence_value.setText(f"{confidence:.2f}")
    
    def load_models(self):
        """Load selected models"""
        # Load YOLO model
        yolo_path = self.ui.lineEdit_yolo_model.text()
        if yolo_path:
            try:
                success = self.model_manager.load_model(yolo_path)
                
                if success:
                    # Update class checkboxes
                    self.update_class_checkboxes()
                    self.logger.info(f"YOLO model loaded: {os.path.basename(yolo_path)}")
                else:
                    QMessageBox.critical(None, "Error", "Failed to load YOLO model")
                    return False
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load YOLO model: {str(e)}")
                return False
        
        # Load SAM model if available and selected
        if self.sam_available:
            sam_path = self.ui.lineEdit_sam_model.text()
            if sam_path:
                try:
                    success = self.annotation_manager.load_sam_model(sam_path)
                    
                    if success:
                        self.sam_model_loaded = True
                        device = self.annotation_manager.sam_device or "unknown"
                        self.logger.info(f"SAM model loaded on {device.upper()}: {os.path.basename(sam_path)}")
                        QMessageBox.information(None, "SAM Model Loaded", 
                            f"SAM model loaded successfully on {device.upper()}!\n\n"
                            f"You can now use 'YOLO+SAM Hybrid' mode for better object boundaries.")
                    else:
                        QMessageBox.critical(None, "Error", "Failed to load SAM model")
                        return False
                except Exception as e:
                    QMessageBox.critical(None, "Error", f"Failed to load SAM model: {str(e)}")
                    return False
        else:
            # SAM not available - show installation instructions
            if self.ui.lineEdit_sam_model.text():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("SAM Not Available")
                msg.setText("Segment Anything Model (SAM) is not installed.")
                msg.setInformativeText(
                    "To use SAM for better object boundaries:\n\n"
                    "1. Install SAM:\n"
                    "   pip install segment-anything\n\n"
                    "2. Download a SAM model checkpoint:\n"
                    "   • vit_h (~2.4GB) - Best quality\n"
                    "   • vit_l (~1.2GB) - Good balance\n"
                    "   • vit_b (~375MB) - Fastest\n\n"
                    "Download from:\n"
                    "https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )
                msg.exec_()
                return False
        
        return True
    
    def update_class_checkboxes(self):
        """Populate the class list with checkboxes based on the loaded YOLO model's class names"""
        # Clear existing checkboxes
        layout = self.ui.verticalLayout_annotation_classes_content
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.class_checkboxes.clear()
        
        # Get class names from model manager
        class_names = self.model_manager.class_names
        
        if class_names:
            for idx, class_name in class_names.items():
                checkbox = QCheckBox(f"{idx}: {class_name}")
                checkbox.setChecked(True)  # Check all by default
                layout.addWidget(checkbox)
                self.class_checkboxes[idx] = checkbox
        else:
            from PyQt5.QtWidgets import QLabel
            label = QLabel("Load YOLO model to see classes.")
            layout.addWidget(label)
        
        layout.addStretch()
    
    def get_selected_classes(self):
        """Returns a list of class IDs for currently selected checkboxes"""
        selected_classes = []
        for class_id, checkbox in self.class_checkboxes.items():
            if checkbox.isChecked():
                selected_classes.append(class_id)
        return selected_classes
    
    def select_all_classes(self):
        """Checks all class checkboxes"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_classes(self):
        """Unchecks all class checkboxes"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)
    
    def load_current_image(self):
        """Load and display the current image with annotations"""
        image_files = self.annotation_manager.image_files
        if not image_files or self.current_image_index < 0 or self.current_image_index >= len(image_files):
            return
        
        img_path = image_files[self.current_image_index]
        
        # Get annotations for this image
        annotations = self.annotation_manager.get_annotations(img_path)
        
        # Draw annotated image if annotations exist
        if annotations:
            self._draw_annotated_image(img_path, annotations)
        else:
            # Load plain image if no annotations
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                # Scale to 50% of original size
                scaled_pixmap = pixmap.scaled(
                    pixmap.width() // 2,
                    pixmap.height() // 2,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ui.label_annotation_viewer.setPixmap(scaled_pixmap)
        
        # Update annotation table
        self.update_annotation_table(img_path)
        
        # Update status
        filename = os.path.basename(img_path)
        index = self.current_image_index + 1
        total = len(image_files)
        self.logger.info(f"Image {index}/{total}: {filename}")
    
    def _draw_annotated_image(self, image_path, annotations):
        """Draw bounding boxes on the image"""
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            self.ui.label_annotation_viewer.setText("Failed to load image")
            return
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Draw bounding boxes for each annotation
        for anno in annotations:
            bbox = anno['bbox']  # [xmin, ymin, xmax, ymax] in pixel coordinates
            class_id = anno['class_id']
            confidence = anno.get('confidence', 1.0)
            
            # Get corner coordinates (already in pixel format)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            
            # Get class name
            class_name = self.model_manager.get_class_name(class_id)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green in RGB
            thickness = 2
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = f"{class_name} {confidence:.2f}"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_rgb, 
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width + 5, y1),
                        color, -1)
            
            # Draw text
            cv2.putText(img_rgb, label_text,
                      (x1 + 2, y1 - baseline - 2),
                      font, font_scale, (255, 255, 255), font_thickness)
        
        # Convert to QPixmap
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Display at 50% size
        if not pixmap.isNull():
            # Scale image to 50% of original size
            scaled_pixmap = pixmap.scaled(
                pixmap.width() // 2, 
                pixmap.height() // 2,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.label_annotation_viewer.setPixmap(scaled_pixmap)
        else:
            self.ui.label_annotation_viewer.setText("Failed to create annotated image")
    
    def update_annotation_table(self, img_path):
        """Update the annotation table with annotations for the current image"""
        # Clear the table
        self.ui.tableWidget_annotations.setRowCount(0)
        
        # Get annotations for this image
        annotations = self.annotation_manager.get_annotations(img_path)
        
        # Add annotations to the table
        for i, anno in enumerate(annotations):
            bbox = anno['bbox']
            class_id = anno['class_id']
            confidence = anno['confidence']
            
            # Get class name
            class_name = self.model_manager.get_class_name(class_id)
            
            # Add row
            row_position = self.ui.tableWidget_annotations.rowCount()
            self.ui.tableWidget_annotations.insertRow(row_position)
            
            self.ui.tableWidget_annotations.setItem(row_position, 0, QTableWidgetItem(class_name))
            self.ui.tableWidget_annotations.setItem(row_position, 1, QTableWidgetItem(f"{confidence:.2f}"))
            self.ui.tableWidget_annotations.setItem(row_position, 2, QTableWidgetItem(
                f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"))
    
    def previous_image(self):
        """Go to the previous image"""
        image_files = self.annotation_manager.image_files
        if not image_files:
            return
        
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        else:
            self.logger.info("No previous image.")
    
    def next_image(self):
        """Go to the next image"""
        image_files = self.annotation_manager.image_files
        if not image_files:
            return
        
        if self.current_image_index < len(image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
        else:
            self.logger.info("No next image.")
    
    def annotate_current(self):
        """Start auto-annotation for the current image"""
        if not self.model_manager.model:
            QMessageBox.warning(None, "No YOLO Model", "Please load a YOLO model to perform detection.")
            return
        
        image_files = self.annotation_manager.image_files
        if not image_files or self.current_image_index < 0 or self.current_image_index >= len(image_files):
            QMessageBox.warning(None, "No Image", "Please load an image folder first.")
            return
        
        img_path = image_files[self.current_image_index]
        
        # Get annotation parameters
        confidence_threshold = self.ui.slider_annotation_confidence.value() / 100.0
        selected_classes = self.get_selected_classes()
        mode_text = self.ui.comboBox_annotation_mode.currentText()
        mode = {"YOLO Only": "yolo", "SAM Only": "sam", "YOLO+SAM Hybrid": "hybrid"}.get(mode_text, "yolo")
        
        self.logger.info(f"Annotating {os.path.basename(img_path)}...")
        
        # Run annotation using annotation manager
        annotations = self.annotation_manager.annotate_image(
            img_path,
            self.model_manager.model,
            confidence_threshold,
            selected_classes,
            mode
        )
        
        if annotations is not None:
            self.load_current_image()  # Reload to show the annotations
            self.logger.info(f"Annotation complete: {len(annotations)} objects found.")
        else:
            self.logger.error("Annotation failed.")
    
    def annotate_batch(self):
        """Start auto-annotation for all images in the folder"""
        if not self.model_manager.model:
            QMessageBox.warning(None, "No YOLO Model", "Please load a YOLO model to perform detection.")
            return
        
        image_files = self.annotation_manager.image_files
        if not image_files:
            QMessageBox.warning(None, "No Images", "Please load an image folder first.")
            return
        
        # Get annotation parameters
        confidence_threshold = self.ui.slider_annotation_confidence.value() / 100.0
        selected_classes = self.get_selected_classes()
        mode_text = self.ui.comboBox_annotation_mode.currentText()
        mode = {"YOLO Only": "yolo", "SAM Only": "sam", "YOLO+SAM Hybrid": "hybrid"}.get(mode_text, "yolo")
        
        # Check if mode is SAM only
        if mode == "sam":
            QMessageBox.warning(
                None,
                "SAM Only Mode",
                "Batch annotation is not supported in 'SAM Only' mode. Use 'YOLO Only' or 'YOLO+SAM Hybrid'."
            )
            return
        
        confirm = QMessageBox.question(
            None,
            "Confirm Batch Annotation",
            f"This will auto-annotate {len(image_files)} images. Proceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm != QMessageBox.Yes:
            return
        
        self.logger.info(f"Starting batch annotation for {len(image_files)} images...")
        
        # Run batch annotation using annotation manager
        self.annotation_manager.annotate_batch(
            self.model_manager.model,
            confidence_threshold,
            selected_classes,
            mode
        )
    
    def on_annotation_progress(self, progress, message):
        """Handle annotation progress event"""
        self.logger.info(message)
    
    def on_annotation_completed(self, path, annotations):
        """Handle annotation completed event"""
        if os.path.isdir(path):
            self.logger.info(f"Batch annotation complete. Processed {len(annotations)} images.")
            self.load_current_image()  # Reload to show current image annotations
        else:
            self.logger.info(f"Annotation complete for {os.path.basename(path)}. Found {len(annotations)} objects.")
            
            # If this is the current image, reload it
            if self.annotation_manager.image_files and path == self.annotation_manager.image_files[self.current_image_index]:
                self.load_current_image()
    
    def save_annotations(self):
        """Save annotations for all images in YOLO format"""
        output_folder = self.ui.lineEdit_output_folder.text()
        if not output_folder:
            QMessageBox.critical(None, "Error", "Please select an output folder to save annotations.")
            return
        
        # Save annotations using annotation manager
        saved_count = self.annotation_manager.save_annotations(output_folder)
        
        if saved_count > 0:
            QMessageBox.information(None, "Save Complete", f"Saved annotations for {saved_count} images.")
            self.logger.info(f"Saved annotations for {saved_count} images to {output_folder}")
        elif saved_count == 0:
            QMessageBox.information(None, "Save Complete", "No annotations to save.")
        else:
            QMessageBox.critical(None, "Error", "Failed to save annotations.")
    
    def on_annotations_saved(self, saved_count, output_folder):
        """Handle annotations saved event"""
        self.logger.info(f"Saved annotations for {saved_count} images to {output_folder}")
    
    def delete_selected_annotation(self):
        """Delete the selected annotation from the current image"""
        current_row = self.ui.tableWidget_annotations.currentRow()
        if current_row < 0:
            QMessageBox.warning(None, "No Selection", "Please select an annotation to delete.")
            return
        
        image_files = self.annotation_manager.image_files
        if not image_files or self.current_image_index < 0 or self.current_image_index >= len(image_files):
            return
        
        img_path = image_files[self.current_image_index]
        
        # Confirm deletion
        confirm = QMessageBox.question(
            None,
            "Confirm Deletion",
            "Are you sure you want to delete this annotation?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            success = self.annotation_manager.delete_annotation(img_path, current_row)
            
            if success:
                self.logger.info(f"Annotation deleted from {os.path.basename(img_path)}")
                self.load_current_image()
            else:
                QMessageBox.critical(None, "Error", "Failed to delete annotation.")
    
    def edit_selected_annotation(self):
        """Open a dialog to edit the selected annotation"""
        QMessageBox.information(None, "Edit Annotation", "Edit annotation feature not yet implemented in PyQt5 version.")
    
    def add_manual_box(self):
        """Enable drawing mode to manually add a bounding box"""
        QMessageBox.information(None, "Add Manual Box", "Manual box drawing feature not yet implemented in PyQt5 version.")
    
    def update_model_info(self, model, model_path, class_names):
        """Update model information"""
        self.ui.lineEdit_yolo_model.setText(model_path)
        self.update_class_checkboxes()
        self.logger.info(f"YOLO model loaded: {os.path.basename(model_path)}")
        
        # Show SAM status if loaded
        if self.sam_model_loaded and hasattr(self.annotation_manager, 'sam_device'):
            device = self.annotation_manager.sam_device or "unknown"
            self.logger.info(f"SAM ready on {device.upper()} - Hybrid mode available")

