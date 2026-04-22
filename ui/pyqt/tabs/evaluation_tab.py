"""Model evaluation tab for PyQt5"""
import os
import cv2
import numpy as np
import threading
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QSplitter
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject

from ui.pyqt.common.ui_utils import main_window_parent
from core.measurement_engine import MeasurementEngine
from utils.image_rendering import (
    draw_edge_measurements, cv_to_pixmap, load_pixmap_scaled,
)


class EvaluationSignals(QObject):
    """Signals for thread-safe UI updates"""
    update_batch_table = pyqtSignal(list)
    update_status = pyqtSignal(str)


class EvaluationTab:
    """Tab for evaluating YOLO models on images and batches"""
    
    def __init__(self, ui, model_manager, logger):
        """
        Initialize the evaluation tab
        
        Args:
            ui: UI object from generated ui_new_mainwindow.py
            model_manager: ModelManager instance
            logger: Logger instance
        """
        self.ui = ui
        self.model_manager = model_manager
        self.logger = logger
        
        # Signals for thread-safe updates
        self.signals = EvaluationSignals()
        self.signals.update_batch_table.connect(self._add_batch_result)
        self.signals.update_status.connect(self._set_status)
        
        # Variables
        self.current_image_path = None
        self.batch_results = []
        # Track last-shown batch preview so Show Labels/Confidence/Measurements can refresh it
        self._last_batch_preview_path = None
        self._last_batch_preview_results = None
        
        # Measurement engine (reusable across tabs / CLI / API)
        self.measurement = MeasurementEngine()
        
        # Expose constants as properties for backward compatibility
        self.MICRONS_PER_PIXEL = self.measurement.microns_per_pixel
        self.BLOCK1_OFFSET = self.measurement.block1_offset
        self.BLOCK2_OFFSET = self.measurement.block2_offset
        self.MEASUREMENT_OFFSET_MICRONS = self.measurement.measurement_offset_microns
        self.judgment_criteria = self.measurement.judgment_criteria
        
        # Connect UI signals to handlers
        self._connect_signals()
        
        # Connect to model manager events
        self.model_manager.on_model_loaded.subscribe(self.update_model_info)
        self.model_manager.on_inference_completed.subscribe(self.on_inference_completed)
        
        # Initialize UI state
        self._initialize_ui()
        
        # Optimize layout for better image preview
        self._optimize_layout()
    
    def _connect_signals(self):
        """Connect Qt signals to handler methods"""
        # Model selection
        self.ui.pushButton_eval_model_browse.clicked.connect(self._browse_model)
        self.ui.pushButton_load_model.clicked.connect(self.load_model)
        
        # Confidence slider
        self.ui.slider_confidence.valueChanged.connect(self._on_confidence_changed)
        
        # Display options
        self.ui.checkBox_show_labels.stateChanged.connect(self.update_display_options)
        self.ui.checkBox_show_conf.stateChanged.connect(self.update_display_options)
        self.ui.checkBox_show_measurements.stateChanged.connect(self.update_display_options)
        
        # Single image tab (removed from UI; connections skipped if widgets absent)
        if hasattr(self.ui, "pushButton_image_browse"):
            self.ui.pushButton_image_browse.clicked.connect(self._browse_image)
        if hasattr(self.ui, "pushButton_evaluate_image"):
            self.ui.pushButton_evaluate_image.clicked.connect(self.evaluate_single_image)
        
        # Batch tab
        self.ui.pushButton_batch_browse.clicked.connect(self._browse_batch_folder)
        self.ui.pushButton_evaluate_batch.clicked.connect(self.evaluate_batch)
        self.ui.tableWidget_batch_results.doubleClicked.connect(self._on_batch_result_double_click)
        
        # Measurement settings - Removed from UI (using default constants)
        # Note: MICRONS_PER_PIXEL, BLOCK1_OFFSET, BLOCK2_OFFSET are set as constants
        # in __init__ and can be modified programmatically if needed
    
    def _initialize_ui(self):
        """Initialize UI with default values"""
        # Set initial model path if available
        if self.model_manager.model_path:
            self.ui.lineEdit_eval_model.setText(self.model_manager.model_path)
        
        # Set initial confidence display
        self._on_confidence_changed(self.ui.slider_confidence.value())
    
    def _optimize_layout(self):
        """Optimize layout to maximize image preview space"""
        try:
            from PyQt5.QtWidgets import QSizePolicy
            
            # Minimize the Image Selection and Detection Results sections
            if hasattr(self.ui, 'groupBox_image_selection'):
                self.ui.groupBox_image_selection.setMaximumHeight(120)
                self.ui.groupBox_image_selection.setSizePolicy(
                    QSizePolicy.Preferred, QSizePolicy.Fixed)
            
            if hasattr(self.ui, 'groupBox_results'):
                self.ui.groupBox_results.setMaximumHeight(120)
                self.ui.groupBox_results.setSizePolicy(
                    QSizePolicy.Preferred, QSizePolicy.Fixed)
            
            # Minimize the Detection Results text area
            if hasattr(self.ui, 'textEdit_results'):
                self.ui.textEdit_results.setMaximumHeight(60)
                self.ui.textEdit_results.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            # Set the Image Preview group box to expand and take all available space
            if hasattr(self.ui, 'groupBox_image_preview'):
                self.ui.groupBox_image_preview.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Set size policies for image viewers to expand
            if hasattr(self.ui, 'label_image_viewer'):
                self.ui.label_image_viewer.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.ui.label_image_viewer.setScaledContents(False)
                self.ui.label_image_viewer.setAlignment(Qt.AlignCenter)
            
            if hasattr(self.ui, 'label_batch_image_viewer'):
                self.ui.label_batch_image_viewer.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.ui.label_batch_image_viewer.setScaledContents(False)
                self.ui.label_batch_image_viewer.setAlignment(Qt.AlignCenter)
            
            # Ensure scroll areas expand properly
            if hasattr(self.ui, 'scrollArea_image_preview'):
                self.ui.scrollArea_image_preview.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            if hasattr(self.ui, 'scrollArea_batch_preview'):
                self.ui.scrollArea_batch_preview.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Set stretch factors on the main vertical layout to prioritize image preview
            if hasattr(self.ui, 'verticalLayout_single_main'):
                # The layout has: horizontalLayout_controls (0), groupBox_image_preview (1)
                self.ui.verticalLayout_single_main.setStretch(0, 0)  # Controls: no stretch
                self.ui.verticalLayout_single_main.setStretch(1, 1)  # Image preview: max stretch
            
            # Minimize spacing and margins
            if hasattr(self.ui, 'horizontalLayout_controls'):
                self.ui.horizontalLayout_controls.setSpacing(5)
                self.ui.horizontalLayout_controls.setContentsMargins(0, 0, 0, 0)
            
            if hasattr(self.ui, 'verticalLayout_single_main'):
                self.ui.verticalLayout_single_main.setSpacing(5)
            
            self.logger.info("Layout optimized for better image preview")
        except Exception as e:
            self.logger.warning(f"Could not fully optimize layout: {str(e)}")
    
    def _browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            main_window_parent(self.logger),
            "Select Model File",
            "",
            "PyTorch Models (*.pt);;All files (*.*)"
        )
        
        if file_path:
            self.ui.lineEdit_eval_model.setText(file_path)
            self.logger.info(f"Model selected: {file_path}")
    
    def _browse_image(self):
        """Browse for image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            main_window_parent(self.logger),
            "Select Image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif);;All files (*.*)"
        )
        
        if file_path:
            self.ui.lineEdit_image_path.setText(file_path)
            self.current_image_path = file_path
            self._load_image_to_viewer(file_path, self.ui.label_image_viewer)
            self.logger.info(f"Image loaded: {os.path.basename(file_path)}")
    
    def _browse_batch_folder(self):
        """Browse for batch folder"""
        folder_path = QFileDialog.getExistingDirectory(
            main_window_parent(self.logger),
            "Select Images Folder"
        )
        
        if folder_path:
            self.ui.lineEdit_batch_folder.setText(folder_path)
    
    def _on_confidence_changed(self, value):
        """Handle confidence slider change"""
        confidence = value / 100.0
        self.ui.label_confidence_value.setText(f"{confidence:.2f}")
    
    def _load_image_to_viewer(self, image_path, label_widget):
        """Load image to a QLabel widget (scrollable) at 50% size"""
        scaled_pixmap = load_pixmap_scaled(image_path, scale=0.5)
        if scaled_pixmap:
            label_widget.setPixmap(scaled_pixmap)
            label_widget.resize(scaled_pixmap.size())
        else:
            label_widget.setText("Failed to load image")
    
    def _calculate_measurements(self, results):
        """
        Calculate Y-difference and judgment from detection results.
        
        Delegates to :class:`MeasurementEngine` for the actual calculation.
        
        Returns:
            tuple: (y_diff_microns, judgment, microns_per_pixel)
        """
        m = self.measurement.calculate_from_results(results, self.model_manager.get_class_name)
        return m["y_diff_microns"], m["judgment"], m["microns_per_pixel"]
    
    def _draw_annotated_image(self, image_path, results, label_widget):
        """Draw annotated image with detection results using shared rendering."""
        img = cv2.imread(image_path)
        if img is None:
            label_widget.setText("Failed to load image")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        show_labels = self.ui.checkBox_show_labels.isChecked()
        show_conf = self.ui.checkBox_show_conf.isChecked()
        show_measurements = self.ui.checkBox_show_measurements.isChecked()

        if hasattr(results.boxes, "xyxy") and len(results.boxes.xyxy) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            boxes_xywh = results.boxes.xywh.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()

            m = self.measurement.calculate(
                boxes_xyxy, boxes_xywh, classes, self.model_manager.get_class_name
            )

            draw_edge_measurements(
                img_rgb, boxes_xyxy, boxes_xywh, classes, confidences,
                self.model_manager.get_class_name, m,
                show_labels=show_labels, show_conf=show_conf,
                show_measurements=show_measurements,
                microns_per_pixel=self.MICRONS_PER_PIXEL,
            )

        pixmap = cv_to_pixmap(img_rgb)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                pixmap.width() // 2, pixmap.height() // 2,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            label_widget.setPixmap(scaled)
            label_widget.resize(scaled.size())
        else:
            label_widget.setText("Failed to create annotated image")
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.ui.lineEdit_eval_model.text()
        if not model_path:
            QMessageBox.critical(None, "Error", "Please select a model file")
            return
        
        # Load the model using the model manager
        success = self.model_manager.load_model(model_path)
        if success:
            QMessageBox.information(None, "Success", f"Model loaded: {os.path.basename(model_path)}")
        else:
            QMessageBox.critical(None, "Error", "Failed to load model")
    
    def update_model_info(self, model, model_path, class_names):
        """Update model information"""
        self.ui.lineEdit_eval_model.setText(model_path)
        self.logger.info(f"Model loaded: {os.path.basename(model_path)}")
    
    def update_display_options(self):
        """Update display options and refresh the current display (batch preview or single image)."""
        try:
            # Refresh batch preview if one was last shown (uses current Show Labels / Confidence / Measurements)
            if self._last_batch_preview_path and self._last_batch_preview_results is not None:
                self._draw_annotated_image(
                    self._last_batch_preview_path,
                    self._last_batch_preview_results,
                    self.ui.label_batch_image_viewer,
                )
            # If we have a single-image path and viewer (e.g. legacy Single Image tab), redraw it
            if (
                self.current_image_path
                and self.model_manager.model
                and hasattr(self.ui, "label_image_viewer")
            ):
                confidence = self.ui.slider_confidence.value() / 100.0
                results = self.model_manager.run_inference(
                    self.current_image_path, confidence=confidence
                )
                if results and len(results) > 0 and hasattr(results[0].boxes, "cls"):
                    self._draw_annotated_image(
                        self.current_image_path, results[0], self.ui.label_image_viewer
                    )
        except Exception as e:
            self.logger.error(f"Error updating display: {str(e)}")
    
    def evaluate_single_image(self):
        """Run inference on a single image"""
        if not self.model_manager.model:
            QMessageBox.critical(None, "Error", "Please load a model first")
            return
        
        if not self.current_image_path:
            QMessageBox.critical(None, "Error", "Please select an image first")
            return
        
        try:
            # Get confidence threshold
            confidence = self.ui.slider_confidence.value() / 100.0
            
            # Run inference
            results = self.model_manager.run_inference(self.current_image_path, confidence=confidence)
            
            if results is None:
                QMessageBox.critical(None, "Error", "Inference failed")
                return
            
            # Display results
            self._display_single_image_results(results)
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to evaluate image: {str(e)}")
            self.logger.error(f"Evaluation error: {str(e)}")
    
    def _display_single_image_results(self, results):
        """Display results for single image evaluation"""
        # Clear previous results
        self.ui.textEdit_results.clear()
        
        if len(results) == 0 or not hasattr(results[0].boxes, 'cls'):
            self.ui.textEdit_results.setText("No objects detected.")
            # Reload plain image if no detections
            if self.current_image_path:
                self._load_image_to_viewer(self.current_image_path, self.ui.label_image_viewer)
            return
        
        # Count detections by class
        class_counts = {}
        for cls in results[0].boxes.cls:
            class_name = self.model_manager.get_class_name(int(cls.item()))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Display detection counts - More compact format
        result_text = f"Detected {len(results[0].boxes)} objects: "
        result_text += ", ".join([f"{name}({count})" for name, count in class_counts.items()])
        
        self.ui.textEdit_results.setText(result_text)
        
        # Draw annotated image
        if self.current_image_path:
            self._draw_annotated_image(self.current_image_path, results[0], self.ui.label_image_viewer)
    
    def on_inference_completed(self, results, image_path):
        """Handle inference completed event"""
        if image_path != self.current_image_path:
            return
        
        self._display_single_image_results(results)
    
    def evaluate_batch(self):
        """Run inference on a batch of images"""
        if not self.model_manager.model:
            QMessageBox.critical(None, "Error", "Please load a model first")
            return
        
        folder_path = self.ui.lineEdit_batch_folder.text()
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.critical(None, "Error", "Please select a valid folder")
            return
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith(image_extensions)]
        
        if not image_files:
            QMessageBox.critical(None, "Error", "No image files found in the selected folder")
            return
        
        # Clear previous results and stored preview (so display options only refresh after a new double-click)
        self.ui.tableWidget_batch_results.setRowCount(0)
        self.batch_results = []
        self._last_batch_preview_path = None
        self._last_batch_preview_results = None

        # Start batch processing in a separate thread
        threading.Thread(
            target=self._batch_processing_thread,
            args=(image_files,),
            daemon=True
        ).start()
    
    def _batch_processing_thread(self, image_files):
        """Process batch of images in a separate thread"""
        total_images = len(image_files)
        confidence = self.ui.slider_confidence.value() / 100.0
        
        self.signals.update_status.emit(f"Processing {total_images} images...")
        
        for i, img_path in enumerate(image_files):
            try:
                self.signals.update_status.emit(
                    f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)}")
                
                # Run inference
                import time
                start_time = time.time()
                results = self.model_manager.run_inference(img_path, confidence=confidence)
                
                if results is None:
                    continue
                
                elapsed = time.time() - start_time
                
                # Analyze results
                num_objects = len(results[0].boxes) if len(results) > 0 and hasattr(results[0].boxes, 'cls') else 0
                classes = {}
                
                if num_objects > 0:
                    for cls in results[0].boxes.cls:
                        class_name = self.model_manager.get_class_name(int(cls.item()))
                        classes[class_name] = classes.get(class_name, 0) + 1
                
                classes_str = ", ".join([f"{name} ({count})" for name, count in classes.items()])
                
                # Calculate Y-diff and judgment
                y_diff_microns, judgment, microns_per_pixel = None, None, None
                y_diff_str = "N/A"
                judgment_str = "N/A"
                
                if len(results) > 0:
                    y_diff_microns, judgment, microns_per_pixel = self._calculate_measurements(results[0])
                    if y_diff_microns is not None:
                        y_diff_str = f"{y_diff_microns:.2f}"
                        judgment_str = judgment
                
                # Store result
                result_item = {
                    "path": img_path,
                    "objects": num_objects,
                    "classes": classes_str if classes_str else "None",
                    "y_diff": y_diff_str,
                    "judgment": judgment_str,
                    "time": f"{elapsed:.2f}s",
                    "results": results
                }
                self.batch_results.append(result_item)
                
                # Add to table (via signal for thread safety)
                self.signals.update_batch_table.emit([
                    os.path.basename(img_path),
                    str(num_objects),
                    classes_str if classes_str else "None",
                    y_diff_str,
                    judgment_str,
                    f"{elapsed:.2f}s"
                ])
                
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
        
        self.signals.update_status.emit(
            f"Batch processing complete. Processed {len(self.batch_results)}/{total_images} images.")
    
    def _add_batch_result(self, row_data):
        """Add a row to the batch results table (called from signal)"""
        row_position = self.ui.tableWidget_batch_results.rowCount()
        self.ui.tableWidget_batch_results.insertRow(row_position)
        
        for col, data in enumerate(row_data):
            self.ui.tableWidget_batch_results.setItem(row_position, col, QTableWidgetItem(str(data)))
    
    def _set_status(self, message):
        """Set status message (called from signal)"""
        self.logger.info(message)
    
    def _on_batch_result_double_click(self, index):
        """Handle double-click on batch result"""
        row = index.row()
        if 0 <= row < len(self.batch_results):
            result = self.batch_results[row]
            img_path = result["path"]
            results = result.get("results")
            # Show the image name directly above the batch preview image
            if hasattr(self.ui, "label_batch_image_name"):
                self.ui.label_batch_image_name.setText(os.path.basename(img_path))
            # Store for update_display_options (Show Labels / Confidence / Measurements)
            self._last_batch_preview_path = img_path
            self._last_batch_preview_results = results[0] if (results and len(results) > 0 and hasattr(results[0].boxes, "cls")) else None
            # Load annotated image to batch viewer (respects current checkbox state)
            if self._last_batch_preview_results is not None:
                self._draw_annotated_image(img_path, self._last_batch_preview_results, self.ui.label_batch_image_viewer)
            else:
                self._load_image_to_viewer(img_path, self.ui.label_batch_image_viewer)

