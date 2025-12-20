"""Model evaluation tab for PyQt5"""
import os
import cv2
import numpy as np
import threading
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject


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
        
        # Measurement constants (matching app_fastapi.py)
        self.MICRONS_PER_PIXEL = 2.3
        self.BLOCK1_OFFSET = 0.0
        self.BLOCK2_OFFSET = 0.0
        self.MEASUREMENT_OFFSET_MICRONS = 5.0  # Offset to apply to final measurement
        self.judgment_criteria = {"good": 10, "acceptable": 20}
        
        # Connect UI signals to handlers
        self._connect_signals()
        
        # Connect to model manager events
        self.model_manager.on_model_loaded.subscribe(self.update_model_info)
        self.model_manager.on_inference_completed.subscribe(self.on_inference_completed)
        
        # Initialize UI state
        self._initialize_ui()
    
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
        
        # Single image tab
        self.ui.pushButton_image_browse.clicked.connect(self._browse_image)
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
    
    def _browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
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
            None,
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
            None,
            "Select Images Folder"
        )
        
        if folder_path:
            self.ui.lineEdit_batch_folder.setText(folder_path)
    
    def _on_confidence_changed(self, value):
        """Handle confidence slider change"""
        confidence = value / 100.0
        self.ui.label_confidence_value.setText(f"{confidence:.2f}")
    
    def _load_image_to_viewer(self, image_path, label_widget):
        """Load image to a QLabel widget (scrollable)"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Set pixmap at full size to enable scrolling
            label_widget.setPixmap(pixmap)
            # Resize label to pixmap size to enable scrolling
            label_widget.resize(pixmap.size())
        else:
            label_widget.setText("Failed to load image")
    
    def _draw_annotated_image(self, image_path, results, label_widget):
        """Draw annotated image with detection results"""
        from PyQt5.QtGui import QPainter, QPen, QFont, QColor
        from datetime import datetime
        
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            label_widget.setText("Failed to load image")
            return
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Get display options
        show_labels = self.ui.checkBox_show_labels.isChecked()
        show_conf = self.ui.checkBox_show_conf.isChecked()
        show_measurements = self.ui.checkBox_show_measurements.isChecked()
        
        # Variables for edge detection and calibration
        block1_edge_y = None
        block2_edge_y = None
        calibration_marker_width_px = None
        microns_per_pixel = self.MICRONS_PER_PIXEL
        
        # Draw bounding boxes and labels
        if hasattr(results.boxes, 'xyxy') and len(results.boxes.xyxy) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            
            # Get xywh format for center calculations
            boxes_xywh = results.boxes.xywh.cpu().numpy()
            
            for i, (box, box_xywh, cls, conf) in enumerate(zip(boxes, boxes_xywh, classes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                x_center, y_center, width_box, height_box = box_xywh
                class_name = self.model_manager.get_class_name(int(cls))
                
                # Calculate center and edge position
                x_center = int(x_center)
                y_center = int(y_center)
                height = int(height_box)
                width = width_box
                
                # Check for specific classes (matching app_fastapi.py)
                if class_name == "block1_edge15":
                    # Calculate edge position (bottom of detection box)
                    edge_y = int(y_center + height / 2)
                    # Apply offset and store edge position (matching app_fastapi.py)
                    block1_edge_y = edge_y + (self.BLOCK1_OFFSET / microns_per_pixel)
                    # Draw line from x_center - 150 to x_center + 150 (matching app_fastapi.py)
                    cv2.line(img_rgb, 
                            (int(x_center - 150), edge_y),
                            (int(x_center + 150), edge_y),
                            (255, 0, 0),  # Red in RGB (matching app_fastapi.py)
                            2)
                    label_y_pos = edge_y
                    color = (255, 0, 0)
                    is_edge_class = True
                elif class_name == "block2_edge15":
                    # Calculate edge position (bottom of detection box)
                    edge_y = int(y_center + height / 2)
                    # Apply offset and store edge position (matching app_fastapi.py)
                    block2_edge_y = edge_y + (self.BLOCK2_OFFSET / microns_per_pixel)
                    # Draw line from x_center - 150 to x_center + 150 (matching app_fastapi.py)
                    cv2.line(img_rgb,
                            (int(x_center - 150), edge_y),
                            (int(x_center + 150), edge_y),
                            (0, 255, 255),  # Cyan in RGB (matching app_fastapi.py)
                            2)
                    label_y_pos = edge_y
                    color = (0, 255, 255)
                    is_edge_class = True
                elif class_name == "cal_mark":
                    # Store calibration marker width (matching app_fastapi.py)
                    # Convert to scalar if it's a numpy array/tensor
                    if hasattr(width, 'item'):
                        calibration_marker_width_px = width.item()
                    else:
                        calibration_marker_width_px = float(width)
                    # Don't draw calibration marker, just store its value
                    continue
                elif class_name == "block1_15" or class_name == "block2_15":
                    # Track block positions but don't draw special lines
                    block_y = int(y_center + height / 2)
                    # Regular bounding box
                    color = (0, 255, 0)  # Green in RGB
                    thickness = 2
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)
                    label_y_pos = y1
                    is_edge_class = False
                else:
                    # Regular bounding box for other classes
                    color = (0, 255, 0)  # Green in RGB
                    thickness = 2
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thickness)
                    label_y_pos = y1
                    is_edge_class = False
                
                # Prepare label text
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_conf:
                    label_parts.append(f"{conf:.2f}")
                
                if label_parts:
                    label_text = " ".join(label_parts)
                    
                    # Calculate text size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9  # Increased from 0.6 for better visibility
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, font, font_scale, font_thickness)
                    
                    # Position label based on detection type
                    if is_edge_class:
                        # For edge classes, position label to the right of the line
                        label_x = x_center + 150
                        label_y = label_y_pos
                    else:
                        # For regular boxes, position at top
                        label_x = x1
                        label_y = label_y_pos
                    
                    # Draw background rectangle for text
                    cv2.rectangle(img_rgb, 
                                (label_x, label_y - text_height - baseline - 5),
                                (label_x + text_width + 5, label_y),
                                color, -1)
                    
                    # Draw text
                    cv2.putText(img_rgb, label_text,
                              (label_x + 2, label_y - baseline - 2),
                              font, font_scale, (255, 255, 255), font_thickness)
                
                # Add measurements if enabled
                if show_measurements:
                    if is_edge_class:
                        # For edge classes, show edge position (Y coordinate)
                        meas_text = f"Edge Y: {label_y_pos}px"
                        meas_x = x_center - 80
                        meas_y = label_y_pos + 25
                    else:
                        # Regular measurements for bounding boxes
                        width_px = x2 - x1
                        height_px = y2 - y1
                        
                        # Convert to microns
                        width_microns = width_px * self.MICRONS_PER_PIXEL
                        height_microns = height_px * self.MICRONS_PER_PIXEL
                        
                        meas_text = f"W:{width_microns:.1f}μm H:{height_microns:.1f}μm"
                        meas_x = x1
                        meas_y = y2 + 20
                    
                    # Draw measurement text
                    cv2.putText(img_rgb, meas_text,
                              (meas_x, meas_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Calculate microns per pixel from calibration marker if detected (matching app_fastapi.py)
            if calibration_marker_width_px:
                microns_per_pixel = 1000.0 / calibration_marker_width_px
                self.logger.info(f"Calibration: cal_mark width = {calibration_marker_width_px:.2f}px, "
                               f"microns/px = {microns_per_pixel:.2f}")
                
                # Check if microns per pixel is too high (matching app_fastapi.py)
                if microns_per_pixel > 10:
                    self.logger.warning("Microns per pixel too high, suggesting focus adjustment")
            else:
                self.logger.warning("cal_mark not detected, using default microns per pixel")
            
            # Display microns per pixel on image (top left)
            cal_text = f"{microns_per_pixel:.2f} um/pixel"
            cv2.putText(img_rgb, cal_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Calculate and display Y-difference if both edges detected
            if block1_edge_y is not None and block2_edge_y is not None:
                # Signed difference: block1_edge - block2_edge (matching FastAPI)
                y_diff_pixels = block1_edge_y - block2_edge_y
                y_diff_microns = (y_diff_pixels * microns_per_pixel) + self.MEASUREMENT_OFFSET_MICRONS
                
                # Determine judgment based on signed difference (matching FastAPI logic)
                if y_diff_microns < self.judgment_criteria["good"]:
                    judgment = "Good"
                    judgment_color = (0, 255, 0)  # Green
                elif y_diff_microns < self.judgment_criteria["acceptable"]:
                    judgment = "Acceptable"
                    judgment_color = (0, 165, 255)  # Orange
                else:
                    judgment = "No Good"
                    judgment_color = (0, 0, 255)  # Red
                
                # Calculate position for text (between the two edges)
                text_x = w // 2 + 250
                text_y = int((block1_edge_y + block2_edge_y) / 2)
                
                # Draw Y-difference text (matching app_fastapi.py format)
                diff_text = f"{y_diff_microns:.2f} microns"
                cv2.putText(img_rgb, diff_text,
                           (text_x - 100, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw judgment text (matching app_fastapi.py format)
                judgment_text = f"Judgment: {judgment}"
                cv2.putText(img_rgb, judgment_text,
                           (text_x - 100, text_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, judgment_color, 2)
                
                # Draw timestamp
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(img_rgb, f"Checked on: {current_datetime}",
                           (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        
        # Convert to QPixmap
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Display at full size to enable scrolling
        if not pixmap.isNull():
            # Set pixmap at full size to enable scrolling
            label_widget.setPixmap(pixmap)
            # Resize label to pixmap size to enable scrolling
            label_widget.resize(pixmap.size())
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
        """Update display options and refresh the current display"""
        # If we have a current image with results, redraw it
        if self.current_image_path and self.model_manager.model:
            try:
                # Get confidence threshold
                confidence = self.ui.slider_confidence.value() / 100.0
                
                # Re-run inference to get results
                results = self.model_manager.run_inference(self.current_image_path, confidence=confidence)
                
                if results and len(results) > 0 and hasattr(results[0].boxes, 'cls'):
                    # Redraw with updated options
                    self._draw_annotated_image(self.current_image_path, results[0], self.ui.label_image_viewer)
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
        
        # Display detection counts
        result_text = f"Found {len(results[0].boxes)} objects:\n\n"
        for class_name, count in class_counts.items():
            result_text += f"- {class_name}: {count}\n"
        
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
        
        # Clear previous results
        self.ui.tableWidget_batch_results.setRowCount(0)
        self.batch_results = []
        
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
                
                # Store result
                result_item = {
                    "path": img_path,
                    "objects": num_objects,
                    "classes": classes_str if classes_str else "None",
                    "y_diff": "N/A",
                    "judgment": "N/A",
                    "time": f"{elapsed:.2f}s",
                    "results": results
                }
                self.batch_results.append(result_item)
                
                # Add to table (via signal for thread safety)
                self.signals.update_batch_table.emit([
                    os.path.basename(img_path),
                    str(num_objects),
                    classes_str if classes_str else "None",
                    "N/A",
                    "N/A",
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
            
            # Load annotated image to batch viewer
            if results and len(results) > 0 and hasattr(results[0].boxes, 'cls'):
                self._draw_annotated_image(img_path, results[0], self.ui.label_batch_image_viewer)
            else:
                self._load_image_to_viewer(img_path, self.ui.label_batch_image_viewer)

