"""Dataset configuration tab for PyQt5"""
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QLineEdit, QHBoxLayout, QWidget, QPushButton, QDialog, QVBoxLayout, QComboBox, QFormLayout
from PyQt5.QtCore import pyqtSignal
import yaml

from utils.event import Event


class DatasetTab:
    """Tab for dataset configuration and management"""
    
    def __init__(self, ui, dataset_manager, logger):
        """
        Initialize the dataset tab
        
        Args:
            ui: UI object from generated ui_new_mainwindow.py
            dataset_manager: DatasetManager instance
            logger: Logger instance
        """
        self.ui = ui
        self.dataset_manager = dataset_manager
        self.logger = logger
        
        # Events
        self.on_dataset_changed = Event()
        
        # Connect UI signals to handlers
        self._connect_signals()
        
        # Initialize UI state
        self._initialize_ui()
    
    def _connect_signals(self):
        """Connect Qt signals to handler methods"""
        # YAML file browser
        self.ui.pushButton_yaml_browse.clicked.connect(self._browse_yaml_file)
        self.ui.pushButton_load_yaml.clicked.connect(self.load_yaml)
        self.ui.pushButton_create_new.clicked.connect(self.create_new_dataset)
        
        # Dataset path browser
        self.ui.pushButton_dataset_browse.clicked.connect(self._browse_dataset_path)
        
        # Save YAML changes
        self.ui.pushButton_save_yaml.clicked.connect(self.save_yaml_changes)
        
        # Quick select combobox
        self.ui.comboBox_quick_select.currentTextChanged.connect(self._on_quick_select_changed)
    
    def _initialize_ui(self):
        """Initialize UI with default values"""
        # Set initial YAML path if available
        yaml_path = self.dataset_manager.get_yaml_path()
        if yaml_path:
            self.ui.lineEdit_yaml_path.setText(yaml_path)
    
    def _browse_yaml_file(self):
        """Browse for YAML file"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select YAML file",
            "",
            "YAML files (*.yaml);;All files (*.*)"
        )
        
        if file_path:
            self.ui.lineEdit_yaml_path.setText(file_path)
            self.dataset_manager.set_yaml_path(file_path)
            self.logger.info(f"YAML file selected: {file_path}")
    
    def _browse_dataset_path(self):
        """Browse for dataset directory"""
        folder_path = QFileDialog.getExistingDirectory(
            None,
            "Select Dataset Folder"
        )
        
        if folder_path:
            self.ui.lineEdit_dataset_path.setText(folder_path)
    
    def _on_quick_select_changed(self, text):
        """Handle quick select combobox change"""
        if text:
            self.ui.lineEdit_yaml_path.setText(text)
    
    def load_yaml(self):
        """Load a YAML file and update the UI"""
        yaml_file = self.ui.lineEdit_yaml_path.text()
        if not yaml_file:
            QMessageBox.critical(None, "Error", "Please select a YAML file")
            return
        
        try:
            # Load YAML file
            dataset_content = self.dataset_manager.load_yaml(yaml_file)
            
            # Update UI with dataset information
            self.ui.lineEdit_dataset_path.setText(dataset_content.get('path', ''))
            
            # Clear previous class entries
            layout = self.ui.verticalLayout_classes_content
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            # Add new class entries
            classes = dataset_content.get('names', [])
            for i, class_name in enumerate(classes):
                class_widget = QWidget()
                class_layout = QHBoxLayout(class_widget)
                class_layout.setContentsMargins(0, 0, 0, 0)
                
                label = QLabel(f"Class {i}:")
                entry = QLineEdit(class_name)
                entry.setObjectName(f"class_entry_{i}")
                
                class_layout.addWidget(label)
                class_layout.addWidget(entry)
                
                layout.addWidget(class_widget)
            
            # Add button for adding a new class
            add_button = QPushButton("Add Class")
            add_button.clicked.connect(self.add_class)
            layout.addWidget(add_button)
            
            # Add spacer
            layout.addStretch()
            
            self.logger.info(f"Loaded YAML file: {os.path.basename(yaml_file)}")
            
            # Trigger event
            self.on_dataset_changed(dataset_content)
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load YAML file: {str(e)}")
            self.logger.error(f"Error loading YAML: {str(e)}")
    
    def add_class(self):
        """Add a new class to the list"""
        layout = self.ui.verticalLayout_classes_content
        
        # Count current class entries
        class_count = 0
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget and isinstance(widget, QWidget):
                # Check if it has a LineEdit (class entry)
                for child in widget.findChildren(QLineEdit):
                    if child.objectName().startswith("class_entry_"):
                        class_count += 1
                        break
        
        # Remove the "Add Class" button and spacer temporarily
        items_to_restore = []
        while layout.count():
            item = layout.takeAt(layout.count() - 1)
            items_to_restore.insert(0, item)
            if isinstance(item.widget(), QPushButton):
                break
        
        # Add new class entry
        class_widget = QWidget()
        class_layout = QHBoxLayout(class_widget)
        class_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(f"Class {class_count}:")
        entry = QLineEdit()
        entry.setObjectName(f"class_entry_{class_count}")
        
        class_layout.addWidget(label)
        class_layout.addWidget(entry)
        
        layout.addWidget(class_widget)
        
        # Restore the "Add Class" button and spacer
        for item in items_to_restore:
            if item.widget():
                layout.addWidget(item.widget())
            elif item.spacerItem():
                layout.addItem(item.spacerItem())
    
    def save_yaml_changes(self):
        """Save changes to the YAML file"""
        yaml_file = self.ui.lineEdit_yaml_path.text()
        if not yaml_file:
            QMessageBox.critical(None, "Error", "No YAML file loaded")
            return
        
        try:
            # Get current dataset content
            dataset_content = self.dataset_manager.get_dataset_content()
            if not dataset_content:
                QMessageBox.critical(None, "Error", "No YAML file loaded")
                return
            
            # Update path
            dataset_content['path'] = self.ui.lineEdit_dataset_path.text()
            
            # Update classes
            classes = []
            layout = self.ui.verticalLayout_classes_content
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget:
                    for line_edit in widget.findChildren(QLineEdit):
                        if line_edit.objectName().startswith("class_entry_"):
                            text = line_edit.text()
                            if text:
                                classes.append(text)
            
            dataset_content['names'] = classes
            
            # Save to file
            self.dataset_manager.save_yaml(yaml_file, dataset_content)
            
            QMessageBox.information(None, "Success", "YAML file updated successfully")
            self.logger.info(f"Saved changes to {os.path.basename(yaml_file)}")
            
            # Trigger event
            self.on_dataset_changed(dataset_content)
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save YAML file: {str(e)}")
            self.logger.error(f"Error saving YAML: {str(e)}")
    
    def create_new_dataset(self):
        """Create a new YAML dataset file"""
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Create New Dataset YAML",
            "",
            "YAML files (*.yaml);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        # Create dataset structure dialog
        dialog = QDialog()
        dialog.setWindowTitle("Create Dataset")
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Form layout for inputs
        form_layout = QFormLayout()
        
        # Dataset path
        path_entry = QLineEdit("./datasets/custom")
        form_layout.addRow("Dataset Path:", path_entry)
        
        # Dataset format
        format_combo = QComboBox()
        format_combo.addItems(["YOLOv8", "YOLOv11"])
        form_layout.addRow("Dataset Format:", format_combo)
        
        layout.addLayout(form_layout)
        
        # Classes input area
        classes_label = QLabel("Classes (one per line):")
        layout.addWidget(classes_label)
        
        from PyQt5.QtWidgets import QTextEdit
        classes_text = QTextEdit()
        classes_text.setPlaceholderText("Enter class names, one per line\nExample:\nclass1\nclass2\nclass3")
        layout.addWidget(classes_text)
        
        # Buttons
        from PyQt5.QtWidgets import QDialogButtonBox
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            # Get class names
            classes_str = classes_text.toPlainText()
            classes = [line.strip() for line in classes_str.split('\n') if line.strip()]
            
            if not classes:
                QMessageBox.critical(None, "Error", "At least one class is required")
                return
            
            # Create YAML content
            content = {
                'path': path_entry.text(),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': classes
            }
            
            # Add YOLOv11 specific settings if selected
            if format_combo.currentText() == "YOLOv11":
                content['format_version'] = 11
                content['advanced_augmentation'] = True
            
            try:
                # Save the YAML file
                self.dataset_manager.save_yaml(file_path, content)
                
                QMessageBox.information(None, "Success", f"Dataset YAML created: {file_path}")
                
                # Update the UI
                self.ui.lineEdit_yaml_path.setText(file_path)
                self.load_yaml()
                
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to create YAML file: {str(e)}")

