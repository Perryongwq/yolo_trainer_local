"""Dataset configuration tab for PyQt5"""
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QLineEdit, QHBoxLayout, QWidget, QPushButton, QDialog, QVBoxLayout, QComboBox, QFormLayout
from PyQt5.QtCore import Qt
import yaml

from utils.event import Event
from ui.pyqt.common.ui_utils import main_window_parent


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
        
        # Dataset path browser
        self.ui.pushButton_dataset_browse.clicked.connect(self._browse_dataset_path)
        
        # Save YAML changes
        self.ui.pushButton_save_yaml.clicked.connect(self.save_yaml_changes)
        
        # Quick select combobox
        self.ui.comboBox_quick_select.currentTextChanged.connect(self._on_quick_select_changed)
    
    def _initialize_ui(self):
        """Initialize UI with default values"""
        # Add Class / Delete Class / Save YAML on one row (outside Classes box)
        self.ui.verticalLayout_3.removeWidget(self.ui.pushButton_save_yaml)
        self._class_buttons_row = QWidget()
        self._class_buttons_row.setObjectName("class_buttons_row")
        button_layout = QHBoxLayout(self._class_buttons_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton("Add Class")
        add_btn.clicked.connect(self.add_class)
        new_yaml_btn = QPushButton("New YAML")
        new_yaml_btn.clicked.connect(self.create_new_dataset)
        save_as_btn = QPushButton("Save As")
        save_as_btn.clicked.connect(self.save_yaml_as)
        button_layout.addWidget(add_btn)
        button_layout.addWidget(new_yaml_btn)
        button_layout.addWidget(save_as_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.ui.pushButton_save_yaml)
        self.ui.verticalLayout_3.insertWidget(2, self._class_buttons_row)

        # Add extra quick-select dataset entries not present in the generated UI
        for extra_yaml in ("config/21dataset.yaml", "config/31dataset.yaml", "config/32dataset.yaml"):
            self._add_to_quick_select(extra_yaml)

        # Set initial YAML path if available
        yaml_path = self.dataset_manager.get_yaml_path()
        if yaml_path:
            self.ui.lineEdit_yaml_path.setText(yaml_path)
    
    def _browse_yaml_file(self):
        """Browse for YAML file"""
        file_path, _ = QFileDialog.getOpenFileName(
            main_window_parent(self.logger),
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
            main_window_parent(self.logger),
            "Select Dataset Folder"
        )
        
        if folder_path:
            self.ui.lineEdit_dataset_path.setText(folder_path)
    
    def _on_quick_select_changed(self, text):
        """Handle quick select combobox change"""
        if text:
            self.ui.lineEdit_yaml_path.setText(text)
            self.dataset_manager.set_yaml_path(text)

    def _classes_in_chronological_order(self, raw_names):
        """Return class names as a list in index order (class 0, 1, 2, ...). Handles YAML 'names' as list or dict."""
        if raw_names is None:
            return []
        if isinstance(raw_names, list):
            return list(raw_names)
        if isinstance(raw_names, dict):
            return [raw_names[k] for k in sorted(raw_names.keys())]
        return []
    
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
            
            # Add new class entries in chronological order (class 0, 1, 2, ...)
            raw_names = dataset_content.get('names', [])
            classes = self._classes_in_chronological_order(raw_names)
            for class_name in classes:
                self._add_class_row(class_name)
            
            # Spacer so class list stays at top
            layout.addStretch()
            
            self.logger.info(f"Loaded YAML file: {os.path.basename(yaml_file)}")
            
            # Trigger event
            self.on_dataset_changed(dataset_content)
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load YAML file: {str(e)}")
            self.logger.error(f"Error loading YAML: {str(e)}")
    
    def _add_class_row(self, class_name=""):
        """Add a single class row (label + text entry + ✕ button) to the classes layout."""
        layout = self.ui.verticalLayout_classes_content

        # Remove trailing spacer so new row is inserted before it
        spacer_item = None
        if layout.count() > 0:
            last = layout.itemAt(layout.count() - 1)
            if last and last.spacerItem():
                spacer_item = layout.takeAt(layout.count() - 1)

        # Count existing entries for index label
        class_count = sum(
            1 for i in range(layout.count())
            if layout.itemAt(i).widget() and
               layout.itemAt(i).widget().findChildren(QLineEdit) and
               any(le.objectName().startswith("class_entry_")
                   for le in layout.itemAt(i).widget().findChildren(QLineEdit))
        )

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(f"Class {class_count}:")
        label.setFixedWidth(65)
        entry = QLineEdit(class_name)
        entry.setObjectName(f"class_entry_{class_count}")

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(28)
        del_btn.setToolTip("Delete this class")
        del_btn.clicked.connect(lambda _checked, rw=row_widget: self._delete_class_row(rw))

        row_layout.addWidget(label)
        row_layout.addWidget(entry)
        row_layout.addWidget(del_btn)

        layout.addWidget(row_widget)

        if spacer_item:
            layout.addItem(spacer_item)

        entry.setFocus()

    def add_class(self):
        """Add a new empty class row to the list."""
        self._add_class_row()

    def _delete_class_row(self, row_widget):
        """Remove a specific class row and renumber the remaining rows."""
        layout = self.ui.verticalLayout_classes_content
        # Require at least one class to remain
        class_rows = [
            layout.itemAt(i).widget() for i in range(layout.count())
            if layout.itemAt(i).widget() and
               any(le.objectName().startswith("class_entry_")
                   for le in layout.itemAt(i).widget().findChildren(QLineEdit))
        ]
        if len(class_rows) <= 1:
            QMessageBox.information(None, "Delete Class", "At least one class must remain.")
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() is row_widget:
                layout.takeAt(i)
                row_widget.setParent(None)
                row_widget.deleteLater()
                break
        self._renumber_class_labels()
        self.logger.info("Removed class row")

    def _renumber_class_labels(self):
        """Renumber Class 0, Class 1, ... and objectNames after a delete."""
        layout = self.ui.verticalLayout_classes_content
        idx = 0
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if not item or not item.widget():
                continue
            w = item.widget()
            for label in w.findChildren(QLabel):
                if label.text().startswith("Class "):
                    label.setText(f"Class {idx}:")
                    break
            for line_edit in w.findChildren(QLineEdit):
                if line_edit.objectName().startswith("class_entry_"):
                    line_edit.setObjectName(f"class_entry_{idx}")
                    idx += 1
                    break
    
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
        """Create a new YAML dataset file from scratch"""
        from PyQt5.QtWidgets import QTextEdit, QDialogButtonBox

        # --- step 1: collect details via dialog ---
        dialog = QDialog(main_window_parent(self.logger))
        dialog.setWindowTitle("New Dataset YAML")
        dialog.resize(420, 320)

        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        path_entry = QLineEdit()
        path_entry.setPlaceholderText("e.g. C:/Common/CT600 Image Data/MyDataset")
        browse_btn = QPushButton("Browse…")
        path_row = QWidget()
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.addWidget(path_entry)
        path_row_layout.addWidget(browse_btn)
        browse_btn.clicked.connect(
            lambda: path_entry.setText(
                QFileDialog.getExistingDirectory(dialog, "Select Dataset Root Folder") or path_entry.text()
            )
        )
        form_layout.addRow("Dataset path:", path_row)
        layout.addLayout(form_layout)

        layout.addWidget(QLabel("Class names (one per line):"))
        classes_text = QTextEdit()
        classes_text.setPlaceholderText("block1\nblock1_edge\nblock2\nblock2_edge\ncal_mark")
        layout.addWidget(classes_text)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return

        classes = [line.strip() for line in classes_text.toPlainText().splitlines() if line.strip()]
        if not classes:
            QMessageBox.critical(None, "Error", "At least one class name is required.")
            return

        # --- step 2: choose save location ---
        save_path, _ = QFileDialog.getSaveFileName(
            main_window_parent(self.logger),
            "Save New Dataset YAML",
            "config/",
            "YAML files (*.yaml);;All files (*.*)"
        )
        if not save_path:
            return

        content = {
            'path': path_entry.text(),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(classes),
            'names': classes,
        }

        try:
            self.dataset_manager.save_yaml(save_path, content)
            self._add_to_quick_select(save_path)
            self.ui.lineEdit_yaml_path.setText(save_path)
            self.load_yaml()
            QMessageBox.information(None, "Success", f"Dataset YAML created:\n{save_path}")
            self.logger.info(f"Created new dataset YAML: {save_path}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to create YAML file: {str(e)}")
            self.logger.error(f"Error creating YAML: {str(e)}")

    def save_yaml_as(self):
        """Save current dataset configuration to a new file location"""
        current_path = self.ui.lineEdit_yaml_path.text()
        save_path, _ = QFileDialog.getSaveFileName(
            main_window_parent(self.logger),
            "Save Dataset YAML As",
            current_path or "config/",
            "YAML files (*.yaml);;All files (*.*)"
        )
        if not save_path:
            return

        # Build content from current UI state
        classes = []
        layout = self.ui.verticalLayout_classes_content
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget:
                for line_edit in widget.findChildren(QLineEdit):
                    if line_edit.objectName().startswith("class_entry_"):
                        text = line_edit.text().strip()
                        if text:
                            classes.append(text)

        dataset_content = self.dataset_manager.get_dataset_content() or {}
        content = {
            'path': self.ui.lineEdit_dataset_path.text(),
            'train': dataset_content.get('train', 'train/images'),
            'val': dataset_content.get('val', 'val/images'),
            'nc': len(classes),
            'names': classes,
        }

        try:
            self.dataset_manager.save_yaml(save_path, content)
            self._add_to_quick_select(save_path)
            self.ui.lineEdit_yaml_path.setText(save_path)
            QMessageBox.information(None, "Success", f"Saved to:\n{save_path}")
            self.logger.info(f"Saved dataset YAML as: {save_path}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to save YAML: {str(e)}")
            self.logger.error(f"Error saving YAML as: {str(e)}")

    def _add_to_quick_select(self, yaml_path):
        """Add a YAML path to the quick-select combobox if not already present"""
        if self.ui.comboBox_quick_select.findText(yaml_path) == -1:
            self.ui.comboBox_quick_select.addItem(yaml_path)

