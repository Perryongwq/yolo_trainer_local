"""File browser widget for PyQt5"""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtCore import pyqtSignal


class FileBrowser(QWidget):
    """A compound widget for file/folder browsing"""
    
    # Signal emitted when a file/folder is selected
    fileSelected = pyqtSignal(str)
    
    def __init__(self, parent=None, label_text="", initial_value="", 
                 file_types=None, is_directory=False, on_select=None):
        """
        Initialize the file browser widget
        
        Args:
            parent: Parent widget
            label_text: Text for the label
            initial_value: Initial path value
            file_types: List of tuples for file type filters e.g. [("Images", "*.jpg *.png")]
            is_directory: Whether to browse for directories instead of files
            on_select: Callback function when file is selected
        """
        super().__init__(parent)
        
        self.file_types = file_types or [("All files", "*.*")]
        self.is_directory = is_directory
        self._on_select_callback = on_select
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        if label_text:
            self.label = QLabel(label_text)
            layout.addWidget(self.label)
        
        # Line edit
        self.line_edit = QLineEdit()
        self.line_edit.setText(initial_value)
        layout.addWidget(self.line_edit)
        
        # Browse button
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._on_browse_clicked)
        layout.addWidget(self.browse_button)
        
        # Connect signal to callback if provided
        if on_select:
            self.fileSelected.connect(on_select)
    
    def _on_browse_clicked(self):
        """Handle browse button click"""
        if self.is_directory:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Directory",
                self.line_edit.text()
            )
        else:
            # Convert file types to Qt format
            filter_str = ";;".join([f"{desc} ({pattern})" for desc, pattern in self.file_types])
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select File",
                self.line_edit.text(),
                filter_str
            )
        
        if path:
            self.line_edit.setText(path)
            self.fileSelected.emit(path)
    
    def get_path(self):
        """Get the current path"""
        return self.line_edit.text()
    
    def set_path(self, path):
        """Set the current path"""
        self.line_edit.setText(path)

