"""Utility functions for PyQt5 UI"""
from PyQt5.QtWidgets import QToolTip, QWidget, QScrollArea, QVBoxLayout
from PyQt5.QtCore import QPoint


def main_window_parent(logger):
    """
    Parent widget for modal QFileDialog / QMessageBox.

    Passing None on Windows often leaves dialogs behind the app or non-modal;
    tabs should use the main QMainWindow from Logger(app=...) instead.
    """
    if logger is None:
        return None
    return getattr(logger, "app", None)


def create_tooltip(widget, text):
    """
    Create a tooltip for a widget
    
    Args:
        widget: The widget to add tooltip to
        text: Tooltip text
    """
    widget.setToolTip(text)


def setup_treeview(tree_widget, columns, widths=None, stretch_columns=None):
    """
    Setup a tree widget (or table widget) with columns
    
    Args:
        tree_widget: QTreeWidget or QTableWidget
        columns: List of column names
        widths: Dictionary of column_name: width
        stretch_columns: List of column names that should stretch
    
    Returns:
        The configured tree widget
    """
    widths = widths or {}
    stretch_columns = stretch_columns or []
    
    # For QTableWidget
    if hasattr(tree_widget, 'setColumnCount'):
        tree_widget.setColumnCount(len(columns))
        tree_widget.setHorizontalHeaderLabels(columns)
        
        # Set column widths
        for idx, col_name in enumerate(columns):
            if col_name in widths:
                tree_widget.setColumnWidth(idx, widths[col_name])
        
        # Enable stretch for specified columns
        if stretch_columns:
            header = tree_widget.horizontalHeader()
            for idx, col_name in enumerate(columns):
                if col_name in stretch_columns:
                    header.setStretchLastSection(idx == len(columns) - 1)
    
    return tree_widget


def create_scrollable_frame(parent):
    """
    Create a scrollable frame
    
    Args:
        parent: Parent widget
    
    Returns:
        Tuple of (scroll_area, content_widget)
    """
    scroll_area = QScrollArea(parent)
    scroll_area.setWidgetResizable(True)
    
    content_widget = QWidget()
    content_layout = QVBoxLayout(content_widget)
    
    scroll_area.setWidget(content_widget)
    
    return scroll_area, content_widget

