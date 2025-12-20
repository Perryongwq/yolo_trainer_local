"""YOLO Training GUI - PyQt5 Application"""
import multiprocessing
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from ui.pyqt.app import YOLOTrainerApp


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


def main():
    """Application entry point with multiprocessing support"""
    # Required for multiprocessing in frozen applications
    multiprocessing.freeze_support()
    
    # Create the PyQt5 application
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Training GUI")
    
    # Set application icon
    try:
        icon_path = resource_path("assets/yolo_icon.ico")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass
    
    # Initialize and show the main window
    main_window = YOLOTrainerApp()
    main_window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
