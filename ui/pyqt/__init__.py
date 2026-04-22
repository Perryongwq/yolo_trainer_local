"""PyQt5 UI components for YOLO Training GUI."""

__all__ = ["YOLOTrainerApp"]


def __getattr__(name):
    """Lazy import to avoid circular/early import issues with generated UI code."""
    if name == "YOLOTrainerApp":
        from ui.pyqt.app import YOLOTrainerApp
        return YOLOTrainerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

