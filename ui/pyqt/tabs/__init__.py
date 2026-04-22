"""PyQt5 tab components."""

__all__ = ["DatasetTab", "TrainingTab", "EvaluationTab", "AnnotationTab",
           "ImageCleaningTab", "DatasetPrepTab"]

_imports = {
    "DatasetTab": "ui.pyqt.tabs.dataset_tab",
    "TrainingTab": "ui.pyqt.tabs.training_tab",
    "EvaluationTab": "ui.pyqt.tabs.evaluation_tab",
    "AnnotationTab": "ui.pyqt.tabs.annotation_tab",
    "ImageCleaningTab": "ui.pyqt.tabs.image_cleaning_tab",
    "DatasetPrepTab": "ui.pyqt.tabs.dataset_prep_tab",
}


def __getattr__(name):
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

