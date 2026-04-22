"""Main application class for PyQt5 YOLO Trainer GUI"""
import os
from PyQt5.QtWidgets import QMainWindow

# Import generated UI
from config.ui_mainwindow import Ui_MainWindow

# Import core managers
from core import (
    ConfigManager, ModelManager, TrainingManager,
    DatasetManager, AnnotationManager,
)

# Import utilities
from utils import Logger

# Import PyQt5 tabs
from ui.pyqt.tabs import DatasetTab, TrainingTab, EvaluationTab, AnnotationTab, ImageCleaningTab, DatasetPrepTab


class YOLOTrainerApp(QMainWindow):
    """Main application class for YOLO Trainer GUI (PyQt5 version)"""
    
    def __init__(self):
        """Initialize the application with required components"""
        super().__init__()
        
        # Setup UI from generated code
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        
        # Initialize core managers
        self.model_manager = ModelManager(self.config_manager)
        self.dataset_manager = DatasetManager(self.config_manager)
        self.training_manager = TrainingManager(self.config_manager)
        self.annotation_manager = AnnotationManager(self.config_manager)
        
        # Initialize logger
        self.logger = Logger(self)
        self.logger.info(f"Application started. Current directory: {os.getcwd()}")
        
        # Initialize tabs with dependency injection
        self.dataset_tab = DatasetTab(
            self.ui,
            self.dataset_manager,
            self.logger
        )
        
        self.training_tab = TrainingTab(
            self.ui,
            self.model_manager,
            self.config_manager,
            self.logger
        )
        
        self.evaluation_tab = EvaluationTab(
            self.ui,
            self.model_manager,
            self.logger
        )
        
        self.annotation_tab = AnnotationTab(
            self.ui,
            self.model_manager,
            self.annotation_manager,
            self.logger
        )
        
        self.image_cleaning_tab = ImageCleaningTab(self.ui, self.logger)
        self.dataset_prep_tab = DatasetPrepTab(self.ui, self.logger)
        
        # Register event listeners
        self._register_events()
    
    def _register_events(self):
        """Register event listeners for inter-component communication"""
        # Dataset events
        self.dataset_manager.on_dataset_loaded.subscribe(self.training_tab.update_dataset_info)
        self.dataset_manager.on_dataset_loaded.subscribe(
            lambda yaml_path, dataset_content: self.annotation_tab.update_model_info(
                None, yaml_path, dataset_content.get('names', [])
            ) if hasattr(self.annotation_tab, 'update_model_info') else None
        )
        
        # Model events
        self.model_manager.on_model_loaded.subscribe(self.training_tab.update_model_info)
        self.model_manager.on_model_loaded.subscribe(self.evaluation_tab.update_model_info)
        self.model_manager.on_model_loaded.subscribe(self.annotation_tab.update_model_info)
        
        # Training events (handled by training_tab which now includes status)
        self.training_manager.on_training_started.subscribe(self.training_tab.on_training_started)
        self.training_manager.on_training_progress.subscribe(self.training_tab.on_training_progress)
        self.training_manager.on_training_completed.subscribe(self.training_tab.on_training_completed)
        self.training_manager.on_training_error.subscribe(self.training_tab.on_training_error)
        
        # Button callbacks
        self.training_tab.on_start_training.subscribe(self.start_training)
    
    def start_training(self):
        """Start the training process"""
        # Stay on training tab (parameters + status are now on same page)
        self.ui.tabWidget.setCurrentWidget(self.ui.tab_training)
        
        # Start training with current configuration
        self.training_manager.start_training(
            model_path=self.config_manager.get("model_path"),
            dataset_path=self.config_manager.get("yaml_path"),
            params=self.config_manager.get_training_params()
        )

