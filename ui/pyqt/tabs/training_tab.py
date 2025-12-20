"""Training parameters tab for PyQt5"""
import torch
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from utils.event import Event


class TrainingTab:
    """Tab for setting training parameters and starting training"""
    
    def __init__(self, ui, model_manager, config_manager, logger):
        """
        Initialize the training tab
        
        Args:
            ui: UI object from generated ui_new_mainwindow.py
            model_manager: ModelManager instance
            config_manager: ConfigManager instance
            logger: Logger instance
        """
        self.ui = ui
        self.model_manager = model_manager
        self.config_manager = config_manager
        self.logger = logger
        
        # Events
        self.on_start_training = Event()
        
        # Connect UI signals to handlers
        self._connect_signals()
        
        # Initialize UI state
        self._initialize_ui()
    
    def _connect_signals(self):
        """Connect Qt signals to handler methods"""
        # Model selection
        self.ui.comboBox_model.currentTextChanged.connect(self._on_model_selected)
        self.ui.pushButton_model_browse.clicked.connect(self._browse_model_file)
        
        # Training parameters
        self.ui.spinBox_epochs.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.epochs', v))
        self.ui.spinBox_imgsz.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.imgsz', v))
        self.ui.doubleSpinBox_lr0.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.lr0', v))
        self.ui.doubleSpinBox_lrf.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.lrf', v))
        self.ui.spinBox_patience.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.patience', v))
        self.ui.comboBox_optimizer.currentTextChanged.connect(
            lambda v: self._on_parameter_changed('training.optimizer', v))
        
        # Advanced options
        self.ui.checkBox_pretrained.stateChanged.connect(
            lambda state: self._on_parameter_changed('training.pretrained', state == Qt.Checked))
        self.ui.spinBox_save_period.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.save_period', v))
        self.ui.comboBox_device.currentTextChanged.connect(
            lambda v: self._on_parameter_changed('training.device', v))
        self.ui.spinBox_batch_size.valueChanged.connect(
            lambda v: self._on_parameter_changed('training.batch_size', v))
        
        # Start training button
        self.ui.pushButton_start_training.clicked.connect(self.start_training)
    
    def _initialize_ui(self):
        """Initialize UI with default values from config"""
        # Set model
        model_path = self.config_manager.get('model_path', 'yolo11l.pt')
        index = self.ui.comboBox_model.findText(model_path)
        if index >= 0:
            self.ui.comboBox_model.setCurrentIndex(index)
        
        # Set training parameters
        self.ui.spinBox_epochs.setValue(self.config_manager.get('training.epochs', 100))
        self.ui.spinBox_imgsz.setValue(self.config_manager.get('training.imgsz', 640))
        self.ui.doubleSpinBox_lr0.setValue(self.config_manager.get('training.lr0', 0.001))
        self.ui.doubleSpinBox_lrf.setValue(self.config_manager.get('training.lrf', 0.2))
        self.ui.spinBox_patience.setValue(self.config_manager.get('training.patience', 0))
        
        optimizer = self.config_manager.get('training.optimizer', 'Adam')
        index = self.ui.comboBox_optimizer.findText(optimizer)
        if index >= 0:
            self.ui.comboBox_optimizer.setCurrentIndex(index)
        
        # Set advanced options
        self.ui.checkBox_pretrained.setChecked(self.config_manager.get('training.pretrained', True))
        self.ui.spinBox_save_period.setValue(self.config_manager.get('training.save_period', 0))
        self.ui.spinBox_batch_size.setValue(self.config_manager.get('training.batch_size', 16))
        
        # Setup device combobox
        self._setup_device_options()
    
    def _setup_device_options(self):
        """Setup device selection options based on GPU availability"""
        self.ui.comboBox_device.clear()
        
        # Add device options
        device_options = []
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            # Add single GPU options
            for i in range(gpu_count):
                device_options.append(str(i))
            # Add multi-GPU options if more than one GPU
            if gpu_count > 1:
                device_options.append(",".join(str(i) for i in range(gpu_count)))
                device_options.append("0,1")  # Common dual GPU setup
        
        device_options.append("cpu")
        device_options.append("auto")
        
        self.ui.comboBox_device.addItems(device_options)
        
        # Set default device
        default_device = self.config_manager.get('training.device', "0" if torch.cuda.is_available() else "cpu")
        index = self.ui.comboBox_device.findText(default_device)
        if index >= 0:
            self.ui.comboBox_device.setCurrentIndex(index)
        
        # Update GPU info label
        if torch.cuda.is_available():
            self.ui.label_gpu_info.setText("GPU available: Yes")
            self.ui.label_gpu_info.setStyleSheet("color: green;")
        else:
            self.ui.label_gpu_info.setText("GPU available: No (using CPU)")
            self.ui.label_gpu_info.setStyleSheet("color: red;")
    
    def _browse_model_file(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Model File",
            "",
            "PyTorch Models (*.pt);;All files (*.*)"
        )
        
        if file_path:
            # Add to combobox if not already there
            if self.ui.comboBox_model.findText(file_path) == -1:
                self.ui.comboBox_model.addItem(file_path)
            
            self.ui.comboBox_model.setCurrentText(file_path)
            self._on_parameter_changed('model_path', file_path)
            self.logger.info(f"Selected model: {file_path}")
    
    def _on_model_selected(self, model_path):
        """Handle model selection from combobox"""
        if model_path:
            self._on_parameter_changed('model_path', model_path)
            self.logger.info(f"Selected model: {model_path}")
    
    def _on_parameter_changed(self, key, value):
        """
        Handle parameter change event
        
        Args:
            key: Configuration key
            value: New value
        """
        self.config_manager.set(key, value)
    
    def update_dataset_info(self, yaml_path, dataset_content):
        """
        Update dataset information
        
        Args:
            yaml_path: Path to the dataset YAML file
            dataset_content: Dataset content dictionary
        """
        self.logger.info(f"Training tab updated with dataset: {yaml_path}")
    
    def update_model_info(self, model, model_path, class_names):
        """
        Update model information
        
        Args:
            model: YOLO model instance
            model_path: Path to the model file
            class_names: Dictionary of class names
        """
        # Add to combobox if not already there
        if self.ui.comboBox_model.findText(model_path) == -1:
            self.ui.comboBox_model.addItem(model_path)
        
        self.ui.comboBox_model.setCurrentText(model_path)
        self._on_parameter_changed('model_path', model_path)
        self.logger.info(f"Training tab updated with model: {model_path}")
    
    def start_training(self):
        """Start the training process"""
        # Validate parameters
        if not self.config_manager.get('yaml_path'):
            QMessageBox.critical(None, "Error", "No dataset YAML file selected. Please configure dataset first.")
            self.logger.error("No dataset YAML file selected. Please configure dataset first.")
            return
        
        # Make sure all parameters are saved to config (they should be auto-saved via signals)
        self.logger.info("Starting training with current configuration")
        
        # Trigger training event
        self.on_start_training()

