"""Training parameters and status tab for PyQt5 (merged)."""
import datetime
import re
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, QTimer

from utils.event import Event
from ui.pyqt.common.ui_utils import main_window_parent


class TrainingTab:
    """Tab for setting training parameters, starting training, and viewing status/logs"""

    def __init__(self, ui, model_manager, config_manager, logger):
        """
        Initialize the training tab (parameters + status).

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

        # Status tracking
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_elapsed_time)

        # Connect UI signals to handlers
        self._connect_signals()

        # Subscribe to logger events
        self.logger.on_log_added.subscribe(self._on_log_added)

        # Initialize UI state
        self._initialize_ui()

    def _connect_signals(self):
        """Connect Qt signals to handler methods"""
        # Model selection
        self.ui.comboBox_model.currentTextChanged.connect(self._on_model_selected)
        self.ui.pushButton_model_browse.clicked.connect(self._browse_model_file)

        # Training parameters
        self.ui.spinBox_epochs.valueChanged.connect(
            lambda v: self._on_parameter_changed("training.epochs", v)
        )
        self.ui.spinBox_imgsz.valueChanged.connect(
            lambda v: self._on_parameter_changed("training.imgsz", v)
        )
        self.ui.doubleSpinBox_lr0.valueChanged.connect(
            lambda v: self._on_parameter_changed("training.lr0", v)
        )
        self.ui.doubleSpinBox_lrf.valueChanged.connect(
            lambda v: self._on_parameter_changed("training.lrf", v)
        )
        self.ui.spinBox_patience.valueChanged.connect(
            lambda v: self._on_parameter_changed("training.patience", v)
        )
        self.ui.comboBox_optimizer.currentTextChanged.connect(
            lambda v: self._on_parameter_changed("training.optimizer", v)
        )

        # Batch size (in Training Parameters)
        self.ui.spinBox_batch_size.valueChanged.connect(
            lambda v: self._on_parameter_changed("training.batch_size", v)
        )

        # Start training button
        self.ui.pushButton_start_training.clicked.connect(self.start_training)

        # Status log buttons
        self.ui.pushButton_clear_log.clicked.connect(self.clear_status)
        self.ui.pushButton_save_log.clicked.connect(self.save_log)

    def _initialize_ui(self):
        """Initialize UI with default values from config"""
        # Set model
        model_path = self.config_manager.get("model_path", "yolo11l.pt")
        index = self.ui.comboBox_model.findText(model_path)
        if index >= 0:
            self.ui.comboBox_model.setCurrentIndex(index)

        # Set training parameters
        self.ui.spinBox_epochs.setValue(self.config_manager.get("training.epochs", 100))
        self.ui.spinBox_imgsz.setValue(self.config_manager.get("training.imgsz", 640))
        self.ui.doubleSpinBox_lr0.setValue(
            self.config_manager.get("training.lr0", 0.001)
        )
        self.ui.doubleSpinBox_lrf.setValue(
            self.config_manager.get("training.lrf", 0.2)
        )
        self.ui.spinBox_patience.setValue(
            self.config_manager.get("training.patience", 0)
        )

        optimizer = self.config_manager.get("training.optimizer", "Adam")
        index = self.ui.comboBox_optimizer.findText(optimizer)
        if index >= 0:
            self.ui.comboBox_optimizer.setCurrentIndex(index)

        # Set batch size
        self.ui.spinBox_batch_size.setValue(
            self.config_manager.get("training.batch_size", 16)
        )

        # Status area
        self.update_status(
            "Ready to train. Configure dataset and parameters, then start training."
        )
        if hasattr(self.ui, "label_epoch_value"):
            self.ui.label_epoch_value.setText("0/0")
        if hasattr(self.ui, "label_elapsed_value"):
            self.ui.label_elapsed_value.setText("00:00:00")

    def _browse_model_file(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            main_window_parent(self.logger),
            "Select Model File",
            "",
            "PyTorch Models (*.pt);;All files (*.*)",
        )

        if file_path:
            if self.ui.comboBox_model.findText(file_path) == -1:
                self.ui.comboBox_model.addItem(file_path)

            self.ui.comboBox_model.setCurrentText(file_path)
            self._on_parameter_changed("model_path", file_path)
            self.logger.info(f"Selected model: {file_path}")

    def _on_model_selected(self, model_path):
        """Handle model selection from combobox"""
        if model_path:
            self._on_parameter_changed("model_path", model_path)
            self.logger.info(f"Selected model: {model_path}")

    def _on_parameter_changed(self, key, value):
        """Handle parameter change event"""
        self.config_manager.set(key, value)

    def _on_log_added(self, log_entry):
        """Handle new log entry event from logger"""
        self.update_status(log_entry["message"])
        self._update_training_progress_from_message(log_entry["message"])

    def _update_training_progress_from_message(self, message):
        """Update training progress info from log message"""
        if "Starting training for" in message and not self.start_time:
            self.start_time = datetime.datetime.now()
            match = re.search(r"Starting training for (\d+) epochs", message)
            if match:
                self.total_epochs = int(match.group(1))
                if hasattr(self.ui, "label_epoch_value"):
                    self.ui.label_epoch_value.setText(f"0/{self.total_epochs}")
            self.timer.start(1000)

        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", message)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            if hasattr(self.ui, "label_epoch_value"):
                self.ui.label_epoch_value.setText(
                    f"{self.current_epoch}/{total_epochs}"
                )

    def _update_elapsed_time(self):
        """Update elapsed time display"""
        if self.start_time and hasattr(self.ui, "label_elapsed_value"):
            elapsed = datetime.datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.ui.label_elapsed_value.setText(
                f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            )

    def update_status(self, message):
        """Add a message to the status log with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.ui.textEdit_status.append(formatted_message.strip())

        if self.ui.checkBox_autoscroll.isChecked():
            cursor = self.ui.textEdit_status.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.ui.textEdit_status.setTextCursor(cursor)

    def clear_status(self):
        """Clear the status log"""
        self.ui.textEdit_status.clear()
        self.update_status("Log cleared")

        self.current_epoch = 0
        self.total_epochs = 0
        if hasattr(self.ui, "label_epoch_value"):
            self.ui.label_epoch_value.setText("0/0")
        if hasattr(self.ui, "label_elapsed_value"):
            self.ui.label_elapsed_value.setText("00:00:00")
        self.start_time = None
        self.timer.stop()

    def save_log(self):
        """Save the status log to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            main_window_parent(self.logger),
            "Save Log",
            "",
            "Text files (*.txt);;All files (*.*)",
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.ui.textEdit_status.toPlainText())
                self.update_status(f"Log saved to {file_path}")
            except Exception as e:
                self.update_status(f"Error saving log: {str(e)}")
                QMessageBox.critical(
                    main_window_parent(self.logger),
                    "Error",
                    f"Failed to save log: {str(e)}",
                )

    # ----- Training event handlers (called by TrainingManager via app) -----
    def on_training_started(self, model_path, dataset_path):
        """Handle training started event"""
        self.start_time = datetime.datetime.now()
        self.current_epoch = 0
        self.update_status(f"Training started with model: {model_path}")
        self.timer.start(1000)

    def on_training_progress(self, message):
        """Handle training progress event"""
        self.update_status(message)

    def on_training_completed(self, success, message, results=None):
        """Handle training completed event"""
        self.timer.stop()
        if success:
            self.update_status("Training complete!")
            self.update_status(
                "Check the 'runs/detect/train' folder for results."
            )
        else:
            self.update_status(f"Training failed: {message}")

    def on_training_error(self, message, error=None):
        """Handle training error event"""
        self.timer.stop()
        self.update_status(f"Error during training: {message}")
        if error:
            self.update_status(f"Error details: {str(error)}")

    # ----- Dataset/Model updates (from other tabs) -----
    def update_dataset_info(self, yaml_path, dataset_content):
        """Update when dataset is loaded"""
        self.logger.info(f"Training tab updated with dataset: {yaml_path}")

    def update_model_info(self, model, model_path, class_names):
        """Update when model is loaded"""
        if self.ui.comboBox_model.findText(model_path) == -1:
            self.ui.comboBox_model.addItem(model_path)
        self.ui.comboBox_model.setCurrentText(model_path)
        self._on_parameter_changed("model_path", model_path)
        self.logger.info(f"Training tab updated with model: {model_path}")

    def start_training(self):
        """Start the training process (validates and fires event)"""
        if not self.config_manager.get("yaml_path"):
            QMessageBox.critical(
                None,
                "Error",
                "No dataset YAML file selected. Please configure dataset first.",
            )
            self.logger.error(
                "No dataset YAML file selected. Please configure dataset first."
            )
            return

        self.logger.info("Starting training with current configuration")
        self.on_start_training()
