"""Training status tab for PyQt5"""
import datetime
import re
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor
from PyQt5.QtCore import Qt, QTimer


class StatusTab:
    """Tab for displaying training status and logs"""
    
    def __init__(self, ui, logger):
        """
        Initialize the status tab
        
        Args:
            ui: UI object from generated ui_new_mainwindow.py
            logger: Logger instance
        """
        self.ui = ui
        self.logger = logger
        
        # Initialize variables
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        
        # Timer for updating elapsed time
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_elapsed_time)
        
        # Connect UI signals to handlers
        self._connect_signals()
        
        # Subscribe to logger events
        self.logger.on_log_added.subscribe(self.on_log_added)
        
        # Initialize UI state
        self._initialize_ui()
    
    def _connect_signals(self):
        """Connect Qt signals to handler methods"""
        self.ui.pushButton_clear_log.clicked.connect(self.clear_status)
        self.ui.pushButton_save_log.clicked.connect(self.save_log)
    
    def _initialize_ui(self):
        """Initialize UI with default values"""
        self.update_status("Ready to train. Configure dataset and parameters, then start training.")
    
    def on_log_added(self, log_entry):
        """
        Handle new log entry event
        
        Args:
            log_entry: Log entry dictionary
        """
        # Format the log message
        self.update_status(log_entry["message"])
        
        # Update training progress if relevant
        self.update_training_progress(log_entry["message"])
    
    def update_training_progress(self, message):
        """
        Update training progress information based on log message
        
        Args:
            message: Log message to analyze
        """
        # Start tracking time if this is the first training message
        if "Starting training for" in message and not self.start_time:
            self.start_time = datetime.datetime.now()
            # Extract total epochs from message
            match = re.search(r'Starting training for (\d+) epochs', message)
            if match:
                self.total_epochs = int(match.group(1))
                self.ui.label_epoch_value.setText(f"0/{self.total_epochs}")
            # Start timer
            self.timer.start(1000)  # Update every second
        
        # Update epoch counter if this is an epoch line
        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', message)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            self.ui.label_epoch_value.setText(f"{self.current_epoch}/{total_epochs}")
    
    def _update_elapsed_time(self):
        """Update elapsed time display"""
        if self.start_time:
            elapsed = datetime.datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.ui.label_elapsed_value.setText(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    def update_status(self, message):
        """
        Add a message to the status log with timestamp
        
        Args:
            message: Message to add to the status log
        """
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Format message with timestamp
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Append to text edit
        self.ui.textEdit_status.append(formatted_message.strip())
        
        # Apply color formatting based on message content
        self._apply_color_formatting(message)
        
        # Auto-scroll if enabled
        if self.ui.checkBox_autoscroll.isChecked():
            cursor = self.ui.textEdit_status.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.ui.textEdit_status.setTextCursor(cursor)
    
    def _apply_color_formatting(self, message):
        """
        Apply color formatting to the message
        
        Args:
            message: Message to format
        """
        # Simple color formatting based on keywords
        # This is a simplified version - full implementation would use QSyntaxHighlighter
        pass
    
    def clear_status(self):
        """Clear the status log"""
        self.ui.textEdit_status.clear()
        self.update_status("Log cleared")
        
        # Reset progress tracking
        self.current_epoch = 0
        self.total_epochs = 0
        self.ui.label_epoch_value.setText("0/0")
        self.ui.label_elapsed_value.setText("00:00:00")
        self.start_time = None
        self.timer.stop()
    
    def save_log(self):
        """Save the status log to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Log",
            "",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.ui.textEdit_status.toPlainText())
                self.update_status(f"Log saved to {file_path}")
            except Exception as e:
                self.update_status(f"Error saving log: {str(e)}")
                QMessageBox.critical(None, "Error", f"Failed to save log: {str(e)}")
    
    # Training event handlers
    def on_training_started(self, model_path, dataset_path):
        """
        Handle training started event
        
        Args:
            model_path: Path to the model
            dataset_path: Path to the dataset
        """
        self.start_time = datetime.datetime.now()
        self.current_epoch = 0
        self.update_status(f"Training started with model: {model_path}")
        self.timer.start(1000)
    
    def on_training_progress(self, message):
        """
        Handle training progress event
        
        Args:
            message: Progress message
        """
        # Update the status display with the training message
        self.update_status(message)
    
    def on_training_completed(self, success, message, results=None):
        """
        Handle training completed event
        
        Args:
            success: Whether training completed successfully
            message: Completion message
            results: Optional training results
        """
        self.timer.stop()
        
        if success:
            self.update_status("Training complete!")
            self.update_status("Check the 'runs/detect/train' folder for results.")
        else:
            self.update_status(f"Training failed: {message}")
    
    def on_training_error(self, message, error=None):
        """
        Handle training error event
        
        Args:
            message: Error message
            error: Optional exception object
        """
        self.timer.stop()
        self.update_status(f"Error during training: {message}")
        if error:
            self.update_status(f"Error details: {str(error)}")

