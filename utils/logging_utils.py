import datetime
import re
from utils.event import Event

class Logger:
    """
    Centralized logging system for the application.
    Handles logging, formatting, and distribution of log messages.
    """
    
    def __init__(self, app=None):
        """
        Initialize the logger
        
        Args:
            app: Optional reference to the main application
        """
        self.app = app
        self.log_entries = []
        
        # Color patterns for syntax highlighting
        self.color_patterns = [
            (r'Epoch\s+\d+/\d+', 'blue'),        # Epoch headers
            (r'GPU_mem', 'dark green'),          # GPU memory
            (r'Class\s+Images\s+Instances', 'purple'),  # Validation headers
            (r'[0-9]+%\|[█▉▊▋▌▍▎▏ ]+\|', 'dark green'),  # Progress bars
            (r'mAP50-95', 'purple'),             # mAP metric
            (r'WARNING', 'orange'),              # Warnings
            (r'Error', 'red'),                   # Errors
        ]
        
        # Events
        self.on_log_added = Event()
    
    def info(self, message):
        """
        Log an informational message
        
        Args:
            message: Message to log
        """
        self._log("INFO", message)
    
    def warning(self, message):
        """
        Log a warning message
        
        Args:
            message: Message to log
        """
        self._log("WARNING", message)
    
    def error(self, message):
        """
        Log an error message
        
        Args:
            message: Message to log
        """
        self._log("ERROR", message)
    
    def debug(self, message):
        """
        Log a debug message
        
        Args:
            message: Message to log
        """
        self._log("DEBUG", message)
    
    def _log(self, level, message):
        """
        Internal logging method
        
        Args:
            level: Log level
            message: Message to log
        """
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "patterns": self._find_patterns(message)
        }
        
        # Add to log entries
        self.log_entries.append(log_entry)
        
        # Trigger event
        self.on_log_added(log_entry)
    
    def _find_patterns(self, message):
        """
        Find color patterns in the message
        
        Args:
            message: Message to check
            
        Returns:
            List of (pattern, start, end, color) tuples
        """
        patterns = []
        
        for pattern, color in self.color_patterns:
            for match in re.finditer(pattern, message):
                patterns.append((pattern, match.start(), match.end(), color))
        
        return patterns
    
    def get_all_logs(self):
        """
        Get all log entries
        
        Returns:
            List of log entries
        """
        return self.log_entries.copy()
    
    def clear_logs(self):
        """Clear all log entries"""
        self.log_entries.clear()
        
    def save_logs(self, file_path):
        """
        Save logs to a file
        
        Args:
            file_path: Path to save the logs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                for entry in self.log_entries:
                    f.write(f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}\n")
            return True
        except Exception as e:
            print(f"Error saving logs: {str(e)}")
            return False
