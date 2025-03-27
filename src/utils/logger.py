"""
Logging module for the Anti-Corruption RAG system.
Configures logging for the application and provides utility functions.
Enhanced with root logger configuration for better visibility of library operations.
"""
import os
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import atexit

# Add parent directory to sys.path to enable imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.config import LOGS_DIR

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate log file name with timestamp
LOG_FILE = LOGS_DIR / f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create shared queue for IPC log handling
log_queue = queue.Queue(-1)  # No limit on size

# Configure queue handler for subprocess logging
queue_handler = logging.handlers.QueueHandler(log_queue)

# Configure the root logger for library visibility
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # Set to INFO to capture most library messages

# Main listener to handle log records from the queue
def log_listener(queue, log_file):
    """
    Listen for log records and write them to file and console.
    
    Args:
        queue (Queue): Queue to read log records from
        log_file (Path): Path to log file
    """
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    # Root logger with both handlers
    root = logging.getLogger()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    
    # Set root logger level to capture library logs
    root.setLevel(logging.INFO)
    
    # Process log records from the queue
    while True:
        try:
            record = queue.get()
            if record is None:  # None is sent as a sentinel to stop the listener
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level check here, already done
        except Exception:
            import traceback
            print(f"Error in log listener: {traceback.format_exc()}")


# Start listener thread
listener = threading.Thread(target=log_listener, args=(log_queue, LOG_FILE))
listener.daemon = True
listener.start()


def setup_logger(name):
    """
    Set up a logger for a module.
    
    Args:
        name (str): Name of the logger, typically __name__
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Add queue handler if not already present
    handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.QueueHandler)]
    if not handlers:
        logger.addHandler(queue_handler)
    
    # Propagate messages to root logger
    logger.propagate = False
    
    return logger


def get_subprocess_logger():
    """
    Get a logger configured for use in a subprocess.
    
    Returns:
        logging.Logger: Configured logger for subprocess
    """
    # Create a logger for the subprocess
    logger = logging.getLogger("subprocess")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Add a queue handler to send logs back to main process
    logger.addHandler(queue_handler)
    
    # Don't propagate to avoid duplicate logs
    logger.propagate = False
    
    return logger


def stop_logging():
    """Stop the logging listener thread."""
    log_queue.put(None)  # Signal to stop
    if listener.is_alive():
        listener.join(1.0)  # Wait for 1 second


# Register stop_logging to be called when the program exits
atexit.register(stop_logging)


# Get the main logger for the application
main_logger = setup_logger("main")
