# logging_config.py

import logging
import sys
from pathlib import Path


class LibraryFilter(logging.Filter):
    """
    Filter to exclude logs from specified libraries
    """

    def __init__(self, excluded_prefixes):
        super().__init__()
        self.excluded_prefixes = excluded_prefixes

    def filter(self, record):
        # Allow logs from our own code, filter out specified libraries
        return not any(record.name.startswith(prefix) for prefix in self.excluded_prefixes)


def setup_logging(log_file='stroke_prediction.log'):
    """
    Set up logging configuration for the stroke prediction project.
    Excludes logs from specified external libraries.

    Args:
        log_file (str): Name of the log file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Libraries to exclude from logging
    excluded_libraries = [
        'matplotlib'
    ]

    # Create the library filter
    lib_filter = LibraryFilter(excluded_libraries)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Create formatters and handlers
    formatter = logging.Formatter(log_format, date_format)

    # File handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(lib_filter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(lib_filter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our configured handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Set specific library log levels to WARNING or higher
    for lib in excluded_libraries:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return root_logger