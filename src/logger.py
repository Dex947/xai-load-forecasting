"""
Structured Logging Module
==========================

Provides centralized logging configuration for the entire system.
All modules should use this logger instead of print statements.

Usage:
    from src.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Error occurred", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import yaml


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    file_rotation: bool = True,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        date_format: Custom date format string
        file_rotation: Enable log file rotation
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Default format if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = log_path / "xai_load_forecasting.log"
    
    if file_rotation:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log initialization
    root_logger.info("=" * 80)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log directory: {log_path.absolute()}")
    root_logger.info("=" * 80)


def setup_logging_from_config(config_path: str = "config/config.yaml") -> None:
    """
    Setup logging using parameters from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging_config = config.get('logging', {})
        paths_config = config.get('paths', {})
        
        setup_logging(
            log_dir=paths_config.get('logs', 'logs'),
            log_level=logging_config.get('level', 'INFO'),
            log_format=logging_config.get('format'),
            date_format=logging_config.get('date_format'),
            file_rotation=logging_config.get('file_rotation', True),
            max_bytes=logging_config.get('max_bytes', 10485760),
            backup_count=logging_config.get('backup_count', 5)
        )
    except Exception as e:
        # Fallback to default logging if config fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.error(f"Failed to load logging config from {config_path}: {e}")
        logging.info("Using default logging configuration")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


# Module-level logger for this file
logger = get_logger(__name__)
