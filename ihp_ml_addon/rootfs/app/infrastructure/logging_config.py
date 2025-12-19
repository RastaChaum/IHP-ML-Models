"""Centralized logging configuration for IHP ML Models addon.

Provides file and console logging with rotation support.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str | None = None,
    log_level: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure logging for the application.

    Sets up both file and console logging with rotation support.

    Args:
        log_dir: Directory for log files (defaults to /data/logs)
        log_level: Logging level (defaults to INFO)
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Get configuration from environment or use defaults
    log_dir = log_dir or os.getenv("LOG_DIR", "/data/logs")
    log_level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    
    # Convert log level string to logging constant
    log_level_value = getattr(logging, log_level_str, logging.INFO)
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir).expanduser().resolve()
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Fallback to /tmp if can't create in specified directory
        print(f"⚠️  Cannot create log directory {log_path}, using /tmp/ihp_ml_logs")
        log_path = Path("/tmp/ihp_ml_logs")
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Define log file paths
    main_log_file = log_path / "ihp_ml.log"
    debug_log_file = log_path / "ihp_ml_debug.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # 1. Console handler (INFO level by default, respects LOG_LEVEL)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_value)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. Main log file (same level as console)
    main_file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    main_file_handler.setLevel(log_level_value)
    main_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(main_file_handler)
    
    # 3. Debug log file (always DEBUG level for troubleshooting)
    debug_file_handler = RotatingFileHandler(
        debug_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_file_handler)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Logging Configuration")
    logger.info("  Log directory: %s", log_dir)
    logger.info("  Console/Main file level: %s", log_level_str)
    logger.info("  Debug file level: DEBUG")
    logger.info("  Main log file: %s", main_log_file)
    logger.info("  Debug log file: %s", debug_log_file)
    logger.info("  Max file size: %.1f MB", max_bytes / (1024 * 1024))
    logger.info("  Backup count: %d", backup_count)
    logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
