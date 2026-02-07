"""
Centralized logging configuration for the RAG application.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """Centralized logger configuration with file and console handlers."""

    _loggers = {}

    @staticmethod
    def setup_logger(
        name: str,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ) -> logging.Logger:
        """
        Setup a logger with both file and console handlers.

        Args:
            name: Logger name (usually __name__ of the module)
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output to console
            file_output: Whether to output to file

        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_formatter = logging.Formatter(
            fmt='%(levelname)s | %(message)s'
        )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if file_output:
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True, parents=True)

            # Create log filename with date
            log_filename = f"rag_app_{datetime.now().strftime('%Y%m%d')}.log"
            log_file = log_path / log_filename

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        # Store logger
        LoggerSetup._loggers[name] = logger

        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get existing logger or create a new one with default settings.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name not in LoggerSetup._loggers:
            return LoggerSetup.setup_logger(name)
        return LoggerSetup._loggers[name]


def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name (typically __name__)
        log_level: Optional logging level override

    Returns:
        Configured logger instance
    """
    if log_level:
        return LoggerSetup.setup_logger(name, log_level=log_level)
    return LoggerSetup.get_logger(name)
