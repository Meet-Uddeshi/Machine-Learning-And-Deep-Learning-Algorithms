# ============================================================================
# Centralized Logging Utility Module
# ============================================================================
# Configures logging for the Statistics and Probability pipeline.
# Provides console output and persistent log file writing with standard formatting.
# ============================================================================

import logging
import os
import sys

from config import LoggingConfig, PathConfig


class LoggerFactory:
    """Factory class to construct and configure logging instances.

    Ensures single-handler registration to avoid duplicated log entries
    and handles file system directory creation for output log persistence.
    """

    @staticmethod
    def create(
        name: str,
        logging_config: LoggingConfig,
        path_config: PathConfig,
    ) -> logging.Logger:
        """Create and return a configured Python Logger instance.

        Args:
            name:           Name identifier for the logger instance.
            logging_config: Logging configuration specifications.
            path_config:    Path configuration detailing output directories.

        Returns:
            A fully configured logging.Logger instance.
        """
        logger = logging.getLogger(name)

        # Convert logging string severity to logging constant level
        numeric_level = getattr(
            logging, logging_config.log_level.upper(), logging.INFO
        )
        logger.setLevel(numeric_level)

        # Clear existing handlers to prevent duplicate messages if re-initialized
        if logger.hasHandlers():
            logger.handlers.clear()

        # Define logging format string
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
