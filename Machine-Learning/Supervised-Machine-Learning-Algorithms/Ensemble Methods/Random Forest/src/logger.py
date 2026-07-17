# ============================================================================
# Logging Module for Random Forest Classification Pipeline
# ============================================================================
# Provides a factory function that produces a fully configured Python logger
# with both console and file handlers. The file handler writes to the
# 'output/' directory alongside model results, keeping all artifacts together.
# ============================================================================

import logging
import os
import sys

from config import LoggingConfig, PathConfig


class LoggerFactory:
    """Factory responsible for creating and configuring pipeline loggers.

    Encapsulates all handler/formatter setup so that consumers simply call
    `LoggerFactory.create(...)` and receive a ready-to-use logger instance.
    """

    _FORMAT = (
        "[%(asctime)s] [%(levelname)-8s] [%(name)-20s] %(message)s"
    )
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def create(
        cls,
        name: str,
        logging_config: LoggingConfig,
        path_config: PathConfig,
    ) -> logging.Logger:
        """Build and return a configured logger.

        Args:
            name:           Logger name (typically __name__ of the caller).
            logging_config: Logging settings (level, file toggle, filename).
            path_config:    Path settings (used to resolve log file location).

        Returns:
            A logging.Logger instance with console (and optionally file)
            handlers attached.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, logging_config.log_level.upper(), logging.DEBUG))

        # Prevent duplicate handlers when the factory is called more than once
        if logger.handlers:
            return logger

        formatter = logging.Formatter(cls._FORMAT, datefmt=cls._DATE_FORMAT)

        # --- Console handler (always active) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # --- File handler (optional) ---
        if logging_config.log_to_file:
            os.makedirs(path_config.output_dir, exist_ok=True)
            log_filepath = os.path.join(path_config.output_dir, logging_config.log_filename)
            file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
