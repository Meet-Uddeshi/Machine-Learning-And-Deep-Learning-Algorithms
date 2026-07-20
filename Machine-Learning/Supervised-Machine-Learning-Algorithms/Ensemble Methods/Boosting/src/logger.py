# ============================================================================
# Logging Module for Boosting Classification Pipeline
# ============================================================================
# Provides a factory function that produces a fully configured Python logger
# with console handlers only. In compliance with data restrictions, logs are
# never written to local files.
# ============================================================================

import logging
import sys

from config import LoggingConfig, PathConfig


class LoggerFactory:
    """Factory responsible for creating and configuring pipeline loggers.

    Encapsulates all handler/formatter setup so that consumers receive a
    ready-to-use console-only logger instance.
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
            name:           Logger name.
            logging_config: Logging settings (level).
            path_config:    Path settings.

        Returns:
            A logging.Logger instance with console stream handler attached.
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, logging_config.log_level.upper(), logging.INFO))

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        formatter = logging.Formatter(cls._FORMAT, datefmt=cls._DATE_FORMAT)

        # --- Console handler ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
