# ============================================================================
# Centralized Logging Utility Module
# ============================================================================
# Configures logging for the Bagging and Boosting ensemble pipeline.
# Provides console stream output only and strictly avoids local disk log persistence.
# ============================================================================

import logging
import sys

from config import LoggingConfig, PathConfig


class LoggerFactory:
    """Factory class to construct and configure logging instances for stdout logging."""

    @staticmethod
    def create(
        name: str,
        logging_config: LoggingConfig,
        path_config: PathConfig,
    ) -> logging.Logger:
        """Create and return a configured Python Logger instance.

        Args:
            name:           Name identifier for the logger instance.
            logging_config: Logging configuration parameters.
            path_config:    Path configuration parameters.

        Returns:
            A fully configured logging.Logger instance.
        """
        logger = logging.getLogger(name)

        numeric_level = getattr(
            logging, logging_config.log_level.upper(), logging.INFO
        )
        logger.setLevel(numeric_level)

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler only (no disk file persistence)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
