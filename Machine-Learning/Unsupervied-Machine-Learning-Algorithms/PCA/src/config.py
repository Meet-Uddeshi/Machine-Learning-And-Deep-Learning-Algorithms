# ============================================================================
# Configuration Module for Principal Component Analysis (PCA) Pipeline
# ============================================================================
# Centralizes all configuration parameters for the PCA pipeline.
# Includes path configurations, data processing parameters, model parameters,
# and logging options.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the PCA project directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where figures and reports are written.
        dataset_file: Full path to the CSV dataset.
    """

    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    data_dir: str = ""
    src_dir: str = ""
    output_dir: str = ""
    dataset_file: str = ""

    def __post_init__(self) -> None:
        """Derive dependent directories and paths from project_root."""
        object.__setattr__(
            self, "data_dir", os.path.join(self.project_root, "data")
        )
        object.__setattr__(
            self, "src_dir", os.path.join(self.project_root, "src")
        )
        object.__setattr__(
            self, "output_dir", os.path.join(self.project_root, "output")
        )
        object.__setattr__(
            self,
            "dataset_file",
            os.path.join(self.project_root, "data", "pca.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Parameters governing dataset loading, target column, and preprocessing.

    Attributes:
        target_column: Name of categorical class label column.
        random_state:  Random seed for reproducibility.
    """

    target_column: str = "class"
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for Principal Component Analysis.

    Attributes:
        variance_threshold: Target cumulative explained variance ratio (e.g. 0.95).
        n_components_2d:    Number of components for 2D visualization (2).
        random_state:       Random seed for reproducibility.
    """

    variance_threshold: float = 0.95
    n_components_2d: int = 2
    random_state: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:   Minimum severity level.
        log_to_file: Whether to write logs to disk (False per user requirements).
    """

    log_level: str = "INFO"
    log_to_file: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregate configuration for the PCA pipeline.

    Usage:
        config = PipelineConfig()
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
