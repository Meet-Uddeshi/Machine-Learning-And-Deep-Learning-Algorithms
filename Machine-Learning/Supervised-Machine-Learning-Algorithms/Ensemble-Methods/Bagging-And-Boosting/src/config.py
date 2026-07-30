# ============================================================================
# Configuration Module for Bagging and Boosting Ensemble Pipeline
# ============================================================================
# Centralizes all configuration parameters for the Ensemble Learning pipeline.
# Includes path configurations, data loading options, model hyperparameters for
# Bagging, AdaBoost, and Gradient Boosting classifiers, and logging options.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Bagging-And-Boosting project root.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results and figures are written.
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
            os.path.join(self.project_root, "data", "heart.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Parameters governing dataset loading, column definitions, and train/test split.

    Attributes:
        target_column: Target outcome column for binary classification.
        test_size:     Fraction of dataset reserved for evaluation.
        random_state:  Random seed for reproducible train/test splits.
    """

    target_column: str = "target"
    test_size: float = 0.20
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for Bagging, AdaBoost, and Gradient Boosting ensemble classifiers.

    Attributes:
        n_estimators:      Number of base estimators in the ensemble.
        learning_rate:     Learning rate shrinks contribution of each tree in boosting.
        max_depth:         Maximum depth of individual decision tree estimators.
        random_state:      Random seed for reproducibility.
    """

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    random_state: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level: Minimum severity level.
        log_to_file: Whether to write logs to disk (False per user requirements).
    """

    log_level: str = "INFO"
    log_to_file: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregate configuration for the Bagging & Boosting ensemble pipeline.

    Usage:
        config = PipelineConfig()
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
