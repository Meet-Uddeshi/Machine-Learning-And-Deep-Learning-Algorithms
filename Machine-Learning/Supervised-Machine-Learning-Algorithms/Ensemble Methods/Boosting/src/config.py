# ============================================================================
# Configuration Module for Boosting Classification Pipeline
# ============================================================================
# Centralizes every tunable parameter so that users can adjust behavior
# without modifying business logic. All paths are resolved relative to the
# project root (one level above 'src/'), making the config portable.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Boosting project directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results are written.
        dataset_file: Full path to the CSV dataset.
    """

    # Resolve project root as parent of 'src/'
    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    data_dir: str = ""
    src_dir: str = ""
    output_dir: str = ""
    dataset_file: str = ""

    def __post_init__(self) -> None:
        """Derive dependent paths from project_root after init."""
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
            os.path.join(self.project_root, "data", "diabetes.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls which columns to use and how to split the data.

    Attributes:
        target_column:  Name of the column to predict (Outcome - 0 or 1).
        drop_columns:   Columns excluded from features (if any).
        test_size:      Fraction of data reserved for test evaluation.
        random_state:   Seed for reproducible train/test splits.
        stratify:       Whether to use stratified splitting on the target.
        scale_features: Whether to apply scaling (disabled by default; tree boosting is scale-invariant).
    """

    target_column: str = "Outcome"
    drop_columns: List[str] = field(default_factory=list)
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    scale_features: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Boosting classifier.

    Attributes:
        boosting_type:          Type of boosting algorithm ('gradient_boosting' or 'adaboost').
        n_estimators:           Number of boosting stages/estimators (default: 100).
        learning_rate:          Step size shrinkage used in update to prevent overfitting (default: 0.1).
        max_depth:              Maximum depth of the individual regression estimators (default: 3).
        min_samples_split:      Minimum samples required to split an internal node.
        min_samples_leaf:       Minimum samples required at a leaf node.
        subsample:              Fraction of samples to be used for fitting individual base learners.
        random_state:           Seed for reproducibility.
    """

    boosting_type: str = "gradient_boosting"  # 'gradient_boosting' or 'adaboost'
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    subsample: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:     Minimum severity level (DEBUG, INFO, WARNING, etc.).
        log_to_file:   Must be False to prevent logs being saved locally.
        log_filename:  Disabled, set to empty string.
    """

    log_level: str = "INFO"
    log_to_file: bool = False
    log_filename: str = ""


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
