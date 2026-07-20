# ============================================================================
# Configuration Module for Support Vector Machine Classification Pipeline
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
        project_root: Absolute path to the Support-Vector-Machine project directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results and logs are written.
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
        # 'frozen=True' requires object.__setattr__ for post-init mutation
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
            os.path.join(
                self.project_root, "data", "breast_cancer_wisconsin.csv"
            ),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls which columns to use and how to split the data.

    Attributes:
        target_column: Name of the column to predict (classification label).
        drop_columns:  Columns excluded from features (identifiers, leakage).
        test_size:     Fraction of data reserved for test evaluation.
        random_state:  Seed for reproducible train/test splits.
        stratify:      Whether to use stratified splitting on the target.
        max_samples:   Maximum number of samples to use (None for all).
    """

    target_column: str = "diagnosis"

    # 'id' is a sample identifier with no predictive value.
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "id",
        ]
    )

    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    max_samples: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Support Vector Classifier.

    Attributes:
        kernel:       Kernel type algorithm ('rbf', 'linear', 'poly', 'sigmoid').
        C:            Regularization parameter trade-off (margin width vs error).
        gamma:        Kernel coefficient ('scale' or 'auto').
        probability:  Whether to enable probability estimates.
        random_state: Random state seed for reproducibility.
        device:       Target execution device: 'cpu', 'gpu', or 'auto'.
    """

    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    probability: bool = True
    random_state: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:    Minimum severity level (DEBUG, INFO, WARNING, etc.).
        log_to_file:  Whether to persist logs to disk in the output directory.
        log_filename: Name of the log file written to the output directory.
    """

    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_filename: str = "svm_classification.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations.

    Usage:
        config = PipelineConfig()       # all defaults
        config = PipelineConfig(        # override specific knobs
            model=ModelConfig(kernel='linear', C=10.0)
        )
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
