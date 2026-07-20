# ============================================================================
# Configuration Module for Decision Tree Classification Pipeline
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
        project_root: Absolute path to the Decision-Tree project directory.
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
            os.path.join(self.project_root, "data", "onlinefraud.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls column selections, downsampling, and train/test splitting.

    Attributes:
        target_column: Name of the target column to predict (isFraud).
        drop_columns:  Columns excluded from feature matrix (identifiers, flags).
        test_size:     Fraction of data reserved for evaluation.
        random_state:  Seed for reproducible data splitting and sampling.
        stratify:      Whether to preserve target class balance when splitting.
        max_samples:   Maximum number of samples to process (default: 50000 for fast runs).
    """

    target_column: str = "isFraud"

    # 'nameOrig' and 'nameDest' are high-cardinality string account identifiers.
    # 'isFlaggedFraud' is a rule-based post-transaction flag causing data leakage.
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "nameOrig",
            "nameDest",
            "isFlaggedFraud",
        ]
    )

    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    max_samples: Optional[int] = 50000


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Decision Tree Classifier.

    Attributes:
        criterion:         Splitting quality function ('gini', 'entropy', 'log_loss').
        max_depth:         Maximum depth of the decision tree (prevents overfitting).
        min_samples_split: Minimum number of samples required to split an internal node.
        min_samples_leaf:  Minimum number of samples required at a leaf node.
        random_state:      Random state seed for deterministic splitting choices.
        device:            Target execution device: 'cpu', 'gpu', or 'auto'.
    """

    criterion: str = "gini"
    max_depth: Optional[int] = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    random_state: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:    Minimum severity level (DEBUG, INFO, WARNING, etc.).
        log_to_file:  Whether to write logs to disk in the output directory.
        log_filename: Log filename written to output folder.
    """

    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_filename: str = "decision_tree_classification.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
