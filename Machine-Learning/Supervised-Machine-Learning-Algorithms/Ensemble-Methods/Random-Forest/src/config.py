# ============================================================================
# Configuration Module for Random Forest Classification Pipeline
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
        project_root: Absolute path to the Random-Forest project directory.
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
            os.path.join(
                self.project_root,
                "data",
                "indian_railway_failure_detection_maintenance_v2.csv",
            ),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls column selections, downsampling, and train/test splitting.

    Attributes:
        target_column: Name of target column to predict (maintenance_required).
        drop_columns:  Columns excluded from feature matrix (identifiers, leakage).
        test_size:     Fraction of data reserved for evaluation.
        random_state:  Seed for reproducible data splitting.
        stratify:      Whether to preserve target class balance when splitting.
        max_samples:   Maximum number of samples to process (default: 50000).
    """

    target_column: str = "maintenance_required"

    # 'train_id' is an identifier.
    # 'failure_type', 'failure_severity', and 'risk_score' are post-outcome
    # failure metrics that cause data leakage if used as predictive features.
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "train_id",
            "failure_type",
            "failure_severity",
            "risk_score",
        ]
    )

    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    max_samples: Optional[int] = 50000


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Random Forest Classifier.

    Attributes:
        n_estimators:      Number of trees in the ensemble forest.
        criterion:         Quality measure for node splits ('gini', 'entropy').
        max_depth:         Maximum depth of individual decision trees.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf:  Minimum samples required at a leaf node.
        bootstrap:         Whether to use bootstrap samples when building trees.
        oob_score:         Whether to compute out-of-bag generalization score.
        n_jobs:            Number of parallel CPU workers (1 to avoid Windows IPC overhead).
        random_state:      Random state seed for reproducibility.
        device:            Target execution device: 'cpu', 'gpu', or 'auto'.
    """

    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: Optional[int] = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    bootstrap: bool = True
    oob_score: bool = True
    n_jobs: int = 1
    random_state: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline."""

    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_filename: str = "random_forest_classification.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
