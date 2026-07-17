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
        project_root: Absolute path to the Random Forest project directory.
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
            os.path.join(self.project_root, "data", "sales_data.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls which columns to use and how to split the data.

    Attributes:
        target_column:  Name of the column to predict (classification label).
        drop_columns:   Columns excluded from features.
        test_size:      Fraction of data reserved for test evaluation.
        random_state:   Seed for reproducible train/test splits.
        stratify:       Whether to use stratified splitting on the target.
        scale_features: Whether to apply scaling (False by default; Random Forest is scale-invariant).
    """

    target_column: str = "Product_Category"
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "Product_ID",
            "Sale_Date",
            "Region_and_Sales_Rep",
        ]
    )
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    scale_features: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Random Forest classifier.

    Attributes:
        n_estimators:       Number of trees in the forest (default: 100).
        criterion:          Function to measure split quality ('gini' or 'entropy').
        max_depth:          Maximum depth of each tree (None = grow until pure).
        min_samples_split:  Minimum samples required to split an internal node.
        min_samples_leaf:   Minimum samples required at a leaf node.
        max_features:       Features considered for best split ('sqrt', 'log2', or None).
        bootstrap:          Whether bootstrap samples are used when building trees.
        oob_score:          Whether to use out-of-bag samples to estimate generalization score.
        random_state:       Seed for reproducibility.
    """

    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    oob_score: bool = True
    random_state: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:     Minimum severity level (DEBUG, INFO, WARNING, etc.).
        log_to_file:   Whether to persist logs to disk in the output directory.
        log_filename:  Name of the log file written to the output directory.
    """

    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_filename: str = "random_forest_sales.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
