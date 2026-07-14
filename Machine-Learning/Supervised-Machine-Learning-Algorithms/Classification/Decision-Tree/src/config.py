# ============================================================================
# Configuration Module for Decision Tree Classification Pipeline
# ============================================================================
# Centralizes every tunable parameter so that users can adjust behavior
# without modifying business logic.  All paths are resolved relative to the
# project root (one level above 'src/'), making the config portable.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Classification project directory.
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
            os.path.join(self.project_root, "data", "oil_sales_assignment_dataset.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls which columns to use and how to split the data.

    Attributes:
        target_column:  Name of the column to predict (classification label).
        drop_columns:   Columns excluded from features (identifiers, leakage).
        test_size:      Fraction of data reserved for test evaluation.
        random_state:   Seed for reproducible train/test splits.
        stratify:       Whether to use stratified splitting on the target.
        max_samples:    Maximum number of samples to use (None for all).
    """

    target_column: str = "price_bracket"

    # 'store_name' has ~1997 unique values in 2000 rows -- it is a near-unique
    # identifier that would cause the tree to memorize store IDs rather than
    # learn generalizable patterns.
    # 'sku' has ~1571 unique values -- same reason; it encodes brand+class+size
    # which are already captured individually by other columns.
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "store_name",
            "sku",
        ]
    )

    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    max_samples: Optional[int] = None


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Decision Tree classifier.

    Attributes:
        criterion:          Function to measure split quality ('gini' or 'entropy').
        max_depth:          Maximum depth of the tree (None = grow until pure).
        min_samples_split:  Minimum samples required to split an internal node.
        min_samples_leaf:   Minimum samples required at a leaf node.
        max_features:       Number of features considered for best split
                            ('sqrt', 'log2', int, float, or None for all).
        class_weight:       Weighting strategy for classes ('balanced' or None).
        random_state:       Seed for reproducibility.
        ccp_alpha:          Complexity parameter for Minimal Cost-Complexity Pruning.
                            Higher values prune more aggressively.
    """

    criterion: str = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[str] = None
    class_weight: Optional[str] = "balanced"
    random_state: int = 42
    ccp_alpha: float = 0.0


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
    log_filename: str = "decision_tree_oil_sales.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations.

    Usage:
        config = PipelineConfig()             # all defaults
        config = PipelineConfig(              # override specific knobs
            model=ModelConfig(max_depth=5)
        )
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
