# ============================================================================
# Configuration Module for Unsupervised Clustering Pipeline
# ============================================================================
# Centralizes all configuration settings for unsupervised clustering algorithms
# (K-Means, DBSCAN, Agglomerative Hierarchical Clustering) applied to the
# Credit Card transaction features dataset (`creditcard.csv`).
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Clustering project directory.
        data_dir:     Absolute path to the 'data/' directory.
        src_dir:      Absolute path to the 'src/' directory.
        output_dir:   Absolute path where figures and reports are stored.
        dataset_file: Absolute path to creditcard.csv dataset.
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
            os.path.join(self.project_root, "data", "creditcard.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Parameters for sampling, feature selection, and scaling.

    Attributes:
        sample_size:   Subsample size for efficient clustering computation.
        random_state:  Random seed for reproducible sampling and initialization.
        drop_columns:  Columns to exclude from clustering features (Time, Class).
    """

    sample_size: int = 5000
    random_state: int = 42
    drop_columns: List[str] = field(
        default_factory=lambda: ["Time", "Class"]
    )


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for K-Means, DBSCAN, and Agglomerative Clustering algorithms.

    Attributes:
        n_clusters:          Target number of clusters k for K-Means and Agglomerative.
        dbscan_eps:          Epsilon radius neighborhood parameter for DBSCAN.
        dbscan_min_samples:  Minimum samples per cluster core point for DBSCAN.
        random_state:        Random seed for reproducibility.
    """

    n_clusters: int = 4
    dbscan_eps: float = 2.5
    dbscan_min_samples: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Controls pipeline logging behaviors.

    Attributes:
        log_level:   Minimum severity level.
        log_to_file: Whether to write logs to disk (False per user requirements).
    """

    log_level: str = "INFO"
    log_to_file: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregate configuration for the Clustering pipeline.

    Usage:
        config = PipelineConfig()
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
