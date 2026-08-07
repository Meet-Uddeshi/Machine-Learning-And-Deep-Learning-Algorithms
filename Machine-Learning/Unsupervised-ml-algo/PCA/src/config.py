# ============================================================================
# Configuration Module for PCA Pipeline
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
        project_root: Absolute path to the PCA directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results and plots are written.
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
            os.path.join(self.project_root, "data", "pca.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls columns, splitting, and scaling settings.

    Attributes:
        target_column:  Target label column (used only for 2D visualization coloring).
        drop_columns:   Columns excluded from PCA decomposition.
        test_size:      Fraction of data reserved for test evaluation.
        random_state:   Seed for reproducible train/test splits.
        stratify:       Stratify by target label to ensure representative classes.
        scale_features: Whether to scale features. Set to True by default since
                        PCA is highly sensitive to variable scales.
    """

    target_column: str = "class"
    drop_columns: List[str] = field(default_factory=list)
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    scale_features: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for Principal Component Analysis.

    Attributes:
        n_components: Number of components to keep (None keeps all components).
        svd_solver:   SVD solver strategy ('auto', 'full', 'arpack', 'randomized').
        whiten:       Whether to multiply components vectors by sqrt(n_samples) and
                      scale by singular values.
        random_state: Seed for randomized SVD solver reproducibility.
    """

    n_components: Optional[int] = None
    svd_solver: str = "auto"
    whiten: bool = False
    random_state: int = 42


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:     Minimum severity level.
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
