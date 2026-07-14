# ============================================================================
# Configuration Module for Logistic Regression Pipeline
# ============================================================================
# Centralizes all configurations for predicting global supply chain disruptions.
# Configures path dependencies, dataset column handling, training hyper-parameters,
# and logging options.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Logistic Regression project directory.
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
        # Use object.__setattr__ to bypass frozen state during initialization
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
            os.path.join(self.project_root, "data", "global_supply_chain_risk_2026.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls which columns to use and how to split the data.

    Attributes:
        target_column:      Name of the binary target column (0 or 1).
        drop_columns:       Columns excluded from features (identifiers or dates).
        test_size:          Fraction of data reserved for test evaluation.
        random_state:       Seed for reproducible train/test splits.
        stratify:           Whether to split using stratified sampling.
        max_samples:        Maximum number of samples to use (None for all).
    """

    target_column: str = "Disruption_Occurred"

    # 'Shipment_ID' is an identifier and must be excluded to prevent overfitting.
    # 'Date' is dropped after feature extraction.
    drop_columns: List[str] = field(
        default_factory=lambda: ["Shipment_ID", "Date"]
    )

    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    max_samples: int = None


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for scikit-learn's Logistic Regression classifier.

    Attributes:
        penalty:      Norm used in the penalization (regularization): 'l1', 'l2', or 'none'.
        C:            Inverse of regularization strength; smaller values specify stronger regularization.
        solver:       Algorithm to use in the optimization problem.
        max_iter:     Maximum number of iterations taken for the solvers to converge.
        random_state: Seed for optimization solver convergence reproducibility.
    """

    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
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
    log_filename: str = "logistic_regression.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations.

    Usage:
        config = PipelineConfig()
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
