# ============================================================================
# Configuration Module for GPU Statistics and Probability Pipeline
# ============================================================================
# Centralizes configuration settings for analyzing the GPU database.
# Includes path configurations, data loading options, analytical parameters,
# and logging options.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Statistics and Probability project directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results, plots, and reports are stored.
        dataset_file: Absolute path to the GPU database CSV.
    """

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
            os.path.join(self.project_root, "data", "gpu_database.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls columns, data preprocessing, and inferential hypothesis parameters.

    Attributes:
        target_column:      Primary target variable for regression (processing_power_gflops).
        feature_columns:    Continuous features for multivariate analysis.
        manufacturer_col:   Categorical grouping variable for manufacturer.
        test_size:          Fraction of dataset reserved for evaluation.
        random_state:       Seed for reproducible splits and sampling.
        confidence_level:   Default confidence level for confidence intervals (0.95).
        alpha_significance: Default significance level for hypothesis tests (0.05).
    """

    target_column: str = "processing_power_gflops"
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "transistors_million",
            "die_size_mm2",
            "core_clock_mhz",
            "tdp_watts",
        ]
    )
    manufacturer_col: str = "manufacturer"
    test_size: float = 0.20
    random_state: int = 42
    confidence_level: float = 0.95
    alpha_significance: float = 0.05


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters for probability distribution modeling and time series decomposition.

    Attributes:
        binomial_n:   Number of trials for Binomial distribution.
        binomial_p:   Probability parameter for Binomial distribution.
        poisson_lambda: Average rate for Poisson distribution.
        normal_mean:  Mean parameter for Normal distribution.
        normal_std:   Standard deviation parameter for Normal distribution.
        exp_lambda:   Rate parameter for Exponential distribution.
    """

    binomial_n: int = 20
    binomial_p: float = 0.5
    poisson_lambda: float = 4.0
    normal_mean: float = 100.0
    normal_std: float = 15.0
    exp_lambda: float = 0.02


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:    Minimum log severity level.
        log_to_file:  Whether to persist logs to disk in the output directory.
        log_filename: Name of the log file written to the output directory.
    """

    log_level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "statistics_and_probability.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregate configuration for the GPU Statistics & Probability pipeline.

    Usage:
        config = PipelineConfig()
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
