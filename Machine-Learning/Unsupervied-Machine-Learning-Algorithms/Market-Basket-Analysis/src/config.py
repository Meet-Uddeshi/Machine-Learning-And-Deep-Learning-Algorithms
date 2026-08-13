# ============================================================================
# Configuration Module for Market Basket Analysis Pipeline
# ============================================================================
# Centralizes all configuration parameters for the Market Basket Analysis
# pipeline, including dataset file paths, filtering parameters, Apriori algorithm
# thresholds (min support, min confidence, min lift), and logging configurations.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from project directory structure.

    Attributes:
        project_root: Absolute path to Market-Basket-Analysis directory.
        data_dir:     Absolute path to data/ directory.
        src_dir:      Absolute path to src/ directory.
        output_dir:   Absolute path to output/ directory for figures and reports.
        dataset_file: Absolute path to Assignment-1_Data.csv.
    """

    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    data_dir: str = ""
    src_dir: str = ""
    output_dir: str = ""
    dataset_file: str = ""

    def __post_init__(self) -> None:
        """Derive dependent paths relative to project_root."""
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
            os.path.join(self.project_root, "data", "Assignment-1_Data.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Parameters for loading, cleaning, and formatting transaction data.

    Attributes:
        delimiter:      CSV file delimiter (semicolon for Assignment-1_Data.csv).
        target_country: Optional country string to filter transactions (e.g. 'United Kingdom').
        min_quantity:   Minimum quantity threshold for valid sales.
        min_price:      Minimum item price threshold.
        sample_size:    Optional number of transactions to subsample for faster execution.
        random_state:   Random seed for sampling reproducibility.
    """

    delimiter: str = ";"
    target_country: Optional[str] = "United Kingdom"
    min_quantity: int = 1
    min_price: float = 0.001
    sample_size: Optional[int] = None
    random_state: int = 42


@dataclass(frozen=True)
class AprioriConfig:
    """Hyperparameters for Apriori frequent itemset mining and association rule generation.

    Attributes:
        min_support:    Minimum itemset support threshold (proportion of total transactions).
        min_confidence: Minimum rule confidence threshold P(B|A).
        min_lift:       Minimum rule lift threshold P(A U B) / (P(A) * P(B)).
        max_len:        Maximum itemset length k to consider during mining.
    """

    min_support: float = 0.02
    min_confidence: float = 0.2
    min_lift: float = 1.0
    max_len: int = 3


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration parameters for pipeline logging.

    Attributes:
        log_level:   Minimum severity level for logging messages.
        log_to_file: Whether to persist logs to disk (False per rules).
    """

    log_level: str = "INFO"
    log_to_file: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate top-level configuration for the Market Basket Analysis pipeline.

    Usage:
        config = PipelineConfig()
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    apriori: AprioriConfig = field(default_factory=AprioriConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
