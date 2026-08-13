# ============================================================================
# Configuration Module for MBA & Apriori Pipeline
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
        project_root: Absolute path to the MBA directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results and plots are written.
        zip_file:     Full path to the zip archive containing retail transactions.
        dataset_file: Full path to the extracted CSV dataset.
    """

    # Resolve project root as parent of 'src/'
    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    data_dir: str = ""
    src_dir: str = ""
    output_dir: str = ""
    zip_file: str = ""
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
            "zip_file",
            os.path.join(self.project_root, "data", "archive (8).zip"),
        )
        object.__setattr__(
            self,
            "dataset_file",
            os.path.join(self.project_root, "data", "Assignment-1_Data.csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls dataset filtering and basket creation settings.

    Attributes:
        csv_filename:   Internal CSV name inside the zip file.
        delimiter:      CSV separator string.
        country_filter: Optional country filter (e.g. 'United Kingdom' or None for all).
        min_quantity:   Minimum item quantity per line item (excludes returns/cancellations).
        min_price:      Minimum unit price (excludes free/postage items).
        sample_baskets: Maximum number of transaction baskets to process (None for all).
    """

    csv_filename: str = "Assignment-1_Data.csv"
    delimiter: str = ";"
    country_filter: Optional[str] = "United Kingdom"
    min_quantity: int = 1
    min_price: float = 0.01
    sample_baskets: Optional[int] = 10000


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Apriori algorithm and Association Rule Mining.

    Attributes:
        min_support:        Minimum itemset support threshold (fraction of transactions).
        min_confidence:     Minimum rule confidence threshold (P(B|A)).
        min_lift:           Minimum rule lift threshold (Ratio of observed to expected support).
        max_itemset_length: Maximum size of itemsets to evaluate.
        top_n_rules:        Number of top rules to display and log.
    """

    min_support: float = 0.015
    min_confidence: float = 0.20
    min_lift: float = 1.0
    max_itemset_length: int = 3
    top_n_rules: int = 20


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
