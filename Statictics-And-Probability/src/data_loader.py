# ============================================================================
# Data Loader and Preprocessing Service Module
# ============================================================================
# Manages loading the GPU database CSV file (`gpu_database.csv`), validating schema,
# handling missing values, parsing dates, extracting features, and preparing clean
# DataFrames for statistical testing and modeling.
# ============================================================================

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service class for loading, validating, and preprocessing the GPU dataset.

    Encapsulates dataset loading for descriptive statistics, hypothesis testing,
    time series analysis, regression modeling, and non-parametric evaluations.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize DataLoaderService with paths and data configurations.

        Args:
            path_config: Immutable path settings.
            data_config: Dataset parameters.
            logger:      Configured logging instance.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load `gpu_database.csv`, validate columns, parse dates, and clean numeric features.

        Preprocessing Steps:
            1. Check file existence.
            2. Parse `launch_date` to Datetime and extract `launch_year`.
            3. Convert numeric columns (transistors, die_size, clocks, power, tdp) to float.
            4. Impute missing numerical values using feature medians.
            5. Normalize/clean manufacturer labels (e.g. ATI -> AMD).

        Returns:
            Preprocessed pandas DataFrame containing GPU observations.
        """
        filepath = self._path_config.dataset_file
        if not os.path.exists(filepath):
            self._logger.error("Dataset file not found at path: %s", filepath)
            raise FileNotFoundError(f"Missing dataset file: {filepath}")

        self._logger.info("Loading GPU dataset from: %s", filepath)
        df = pd.read_csv(filepath)
        self._logger.info("Raw dataset shape: %s", df.shape)

        # Step 1: Parse Datetime and Year
        if "launch_date" in df.columns:
            dt_series = pd.to_datetime(df["launch_date"], errors="coerce")
            df["launch_date"] = dt_series
            df["launch_year"] = pd.DatetimeIndex(dt_series).year
            # Fill missing launch years with median year
            median_year = int(df["launch_year"].median())
            df["launch_year"] = df["launch_year"].fillna(median_year).astype(int)

        # Step 2: Clean and Impute Numeric Features
        numeric_cols = [
            "transistors_million",
            "die_size_mm2",
            "core_clock_mhz",
            "memory_clock_mhz",
            "processing_power_gflops",
            "tdp_watts",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                col_median = df[col].median()
                df[col] = df[col].fillna(col_median)

        # Step 3: Clean Manufacturer Categories
        if "manufacturer" in df.columns:
            df["manufacturer"] = df["manufacturer"].astype(str).str.strip()
            # Map legacy ATI to AMD for consistency
            df["manufacturer"] = df["manufacturer"].replace({"ATI": "AMD"})
            # Filter top manufacturers for clean comparisons
            top_manufacturers = ["Nvidia", "AMD", "Intel"]
            df["manufacturer_clean"] = df["manufacturer"].apply(
                lambda m: m if m in top_manufacturers else "Other"
            )

        self._logger.info("Dataset preprocessing completed successfully.")
        self._logger.info("Cleaned dataset shape: %s", df.shape)
        return df

    def get_time_series_aggregated(self, df: pd.DataFrame) -> pd.Series:
        """Aggregate yearly median GPU processing power (GFLOPS) for time series analysis.

        Args:
            df: Cleaned GPU DataFrame.

        Returns:
            Pandas Series indexed by launch year.
        """
        self._logger.info("Aggregating time series data by launch_year...")
        ts = (
            df.groupby("launch_year")["processing_power_gflops"]
            .median()
            .sort_index()
        )
        return ts
