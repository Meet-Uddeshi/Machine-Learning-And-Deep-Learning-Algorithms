# ============================================================================
# Data Loader and Preprocessing Service Module
# ============================================================================
# Manages loading the Credit Card transaction dataset (`creditcard.csv`),
# sampling representative observations, selecting features (`V1-V28`, `Amount`),
# and standardizing feature columns for clustering algorithms.
# ============================================================================

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service class for loading, validating, sampling, and standardizing clustering data."""

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize DataLoaderService.

        Args:
            path_config: Immutable path settings.
            data_config: Dataset parameters.
            logger:      Logger instance.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger
        self._scaler = StandardScaler()
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        """Return list of selected clustering feature names."""
        return self._feature_names

    def load_and_prepare(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load `creditcard.csv`, sample representative subset, and scale features.

        Returns:
            Tuple of (scaled_feature_matrix, sampled_dataframe).
        """
        filepath = self._path_config.dataset_file
        if not os.path.exists(filepath):
            self._logger.error("Dataset file not found at: %s", filepath)
            raise FileNotFoundError(f"Dataset missing: {filepath}")

        self._logger.info("Loading credit card dataset from: %s", filepath)
        df = pd.read_csv(filepath)
        self._logger.info("Raw dataset shape: %s", df.shape)

        # Sample representative subset for efficient distance calculations
        sample_n = min(self._data_config.sample_size, len(df))
        df_sampled = df.sample(
            n=sample_n, random_state=self._data_config.random_state
        ).reset_index(drop=True)
        self._logger.info("Subsampled dataset shape: %s", df_sampled.shape)

        # Drop non-clustering columns (e.g. Time, Class)
        drop_cols = [
            c for c in self._data_config.drop_columns if c in df_sampled.columns
        ]
        x_df = df_sampled.drop(columns=drop_cols)
        self._feature_names = x_df.columns.tolist()

        # Standardize all features (zero mean, unit variance)
        x_scaled = self._scaler.fit_transform(x_df.values)
        self._logger.info("Features standardized successfully.")

        return x_scaled, df_sampled
