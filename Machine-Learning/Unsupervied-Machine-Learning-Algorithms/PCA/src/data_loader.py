# ============================================================================
# Data Loader and Preprocessing Service Module
# ============================================================================
# Manages loading the Vehicle Silhouette dataset (`pca.csv`), handling missing
# values, separating continuous features from categorical targets, and standardizing
# feature vectors using StandardScaler.
# ============================================================================

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service class for loading, validating, cleaning, and standardizing PCA data."""

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize DataLoaderService.

        Args:
            path_config: Path configuration settings.
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
        """Return list of continuous feature column names."""
        return self._feature_names

    def load_and_prepare(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load `pca.csv`, impute missing numerical values, scale features, and separate class targets.

        Returns:
            Tuple of (scaled_feature_matrix, target_labels, cleaned_dataframe).
        """
        filepath = self._path_config.dataset_file
        if not os.path.exists(filepath):
            self._logger.error("Dataset file not found at: %s", filepath)
            raise FileNotFoundError(f"Dataset missing: {filepath}")

        self._logger.info("Loading PCA dataset from: %s", filepath)
        df = pd.read_csv(filepath)
        self._logger.info("Raw dataset shape: %s", df.shape)

        target_col = self._data_config.target_column
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset.")

        # Step 1: Separate features and target
        x_df = df.drop(columns=[target_col]).copy()
        y_series = df[target_col].astype(str).str.strip().copy()
        self._feature_names = x_df.columns.tolist()

        # Step 2: Handle missing numerical values using median imputation
        for col in x_df.columns:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
            if x_df[col].isnull().sum() > 0:
                col_median = x_df[col].median()
                self._logger.info("Imputing missing values in '%s' with median %.2f", col, col_median)
                x_df[col] = x_df[col].fillna(col_median)

        # Step 3: Standardize features (zero mean, unit variance)
        x_scaled = self._scaler.fit_transform(x_df.values)
        self._logger.info("Standardized %d features across %d samples.", x_scaled.shape[1], x_scaled.shape[0])

        df_cleaned = x_df.copy()
        df_cleaned[target_col] = y_series.values

        return x_scaled, y_series.values, df_cleaned
