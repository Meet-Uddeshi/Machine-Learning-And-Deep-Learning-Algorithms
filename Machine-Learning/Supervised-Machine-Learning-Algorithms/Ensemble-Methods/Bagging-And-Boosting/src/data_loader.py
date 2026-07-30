# ============================================================================
# Data Loader and Preprocessing Service Module
# ============================================================================
# Manages loading the Heart Disease dataset (`heart.csv`), schema validation,
# numerical feature scaling, and stratified train/test splitting.
# ============================================================================

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service class for loading, validating, scaling, and splitting classification data."""

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
        """Return names of input feature columns."""
        return self._feature_names

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load `heart.csv`, split features and target, scale features, and split train/test sets.

        Returns:
            Tuple of (x_train, x_test, y_train, y_test) arrays.
        """
        filepath = self._path_config.dataset_file
        if not os.path.exists(filepath):
            self._logger.error("Dataset file not found at: %s", filepath)
            raise FileNotFoundError(f"Dataset missing: {filepath}")

        self._logger.info("Loading dataset from: %s", filepath)
        df = pd.read_csv(filepath)
        self._logger.info("Dataset shape: %s", df.shape)

        target_col = self._data_config.target_column
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataset.")

        # Drop duplicate rows if present to prevent data leakage
        df = df.drop_duplicates().reset_index(drop=True)
        self._logger.info("Shape after dropping duplicate records: %s", df.shape)

        x_df = df.drop(columns=[target_col])
        y_vec = df[target_col].values
        self._feature_names = x_df.columns.tolist()

        x_train, x_test, y_train, y_test = train_test_split(
            x_df.values,
            y_vec,
            test_size=self._data_config.test_size,
            random_state=self._data_config.random_state,
            stratify=y_vec,
        )

        self._logger.info("Splitting dataset:")
        self._logger.info("  Training samples : %d", x_train.shape[0])
        self._logger.info("  Testing samples  : %d", x_test.shape[0])

        # Scale continuous features
        x_train_scaled = self._scaler.fit_transform(x_train)
        x_test_scaled = self._scaler.transform(x_test)

        return x_train_scaled, x_test_scaled, y_train, y_test
