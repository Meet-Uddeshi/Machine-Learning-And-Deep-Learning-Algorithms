# ============================================================================
# Data Loader Module for Random Forest Classification Pipeline
# ============================================================================
# Handles all data I/O, validation, exploration, downsampling, and preprocessing.
# Follows the Service Layer pattern: DataLoaderService owns data preparation.
# ============================================================================

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service responsible for loading, validating, downsampling, and preprocessing dataset.

    Responsibilities:
        1. Read CSV from disk and validate schema.
        2. Perform stratified downsampling if max_samples is configured.
        3. Log exploratory statistics (shape, dtypes, missing values, target ratio).
        4. Encode categorical variables and scale numerical variables.
        5. Split into train/test subsets with stratification.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize data loader service."""
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

        self._feature_names: List[str] = []
        self._label_names: List[str] = []
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._scaler: StandardScaler = StandardScaler()

    @property
    def feature_names(self) -> List[str]:
        """Column names of features post-encoding."""
        return self._feature_names

    @property
    def label_names(self) -> List[str]:
        """Ordered class label names."""
        return self._label_names

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute full data pipeline: load -> validate -> downsample -> preprocess -> split."""
        dataframe = self._load_csv()
        self._validate_schema(dataframe)
        dataframe = self._downsample(dataframe)
        self._log_data_summary(dataframe)
        features, target = self._preprocess(dataframe)
        return self._split(features, target)

    def _load_csv(self) -> pd.DataFrame:
        """Read CSV file and check integrity."""
        self._logger.info(
            "Loading dataset from: %s", self._path_config.dataset_file
        )
        dataframe = pd.read_csv(self._path_config.dataset_file)

        if dataframe.empty:
            raise ValueError(
                f"Dataset is empty: {self._path_config.dataset_file}"
            )

        self._logger.info(
            "Dataset loaded successfully -- shape: %s", dataframe.shape
        )
        return dataframe

    def _validate_schema(self, dataframe: pd.DataFrame) -> None:
        """Validate presence of target and drop columns."""
        if self._data_config.target_column not in dataframe.columns:
            raise KeyError(
                f"Target column '{self._data_config.target_column}' missing from dataset. "
                f"Available columns: {list(dataframe.columns)}"
            )

        missing_drops = [
            col
            for col in self._data_config.drop_columns
            if col not in dataframe.columns
        ]
        if missing_drops:
            raise KeyError(
                f"Columns configured for dropping are absent: {missing_drops}"
            )

        self._logger.info("Schema validation passed.")

    def _downsample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Perform stratified downsampling if max_samples is configured."""
        max_samples = self._data_config.max_samples
        if max_samples is None or max_samples >= len(dataframe):
            return dataframe

        self._logger.info(
            "Downsampling dataset from %d to %d samples...",
            len(dataframe),
            max_samples,
        )

        target = self._data_config.target_column
        fraction = max_samples / len(dataframe)

        downsampled = dataframe.groupby(target, group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(np.round(len(x) * fraction))),
                random_state=self._data_config.random_state,
            )
        )

        self._logger.info(
            "Downsampling complete. New dataset shape: %s", downsampled.shape
        )
        return downsampled

    def _log_data_summary(self, dataframe: pd.DataFrame) -> None:
        """Log summary of dataset shape, types, missing values, and target distribution."""
        self._logger.info("=" * 70)
        self._logger.info("DATASET SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info("Rows: %d  |  Columns: %d", *dataframe.shape)

        self._logger.info("-" * 70)
        self._logger.info("%-35s  %-15s  %s", "Column", "Dtype", "Nulls")
        self._logger.info("-" * 70)
        for col in dataframe.columns:
            null_count = dataframe[col].isna().sum()
            self._logger.info(
                "%-35s  %-15s  %d", col, str(dataframe[col].dtype), null_count
            )

        self._logger.info("-" * 70)
        self._logger.info(
            "TARGET CLASS DISTRIBUTION ('%s'):", self._data_config.target_column
        )
        target_counts = dataframe[
            self._data_config.target_column
        ].value_counts()
        total = len(dataframe)
        for label, count in target_counts.items():
            pct = (count / total) * 100
            self._logger.info("  %-15s : %6d  (%.1f%%)", str(label), count, pct)
        self._logger.info("=" * 70)

    def _preprocess(
        self, dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform raw data into model-ready arrays."""
        self._logger.info("Starting preprocessing pipeline...")

        target_series = dataframe[self._data_config.target_column].copy()
        features_df = dataframe.drop(
            columns=[self._data_config.target_column]
            + self._data_config.drop_columns
        )
        self._logger.info(
            "Dropped columns: %s",
            [self._data_config.target_column] + self._data_config.drop_columns,
        )

        numeric_cols = features_df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        categorical_cols = features_df.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        # Missing value imputation
        for col in numeric_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with median=%.4f",
                    null_count,
                    col,
                    median_val,
                )

        for col in categorical_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                mode_val = features_df[col].mode()[0]
                features_df[col] = features_df[col].fillna(mode_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with mode='%s'",
                    null_count,
                    col,
                    mode_val,
                )

        # Label encode target
        target_encoded = self._label_encoder.fit_transform(target_series)
        self._label_names = [str(cls) for cls in self._label_encoder.classes_]

        # Label encode categorical features
        for col in categorical_cols:
            col_encoder = LabelEncoder()
            features_df[col] = col_encoder.fit_transform(features_df[col])

        self._feature_names = features_df.columns.tolist()

        # Standardize features
        features_scaled = self._scaler.fit_transform(features_df.values)
        self._logger.info(
            "Features preprocessed and scaled. Shape: %s", features_scaled.shape
        )

        return features_scaled, target_encoded

    def _split(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test subsets."""
        stratify_arg = target if self._data_config.stratify else None

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self._data_config.test_size,
            random_state=self._data_config.random_state,
            stratify=stratify_arg,
        )

        self._logger.info(
            "Train/Test split complete -- "
            "Train: %d samples | Test: %d samples | Test ratio: %.0f%%",
            x_train.shape[0],
            x_test.shape[0],
            self._data_config.test_size * 100,
        )

        return x_train, x_test, y_train, y_test
