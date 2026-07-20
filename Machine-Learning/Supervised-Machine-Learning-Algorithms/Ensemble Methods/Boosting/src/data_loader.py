# ============================================================================
# Data Loader Module for Boosting Classification Pipeline
# ============================================================================
# Handles all data I/O, validation, exploration, encoding, and preprocessing.
# Follows the Service Layer pattern: the DataLoaderService class owns the
# entire data preparation lifecycle and exposes a clean interface to callers.
# ============================================================================

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service responsible for loading, validating, and preprocessing the dataset.

    Responsibilities:
        1. Read the CSV from disk and validate its schema.
        2. Log an exploratory summary (shape, types, nulls, class distribution).
        3. Encode categorical features and target labels.
        4. Split into train/test sets with optional stratification.
        5. Apply scaling if configured.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the data loader with configuration and logger.

        Args:
            path_config: Path settings for locating the dataset file.
            data_config: Data settings for columns, splits, and target.
            logger:      Logger instance for detailed progress reporting.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

        # Populated during preprocessing
        self._feature_names: list = []
        self._label_names: list = []
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._scaler: StandardScaler = StandardScaler()

    # -- Public properties ---------------------------------------------------

    @property
    def feature_names(self) -> list:
        """Column names of the final feature matrix."""
        return self._feature_names

    @property
    def label_names(self) -> list:
        """Ordered class labels produced by the label encoder."""
        return self._label_names

    # -- Public workflow methods ---------------------------------------------

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full data pipeline: load -> validate -> preprocess -> split -> scale.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        dataframe = self._load_csv()
        self._validate_schema(dataframe)
        self._log_data_summary(dataframe)
        features, target = self._preprocess(dataframe)
        
        # Split features and target
        x_train, x_test, y_train, y_test = self._split(features, target)
        
        # Apply scaling if configured
        if self._data_config.scale_features:
            self._logger.info("Feature scaling is activated in configuration.")
            x_train_scaled = self._scaler.fit_transform(x_train)
            x_test_scaled = self._scaler.transform(x_test)
            return x_train_scaled, x_test_scaled, y_train, y_test
        
        return x_train, x_test, y_train, y_test

    # -- Private implementation methods --------------------------------------

    def _load_csv(self) -> pd.DataFrame:
        """Read the CSV file and perform basic integrity checks.

        Returns:
            Raw pandas DataFrame.
        """
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
        """Ensure the target column and droppable columns exist in the data.

        Args:
            dataframe: The raw DataFrame to validate.
        """
        if self._data_config.target_column not in dataframe.columns:
            raise KeyError(
                f"Target column '{self._data_config.target_column}' "
                f"not found in dataset. Available columns: "
                f"{list(dataframe.columns)}"
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

    def _log_data_summary(self, dataframe: pd.DataFrame) -> None:
        """Log a detailed exploratory overview of the dataset.

        Args:
            dataframe: The raw DataFrame to summarize.
        """
        self._logger.info("=" * 70)
        self._logger.info("DATASET SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info("Rows: %d  |  Columns: %d", *dataframe.shape)

        # Column-level details
        self._logger.info("-" * 70)
        self._logger.info("%-35s  %-15s  %s", "Column", "Dtype", "Nulls")
        self._logger.info("-" * 70)
        for col in dataframe.columns:
            null_count = dataframe[col].isna().sum()
            self._logger.info(
                "%-35s  %-15s  %d", col, str(dataframe[col].dtype), null_count
            )

        # Target distribution
        self._logger.info("-" * 70)
        self._logger.info(
            "TARGET CLASS DISTRIBUTION ('%s'):", self._data_config.target_column
        )
        target_counts = dataframe[self._data_config.target_column].value_counts()
        total = len(dataframe)
        for label, count in target_counts.items():
            pct = (count / total) * 100
            self._logger.info("  %-15s : %6d  (%.1f%%)", str(label), count, pct)
        self._logger.info("=" * 70)

    def _preprocess(
        self, dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform raw data into model-ready numeric arrays.

        Args:
            dataframe: Validated raw DataFrame.

        Returns:
            Tuple of (features_array, target_array).
        """
        self._logger.info("Starting preprocessing pipeline...")

        # Separate and drop
        target_series = dataframe[self._data_config.target_column].copy()
        drop_cols = [self._data_config.target_column] + self._data_config.drop_columns
        features_df = dataframe.drop(columns=drop_cols)
        self._logger.info("Dropped columns: %s", drop_cols)
        self._logger.info("Feature matrix shape: %s", features_df.shape)

        # Impute missing values (if any)
        numeric_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = features_df.select_dtypes(exclude=["number"]).columns.tolist()

        for col in numeric_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with median=%.4f",
                    null_count, col, median_val,
                )

        for col in categorical_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                mode_val = features_df[col].mode()[0]
                features_df[col] = features_df[col].fillna(mode_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with mode='%s'",
                    null_count, col, mode_val,
                )

        # Encode target labels
        target_encoded = self._label_encoder.fit_transform(target_series)
        self._label_names = [str(c) for c in self._label_encoder.classes_]
        self._logger.info("Target labels encoded: %s", self._label_names)

        # Encode categorical features
        if categorical_cols:
            self._logger.info("Encoding categorical features...")
            for col in categorical_cols:
                col_encoder = LabelEncoder()
                features_df[col] = col_encoder.fit_transform(features_df[col])
                unique_map = dict(
                    zip(
                        col_encoder.classes_,
                        col_encoder.transform(col_encoder.classes_),
                    )
                )
                self._logger.debug("  %-30s -> %s", col, unique_map)

        self._feature_names = features_df.columns.tolist()
        self._logger.info("Final feature columns: %s", self._feature_names)
        return features_df.values, target_encoded

    def _split(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partition data into training and test sets.

        Args:
            features: Preprocessed feature matrix.
            target:   Encoded target vector.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
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
