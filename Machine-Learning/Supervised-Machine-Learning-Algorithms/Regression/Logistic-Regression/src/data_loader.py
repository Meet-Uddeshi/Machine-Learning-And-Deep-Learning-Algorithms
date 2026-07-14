# ============================================================================
# Data Loader Module for Logistic Regression Pipeline
# ============================================================================
# Handles all supply chain dataset ingestion, validation, preprocessing,
# categorical encoding, scaling, and train-test splits.
# Exposes a clean interface to downstream model orchestration components.
# ============================================================================

import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service responsible for loading, validating, and preprocessing the supply chain risk dataset.

    Responsibilities:
        1. Ingest dataset and validate column requirements.
        2. Extract features from dates (Month, Year).
        3. Encode categorical columns using individual LabelEncoders.
        4. Impute missing values with median (numeric) or mode (categorical).
        5. Scale features using StandardScaler.
        6. Return train-test splits with optional target stratification.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the data loader service.

        Args:
            path_config: Paths for dataset discovery.
            data_config: Preprocessing configurations.
            logger:      Logger pipeline.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

        # Feature properties
        self._feature_names: list = []
        self._label_names: list = []
        self._categorical_encoders: Dict[str, LabelEncoder] = {}
        self._target_encoder: LabelEncoder = LabelEncoder()
        self._scaler: StandardScaler = StandardScaler()

    @property
    def feature_names(self) -> list:
        """Names of features processed and scale-transformed."""
        return self._feature_names

    @property
    def label_names(self) -> list:
        """Label names for output classes (e.g., [No Disruption, Disruption])."""
        return self._label_names

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Orchestrate the full data loaders pipeline.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        dataframe = self._load_csv()
        self._validate_schema(dataframe)
        dataframe = self._downsample(dataframe)
        self._log_data_summary(dataframe)
        
        features, target = self._preprocess(dataframe)
        return self._split(features, target)

    def _load_csv(self) -> pd.DataFrame:
        """Read CSV from target files and handle basic errors."""
        self._logger.info(
            "Loading Global Supply Chain Risk dataset from: %s", self._path_config.dataset_file
        )
        if not os.path.exists(self._path_config.dataset_file):
            raise FileNotFoundError(f"Dataset file not found at: {self._path_config.dataset_file}")

        dataframe = pd.read_csv(self._path_config.dataset_file)
        if dataframe.empty:
            raise ValueError(f"Dataset is empty: {self._path_config.dataset_file}")

        self._logger.info(
            "Dataset loaded successfully -- shape: %s", dataframe.shape
        )
        return dataframe

    def _validate_schema(self, dataframe: pd.DataFrame) -> None:
        """Ensure columns required by configuration are present.

        Args:
            dataframe: The raw dataframe.
        """
        if self._data_config.target_column not in dataframe.columns:
            raise KeyError(
                f"Target column '{self._data_config.target_column}' missing."
            )

        for col in self._data_config.drop_columns:
            if col not in dataframe.columns:
                raise KeyError(
                    f"Configured drop column '{col}' missing from the dataset."
                )

        self._logger.info("Schema validation passed.")

    def _downsample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Stratified downsampling to keep class balances equivalent.

        Args:
            dataframe: Inputs.

        Returns:
            The downsampled dataframe.
        """
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
                n=int(np.round(len(x) * fraction)),
                random_state=self._data_config.random_state
            )
        )
        self._logger.info(
            "Downsampling complete. New shape: %s", downsampled.shape
        )
        return downsampled

    def _log_data_summary(self, dataframe: pd.DataFrame) -> None:
        """Log column types, shapes, and class distribution of target."""
        self._logger.info("=" * 70)
        self._logger.info("DATASET SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info("Rows: %d  |  Columns: %d", *dataframe.shape)

        self._logger.info("-" * 70)
        self._logger.info("%-30s  %-15s  %s", "Column", "Dtype", "Nulls")
        self._logger.info("-" * 70)
        for col in dataframe.columns:
            null_count = dataframe[col].isna().sum()
            self._logger.info(
                "%-30s  %-15s  %d", col, str(dataframe[col].dtype), null_count
            )

        self._logger.info("-" * 70)
        self._logger.info("TARGET CLASS DISTRIBUTION ('%s'):", self._data_config.target_column)
        counts = dataframe[self._data_config.target_column].value_counts()
        total = len(dataframe)
        for label, count in counts.items():
            pct = (count / total) * 100
            self._logger.info("  %-15s : %6d  (%.1f%%)", str(label), count, pct)
        self._logger.info("=" * 70)

    def _preprocess(
        self, dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply conversions, label encoding, null imputation, and scaling.

        Args:
            dataframe: Validated raw DataFrame.

        Returns:
            Tuple of feature matrix (scaled) and target vector.
        """
        self._logger.info("Beginning preprocessing pipeline...")
        df_processed = dataframe.copy()

        # Step 1: Feature Engineering on Date (format: YYYY-MM-DD)
        if "Date" in df_processed.columns:
            self._logger.info("Extracting Month & Year from 'Date' column...")
            df_processed["Date_Parsed"] = pd.to_datetime(
                df_processed["Date"], format="%Y-%m-%d", errors="coerce"
            )
            
            # Fallback in case of parsing failures due to format differences
            if df_processed["Date_Parsed"].isna().any():
                self._logger.warning("Some dates failed to parse with standard format. Using automatic inference.")
                df_processed["Date_Parsed"] = pd.to_datetime(
                    df_processed["Date"], errors="coerce"
                )

            # Impute invalid dates with mode parsed date to avoid NaN features
            if df_processed["Date_Parsed"].isna().any():
                mode_date = df_processed["Date_Parsed"].mode()[0]
                df_processed["Date_Parsed"] = df_processed["Date_Parsed"].fillna(mode_date)

            df_processed["Month"] = df_processed["Date_Parsed"].dt.month
            df_processed["Year"] = df_processed["Date_Parsed"].dt.year
            df_processed = df_processed.drop(columns=["Date_Parsed"])

        # Separate targets
        target_raw = df_processed[self._data_config.target_column].copy()
        
        # Drop columns
        features_df = df_processed.drop(
            columns=[self._data_config.target_column] + self._data_config.drop_columns
        )

        # Separate numerical and categorical columns
        numeric_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = features_df.select_dtypes(exclude=["number"]).columns.tolist()

        # Impute numeric null values with median
        for col in numeric_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with median=%.4f",
                    null_count, col, median_val,
                )

        # Impute categorical null values with mode
        for col in categorical_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                mode_val = features_df[col].mode()[0]
                features_df[col] = features_df[col].fillna(mode_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with mode='%s'",
                    null_count, col, mode_val,
                )

        # Encode target classes (usually maps 0 -> 0, 1 -> 1 but ensures consistency)
        target_encoded = self._target_encoder.fit_transform(target_raw)
        self._label_names = [str(c) for c in self._target_encoder.classes_]
        self._logger.info("Target classes encoded: %s", self._label_names)

        # Encode categorical features
        # Each column gets an independent encoder instance
        self._logger.info("Encoding categorical features...")
        for col in categorical_cols:
            col_encoder = LabelEncoder()
            features_df[col] = col_encoder.fit_transform(features_df[col].astype(str))
            self._categorical_encoders[col] = col_encoder
            
            mapping = dict(
                zip(
                    col_encoder.classes_,
                    col_encoder.transform(col_encoder.classes_),
                )
            )
            self._logger.debug("  %-25s -> %s", col, mapping)

        self._feature_names = features_df.columns.tolist()
        self._logger.info("Final feature columns: %s", self._feature_names)

        # Standardize features
        # Scaled values are mandatory for stable convergence of solvers in LogisticRegression
        features_scaled = self._scaler.fit_transform(features_df.values)
        self._logger.info(
            "Feature standardization complete. Matrix shape: %s",
            features_scaled.shape,
        )

        return features_scaled, target_encoded

    def _split(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data with optional stratification.

        Args:
            features: Scaled feature matrix.
            target:   Encoded target vector.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test).
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
            "Train/Test split complete -- Train size: %d, Test size: %d (stratify=%s)",
            x_train.shape[0],
            x_test.shape[0],
            self._data_config.stratify,
        )

        return x_train, x_test, y_train, y_test
