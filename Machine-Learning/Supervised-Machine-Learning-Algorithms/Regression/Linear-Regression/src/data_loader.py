# ============================================================================
# Data Loader Module for Linear Regression Pipeline
# ============================================================================
# Handles all data loading, schema validation, feature engineering (extracting
# Month and Year from Date), feature scaling, and train-test splitting.
# Follows the Service Layer pattern to isolate data preprocessing from business logic.
# ============================================================================

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service responsible for loading, validating, and preprocessing the Walmart Sales dataset.

    Responsibilities:
        1. Ingest the dataset from CSV and check schema consistency.
        2. Perform feature engineering on date columns to extract Month and Year.
        3. Impute missing values (median for numeric features).
        4. Standardize numerical feature columns to zero mean and unit variance.
        5. Split the data into train and test splits for validation.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the data loader service.

        Args:
            path_config: Path configuration for locating the dataset.
            data_config: Data preprocessing parameters and column rules.
            logger:      Logger instance to report preparation progress.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

        # Feature columns and scaling objects
        self._feature_names: list = []
        self._scaler: StandardScaler = StandardScaler()

    @property
    def feature_names(self) -> list:
        """Column names of the final feature matrix."""
        return self._feature_names

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full data preparation pipeline.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test) as numpy arrays.

        Raises:
            FileNotFoundError: If the raw dataset cannot be found.
            KeyError:         If expected columns are missing from the schema.
            ValueError:        If the loaded dataset is empty.
        """
        dataframe = self._load_csv()
        self._validate_schema(dataframe)
        dataframe = self._downsample(dataframe)
        
        # Log summary stats before transformations
        self._log_data_summary(dataframe)
        
        # Preprocess: feature engineering + imputation + scaling
        features, target = self._preprocess(dataframe)
        
        return self._split(features, target)

    def _load_csv(self) -> pd.DataFrame:
        """Read the CSV file and handle basic file verification.

        Returns:
            The raw pandas DataFrame.
        """
        self._logger.info(
            "Loading Walmart Sales dataset from: %s", self._path_config.dataset_file
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
        """Ensure the target column and drop columns are present in the dataframe.

        Args:
            dataframe: The raw dataframe.
        """
        if self._data_config.target_column not in dataframe.columns:
            raise KeyError(
                f"Target column '{self._data_config.target_column}' "
                f"not found in dataset. Available columns: {list(dataframe.columns)}"
            )

        # Confirm all drop columns (other than 'Date' which is engineered) are present
        for col in self._data_config.drop_columns:
            if col not in dataframe.columns:
                raise KeyError(
                    f"Configured drop column '{col}' is missing from the dataset."
                )

        self._logger.info("Schema validation passed.")

    def _downsample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Downsample the dataset if max_samples is configured.

        Args:
            dataframe: Input dataframe.

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
        return dataframe.sample(
            n=max_samples,
            random_state=self._data_config.random_state
        )

    def _log_data_summary(self, dataframe: pd.DataFrame) -> None:
        """Log a summary of the dataset for debugging and auditing.

        Args:
            dataframe: Input dataframe.
        """
        self._logger.info("=" * 70)
        self._logger.info("DATASET SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info("Rows: %d  |  Columns: %d", *dataframe.shape)

        self._logger.info("-" * 70)
        self._logger.info("%-25s  %-15s  %s", "Column", "Dtype", "Nulls")
        self._logger.info("-" * 70)
        for col in dataframe.columns:
            null_count = dataframe[col].isna().sum()
            self._logger.info(
                "%-25s  %-15s  %d", col, str(dataframe[col].dtype), null_count
            )
        
        # Describe target column
        target = self._data_config.target_column
        desc = dataframe[target].describe()
        self._logger.info("-" * 70)
        self._logger.info("TARGET COLUMN METRICS ('%s'):", target)
        self._logger.info("  Mean  : %.2f", desc["mean"])
        self._logger.info("  Min   : %.2f", desc["min"])
        self._logger.info("  Max   : %.2f", desc["max"])
        self._logger.info("  StdDev: %.2f", desc["std"])
        self._logger.info("=" * 70)

    def _preprocess(
        self, dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature engineering, imputation, and standardization.

        Args:
            dataframe: Raw DataFrame.

        Returns:
            A tuple of feature matrix (scaled) and target vector (unscaled).
        """
        self._logger.info("Beginning data preprocessing pipeline...")
        df_processed = dataframe.copy()

        # Step 1: Feature Engineering on Date
        # Why extract Month and Year?
        # Linear Regression requires numeric features. Dates are highly cyclical
        # and seasonal (e.g. holiday peaks). Converting date strings to numerical
        # month and year preserves this seasonal pattern for the regressor.
        if "Date" in df_processed.columns:
            self._logger.info("Parsing 'Date' column and extracting Month & Year features...")
            df_processed["Date_Parsed"] = pd.to_datetime(
                df_processed["Date"], format="%d-%m-%Y", errors="coerce"
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

        # Separate features and target
        target = df_processed[self._data_config.target_column].values
        
        # Drop columns specified in data config
        features_df = df_processed.drop(
            columns=[self._data_config.target_column] + self._data_config.drop_columns
        )

        self._logger.info(
            "Dropped target column '%s' and drop columns: %s",
            self._data_config.target_column,
            self._data_config.drop_columns,
        )

        # Step 2: Handle missing values via median imputation
        for col in features_df.columns:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
                self._logger.info(
                    "Imputed %d nulls in feature '%s' with median=%.4f",
                    null_count, col, median_val,
                )

        self._feature_names = features_df.columns.tolist()
        self._logger.info("Final feature columns: %s", self._feature_names)

        # Step 3: Feature Standardization
        # Why scale features?
        # Ordinary Least Squares computes coefficients to minimize squared errors.
        # Scaling variables to similar magnitudes ensures that coefficients are stable
        # and directly comparable, improving numerical performance.
        features_scaled = self._scaler.fit_transform(features_df.values)
        self._logger.info(
            "Feature standardization complete (mean~0, std~1). Shape: %s",
            features_scaled.shape,
        )

        return features_scaled, target

    def _split(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform train-test split on features and targets.

        Args:
            features: Scaled feature matrix.
            target:   Unscaled target vector.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test).
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self._data_config.test_size,
            random_state=self._data_config.random_state,
        )

        self._logger.info(
            "Train/Test split complete -- Train size: %d, Test size: %d (ratio: %.0f%%)",
            x_train.shape[0],
            x_test.shape[0],
            self._data_config.test_size * 100,
        )

        return x_train, x_test, y_train, y_test
