# ============================================================================
# Data Loader Module for Market Basket Analysis Pipeline
# ============================================================================
# Handles transaction data I/O from Zip archive or CSV, schema validation,
# data cleaning, cancellation removal, and basket extraction.
# ============================================================================

import logging
import os
import zipfile
from typing import List, Set, Tuple

import pandas as pd

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service responsible for loading, cleaning, and preparing transaction baskets.

    Responsibilities:
        1. Read transaction dataset from Zip archive or raw CSV file.
        2. Validate required columns (BillNo, Itemname, Quantity, Price, Country).
        3. Filter out invalid rows, missing descriptions, and cancellation orders.
        4. Apply optional geographic filtering.
        5. Group line items by transaction (BillNo) to build unique item sets.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the data loader with configuration and logger.

        Args:
            path_config: Path settings for locating data files.
            data_config: Data filtering parameters.
            logger:      Logger instance for progress reporting.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

    # -- Public workflow methods ---------------------------------------------

    def load_and_prepare_baskets(self) -> Tuple[List[Set[str]], pd.DataFrame]:
        """Execute full data loading pipeline: read -> validate -> clean -> group.

        Returns:
            Tuple of (baskets_list, cleaned_dataframe).
            Where baskets_list is a list of sets containing item names per transaction.
        """
        dataframe = self._load_data()
        self._validate_schema(dataframe)
        cleaned_df = self._clean_data(dataframe)
        self._log_data_summary(cleaned_df)
        baskets = self._build_baskets(cleaned_df)
        return baskets, cleaned_df

    # -- Private implementation methods --------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """Read data from Zip file or standalone CSV file.

        Returns:
            Raw pandas DataFrame.
        """
        zip_path = self._path_config.zip_file
        csv_path = self._path_config.dataset_file

        if os.path.exists(csv_path):
            self._logger.info("Loading dataset directly from CSV: %s", csv_path)
            df = pd.read_csv(csv_path, sep=self._data_config.delimiter, low_memory=False)
        elif os.path.exists(zip_path):
            self._logger.info("Loading dataset from Zip archive: %s", zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(self._data_config.csv_filename) as f:
                    df = pd.read_csv(f, sep=self._data_config.delimiter, low_memory=False)
        else:
            raise FileNotFoundError(
                f"Neither dataset CSV ({csv_path}) nor Zip archive ({zip_path}) was found."
            )

        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        self._logger.info("Raw dataset loaded -- shape: %s", df.shape)
        return df

    def _validate_schema(self, dataframe: pd.DataFrame) -> None:
        """Ensure required transaction columns are present.

        Args:
            dataframe: Raw DataFrame to validate.
        """
        required_cols = ["BillNo", "Itemname", "Quantity", "Price", "Country"]
        missing = [col for col in required_cols if col not in dataframe.columns]
        if missing:
            raise KeyError(
                f"Missing required columns: {missing}. Available: {list(dataframe.columns)}"
            )

        self._logger.info("Schema validation passed.")

    def _clean_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Clean raw transactions: remove nulls, cancellations, and apply filters.

        Args:
            dataframe: Validated DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        self._logger.info("Cleaning transaction records...")
        df = dataframe.copy()

        # Drop missing BillNo or Itemname
        initial_count = len(df)
        df = df.dropna(subset=["BillNo", "Itemname"])
        
        # Clean text columns and parse numeric columns
        df["BillNo"] = df["BillNo"].astype(str).str.strip()
        df["Itemname"] = df["Itemname"].astype(str).str.strip().str.upper()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "."), errors="coerce")

        # Remove cancellations (BillNo starting with 'C'), nulls, and invalid quantities/prices
        df = df.dropna(subset=["Quantity", "Price"])
        df = df[~df["BillNo"].str.startswith("C")]
        df = df[df["Quantity"] >= self._data_config.min_quantity]
        df = df[df["Price"] >= self._data_config.min_price]

        # Apply country filter if specified
        if self._data_config.country_filter:
            self._logger.info(
                "Filtering transactions for country: '%s'", self._data_config.country_filter
            )
            df = df[df["Country"].astype(str).str.strip() == self._data_config.country_filter]

        self._logger.info(
            "Data cleaning complete -- retained %d / %d rows (%.1f%%)",
            len(df), initial_count, (len(df) / initial_count) * 100
        )
        return df

    def _log_data_summary(self, dataframe: pd.DataFrame) -> None:
        """Log summary statistics of cleaned transaction data.

        Args:
            dataframe: Cleaned DataFrame.
        """
        self._logger.info("=" * 70)
        self._logger.info("TRANSACTION DATASET SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info("Cleaned Rows     : %d", len(dataframe))
        self._logger.info("Unique Orders    : %d", dataframe["BillNo"].nunique())
        self._logger.info("Unique Products  : %d", dataframe["Itemname"].nunique())
        self._logger.info("Countries        : %s", list(dataframe["Country"].unique()))

        # Top 5 most purchased items by frequency
        top_items = dataframe["Itemname"].value_counts().head(5)
        self._logger.info("-" * 70)
        self._logger.info("TOP 5 LINE-ITEM PRODUCTS:")
        for item, count in top_items.items():
            self._logger.info("  %-45s : %d lines", item, count)
        self._logger.info("=" * 70)

    def _build_baskets(self, dataframe: pd.DataFrame) -> List[Set[str]]:
        """Group transaction lines into item set baskets per unique BillNo.

        Args:
            dataframe: Cleaned DataFrame.

        Returns:
            List of item sets representing individual transaction baskets.
        """
        self._logger.info("Grouping transaction lines into customer baskets...")

        grouped = dataframe.groupby("BillNo")["Itemname"].apply(lambda x: set(x.unique()))
        baskets = [basket for basket in grouped if len(basket) > 0]

        # Limit total baskets if sampling is configured
        if self._data_config.sample_baskets and self._data_config.sample_baskets < len(baskets):
            self._logger.info(
                "Limiting processing to top %d baskets for efficiency.",
                self._data_config.sample_baskets
            )
            baskets = baskets[: self._data_config.sample_baskets]

        self._logger.info("Extracted %d transaction baskets.", len(baskets))
        return baskets
