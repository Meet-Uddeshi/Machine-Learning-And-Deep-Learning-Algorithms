# ============================================================================
# Data Loader and Preprocessing Service Module
# ============================================================================
# Responsible for loading retail transaction records from CSV/XLSX format,
# performing data cleaning (stripping strings, handling European price decimals,
# filtering cancelled orders), and constructing one-hot encoded transaction matrices
# for market basket analysis.
# ============================================================================

import logging
import os
from typing import List, Tuple

import pandas as pd

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service class for loading, cleaning, and formatting transaction data for Apriori mining."""

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize DataLoaderService.

        Args:
            path_config: Immutable path settings.
            data_config: Dataset loading and filtering parameters.
            logger:      Logger instance.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

    def load_and_prepare(self) -> Tuple[pd.DataFrame, List[set]]:
        """Load, clean, and convert transaction records into binary basket format.

        Returns:
            Tuple containing:
                - basket_df: One-hot encoded DataFrame (rows=BillNo, cols=Itemname, values=0 or 1).
                - transactions: List of item sets, where each set represents items in a transaction.

        Raises:
            FileNotFoundError: If the dataset file is not present at the configured path.
        """
        filepath = self._path_config.dataset_file
        if not os.path.exists(filepath):
            self._logger.error("Dataset file not found at: %s", filepath)
            raise FileNotFoundError(f"Dataset missing: {filepath}")

        self._logger.info("Loading transaction dataset from: %s", filepath)

        # Load CSV using configured delimiter
        df = pd.read_csv(filepath, sep=self._data_config.delimiter, low_memory=False)
        self._logger.info("Raw dataset shape: %s", df.shape)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Ensure required columns exist
        required_cols = ["BillNo", "Itemname", "Quantity", "Price"]
        for col in required_cols:
            if col not in df.columns:
                self._logger.error("Required column '%s' missing from dataset.", col)
                raise ValueError(f"Missing required column: {col}")

        # Drop records with missing BillNo or Itemname
        df = df.dropna(subset=["BillNo", "Itemname"]).copy()

        # Format Itemname: strip spaces and convert to uppercase
        df["Itemname"] = df["Itemname"].astype(str).str.strip().str.upper()

        # Filter out postage, bank charges, or generic non-product descriptions
        invalid_items = ["POSTAGE", "DOTCOM POSTAGE", "MANUAL", "BANK CHARGES", "AMAZON FEE"]
        df = df[~df["Itemname"].isin(invalid_items)]

        # Clean Price column (handle European decimal format with comma)
        if df["Price"].dtype == object:
            df["Price"] = (
                df["Price"].astype(str).str.replace(",", ".").astype(float)
            )

        # Ensure Quantity is numeric
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

        # Exclude cancelled transactions (BillNo starting with 'C') and non-positive quantities/prices
        df["BillNo_str"] = df["BillNo"].astype(str).str.strip()
        df = df[~df["BillNo_str"].str.startswith("C")]
        df = df[df["Quantity"] >= self._data_config.min_quantity]
        df = df[df["Price"] >= self._data_config.min_price]

        # Filter by Target Country if specified
        if self._data_config.target_country and "Country" in df.columns:
            target_c = self._data_config.target_country.strip().lower()
            df = df[df["Country"].astype(str).str.strip().str.lower() == target_c]
            self._logger.info(
                "Filtered transactions for country '%s': %d records remaining.",
                self._data_config.target_country,
                len(df),
            )

        # Subsample transactions if sample_size is specified
        if (
            self._data_config.sample_size
            and len(df["BillNo_str"].unique()) > self._data_config.sample_size
        ):
            unique_bills = pd.Series(df["BillNo_str"].unique()).sample(
                n=self._data_config.sample_size,
                random_state=self._data_config.random_state,
            )
            df = df[df["BillNo_str"].isin(unique_bills)]
            self._logger.info(
                "Subsampled dataset to %d unique transactions.",
                self._data_config.sample_size,
            )

        self._logger.info(
            "Cleaned dataset contains %d line items across %d unique transactions.",
            len(df),
            df["BillNo_str"].nunique(),
        )

        # Group items by transaction (BillNo) to build item set list
        grouped = df.groupby("BillNo_str")["Itemname"].apply(set)
        transactions = grouped.tolist()

        # Build one-hot encoded matrix for basket analysis
        self._logger.info("Constructing one-hot encoded transaction basket matrix...")
        basket_df = (
            df.groupby(["BillNo_str", "Itemname"])["Quantity"]
            .sum()
            .unstack()
            .fillna(0)
        )

        # Binarize quantities (1 if purchased, 0 otherwise)
        basket_df = (basket_df > 0).astype(int)
        self._logger.info("Basket matrix shape: %s", basket_df.shape)

        return basket_df, transactions
