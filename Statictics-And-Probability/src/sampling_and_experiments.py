# ============================================================================
# Sampling Methods and Experimental Design Service Module
# ============================================================================
# Implements statistical sampling strategies (Simple Random, Systematic,
# Stratified, Cluster) and outlines experimental design structures (Completely
# Randomized, Randomized Block, Factorial) per Statistics infographic specifications.
# ============================================================================

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class SamplingAndDesignService:
    """Service encapsulating statistical sampling methods and experimental design setup.

    Responsibilities:
        1. Simple Random Sampling: Select n random units with equal probability.
        2. Systematic Sampling: Select every k-th element starting from random seed.
        3. Stratified Sampling: Sample proportionally across category strata.
        4. Cluster Sampling: Randomly choose full clusters and sample all elements within.
        5. Define experimental design allocations.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize SamplingAndDesignService.

        Args:
            logger: Logging instance.
        """
        self._logger = logger

    def simple_random_sample(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Perform Simple Random Sampling without replacement.

        Args:
            df:        Full population dataset.
            n_samples: Number of observations to draw.

        Returns:
            Sampled DataFrame subset.
        """
        self._logger.info("Performing Simple Random Sampling (n=%d)...", n_samples)
        if n_samples > len(df):
            raise ValueError("Sample size cannot exceed population size.")
        return df.sample(n=n_samples, replace=False, random_state=42)

    def systematic_sample(self, df: pd.DataFrame, step_k: int) -> pd.DataFrame:
        """Perform Systematic Sampling by taking every k-th row.

        Args:
            df:     Full population dataset.
            step_k: Step interval k (e.g. k=5).

        Returns:
            Sampled DataFrame subset.
        """
        self._logger.info("Performing Systematic Sampling (step_k=%d)...", step_k)
        if step_k <= 0:
            raise ValueError("Step interval k must be positive.")

        indices = np.arange(0, len(df), step_k)
        return df.iloc[indices].copy()

    def stratified_sample(
        self, df: pd.DataFrame, strata_column: str, frac: float = 0.2
    ) -> pd.DataFrame:
        """Perform Stratified Sampling proportionally across strata groups.

        Args:
            df:            Full population dataset.
            strata_column: Column name defining subgroups/strata.
            frac:          Fraction of samples to draw from each stratum.

        Returns:
            Sampled DataFrame subset.
        """
        self._logger.info(
            "Performing Stratified Sampling on column '%s' (frac=%.2f)...",
            strata_column,
            frac,
        )
        if strata_column not in df.columns:
            raise KeyError(f"Strata column '{strata_column}' not found in DataFrame.")

        sampled_df = (
            df.groupby(strata_column, group_keys=False)
            .apply(lambda x: x.sample(frac=frac, random_state=42))
            .copy()
        )
        return sampled_df

    def cluster_sample(
        self, df: pd.DataFrame, cluster_column: str, num_clusters_to_select: int = 1
    ) -> pd.DataFrame:
        """Perform Cluster Sampling by selecting entire random clusters.

        Args:
            df:                      Full population dataset.
            cluster_column:          Column name identifying cluster IDs.
            num_clusters_to_select:  Number of clusters to randomly pick.

        Returns:
            Sampled DataFrame containing all observations in chosen clusters.
        """
        self._logger.info(
            "Performing Cluster Sampling on '%s' (selecting %d clusters)...",
            cluster_column,
            num_clusters_to_select,
        )
        unique_clusters = df[cluster_column].unique()
        if num_clusters_to_select > len(unique_clusters):
            raise ValueError("Number of clusters to select exceeds available clusters.")

        chosen_clusters = np.random.choice(
            unique_clusters, size=num_clusters_to_select, replace=False
        )
        return df[df[cluster_column].isin(chosen_clusters)].copy()

    def get_experimental_designs_summary() -> Dict[str, str]:
        """Return structural definitions for standard experimental design models.

        Designs:
            - Completely Randomized Design (CRD): Treatments assigned completely at random.
            - Randomized Block Design (RBD): Units blocked by nuisance variable before randomizing.
            - Factorial Design: Evaluates main effects and interactions across multiple factors.
            - Latin Square Design: Controls two orthogonal nuisance sources of variation.
            - Split-Plot Design: Factorial structure with different randomization units.

        Returns:
            Dictionary mapping design names to their descriptions.
        """
        return {
            "Completely_Randomized_Design": "Treatments assigned completely at random across all homogeneous experimental units.",
            "Randomized_Block_Design": "Experimental units grouped into homogeneous blocks to control nuisance variability.",
            "Factorial_Design": "Tests all possible combinations of factor levels simultaneously to measure interaction effects.",
            "Latin_Square_Design": "Two-way blocking structure controlling for row and column confounding sources.",
            "Split_Plot_Design": "Used when factor level changes require large experimental units embedded inside smaller plots.",
        }
