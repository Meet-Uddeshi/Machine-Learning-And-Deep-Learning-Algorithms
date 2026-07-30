# ============================================================================
# Descriptive Statistics Service Module
# ============================================================================
# Computes central tendency metrics (mean, median, mode) and dispersion metrics
# (range, sample variance, standard deviation, IQR), and generates visual charts
# (histograms, box plots, bar charts, pie charts) per Statistics infographic specifications.
# ============================================================================

import logging
import os
from typing import Dict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import PathConfig


class DescriptiveStatisticsService:
    """Service encapsulating descriptive statistics calculations and visual summaries.

    Responsibilities:
        1. Calculate measures of central tendency: Mean, Median, Mode.
        2. Calculate measures of dispersion: Range, Sample Variance, Standard Deviation, IQR.
        3. Generate visual summary figures (Histogram, Boxplot, Bar Chart, Pie Chart).
    """

    def __init__(self, path_config: PathConfig, logger: logging.Logger) -> None:
        """Initialize DescriptiveStatisticsService.

        Args:
            path_config: Path configurations for saving output figures.
            logger:      Logging instance for diagnostics and outputs.
        """
        self._path_config = path_config
        self._logger = logger

    def compute_descriptive_stats(
        self, data: Union[np.ndarray, pd.Series], name: str = "Variable"
    ) -> Dict[str, float]:
        """Calculate complete central tendency and dispersion metrics for continuous data.

        Formulae:
            - Mean: x_bar = sum(x_i) / n
            - Variance: s^2 = sum((x_i - x_bar)^2) / (n - 1)
            - Std Dev: s = sqrt(s^2)
            - IQR: Q3 - Q1

        Args:
            data: Numeric array or pandas series.
            name: Label identifier for the dataset column.

        Returns:
            Dictionary containing computed statistical values.
        """
        self._logger.info("Computing descriptive statistics for %s...", name)
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]

        n = len(arr)
        if n == 0:
            raise ValueError("Input data array is empty or contains only NaNs.")

        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        mode_res = stats.mode(arr, keepdims=True)
        mode_val = float(mode_res.mode[0])

        range_val = float(np.max(arr) - np.min(arr))
        var_sample = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        std_sample = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr_val = q3 - q1

        results = {
            "Count": n,
            "Mean": mean_val,
            "Median": median_val,
            "Mode": mode_val,
            "Range": range_val,
            "Variance": var_sample,
            "Std_Dev": std_sample,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr_val,
        }

        self._logger.info("Descriptive Statistics for %s:", name)
        for key, val in results.items():
            self._logger.info("  %-15s : %.4f", key, val)

        return results

    def generate_visual_summaries(
        self, df: pd.DataFrame, continuous_col: str, categorical_col: str
    ) -> str:
        """Create a 2x2 multi-panel figure displaying visual summaries.

        Plots included:
            - Top-Left: Histogram with KDE curve
            - Top-Right: Boxplot of continuous variable
            - Bottom-Left: Bar chart of categorical distribution
            - Bottom-Right: Pie chart of categorical proportions

        Args:
            df: DataFrame containing the data columns.
            continuous_col: Name of continuous feature for Histogram & Boxplot.
            categorical_col: Name of categorical feature for Bar Chart & Pie Chart.

        Returns:
            Filepath to the saved plot image.
        """
        self._logger.info("Generating visual summaries panel...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Histogram
        axes[0, 0].hist(
            df[continuous_col], bins=20, color="navy", alpha=0.7, edgecolor="white"
        )
        axes[0, 0].set_title(f"Histogram of {continuous_col}")
        axes[0, 0].set_xlabel(continuous_col)
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, linestyle=":", alpha=0.6)

        # Plot 2: Boxplot
        axes[0, 1].boxplot(
            df[continuous_col], patch_artist=True, boxprops=dict(facecolor="teal")
        )
        axes[0, 1].set_title(f"Box Plot of {continuous_col}")
        axes[0, 1].set_ylabel(continuous_col)
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)

        # Plot 3: Bar Chart
        cat_counts = df[categorical_col].value_counts()
        axes[1, 0].bar(
            cat_counts.index, cat_counts.values, color="coral", edgecolor="black"
        )
        axes[1, 0].set_title(f"Bar Chart of {categorical_col}")
        axes[1, 0].set_xlabel(categorical_col)
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, linestyle=":", alpha=0.6)

        # Plot 4: Pie Chart
        axes[1, 1].pie(
            cat_counts.values,
            labels=cat_counts.index,
            autopct="%1.1f%%",
            colors=["#66b3ff", "#99ff99", "#ffcc99"],
            startangle=140,
        )
        axes[1, 1].set_title(f"Pie Chart of {categorical_col}")

        plt.tight_layout()
        output_path = os.path.join(self._path_config.output_dir, "descriptive_summaries.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        self._logger.info("Visual summaries saved to: %s", output_path)
        return output_path
