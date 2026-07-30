# ============================================================================
# Time Series Analysis Service Module
# ============================================================================
# Implements time series decomposition (Trend, Seasonality, Residuals) under Additive
# and Multiplicative models using pure pandas/numpy moving averages (avoiding external
# statsmodels dependencies), Simple Moving Averages (SMA), Exponential Smoothing (SES),
# and generates time series diagnostic plots per Statistics infographic specifications.
# ============================================================================

import logging
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PathConfig


class TimeSeriesService:
    """Service encapsulating time series decomposition and smoothing techniques.

    Responsibilities:
        1. Decompose time series into Trend (T_t), Seasonality (S_t), and Irregular (I_t) components.
        2. Support both Additive (Y = T + S + I) and Multiplicative (Y = T * S * I) models.
        3. Compute Simple Moving Average (SMA) smoothing.
        4. Compute Simple Exponential Smoothing (SES).
        5. Generate diagnostic multi-panel decomposition plots.
    """

    def __init__(self, path_config: PathConfig, logger: logging.Logger) -> None:
        """Initialize TimeSeriesService.

        Args:
            path_config: Output file path configurations.
            logger:      Logging instance.
        """
        self._path_config = path_config
        self._logger = logger

    def compute_moving_average(
        self, series: pd.Series, window: int = 7
    ) -> pd.Series:
        """Calculate Simple Moving Average (SMA) for time series smoothing.

        Args:
            series: Input time-ordered pandas series.
            window: Rolling window size.

        Returns:
            Smoothed pandas Series.
        """
        self._logger.info("Computing Simple Moving Average (window=%d)...", window)
        return series.rolling(window=window, min_periods=1).mean()

    def compute_exponential_smoothing(
        self, series: pd.Series, alpha: float = 0.3
    ) -> pd.Series:
        """Calculate Simple Exponential Smoothing (SES).

        Formula:
            S_t = alpha * Y_t + (1 - alpha) * S_{t-1}

        Args:
            series: Input time-ordered pandas series.
            alpha:  Smoothing factor (0 < alpha <= 1).

        Returns:
            Exponentially smoothed pandas Series.
        """
        self._logger.info("Computing Simple Exponential Smoothing (alpha=%.2f)...", alpha)
        return series.ewm(alpha=alpha, adjust=False).mean()

    def decompose_time_series(
        self, series: pd.Series, model: str = "additive", period: int = 3
    ) -> Dict[str, pd.Series]:
        """Decompose time series into Trend, Seasonality, and Residual components using pandas.

        Formulae:
            - Trend (T_t): Centered rolling mean with window = period.
            - Detrended:
                - Additive: Y_t - T_t
                - Multiplicative: Y_t / T_t
            - Seasonal (S_t): Average detrended values per period phase.
            - Residual (I_t):
                - Additive: Y_t - T_t - S_t
                - Multiplicative: Y_t / (T_t * S_t)

        Args:
            series: Time series indexed by datetime or consecutive integer.
            model:  'additive' or 'multiplicative'.
            period: Seasonal period frequency.

        Returns:
            Dictionary containing trend, seasonal, resid components and plot output path.
        """
        self._logger.info(
            "Decomposing time series (%s model, period=%d)...", model, period
        )
        values = series.astype(float).values
        n = len(values)

        # Step 1: Trend Component (Rolling Mean)
        trend_series = series.rolling(window=period, center=True, min_periods=1).mean()
        trend_vals = trend_series.values

        # Step 2: Detrended Component
        if model.lower() == "multiplicative":
            # Avoid division by zero
            safe_trend = np.where(trend_vals == 0, 1e-6, trend_vals)
            detrended = values / safe_trend
        else:
            detrended = values - trend_vals

        # Step 3: Seasonal Component (Phase Means)
        period_indices = np.arange(n) % period
        seasonal_means = pd.Series(detrended).groupby(period_indices).transform("mean").values

        if model.lower() == "multiplicative":
            # Normalize seasonal means to average 1.0
            seasonal_means = seasonal_means / (np.mean(seasonal_means) if np.mean(seasonal_means) != 0 else 1.0)
            residual_vals = values / (safe_trend * seasonal_means)
        else:
            # Center seasonal means around zero
            seasonal_means = seasonal_means - np.mean(seasonal_means)
            residual_vals = values - trend_vals - seasonal_means

        trend_res = pd.Series(trend_vals, index=series.index)
        seasonal_res = pd.Series(seasonal_means, index=series.index)
        resid_res = pd.Series(residual_vals, index=series.index)

        output_plot = self._plot_decomposition(
            series, trend_res, seasonal_res, resid_res, model
        )

        return {
            "Trend": trend_res,
            "Seasonal": seasonal_res,
            "Residual": resid_res,
            "Plot_Path": output_plot,
        }

    def _plot_decomposition(
        self,
        observed: pd.Series,
        trend: pd.Series,
        seasonal: pd.Series,
        resid: pd.Series,
        model: str,
    ) -> str:
        """Plot time series components in a 4-row layout.

        Args:
            observed: Raw time series data.
            trend:    Extracted trend component.
            seasonal: Periodic seasonal component.
            resid:    Irregular noise / residual component.
            model:    Model type string.

        Returns:
            Filepath to the saved plot image.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(observed, color="navy", label="Observed Y_t")
        axes[0].set_title(f"Time Series Decomposition ({model.capitalize()} Model)")
        axes[0].set_ylabel("Observed")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].plot(trend, color="crimson", label="Trend T_t")
        axes[1].set_ylabel("Trend")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        axes[2].plot(seasonal, color="teal", label="Seasonal S_t")
        axes[2].set_ylabel("Seasonal")
        axes[2].grid(True, linestyle=":", alpha=0.6)

        axes[3].plot(resid, color="darkorange", label="Residual I_t")
        axes[3].set_ylabel("Residual")
        axes[3].grid(True, linestyle=":", alpha=0.6)
        axes[3].set_xlabel("Time Index / Date")

        plt.tight_layout()
        output_path = os.path.join(
            self._path_config.output_dir, f"time_series_decomposition_{model}.png"
        )
        plt.savefig(output_path, dpi=300)
        plt.close()

        self._logger.info("Time series plot saved to: %s", output_path)
        return output_path
