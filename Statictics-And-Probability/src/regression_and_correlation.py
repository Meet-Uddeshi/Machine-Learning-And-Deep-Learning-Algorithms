# ============================================================================
# Correlation & Linear Regression Service Module
# ============================================================================
# Computes Pearson Correlation Coefficient r, fits Simple OLS Linear Regression
# (beta0 intercept, beta1 slope), evaluates Goodness-of-Fit R^2, and creates
# regression fitted line and residual diagnostic plots per Statistics infographic specs.
# ============================================================================

import logging
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig


class RegressionCorrelationService:
    """Service encapsulating correlation metrics, OLS linear regression, and diagnostic plots.

    Responsibilities:
        1. Calculate Pearson correlation coefficient r (-1 <= r <= 1).
        2. Fit Simple OLS Linear Regression model: y_hat = beta0 + beta1 * x.
        3. Compute SSE, SST, and R^2 Goodness of Fit coefficient.
        4. Plot regression fit line and residual distribution charts.
    """

    def __init__(self, path_config: PathConfig, logger: logging.Logger) -> None:
        """Initialize RegressionCorrelationService.

        Args:
            path_config: Output file path configurations.
            logger:      Logging instance.
        """
        self._path_config = path_config
        self._logger = logger

    def compute_pearson_correlation(
        self, x: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute Pearson Correlation Coefficient r.

        Formula:
            r = sum((x_i - x_bar)*(y_i - y_bar)) / sqrt( sum((x_i - x_bar)^2) * sum((y_i - y_bar)^2) )

        Args:
            x: Input feature vector.
            y: Target outcome vector.

        Returns:
            Pearson correlation value r (-1.0 to 1.0).
        """
        x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        x_mean, y_mean = np.mean(x_arr), np.mean(y_arr)

        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sqrt(
            np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2)
        )

        if denominator == 0:
            return 0.0

        r_val = float(numerator / denominator)
        self._logger.info("Pearson Correlation Coefficient r = %.4f", r_val)
        return r_val

    def fit_simple_linear_regression(
        self, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Fit OLS Simple Linear Regression model: y = beta0 + beta1 * x + epsilon.

        Estimators:
            beta1 = sum((x_i - x_bar)*(y_i - y_bar)) / sum((x_i - x_bar)^2)
            beta0 = y_bar - beta1 * x_bar
            R^2 = 1 - (SSE / SST)

        Args:
            x: Independent variable array.
            y: Dependent variable array.

        Returns:
            Dictionary with beta0, beta1, R2, SSE, SST, and predictions.
        """
        self._logger.info("Fitting Simple OLS Linear Regression model...")
        x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        x_mean, y_mean = np.mean(x_arr), np.mean(y_arr)

        beta1 = np.sum((x_arr - x_mean) * (y_arr - y_mean)) / np.sum(
            (x_arr - x_mean) ** 2
        )
        beta0 = y_mean - beta1 * x_mean

        predictions = beta0 + beta1 * x_arr
        residuals = y_arr - predictions

        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((y_arr - y_mean) ** 2))
        r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

        results = {
            "Intercept_Beta0": float(beta0),
            "Slope_Beta1": float(beta1),
            "R_Squared": r2,
            "SSE": sse,
            "SST": sst,
            "Predictions": predictions,
            "Residuals": residuals,
        }

        self._logger.info("Regression Parameters:")
        self._logger.info("  y_hat = %.4f + %.4f * x", beta0, beta1)
        self._logger.info("  R-Squared (R^2) = %.4f", r2)

        self._generate_regression_plots(x_arr, y_arr, predictions, residuals)

        return results

    def _generate_regression_plots(
        self,
        x: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        residuals: np.ndarray,
    ) -> str:
        """Create fitted regression line plot and residual scatter plot.

        Args:
            x:           Independent variable array.
            y:           Observed targets.
            predictions: Predicted y_hat values.
            residuals:   Error vector (y - y_hat).

        Returns:
            Saved image path.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Regression Fit Line
        axes[0].scatter(x, y, color="teal", alpha=0.5, label="Observed Data")
        # Sort for clean linear line display
        sort_idx = np.argsort(x)
        axes[0].plot(x[sort_idx], predictions[sort_idx], color="crimson", lw=2, label="OLS Line")
        axes[0].set_title("Simple Linear Regression Fit")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Plot 2: Residuals Scatter Plot
        axes[1].scatter(predictions, residuals, color="darkorange", alpha=0.5)
        axes[1].axhline(y=0, color="navy", linestyle="--", lw=2)
        axes[1].set_title("Residuals vs. Fitted Values")
        axes[1].set_xlabel("Fitted Values (y_hat)")
        axes[1].set_ylabel("Residuals (y - y_hat)")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        output_path = os.path.join(self._path_config.output_dir, "regression_analysis_plots.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        self._logger.info("Regression diagnostic plots saved to: %s", output_path)
        return output_path
