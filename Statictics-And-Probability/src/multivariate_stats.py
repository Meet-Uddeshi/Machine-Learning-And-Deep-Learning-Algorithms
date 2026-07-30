# ============================================================================
# Multivariate Statistics Service Module
# ============================================================================
# Implements Multiple Linear Regression analysis, covariance matrix evaluation,
# and multivariate correlation metrics per Statistics infographic specifications
# (excluding PCA and K-Means per user custom instructions).
# ============================================================================

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from config import DataConfig


class MultivariateStatsService:
    """Service encapsulating multivariate statistical modeling and covariance evaluation.

    Responsibilities:
        1. Fit Multiple OLS Linear Regression: y = X * beta + epsilon.
        2. Compute Multivariate Covariance Matrix (Sigma) and Correlation Matrix (R).
        3. Provide structural summaries for Factor Analysis and Discriminant Analysis.
    """

    def __init__(self, data_config: DataConfig, logger: logging.Logger) -> None:
        """Initialize MultivariateStatsService.

        Args:
            data_config: Dataset parameters.
            logger:      Logging instance.
        """
        self._config = data_config
        self._logger = logger

    def fit_multiple_linear_regression(
        self, x_matrix: np.ndarray, y_vector: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Union[float, List[float], Dict[str, float]]]:
        """Fit Multiple OLS Linear Regression using matrix inversion: beta = (X^T * X)^(-1) * X^T * y.

        Args:
            x_matrix:      Feature matrix of shape (n_samples, n_features).
            y_vector:      Target outcome vector of shape (n_samples,).
            feature_names: List of column labels corresponding to x_matrix.

        Returns:
            Dictionary containing coefficients map, intercept, and R^2 goodness of fit.
        """
        self._logger.info("Fitting Multiple Linear Regression on %d features...", x_matrix.shape[1])
        n = x_matrix.shape[0]

        # Add column of ones for intercept beta0
        x_design = np.hstack([np.ones((n, 1)), x_matrix])

        # Closed form OLS solution: beta = (X^T X)^(-1) X^T y
        beta = np.linalg.inv(x_design.T @ x_design) @ x_design.T @ y_vector

        intercept = float(beta[0])
        coefficients = beta[1:].tolist()

        predictions = x_design @ beta
        residuals = y_vector - predictions

        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((y_vector - np.mean(y_vector)) ** 2))
        r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

        coef_map = {name: float(c) for name, c in zip(feature_names, coefficients)}

        self._logger.info("Multiple Regression Results:")
        self._logger.info("  Intercept : %.4f", intercept)
        for name, coef in coef_map.items():
            self._logger.info("  %-15s : %+.4f", name, coef)
        self._logger.info("  R-Squared (R^2) : %.4f", r2)

        return {
            "Intercept": intercept,
            "Coefficients": coef_map,
            "R_Squared": r2,
            "SSE": sse,
            "SST": sst,
        }

    def compute_covariance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the covariance matrix for all continuous features.

        Args:
            df: DataFrame containing continuous variables.

        Returns:
            Symmetric Covariance Matrix DataFrame.
        """
        self._logger.info("Computing multivariate covariance matrix...")
        numeric_df = df.select_dtypes(include=[np.number])
        cov_matrix = numeric_df.cov()
        return cov_matrix

    def get_multivariate_techniques_summary() -> Dict[str, str]:
        """Provide analytical definitions for advanced multivariate techniques.

        Techniques:
            - Factor Analysis: Model observed variables in terms of fewer underlying unobserved factors.
            - Discriminant Analysis: Classify observations into groups based on continuous linear predictors.

        Returns:
            Dictionary mapping multivariate technique to its definition.
        """
        return {
            "Factor_Analysis": "Identifies unobserved latent factors that explain pattern correlations across observed variables.",
            "Discriminant_Analysis": "Finds linear combinations of features that best separate two or more categorical classes.",
        }
