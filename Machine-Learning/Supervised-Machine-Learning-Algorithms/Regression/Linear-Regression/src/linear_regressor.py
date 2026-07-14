# ============================================================================
# Linear Regressor Service for Regression Pipeline
# ============================================================================
# Manages the full model lifecycle: training scikit-learn LinearRegression,
# performing predictions on the test set, computing regression-specific metrics,
# generating visualizations (residuals and predictions), and documenting findings.
# ============================================================================

import logging
import os
import time
from typing import List

# Use Agg non-interactive backend to avoid issues on servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import ModelConfig, PathConfig


class LinearRegressorService:
    """Service encapsulating training, evaluation, and logging for Linear Regression.

    Responsibilities:
        1. Fit the OLS LinearRegression model on training features and targets.
        2. Evaluate performance using MAE, MSE, RMSE, and R2 metrics.
        3. Save performance metrics and model coefficients to disk.
        4. Plot prediction accuracy and residual distribution.
        5. Write a markdown explanation report analyzing the output.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the regressor service.

        Args:
            model_config:  Model hyperparameters (fit_intercept, n_jobs).
            path_config:   Path settings for output files.
            feature_names: Names of feature columns for coefficient mapping.
            logger:        Logger instance for pipelines.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._feature_names = feature_names
        self._logger = logger

        # Instantiate linear regression model
        self._model = LinearRegression(
            fit_intercept=self._model_config.fit_intercept,
            n_jobs=self._model_config.n_jobs,
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the linear regression model on training data.

        Args:
            x_train: Training feature matrix.
            y_train: Training continuous target vector.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._logger.info("Configured Hyperparameters:")
        self._logger.info("  fit_intercept : %s", self._model_config.fit_intercept)
        self._logger.info("  n_jobs        : %d", self._model_config.n_jobs)

        self._logger.info(
            "Fitting OLS model on %d samples with %d features...",
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Training completed successfully in %.4f seconds.", elapsed
        )

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Predict on the test set and calculate OLS evaluation metrics.

        Args:
            x_test: Test feature matrix.
            y_test: Test target vector.

        Returns:
            A dictionary containing MAE, MSE, RMSE, R2, and predicted values.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL EVALUATION")
        self._logger.info("=" * 70)

        start_time = time.perf_counter()
        predictions = self._model.predict(x_test)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Predictions on %d test samples completed in %.4f seconds.",
            x_test.shape[0],
            elapsed,
        )

        # Compute regression metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "predictions": predictions,
            "coefficients": self._model.coef_.tolist(),
            "intercept": float(self._model.intercept_),
        }

        # Log metrics and model architecture details
        self._log_results(results)
        self._save_results(results)
        self._generate_plots(results, y_test)
        self._save_analysis(results, y_test)

        return results

    def _log_results(self, results: dict) -> None:
        """Print results and model parameters to standard output.

        Args:
            results: Dictionary of computed results.
        """
        self._logger.info("Overall Performance Metrics:")
        self._logger.info("  Mean Absolute Error (MAE)    : %.4f", results["mae"])
        self._logger.info("  Mean Squared Error (MSE)     : %.4f", results["mse"])
        self._logger.info("  Root Mean Squared Error (RMSE): %.4f", results["rmse"])
        self._logger.info("  Coefficient of Determination (R2): %.4f", results["r2"])

        self._logger.info("-" * 70)
        self._logger.info("Model Equations Coefficients:")
        self._logger.info("  Intercept (bias)             : %.4f", results["intercept"])
        
        # Log coefficients for each feature
        self._logger.info("  Coefficients by Feature:")
        for name, coef in zip(self._feature_names, results["coefficients"]):
            self._logger.info("    %-25s : %+.4f", name, coef)
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Write model coefficients and results summary to a text file.

        Args:
            results: Results dictionary.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        filepath = os.path.join(self._path_config.output_dir, "regression_results.txt")

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("LINEAR REGRESSION PERFORMANCE REPORT\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("MODEL CONFIGURATION\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  fit_intercept : {self._model_config.fit_intercept}\n")
            fh.write(f"  n_jobs        : {self._model_config.n_jobs}\n\n")

            fh.write("EVALUATION METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Mean Absolute Error (MAE)     : {results['mae']:.4f}\n")
            fh.write(f"  Mean Squared Error (MSE)      : {results['mse']:.4f}\n")
            fh.write(f"  Root Mean Squared Error (RMSE): {results['rmse']:.4f}\n")
            fh.write(f"  R-Squared (R2 Score)          : {results['r2']:.4f}\n\n")

            fh.write("COEFFICIENTS AND BIAS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Intercept (bias term) : {results['intercept']:.4f}\n")
            for name, coef in zip(self._feature_names, results["coefficients"]):
                fh.write(f"  {name:<25} : {coef:+.4f}\n")
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Regression results saved to: %s", filepath)

    def _generate_plots(self, results: dict, y_test: np.ndarray) -> None:
        """Create and save residual plots and actual vs predicted sales charts.

        Args:
            results: Results dictionary.
            y_test:  True values.
        """
        self._logger.info("Generating evaluation plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        predictions = results["predictions"]
        residuals = y_test - predictions

        # Plot 1: Actual vs Predicted Sales
        # Why generate this plot?
        # A scatter plot of Actual vs Predicted sales visually demonstrates how closely
        # model predictions align with target reality. Perfect predictions fall
        # along the diagonal Y=X line.
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, y_test, alpha=0.5, color="teal", edgecolor="none")
        
        # Add diagonal reference line
        min_val = min(predictions.min(), y_test.min())
        max_val = max(predictions.max(), y_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", lw=2)
        
        plt.xlabel("Predicted Sales ($)")
        plt.ylabel("Actual Sales ($)")
        plt.title("Actual vs. Predicted Weekly Sales")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        pred_path = os.path.join(self._path_config.output_dir, "actual_vs_predicted.png")
        plt.savefig(pred_path, dpi=300)
        plt.close()
        self._logger.info("Actual vs Predicted plot saved to: %s", pred_path)

        # Plot 2: Residuals Plot
        # Why generate a residuals plot?
        # Residual analysis is a critical diagnostic for regression. It reveals whether
        # errors are independent with constant variance (homoscedasticity) or display
        # patterns suggesting non-linear structures or heteroscedasticity.
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, alpha=0.5, color="darkorange", edgecolor="none")
        plt.axhline(y=0, color="navy", linestyle="--", lw=2)
        plt.xlabel("Predicted Sales ($)")
        plt.ylabel("Residuals (Actual - Predicted) ($)")
        plt.title("Residuals vs. Predicted Values")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        res_path = os.path.join(self._path_config.output_dir, "residuals_plot.png")
        plt.savefig(res_path, dpi=300)
        plt.close()
        self._logger.info("Residuals plot saved to: %s", res_path)

    def _save_analysis(self, results: dict, y_test: np.ndarray) -> None:
        """Write a comprehensive analytical markdown report explaining the OLS output.

        Args:
            results: Results dictionary.
            y_test:  True target vector.
        """
        self._logger.info("Writing regression explanation report...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        report_path = os.path.join(self._path_config.output_dir, "regression_analysis.md")

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Linear Regression Analysis Report (Walmart Weekly Sales)\n\n")
            fh.write("## 1. Executive Summary\n")
            fh.write("This report presents a diagnostic analysis of the Ordinary Least Squares (OLS) Linear Regression model ")
            fh.write("built to predict Walmart Weekly Sales based on economic indicator features and seasonal variables. ")
            fh.write(f"The model explains **{results['r2'] * 100:.2f}%** of the variance in sales on the test set, ")
            fh.write(f"with a Root Mean Squared Error (RMSE) of **${results['rmse']:,.2f}** and a Mean Absolute Error (MAE) ")
            fh.write(f"of **${results['mae']:,.2f}**.\n\n")

            fh.write("## 2. Evaluation Metrics Interpretation\n")
            fh.write("Unlike classification, regression models produce continuous outputs, requiring error-magnitude metrics:\n\n")
            
            fh.write("| Metric | Value | Meaning |\n")
            fh.write("|--------|-------|---------|\n")
            fh.write(f"| **MAE** | ${results['mae']:,.2f} | On average, predictions deviate from actual sales by this dollar amount. |\n")
            fh.write(f"| **MSE** | {results['mse']:,.2f} | The average squared error; places a heavier penalty on larger sales errors. |\n")
            fh.write(f"| **RMSE** | ${results['rmse']:,.2f} | Standard deviation of the residuals; indicating the typical spread of predictions. |\n")
            fh.write(f"| **R-Squared ($R^2$)** | {results['r2']:.4f} | Proportion of variance in Weekly Sales explained by the input features. |\n\n")

            fh.write("## 3. Model Interpretation (Coefficients & Intercept)\n")
            fh.write(f"The model's base intercept term is **${results['intercept']:,.2f}** (which represents the expected weekly sales ")
            fh.write("when all standardized features are zero). Since all features were standardized prior to model training, ")
            fh.write("the coefficients indicate the direct relative strength and direction of each input feature on weekly sales:\n\n")

            fh.write("| Feature | Coefficient (Impact per 1 Std Dev) | Direction |\n")
            fh.write("|---------|-----------------------------|-----------|\n")
            for name, coef in zip(self._feature_names, results["coefficients"]):
                direction = "Positive" if coef >= 0 else "Negative"
                fh.write(f"| **{name}** | ${coef:,.2f} | {direction} |\n")
            fh.write("\n")

            fh.write("### Qualitative Feature Impact Analysis:\n")
            for name, coef in zip(self._feature_names, results["coefficients"]):
                if name == "Holiday_Flag":
                    fh.write(f"- **Holiday Flag**: Sales on holiday weeks are impact-shifted by **${coef:,.2f}** compared to non-holiday weeks. ")
                    fh.write("This aligns with retail trends showing elevated transaction volumes during major national holidays.\n")
                elif name == "Temperature":
                    fh.write(f"- **Temperature**: A standardized increase in temperature leads to an average change of **${coef:,.2f}** in sales. ")
                    fh.write("Extreme weather (hot or cold) can affect consumer foot-traffic patterns.\n")
                elif name == "Unemployment":
                    fh.write(f"- **Unemployment**: A standard deviation increase in local unemployment is associated with a **${coef:,.2f}** change in sales. ")
                    fh.write("Elevated unemployment typically reduces consumer disposable income and spending.\n")
                elif name == "CPI":
                    fh.write(f"- **Consumer Price Index (CPI)**: CPI changes are associated with a **${coef:,.2f}** shift in sales. ")
                    fh.write("This reflects macroeconomic conditions, such as inflation or purchasing power variation.\n")
                elif name == "Month":
                    fh.write(f"- **Month**: Seasonal effects (expressed numerically as month) result in a **${coef:,.2f}** shift in sales, ")
                    fh.write("capturing quarterly shopping habits.\n")

            fh.write("\n## 4. Residual Diagnostics\n")
            fh.write("The residuals plot (`residuals_plot.png`) evaluates model assumptions:\n")
            fh.write("- **Zero Mean Assumption**: The residuals should cluster around the horizontal $y=0$ reference line. ")
            fh.write("Any persistent upward or downward deviation would indicate prediction bias.\n")
            fh.write("- **Homoscedasticity vs. Heteroscedasticity**: Homoscedasticity means the error term has constant variance across all predicted values. ")
            fh.write("If the residuals scatter forms a 'funnel' shape (widening at higher sales), heteroscedasticity is present, ")
            fh.write("meaning the model makes larger errors on high-volume stores/weeks.\n")
            fh.write("- **Unmodeled Non-Linearity**: If the residuals show a curved (parabolic) pattern, it indicates ")
            fh.write("that the relationship between the features and sales is non-linear and would benefit from polynomial terms or non-linear models.\n")

        self._logger.info("Technical explanation report saved to: %s", report_path)
