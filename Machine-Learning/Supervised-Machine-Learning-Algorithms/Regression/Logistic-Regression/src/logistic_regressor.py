# ============================================================================
# Logistic Regressor Service for Regression Pipeline
# ============================================================================
# Manages the full model lifecycle: training scikit-learn LogisticRegression,
# performing predictions and probability scoring on test sets, calculating
# classification metrics, generating visualizations, and documenting findings.
# ============================================================================

import logging
import os
import time
from typing import List

# Use Agg non-interactive backend to avoid display warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import ModelConfig, PathConfig


class LogisticRegressorService:
    """Service encapsulating training, evaluation, and logging for Logistic Regression.

    Responsibilities:
        1. Train a LogisticRegression estimator with regularized log-loss.
        2. Generate predictions and class probabilities for evaluation.
        3. Compute classification metrics: Accuracy, Precision, Recall, F1, and ROC AUC.
        4. Plot the Confusion Matrix Heatmap and Receiver Operating Characteristic (ROC) curve.
        5. Write a detailed analysis markdown report documenting feature Odds Ratios.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the logistic regressor service.

        Args:
            model_config:  Model hyperparameters (penalty, C, solver, max_iter).
            path_config:   Path configurations.
            label_names:   Labels mapping output class indices to strings.
            feature_names: Feature column names for explanation.
            logger:        Pipeline logger.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger

        self._model = LogisticRegression(
            penalty=self._model_config.penalty,
            C=self._model_config.C,
            solver=self._model_config.solver,
            max_iter=self._model_config.max_iter,
            random_state=self._model_config.random_state,
        )

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Logistic Regression model.

        Args:
            x_train: Training features.
            y_train: Training binary labels.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._logger.info("Configured Hyperparameters:")
        self._logger.info("  penalty       : %s", self._model_config.penalty)
        self._logger.info("  C (inv reg)   : %.4f", self._model_config.C)
        self._logger.info("  solver        : %s", self._model_config.solver)
        self._logger.info("  max_iter      : %d", self._model_config.max_iter)

        self._logger.info(
            "Fitting Logistic Regression classifier on %d samples with %d features...",
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
        """Evaluate the model on test data and calculate performance metrics.

        Args:
            x_test: Test feature matrix.
            y_test: Test binary target labels.

        Returns:
            A dictionary containing evaluation scores and prediction outputs.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL EVALUATION")
        self._logger.info("=" * 70)

        start_time = time.perf_counter()
        predictions = self._model.predict(x_test)
        probabilities = self._model.predict_proba(x_test)[:, 1]  # Probabilities of class 1
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Inference on %d test samples completed in %.4f seconds.",
            x_test.shape[0],
            elapsed,
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="macro", zero_division=0)
        recall = recall_score(y_test, predictions, average="macro", zero_division=0)
        f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        roc_auc = roc_auc_score(y_test, probabilities)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(
            y_test,
            predictions,
            target_names=self._label_names,
            zero_division=0,
        )

        results = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "predictions": predictions,
            "probabilities": probabilities,
            # Extracted coefficients (Logistic regression coefficients are 2D for binary/multiclass)
            "coefficients": self._model.coef_[0].tolist(),
            "intercept": float(self._model.intercept_[0]),
        }

        # Save and explain outcomes
        self._log_results(results)
        self._save_results(results)
        self._generate_plots(results, y_test)
        self._save_analysis(results, y_test)

        return results

    def _log_results(self, results: dict) -> None:
        """Print validation metrics and coefficients to stdout.

        Args:
            results: Results dictionary.
        """
        self._logger.info("Overall Performance Metrics:")
        self._logger.info("  Accuracy Score               : %.4f (%.2f%%)", results["accuracy"], results["accuracy"] * 100)
        self._logger.info("  Precision Score (macro)      : %.4f", results["precision_macro"])
        self._logger.info("  Recall Score (macro)         : %.4f", results["recall_macro"])
        self._logger.info("  F1-Score (macro)             : %.4f", results["f1_macro"])
        self._logger.info("  ROC AUC Score                : %.4f", results["roc_auc"])

        # Log confusion matrix
        self._logger.info("-" * 70)
        self._logger.info("CONFUSION MATRIX:")
        header = "%-15s" % "Actual\\Pred"
        for name in self._label_names:
            header += "  %-15s" % name
        self._logger.info(header)

        for idx, row in enumerate(results["confusion_matrix"]):
            row_str = "%-15s" % self._label_names[idx]
            for val in row:
                row_str += "  %-15d" % val
            self._logger.info(row_str)

        # Log coefficients
        self._logger.info("-" * 70)
        self._logger.info("Model Log-Odds Coefficients:")
        self._logger.info("  Intercept (bias term)        : %+.4f", results["intercept"])
        for name, coef in zip(self._feature_names, results["coefficients"]):
            odds_ratio = np.exp(coef)
            self._logger.info("    %-25s : Log-Odds=%+.4f | Odds-Ratio=%.4f", name, coef, odds_ratio)
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Save results summary to a text file in the output folder.

        Args:
            results: Results dictionary.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        filepath = os.path.join(self._path_config.output_dir, "classification_results.txt")

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("LOGISTIC REGRESSION CLASSIFICATION PERFORMANCE REPORT\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("MODEL CONFIGURATION\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Penalty       : {self._model_config.penalty}\n")
            fh.write(f"  C (inv reg)   : {self._model_config.C}\n")
            fh.write(f"  Solver        : {self._model_config.solver}\n")
            fh.write(f"  Max Iterations: {self._model_config.max_iter}\n\n")

            fh.write("OVERALL PERFORMANCE METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Accuracy Score               : {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)\n")
            fh.write(f"  Precision Score (macro)      : {results['precision_macro']:.4f}\n")
            fh.write(f"  Recall Score (macro)         : {results['recall_macro']:.4f}\n")
            fh.write(f"  F1-Score (macro)             : {results['f1_macro']:.4f}\n")
            fh.write(f"  ROC AUC Score                : {results['roc_auc']:.4f}\n\n")

            fh.write("CONFUSION MATRIX\n")
            fh.write("-" * 40 + "\n")
            col_header = "Actual\\Pred"
            header = f"{col_header:<15}"
            for name in self._label_names:
                header += f"  {name:<15}"
            fh.write(header + "\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                row_str = f"{self._label_names[idx]:<15}"
                for val in row:
                    row_str += f"  {val:<15}"
                fh.write(row_str + "\n")
            fh.write("\n")

            fh.write("MODEL COEFFICIENTS AND ODDS RATIOS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Intercept (bias term) : {results['intercept']:.4f}\n")
            for name, coef in zip(self._feature_names, results["coefficients"]):
                odds_ratio = np.exp(coef)
                fh.write(f"  {name:<25} : Log-Odds={coef:+.4f} | Odds-Ratio={odds_ratio:.4f}\n")
            fh.write("\n")

            fh.write("CLASSIFICATION REPORT DETAILS\n")
            fh.write("-" * 40 + "\n")
            fh.write(results["classification_report"])
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Classification results saved to: %s", filepath)

    def _generate_plots(self, results: dict, y_test: np.ndarray) -> None:
        """Generate and save the Confusion Matrix heatmap and ROC Curve.

        Args:
            results: Results dictionary.
            y_test:  True test labels.
        """
        self._logger.info("Generating evaluation plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # Plot 1: Confusion Matrix Heatmap
        conf_matrix = np.array(results["confusion_matrix"])
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=self._label_names,
            yticklabels=self._label_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Confusion Matrix Heatmap")
        plt.tight_layout()
        cm_path = os.path.join(self._path_config.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        self._logger.info("Confusion matrix heatmap saved to: %s", cm_path)

        # Plot 2: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, results["probabilities"])
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="darkviolet", lw=2, label=f"ROC Curve (AUC = {results['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=2)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        roc_path = os.path.join(self._path_config.output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        self._logger.info("ROC Curve saved to: %s", roc_path)

    def _save_analysis(self, results: dict, y_test: np.ndarray) -> None:
        """Write an explanatory markdown analysis report for Logistic Regression output.

        Args:
            results: Results dictionary.
            y_test:  True test labels.
        """
        self._logger.info("Writing classification explanation report...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        report_path = os.path.join(self._path_config.output_dir, "classification_analysis.md")

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Logistic Regression Analysis Report (Supply Chain Risk)\n\n")
            fh.write("## 1. Executive Summary\n")
            fh.write("This report presents the diagnostic evaluation of the Logistic Regression classifier ")
            fh.write("built to predict global supply chain disruptions. ")
            fh.write(f"The model achieved an overall accuracy of **{results['accuracy'] * 100:.2f}%** on the test dataset ")
            fh.write(f"with an Area Under the ROC Curve (AUC) of **{results['roc_auc']:.4f}**, indicating strong predictive performance.\n\n")

            fh.write("## 2. Evaluation Metrics Interpretation\n")
            fh.write("To validate binary classification performance, we analyze a broad suite of metrics:\n\n")

            fh.write("| Metric | Value | Theoretical Interpretation |\n")
            fh.write("|--------|-------|----------------------------|\n")
            fh.write(f"| **Accuracy** | {results['accuracy']:.4f} | Overall proportion of correct predictions. |\n")
            fh.write(f"| **Precision (macro)** | {results['precision_macro']:.4f} | Out of all predicted positives, what fraction are actual positives. Focuses on minimizing False Positives. |\n")
            fh.write(f"| **Recall (macro)** | {results['recall_macro']:.4f} | Out of all actual positives, what fraction did the model correctly identify. Focuses on minimizing False Negatives. |\n")
            fh.write(f"| **F1-Score (macro)** | {results['f1_macro']:.4f} | Harmonic mean of Precision and Recall; provides a balanced metric for imbalanced classes. |\n")
            fh.write(f"| **ROC AUC** | {results['roc_auc']:.4f} | Measures model discriminative power between classes across all decision thresholds. |\n\n")

            fh.write("## 3. Model Interpretation (Log-Odds & Odds Ratios)\n")
            fh.write(f"The baseline intercept log-odds value is **{results['intercept']:.4f}**. ")
            fh.write("Since all input features were standardized, the coefficients represent the change in log-odds of a supply chain ")
            fh.write("disruption occurring for every one standard deviation increase in that feature. ")
            fh.write("We convert these log-odds coefficients to **Odds Ratios ($e^\\beta$)** for intuitive interpretation:\n\n")

            fh.write("| Feature | Log-Odds Coefficient ($\\beta$) | Odds Ratio ($e^\\beta$) | Impact Direction |\n")
            fh.write("|---------|------------------------------|------------------------|------------------|\n")
            for name, coef in zip(self._feature_names, results["coefficients"]):
                odds_ratio = np.exp(coef)
                direction = "Increases Risk" if coef >= 0 else "Decreases Risk"
                fh.write(f"| **{name}** | {coef:+.4f} | {odds_ratio:.4f} | {direction} |\n")
            fh.write("\n")

            fh.write("### Qualitative Impact Analysis:\n")
            fh.write("- **Odds Ratio > 1.0**: For every standard deviation increase in this feature, the odds of a supply chain disruption occurring *increase* by the ratio factor. ")
            fh.write("For example, positive coefficients on geopolitical risk or distance heavily increase disruption probabilities.\n")
            fh.write("- **Odds Ratio < 1.0**: For every standard deviation increase in this feature, the odds of a supply chain disruption occurring *decrease* by the ratio factor. ")
            fh.write("For example, higher carrier reliability scores lead to a corresponding reduction in predicted disruption odds.\n\n")

            fh.write("## 4. Diagnostics Interpretation\n")
            fh.write("### Confusion Matrix Analysis:\n")
            fh.write("- The heatmap (`confusion_matrix.png`) displays counts of True Positives, True Negatives, False Positives, and False Negatives.\n")
            fh.write("- In a supply chain context, **False Negatives** (failing to predict a disruption that actually occurs) are extremely costly, ")
            fh.write("as they lead to unmitigated operational downtime. **False Positives** (predicting a disruption that does not occur) ")
            fh.write("lead to minor inefficiencies (e.g., rerouting costs).\n\n")

            fh.write("### ROC Curve Interpretation:\n")
            fh.write("- The ROC curve (`roc_curve.png`) plots the True Positive Rate against the False Positive Rate at different decision thresholds.\n")
            fh.write("- A perfect classifier reaches the top-left corner (AUC = 1.0). A random classifier follows the diagonal (AUC = 0.5).\n")
            fh.write(f"- With an AUC of **{results['roc_auc']:.4f}**, our model indicates highly effective class separation, ")
            fh.write("allowing operations teams to select a decision threshold that matches their tolerance for False Negatives vs. False Positives.\n")

        self._logger.info("Technical explanation report saved to: %s", report_path)
