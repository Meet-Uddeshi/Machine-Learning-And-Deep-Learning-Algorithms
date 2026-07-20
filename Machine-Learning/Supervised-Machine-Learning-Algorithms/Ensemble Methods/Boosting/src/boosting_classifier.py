# ============================================================================
# Boosting Classifier Service for Classification Pipeline
# ============================================================================
# Owns the entire model lifecycle: training, prediction, evaluation, and
# structured result reporting. Follows the Service Layer pattern so that
# model logic is isolated from data preparation and orchestration.
# ============================================================================

import logging
import os
import time
from typing import List, Optional

# Set non-interactive backend for matplotlib to avoid GUI initialization issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from config import ModelConfig, PathConfig


class BoostingClassifierService:
    """Service encapsulating Boosting classification.

    Responsibilities:
        1. Train a Boosting Classifier (Gradient Boosting or AdaBoost) with configurations.
        2. Generate predictions on unseen test data.
        3. Compute and log a comprehensive evaluation report.
        4. Visualise confusion matrix, metrics, feature importances, and stage loss.
        5. Persist all output artifacts to the output directory.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the classifier service.

        Args:
            model_config:  Boosting hyperparameters.
            path_config:   Path settings for saving output artifacts.
            label_names:   Human-readable class labels in encoder order.
            feature_names: Feature column names.
            logger:        Logger instance for progress and results.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger
        self._model = self._build_model()

    # -- Public workflow methods ---------------------------------------------

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Boosting model on the training data.

        Args:
            x_train: Training feature matrix.
            y_train: Training target vector.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        self._logger.info(
            "Training %s on %d samples with %d features...",
            self._model_config.boosting_type.upper(),
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info("Training completed in %.3f seconds.", elapsed)
        self._logger.info(
            "Number of estimators: %d",
            len(self._model.estimators_),
        )

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray, x_train: np.ndarray = None, y_train: np.ndarray = None
    ) -> dict:
        """Predict on the test set and compute evaluation metrics.

        Args:
            x_test: Test feature matrix.
            y_test: True test labels.
            x_train: Training feature matrix (optional, for deviance plotting).
            y_train: True training labels (optional, for deviance plotting).

        Returns:
            Dictionary with evaluation metrics and predictions.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL EVALUATION")
        self._logger.info("=" * 70)

        start_time = time.perf_counter()
        predictions = self._model.predict(x_test)
        predict_elapsed = time.perf_counter() - start_time
        self._logger.info(
            "Prediction on %d test samples completed in %.3f seconds.",
            x_test.shape[0],
            predict_elapsed,
        )

        # Compute metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
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
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "predictions": predictions,
        }

        self._log_evaluation(results)
        self._save_results(results)
        self._generate_plots(results, y_test, x_train, y_train, x_test)
        self._save_analysis(results, y_test)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self):
        """Construct the Boosting estimator from configuration.

        Returns:
            An unfitted sklearn Boosting estimator.
        """
        self._logger.info("Instantiating Boosting Classifier (%s).", self._model_config.boosting_type)
        
        if self._model_config.boosting_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self._model_config.n_estimators,
                learning_rate=self._model_config.learning_rate,
                max_depth=self._model_config.max_depth,
                min_samples_split=self._model_config.min_samples_split,
                min_samples_leaf=self._model_config.min_samples_leaf,
                subsample=self._model_config.subsample,
                random_state=self._model_config.random_state,
            )
        elif self._model_config.boosting_type == "adaboost":
            # For AdaBoost, base estimator is DecisionTreeClassifier with max_depth
            base_estimator = DecisionTreeClassifier(
                max_depth=self._model_config.max_depth,
                random_state=self._model_config.random_state,
            )
            return AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=self._model_config.n_estimators,
                learning_rate=self._model_config.learning_rate,
                random_state=self._model_config.random_state,
            )
        else:
            raise ValueError(
                f"Unknown boosting type: {self._model_config.boosting_type}"
            )

    def _log_hyperparameters(self) -> None:
        """Log all model hyperparameters for reproducibility."""
        self._logger.info("Boosting Hyperparameters:")
        self._logger.info("  boosting_type    : %s", self._model_config.boosting_type)
        self._logger.info("  n_estimators     : %d", self._model_config.n_estimators)
        self._logger.info("  learning_rate    : %.4f", self._model_config.learning_rate)
        self._logger.info("  max_depth        : %d", self._model_config.max_depth)
        if self._model_config.boosting_type == "gradient_boosting":
            self._logger.info("  min_samples_split: %d", self._model_config.min_samples_split)
            self._logger.info("  min_samples_leaf : %d", self._model_config.min_samples_leaf)
            self._logger.info("  subsample        : %.4f", self._model_config.subsample)
        self._logger.info("  random_state     : %d", self._model_config.random_state)

    def _log_evaluation(self, results: dict) -> None:
        """Log evaluation metrics."""
        self._logger.info("-" * 70)
        self._logger.info("OVERALL METRICS:")
        self._logger.info(
            "  Accuracy         : %.4f  (%.2f%%)",
            results["accuracy"],
            results["accuracy"] * 100,
        )
        self._logger.info(
            "  Precision (macro): %.4f", results["precision_macro"]
        )
        self._logger.info(
            "  Recall    (macro): %.4f", results["recall_macro"]
        )
        self._logger.info(
            "  F1-Score  (macro): %.4f", results["f1_macro"]
        )

        # Confusion matrix
        self._logger.info("-" * 70)
        self._logger.info("CONFUSION MATRIX:")
        header = "%-15s" % "Actual\\Pred"
        for name in self._label_names:
            header += "  %-10s" % name
        self._logger.info(header)

        for idx, row in enumerate(results["confusion_matrix"]):
            row_str = "%-15s" % self._label_names[idx]
            for val in row:
                row_str += "  %-10d" % val
            self._logger.info(row_str)

        # Per-class report
        self._logger.info("-" * 70)
        self._logger.info("PER-CLASS CLASSIFICATION REPORT:")
        for line in results["classification_report"].split("\n"):
            if line.strip():
                self._logger.info("  %s", line)
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Persist evaluation results to classification_results.txt."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        output_path = os.path.join(
            self._path_config.output_dir, "classification_results.txt"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write(f"BOOSTING CLASSIFICATION RESULTS ({self._model_config.boosting_type.upper()})\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  boosting_type    : {self._model_config.boosting_type}\n")
            fh.write(f"  n_estimators     : {self._model_config.n_estimators}\n")
            fh.write(f"  learning_rate    : {self._model_config.learning_rate}\n")
            fh.write(f"  max_depth        : {self._model_config.max_depth}\n")
            if self._model_config.boosting_type == "gradient_boosting":
                fh.write(f"  min_samples_split: {self._model_config.min_samples_split}\n")
                fh.write(f"  min_samples_leaf : {self._model_config.min_samples_leaf}\n")
                fh.write(f"  subsample        : {self._model_config.subsample}\n")
            fh.write(f"  random_state     : {self._model_config.random_state}\n\n")

            fh.write("FEATURES USED\n")
            fh.write("-" * 40 + "\n")
            for i, name in enumerate(self._feature_names, start=1):
                fh.write(f"  {i:2d}. {name}\n")
            fh.write("\n")

            # Feature importances
            fh.write("FEATURE IMPORTANCES\n")
            fh.write("-" * 40 + "\n")
            importances = self._model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_idx, start=1):
                fh.write(
                    f"  {rank:2d}. {self._feature_names[idx]:<25s}: "
                    f"{importances[idx]:.4f}\n"
                )
            fh.write("\n")

            fh.write("OVERALL METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(
                f"  Accuracy         : {results['accuracy']:.4f}  "
                f"({results['accuracy'] * 100:.2f}%)\n"
            )
            fh.write(
                f"  Precision (macro): {results['precision_macro']:.4f}\n"
            )
            fh.write(
                f"  Recall    (macro): {results['recall_macro']:.4f}\n"
            )
            fh.write(
                f"  F1-Score  (macro): {results['f1_macro']:.4f}\n\n"
            )

            fh.write("CONFUSION MATRIX\n")
            fh.write("-" * 40 + "\n")
            col_header = 'Actual\\Pred'
            header = f"{col_header:<15}"
            for name in self._label_names:
                header += f"  {name:<10}"
            fh.write(header + "\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                row_str = f"{self._label_names[idx]:<15}"
                for val in row:
                    row_str += f"  {val:<10}"
                fh.write(row_str + "\n")
            fh.write("\n")

            fh.write("PER-CLASS CLASSIFICATION REPORT\n")
            fh.write("-" * 40 + "\n")
            fh.write(results["classification_report"])
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Results saved to: %s", output_path)

    def _generate_plots(
        self, results: dict, y_test: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> None:
        """Generate evaluation plots: confusion matrix, metrics, feature importances, stage loss."""
        self._logger.info("Generating evaluation plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # Plot 1: Confusion Matrix Heatmap
        conf_matrix = np.array(results["confusion_matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self._label_names,
            yticklabels=self._label_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f"Boosting ({self._model_config.boosting_type.upper()}) -- Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(
            self._path_config.output_dir, "confusion_matrix.png"
        )
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # Plot 2: Per-Class Metrics Bar Chart
        precisions = precision_score(
            y_test, results["predictions"], average=None, zero_division=0
        )
        recalls = recall_score(
            y_test, results["predictions"], average=None, zero_division=0
        )
        f1_scores = f1_score(
            y_test, results["predictions"], average=None, zero_division=0
        )

        data = []
        for idx, name in enumerate(self._label_names):
            data.append({"Class": name, "Metric": "Precision", "Value": precisions[idx]})
            data.append({"Class": name, "Metric": "Recall",    "Value": recalls[idx]})
            data.append({"Class": name, "Metric": "F1-Score",  "Value": f1_scores[idx]})

        df_metrics = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Class", y="Value", hue="Metric", data=df_metrics, palette="pastel")
        plt.ylim(0, 1.05)
        plt.title(f"Boosting ({self._model_config.boosting_type.upper()}) -- Classification Metrics")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(loc="lower right")
        plt.tight_layout()
        report_fig_path = os.path.join(
            self._path_config.output_dir, "classification_report.png"
        )
        plt.savefig(report_fig_path, dpi=300)
        plt.close()

        # Plot 3: Feature Importance Bar Chart
        importances = self._model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [self._feature_names[i] for i in sorted_idx]
        sorted_imp = importances[sorted_idx]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=sorted_imp, y=sorted_names, hue=sorted_names, palette="rocket", legend=False)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Boosting ({self._model_config.boosting_type.upper()}) -- Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(
            self._path_config.output_dir, "feature_importance.png"
        )
        plt.savefig(fi_path, dpi=300)
        plt.close()

        # Plot 4: Stage Loss / Stage Error over Iterations
        plt.figure(figsize=(10, 6))
        n_est = self._model_config.n_estimators
        
        if self._model_config.boosting_type == "gradient_boosting":
            # Plot training deviance (loss)
            plt.plot(
                np.arange(n_est) + 1,
                self._model.train_score_,
                label="Training Set Loss (Deviance)",
                color="navy",
                linewidth=2,
            )
            # Compute and plot test deviance if test features are provided
            test_deviance = np.zeros((n_est,), dtype=np.float64)
            for i, y_pred in enumerate(self._model.staged_predict_proba(x_test)):
                test_deviance[i] = log_loss(y_test, y_pred)
            
            plt.plot(
                np.arange(n_est) + 1,
                test_deviance,
                label="Test Set Loss (Deviance)",
                color="orange",
                linewidth=2,
                linestyle="--",
            )
            plt.ylabel("Loss (Deviance)")
            plt.title("Gradient Boosting -- Deviance Loss Over Stages")
            
        elif self._model_config.boosting_type == "adaboost":
            # For AdaBoost, plot staging error rates
            train_err = np.zeros((n_est,), dtype=np.float64)
            test_err = np.zeros((n_est,), dtype=np.float64)
            
            for i, y_pred in enumerate(self._model.staged_predict(x_train)):
                train_err[i] = 1.0 - accuracy_score(y_train, y_pred)
            for i, y_pred in enumerate(self._model.staged_predict(x_test)):
                test_err[i] = 1.0 - accuracy_score(y_test, y_pred)
                
            plt.plot(
                np.arange(n_est) + 1,
                train_err,
                label="Training Set Error",
                color="navy",
                linewidth=2,
            )
            plt.plot(
                np.arange(n_est) + 1,
                test_err,
                label="Test Set Error",
                color="orange",
                linewidth=2,
                linestyle="--",
            )
            plt.ylabel("Classification Error Rate")
            plt.title("AdaBoost -- Classification Error Over Stages")

        plt.xlabel("Boosting Stages / Iterations")
        plt.grid(True, linestyle=":")
        plt.legend(loc="upper right")
        plt.tight_layout()
        loss_path = os.path.join(
            self._path_config.output_dir, "boosting_stage_loss.png"
        )
        plt.savefig(loss_path, dpi=300)
        plt.close()

        self._logger.info("Evaluation plots saved successfully.")

    def _save_analysis(self, results: dict, y_test: np.ndarray) -> None:
        """Generate and save technical markdown report classification_analysis.md."""
        report_path = os.path.join(
            self._path_config.output_dir, "classification_analysis.md"
        )

        precisions = precision_score(
            y_test, results["predictions"], average=None, zero_division=0
        )
        recalls = recall_score(
            y_test, results["predictions"], average=None, zero_division=0
        )
        f1_scores = f1_score(
            y_test, results["predictions"], average=None, zero_division=0
        )

        importances = self._model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Boosting ({self._model_config.boosting_type.upper()}) Classification Analysis Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                f"This report provides a formal technical analysis of the Boosting model's performance "
                f"on predicting diabetes onset using the Pima Indians Diabetes Dataset. "
                f"The model achieved an overall test accuracy of **{results['accuracy'] * 100:.2f}%**.\n\n"
            )

            fh.write("## 2. Model Configuration\n\n")
            fh.write("| Hyperparameter | Value |\n")
            fh.write("|----------------|-------|\n")
            fh.write(f"| `boosting_type` | `{self._model_config.boosting_type}` |\n")
            fh.write(f"| `n_estimators` | `{self._model_config.n_estimators}` |\n")
            fh.write(f"| `learning_rate` | `{self._model_config.learning_rate}` |\n")
            fh.write(f"| `max_depth` | `{self._model_config.max_depth}` |\n")
            if self._model_config.boosting_type == "gradient_boosting":
                fh.write(f"| `min_samples_split` | `{self._model_config.min_samples_split}` |\n")
                fh.write(f"| `min_samples_leaf` | `{self._model_config.min_samples_leaf}` |\n")
                fh.write(f"| `subsample` | `{self._model_config.subsample}` |\n")
            fh.write(f"| `random_state` | `{self._model_config.random_state}` |\n\n")

            fh.write("## 3. Evaluation Metrics Breakdown\n\n")
            fh.write("| Class | Precision | Recall | F1-Score |\n")
            fh.write("|-------|-----------|--------|----------|\n")
            for idx, name in enumerate(self._label_names):
                fh.write(
                    f"| {name} (Class {idx}) | {precisions[idx]:.4f} | "
                    f"{recalls[idx]:.4f} | {f1_scores[idx]:.4f} |\n"
                )
            fh.write("\n")

            fh.write("### Theoretical Interpretation of Metrics\n")
            fh.write(
                "- **Precision**: Shows the proportion of predicted positive cases that are actually positive. "
                "High precision indicates a low false positive rate.\n"
                "- **Recall**: Indicates the proportion of actual positive cases that were correctly identified. "
                "High recall implies a low false negative rate.\n"
                "- **F1-Score**: The harmonic mean of precision and recall. It is a balanced evaluation metric "
                "particularly suited for datasets with slight class imbalances.\n\n"
            )

            fh.write("## 4. Confusion Matrix Analysis\n\n")
            fh.write("| Actual \\ Predicted |")
            for name in self._label_names:
                fh.write(f" Class {name} |")
            fh.write("\n| --- |")
            for _ in self._label_names:
                fh.write(" --- |")
            fh.write("\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                fh.write(f"| **Class {self._label_names[idx]}** |")
                for val in row:
                    fh.write(f" {val} |")
                fh.write("\n")
            fh.write("\n")

            fh.write("### Analysis of Misclassifications\n")
            fh.write(
                "Boosting models reduce bias by sequentially training weak learners to correct the errors "
                "made by preceding models. Misclassifications in the final ensemble typically stem from "
                "regions of high overlap in the feature distributions (e.g. boundary values of Glucose and BMI) "
                "or extreme outliers in numerical columns that distort the stage-wise gradient steps.\n\n"
            )

            fh.write("## 5. Feature Importances\n\n")
            fh.write("| Rank | Feature | Importance |\n")
            fh.write("|------|---------|------------|\n")
            for rank, idx in enumerate(sorted_idx, start=1):
                fh.write(
                    f"| {rank} | {self._feature_names[idx]} | "
                    f"{importances[idx]:.4f} |\n"
                )
            fh.write("\n")

            fh.write("## 6. Output Artifacts\n\n")
            fh.write("| File | Description |\n")
            fh.write("|------|-------------|\n")
            fh.write("| `classification_results.txt` | Hyperparameters, confusion matrix, and numeric metrics |\n")
            fh.write("| `classification_analysis.md` | This technical analysis report |\n")
            fh.write("| `confusion_matrix.png` | Heatmap visualization of predictions vs truth |\n")
            fh.write("| `classification_report.png` | Grouped bar chart of precision, recall, and F1-score |\n")
            fh.write("| `feature_importance.png` | Horizontal bar chart of sorted feature importances |\n")
            fh.write("| `boosting_stage_loss.png` | Staged training vs validation loss/error curve |\n")

        self._logger.info("Technical analysis report saved to: %s", report_path)
