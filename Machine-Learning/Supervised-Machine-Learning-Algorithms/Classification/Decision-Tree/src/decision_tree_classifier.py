# ============================================================================
# Decision Tree Classifier Service for Classification Pipeline
# ============================================================================
# Owns model lifecycle: training, evaluation, feature importance extraction,
# plot generation, text summary export, and technical markdown reporting.
# ============================================================================

import logging
import os
import time
from typing import List, Union

import matplotlib
matplotlib.use("Agg")
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
)
from sklearn.tree import DecisionTreeClassifier

from config import ModelConfig, PathConfig

# cuML GPU DecisionTreeClassifier support check
try:
    from cuml.tree import DecisionTreeClassifier as GPUDecisionTreeClassifier
    _HAS_GPU_DT = True
except ImportError:
    _HAS_GPU_DT = False


class DecisionTreeClassifierService:
    """Service encapsulating Decision Tree classification and analysis.

    Responsibilities:
        1. Train a DecisionTreeClassifier (scikit-learn or cuML GPU).
        2. Generate predictions on unseen test data.
        3. Compute and log classification metrics.
        4. Extract feature importances.
        5. Generate visualization plots (Confusion Matrix, Feature Importance, Metrics Bar Chart).
        6. Persist text and markdown evaluation reports.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the decision tree classifier service.

        Args:
            model_config:  Decision Tree hyperparameters (criterion, max_depth, etc.).
            path_config:   Path settings for writing output artifacts.
            label_names:   Class label names in encoder order.
            feature_names: Feature names for importance reporting.
            logger:        Logger instance.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger
        self._model = self._build_model()

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Decision Tree classifier on training data."""
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        self._logger.info(
            "Training Decision Tree on %d samples with %d features...",
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Training completed in %.3f seconds.", elapsed
        )

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Predict on the test set and compute metrics."""
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

        # Extract feature importances if supported
        feature_importances = None
        if hasattr(self._model, "feature_importances_"):
            feature_importances = self._model.feature_importances_.tolist()

        results = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "feature_importances": feature_importances,
            "predictions": predictions,
        }

        self._log_evaluation(results)
        self._save_results(results)
        self._generate_plots(results, y_test)
        self._save_analysis(results, y_test)

        return results

    def _build_model(self) -> Union[DecisionTreeClassifier, 'GPUDecisionTreeClassifier']:
        """Construct the decision tree estimator from model configuration."""
        device = self._model_config.device.lower()
        use_gpu = False

        if device == "gpu":
            if _HAS_GPU_DT:
                use_gpu = True
            else:
                self._logger.warning(
                    "GPU requested, but RAPIDS cuML is not available. Falling back to CPU."
                )
        elif device == "auto":
            if _HAS_GPU_DT:
                use_gpu = True
                self._logger.info("GPU detected and selected for Decision Tree acceleration.")
            else:
                self._logger.info("Using CPU execution (cuML not available or no GPU).")
        else:
            self._logger.info("Using CPU execution as configured.")

        if use_gpu:
            self._logger.info("Instantiating GPU Decision Tree Classifier (RAPIDS cuML).")
            return GPUDecisionTreeClassifier(
                max_depth=self._model_config.max_depth or 16,
                min_samples_split=self._model_config.min_samples_split,
                min_samples_leaf=self._model_config.min_samples_leaf,
            )

        self._logger.info("Instantiating CPU Decision Tree Classifier (scikit-learn).")
        return DecisionTreeClassifier(
            criterion=self._model_config.criterion,
            max_depth=self._model_config.max_depth,
            min_samples_split=self._model_config.min_samples_split,
            min_samples_leaf=self._model_config.min_samples_leaf,
            random_state=self._model_config.random_state,
        )

    def _log_hyperparameters(self) -> None:
        """Log model hyperparameters for auditability."""
        self._logger.info("Decision Tree Hyperparameters:")
        self._logger.info("  criterion         : %s", self._model_config.criterion)
        self._logger.info("  max_depth         : %s", str(self._model_config.max_depth))
        self._logger.info("  min_samples_split : %d", self._model_config.min_samples_split)
        self._logger.info("  min_samples_leaf  : %d", self._model_config.min_samples_leaf)

    def _log_evaluation(self, results: dict) -> None:
        """Log evaluation summary and confusion matrix to console and file."""
        self._logger.info("-" * 70)
        self._logger.info("OVERALL METRICS:")
        self._logger.info(
            "  Accuracy         : %.4f  (%.2f%%)",
            results["accuracy"],
            results["accuracy"] * 100,
        )
        self._logger.info(
            "  Precision (macro): %.4f",
            results["precision_macro"],
        )
        self._logger.info(
            "  Recall    (macro): %.4f",
            results["recall_macro"],
        )
        self._logger.info(
            "  F1-Score  (macro): %.4f",
            results["f1_macro"],
        )

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

        self._logger.info("-" * 70)
        self._logger.info("PER-CLASS CLASSIFICATION REPORT:")
        for line in results["classification_report"].split("\n"):
            if line.strip():
                self._logger.info("  %s", line)
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Save text summary of metrics and feature importances."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        output_path = os.path.join(
            self._path_config.output_dir, "classification_results.txt"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("DECISION TREE CLASSIFICATION RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  criterion         : {self._model_config.criterion}\n")
            fh.write(f"  max_depth         : {self._model_config.max_depth}\n")
            fh.write(f"  min_samples_split : {self._model_config.min_samples_split}\n")
            fh.write(f"  min_samples_leaf  : {self._model_config.min_samples_leaf}\n\n")

            fh.write("OVERALL METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(
                f"  Accuracy         : {results['accuracy']:.4f}  "
                f"({results['accuracy'] * 100:.2f}%)\n"
            )
            fh.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
            fh.write(f"  Recall    (macro): {results['recall_macro']:.4f}\n")
            fh.write(f"  F1-Score  (macro): {results['f1_macro']:.4f}\n\n")

            if results["feature_importances"]:
                fh.write("FEATURE IMPORTANCES\n")
                fh.write("-" * 40 + "\n")
                sorted_imp = sorted(
                    zip(self._feature_names, results["feature_importances"]),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for name, imp in sorted_imp:
                    fh.write(f"  {name:<30} : {imp:.6f}\n")
                fh.write("\n")

            fh.write("CONFUSION MATRIX\n")
            fh.write("-" * 40 + "\n")
            col_header = "Actual\\Pred"
            header_str = f"{col_header:<15}"
            for name in self._label_names:
                header_str += f"  {name:<10}"
            fh.write(header_str + "\n")
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

    def _generate_plots(self, results: dict, y_test: np.ndarray) -> None:
        """Generate confusion matrix, metrics bar chart, and feature importance plot."""
        self._logger.info("Generating evaluation plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # Plot 1: Confusion Matrix
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
        plt.title("Decision Tree Confusion Matrix Heatmap")
        plt.tight_layout()
        cm_path = os.path.join(self._path_config.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # Plot 2: Per-Class Metrics
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
            data.append({"Class": name, "Metric": "Recall", "Value": recalls[idx]})
            data.append({"Class": name, "Metric": "F1-Score", "Value": f1_scores[idx]})

        df_metrics = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Class", y="Value", hue="Metric", data=df_metrics, palette="muted"
        )
        plt.ylim(0, 1.05)
        plt.title("Decision Tree Per-Class Metrics")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(loc="lower right")
        plt.tight_layout()
        report_fig_path = os.path.join(
            self._path_config.output_dir, "classification_report.png"
        )
        plt.savefig(report_fig_path, dpi=300)
        plt.close()

        # Plot 3: Feature Importances
        if results["feature_importances"]:
            df_imp = pd.DataFrame(
                {
                    "Feature": self._feature_names,
                    "Importance": results["feature_importances"],
                }
            ).sort_values("Importance", ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(df_imp["Feature"], df_imp["Importance"], color="skyblue")
            plt.xlabel("Gini Importance")
            plt.title("Decision Tree Feature Importances")
            plt.tight_layout()
            imp_path = os.path.join(
                self._path_config.output_dir, "feature_importance.png"
            )
            plt.savefig(imp_path, dpi=300)
            plt.close()
            self._logger.info("Feature importance plot saved to: %s", imp_path)

    def _save_analysis(self, results: dict, y_test: np.ndarray) -> None:
        """Write technical markdown explanation report."""
        self._logger.info("Writing classification explanation report...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        report_path = os.path.join(
            self._path_config.output_dir, "classification_analysis.md"
        )

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Decision Tree Classification Analysis Report\n\n")
            fh.write("## 1. Executive Summary\n")
            fh.write(
                "This report analyzes the Decision Tree classifier performance on online fraud detection. "
            )
            fh.write(
                f"The model achieved an overall accuracy of **{results['accuracy'] * 100:.2f}%**.\n\n"
            )

            fh.write("## 2. Confusion Matrix & Metrics\n")
            fh.write(f"- Accuracy: {results['accuracy']:.4f}\n")
            fh.write(f"- Macro Precision: {results['precision_macro']:.4f}\n")
            fh.write(f"- Macro Recall: {results['recall_macro']:.4f}\n")
            fh.write(f"- Macro F1-Score: {results['f1_macro']:.4f}\n\n")

            if results["feature_importances"]:
                fh.write("## 3. Top Predictive Features\n")
                sorted_imp = sorted(
                    zip(self._feature_names, results["feature_importances"]),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for name, imp in sorted_imp[:5]:
                    fh.write(f"- **{name}**: Gini Importance = `{imp:.6f}`\n")
                fh.write("\n")

            fh.write("## 4. Hyperparameter Design Rationale\n")
            fh.write(
                f"- **Criterion (`{self._model_config.criterion}`)**: Evaluates information gain at each node split.\n"
            )
            fh.write(
                f"- **Max Depth (`{self._model_config.max_depth}`)**: Prunes recursive tree growth to avoid memorizing noise.\n"
            )
            fh.write(
                f"- **Min Samples Split (`{self._model_config.min_samples_split}`)**: Prevents splits on small sub-branches.\n"
            )

        self._logger.info("Explanation report saved to: %s", report_path)
