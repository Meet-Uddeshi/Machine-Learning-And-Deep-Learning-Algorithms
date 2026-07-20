# ============================================================================
# Random Forest Classifier Service for Ensemble Classification Pipeline
# ============================================================================
# Owns model lifecycle: training, OOB error evaluation, prediction,
# feature importance ranking, plot generation, and structured reporting.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import ModelConfig, PathConfig

# cuML GPU RandomForestClassifier support check
try:
    from cuml.ensemble import RandomForestClassifier as GPURandomForestClassifier
    _HAS_GPU_RF = True
except ImportError:
    _HAS_GPU_RF = False


class RandomForestClassifierService:
    """Service encapsulating Random Forest ensemble classification.

    Responsibilities:
        1. Train a RandomForestClassifier with bagging and random feature selection.
        2. Evaluate OOB (Out-of-Bag) generalization score if enabled.
        3. Predict on unseen test data.
        4. Compute accuracy, precision, recall, F1-score, and confusion matrix.
        5. Extract and plot feature importances.
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
        """Initialize the random forest classifier service.

        Args:
            model_config:  Random Forest hyperparameters (n_estimators, max_depth, etc.).
            path_config:   Path settings for output files.
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
        """Fit the Random Forest ensemble on training data."""
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        self._logger.info(
            "Training Random Forest (%d trees) on %d samples with %d features...",
            self._model_config.n_estimators,
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Training completed in %.3f seconds.", elapsed
        )

        # Log OOB Score if available
        if hasattr(self._model, "oob_score_") and self._model.oob_score_:
            self._logger.info(
                "Out-Of-Bag (OOB) Generalization Score: %.4f (%.2f%%)",
                self._model.oob_score_,
                self._model.oob_score_ * 100,
            )

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Predict on test set and compute metrics."""
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

        oob_score = getattr(self._model, "oob_score_", None)
        feature_importances = None
        if hasattr(self._model, "feature_importances_"):
            feature_importances = self._model.feature_importances_.tolist()

        results = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "oob_score": oob_score,
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

    def _build_model(self) -> Union[RandomForestClassifier, 'GPURandomForestClassifier']:
        """Construct estimator based on device configuration."""
        device = self._model_config.device.lower()
        use_gpu = False

        if device == "gpu":
            if _HAS_GPU_RF:
                use_gpu = True
            else:
                self._logger.warning(
                    "GPU requested, but RAPIDS cuML is not available. Falling back to CPU."
                )
        elif device == "auto":
            if _HAS_GPU_RF:
                use_gpu = True
                self._logger.info("GPU detected and selected for Random Forest acceleration.")
            else:
                self._logger.info("Using CPU execution (cuML not available or no GPU).")
        else:
            self._logger.info("Using CPU execution as configured.")

        if use_gpu:
            self._logger.info("Instantiating GPU Random Forest Classifier (RAPIDS cuML).")
            return GPURandomForestClassifier(
                n_estimators=self._model_config.n_estimators,
                max_depth=self._model_config.max_depth or 16,
            )

        self._logger.info("Instantiating CPU Random Forest Classifier (scikit-learn).")
        return RandomForestClassifier(
            n_estimators=self._model_config.n_estimators,
            criterion=self._model_config.criterion,
            max_depth=self._model_config.max_depth,
            min_samples_split=self._model_config.min_samples_split,
            min_samples_leaf=self._model_config.min_samples_leaf,
            bootstrap=self._model_config.bootstrap,
            oob_score=self._model_config.oob_score,
            n_jobs=self._model_config.n_jobs,
            random_state=self._model_config.random_state,
        )

    def _log_hyperparameters(self) -> None:
        """Log ensemble hyperparameters."""
        self._logger.info("Random Forest Hyperparameters:")
        self._logger.info("  n_estimators      : %d", self._model_config.n_estimators)
        self._logger.info("  criterion         : %s", self._model_config.criterion)
        self._logger.info("  max_depth         : %s", str(self._model_config.max_depth))
        self._logger.info("  bootstrap         : %s", str(self._model_config.bootstrap))
        self._logger.info("  oob_score         : %s", str(self._model_config.oob_score))

    def _log_evaluation(self, results: dict) -> None:
        """Log metrics summary to console and file."""
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
        if results.get("oob_score"):
            self._logger.info(
                "  OOB Score        : %.4f  (%.2f%%)",
                results["oob_score"],
                results["oob_score"] * 100,
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
        """Write summary file containing hyperparameters, metrics, and feature importances."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        output_path = os.path.join(
            self._path_config.output_dir, "classification_results.txt"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("RANDOM FOREST CLASSIFICATION RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  n_estimators      : {self._model_config.n_estimators}\n")
            fh.write(f"  criterion         : {self._model_config.criterion}\n")
            fh.write(f"  max_depth         : {self._model_config.max_depth}\n")
            fh.write(f"  bootstrap         : {self._model_config.bootstrap}\n")
            fh.write(f"  oob_score         : {self._model_config.oob_score}\n\n")

            fh.write("OVERALL METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(
                f"  Accuracy         : {results['accuracy']:.4f}  "
                f"({results['accuracy'] * 100:.2f}%)\n"
            )
            fh.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
            fh.write(f"  Recall    (macro): {results['recall_macro']:.4f}\n")
            fh.write(f"  F1-Score  (macro): {results['f1_macro']:.4f}\n")
            if results.get("oob_score"):
                fh.write(f"  OOB Score        : {results['oob_score']:.4f}\n")
            fh.write("\n")

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
        """Save confusion matrix heatmap, per-class metrics bar chart, and feature importances."""
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
        plt.title("Random Forest Confusion Matrix Heatmap")
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
        plt.title("Random Forest Per-Class Metrics")
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

            plt.figure(figsize=(10, 8))
            plt.barh(df_imp["Feature"], df_imp["Importance"], color="seagreen")
            plt.xlabel("Gini Importance")
            plt.title("Random Forest Feature Importances")
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
            fh.write("# Random Forest Classification Analysis Report\n\n")
            fh.write("## 1. Executive Summary\n")
            fh.write(
                "This report details the Random Forest ensemble model performance on Indian Railway failure detection and maintenance requirements. "
            )
            fh.write(
                f"The model achieved an overall accuracy of **{results['accuracy'] * 100:.2f}%**.\n\n"
            )

            fh.write("## 2. Metrics & Out-Of-Bag Evaluation\n")
            fh.write(f"- Accuracy: {results['accuracy']:.4f}\n")
            fh.write(f"- Macro Precision: {results['precision_macro']:.4f}\n")
            fh.write(f"- Macro Recall: {results['recall_macro']:.4f}\n")
            fh.write(f"- Macro F1-Score: {results['f1_macro']:.4f}\n")
            if results.get("oob_score"):
                fh.write(f"- Out-of-Bag (OOB) Score: `{results['oob_score']:.4f}`\n")
            fh.write("\n")

            if results["feature_importances"]:
                fh.write("## 3. Key Predictive Maintenance Indicators\n")
                sorted_imp = sorted(
                    zip(self._feature_names, results["feature_importances"]),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for name, imp in sorted_imp[:5]:
                    fh.write(f"- **{name}**: Importance = `{imp:.6f}`\n")
                fh.write("\n")

            fh.write("## 4. Ensemble Theory & Hyperparameters\n")
            fh.write(
                f"- **Trees (`{self._model_config.n_estimators}`)**: Combines predictions across {self._model_config.n_estimators} independent decision trees via majority voting.\n"
            )
            fh.write(
                "- **Bootstrap Aggregation (Bagging)**: Each tree is trained on a random sample drawn with replacement, reducing variance.\n"
            )
            fh.write(
                "- **Random Subspace Selection**: At each node, a random subset of features is evaluated, decorrelating individual trees.\n"
            )

        self._logger.info("Explanation report saved to: %s", report_path)
