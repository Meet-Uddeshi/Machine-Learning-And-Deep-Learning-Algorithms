# ============================================================================
# SVM Classifier Service for Classification Pipeline
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
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from config import ModelConfig, PathConfig


class SVMClassifierService:
    """Service encapsulating Support Vector Machine (SVM) classification.

    Responsibilities:
        1. Train a scikit-learn SVC with configurable hyperparameters.
        2. Generate predictions on unseen test data.
        3. Compute and log a comprehensive evaluation report.
        4. Plot the confusion matrix, performance metrics, feature permutation importance,
           and the 2D PCA decision boundary showing support vectors.
        5. Persist all output artifacts (reports, text details, and graphs).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the SVM classifier service.

        Args:
            model_config:  SVM hyperparameters.
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
        self._model: SVC = self._build_model()

    # -- Public workflow methods ---------------------------------------------

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the SVM classifier on the standardized training data.

        Args:
            x_train: Training feature matrix (n_samples, n_features).
            y_train: Training target vector (n_samples,).
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        self._logger.info(
            "Training on %d samples with %d features...",
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info("Training completed in %.3f seconds.", elapsed)
        
        # Log support vector statistics
        total_sv = self._model.support_.shape[0]
        self._logger.info("Total support vectors: %d", total_sv)
        for idx, count in enumerate(self._model.n_support_):
            self._logger.info("  Class '%s' support vectors: %d", self._label_names[idx], count)

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> dict:
        """Predict on the test set and compute evaluation metrics.

        Args:
            x_test:  Test feature matrix.
            y_test:  True test labels.
            x_train: Training feature matrix (needed for decision boundary PCA).
            y_train: Training target vector (needed for decision boundary PCA).

        Returns:
            Dictionary containing metrics and predictions.
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

        # Calculate Permutation Importance (since SVMs do not have a built-in feature importance)
        self._logger.info("Calculating permutation feature importances on test set...")
        perm_importance = permutation_importance(
            self._model,
            x_test,
            y_test,
            n_repeats=10,
            random_state=self._model_config.random_state,
        )
        importances = perm_importance.importances_mean

        results = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "predictions": predictions,
            "feature_importances": importances,
        }

        self._log_evaluation(results)
        self._save_results(results)
        self._save_model_details()
        self._generate_plots(results, y_test, x_train, y_train, x_test)
        self._save_analysis(results, y_test)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self) -> SVC:
        """Construct the SVC estimator from configuration.

        Returns:
            An unfitted SVC model.
        """
        self._logger.info("Instantiating Support Vector Classifier (scikit-learn).")
        return SVC(
            C=self._model_config.C,
            kernel=self._model_config.kernel,
            degree=self._model_config.degree,
            gamma=self._model_config.gamma,
            probability=self._model_config.probability,
            random_state=self._model_config.random_state,
        )

    def _log_hyperparameters(self) -> None:
        """Log model hyperparameters."""
        self._logger.info("SVM Hyperparameters:")
        self._logger.info("  C            : %.4f", self._model_config.C)
        self._logger.info("  kernel       : %s", self._model_config.kernel)
        if self._model_config.kernel == "poly":
            self._logger.info("  degree       : %d", self._model_config.degree)
        self._logger.info("  gamma        : %s", self._model_config.gamma)
        self._logger.info("  probability  : %s", self._model_config.probability)
        self._logger.info("  random_state : %d", self._model_config.random_state)

    def _log_evaluation(self, results: dict) -> None:
        """Log evaluation metrics in console and log file."""
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
        """Persist evaluation results into a text file.

        Args:
            results: Results dictionary containing performance scores.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        output_path = os.path.join(
            self._path_config.output_dir, "classification_results.txt"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("SVM CLASSIFICATION RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  C            : {self._model_config.C}\n")
            fh.write(f"  kernel       : {self._model_config.kernel}\n")
            fh.write(f"  degree       : {self._model_config.degree}\n")
            fh.write(f"  gamma        : {self._model_config.gamma}\n")
            fh.write(f"  probability  : {self._model_config.probability}\n")
            fh.write(f"  random_state : {self._model_config.random_state}\n\n")

            fh.write("SUPPORT VECTOR INFORMATION\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Total support vectors: {self._model.support_.shape[0]}\n")
            for idx, count in enumerate(self._model.n_support_):
                fh.write(f"  Class '{self._label_names[idx]}' support vectors: {count}\n")
            fh.write("\n")

            fh.write("FEATURES USED\n")
            fh.write("-" * 40 + "\n")
            for i, name in enumerate(self._feature_names, start=1):
                fh.write(f"  {i:2d}. {name}\n")
            fh.write("\n")

            # Permutation importances
            fh.write("PERMUTATION FEATURE IMPORTANCES\n")
            fh.write("-" * 40 + "\n")
            importances = results["feature_importances"]
            sorted_idx = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_idx, start=1):
                fh.write(
                    f"  {rank:2d}. {self._feature_names[idx]:<35s}: "
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

    def _save_model_details(self) -> None:
        """Export the support vectors, indices, dual coefficients, and intercepts.

        This acts as the structural text detail of the trained SVM model.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        details_path = os.path.join(
            self._path_config.output_dir, "svm_model_details.txt"
        )

        with open(details_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("SUPPORT VECTOR MACHINE INTERNAL MODEL DETAILS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write(f"Intercepts per decision boundary:\n")
            for idx, val in enumerate(self._model.intercept_):
                fh.write(f"  Boundary {idx}: {val:.6f}\n")
            fh.write("\n")

            fh.write(f"Number of support vectors per class:\n")
            for idx, name in enumerate(self._label_names):
                fh.write(f"  Class '{name}': {self._model.n_support_[idx]}\n")
            fh.write("\n")

            fh.write(f"Indices of support vectors in training set:\n")
            fh.write(f"  {self._model.support_.tolist()}\n\n")

            if self._model_config.kernel == "linear":
                fh.write("Linear SVM Coefficients (Weight Vector):\n")
                # coef_ is shape (n_classes * (n_classes - 1) / 2, n_features)
                for idx, coef in enumerate(self._model.coef_):
                    fh.write(f"  Boundary {idx} weights:\n")
                    for feat_idx, val in enumerate(coef):
                        fh.write(f"    {self._feature_names[feat_idx]:<35s}: {val:.6f}\n")
                fh.write("\n")

            fh.write("Dual Coefficients (Alpha * Y) for Support Vectors:\n")
            # dual_coef_ is shape (n_classes - 1, n_SV)
            fh.write(f"  Shape: {self._model.dual_coef_.shape}\n")
            fh.write(f"  First few dual coefficients (absolute values represent alpha penalty):\n")
            # Print a representative subset of dual coefficients
            n_print = min(15, self._model.dual_coef_.shape[1])
            for class_idx in range(self._model.dual_coef_.shape[0]):
                fh.write(f"    Decision Function {class_idx} (SV 0 to {n_print-1}):\n")
                fh.write(f"      {self._model.dual_coef_[class_idx][:n_print].tolist()} ...\n")
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Model details saved to: %s", details_path)

    def _generate_plots(
        self,
        results: dict,
        y_test: np.ndarray,
        x_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        x_test: np.ndarray,
    ) -> None:
        """Generate evaluation plots.

        Plots:
            1. Confusion Matrix Heatmap
            2. Per-Class Metrics Bar Chart
            3. Permutation Feature Importance Bar Chart
            4. 2D PCA Decision Boundary Plot (with support vectors highlighted)
        """
        self._logger.info("Generating evaluation plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # 1. Confusion Matrix
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
        plt.title("SVM -- Confusion Matrix Heatmap")
        plt.tight_layout()
        cm_path = os.path.join(
            self._path_config.output_dir, "confusion_matrix.png"
        )
        plt.savefig(cm_path, dpi=300)
        plt.close()

        # 2. Per-Class Metrics
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
        plt.title("SVM -- Per-Class Classification Metrics")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(loc="lower right")
        plt.tight_layout()
        report_fig_path = os.path.join(
            self._path_config.output_dir, "classification_report.png"
        )
        plt.savefig(report_fig_path, dpi=300)
        plt.close()

        # 3. Permutation Feature Importance Plot
        importances = results["feature_importances"]
        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [self._feature_names[i] for i in sorted_idx]
        sorted_imp = importances[sorted_idx]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=sorted_imp, y=sorted_names, hue=sorted_names, palette="viridis", legend=False)
        plt.xlabel("Permutation Importance (Drop in Test Accuracy)")
        plt.ylabel("Feature")
        plt.title("SVM -- Feature Permutation Importance")
        plt.tight_layout()
        fi_path = os.path.join(
            self._path_config.output_dir, "feature_importance.png"
        )
        plt.savefig(fi_path, dpi=300)
        plt.close()

        # 4. 2D PCA Decision Boundary Plot
        if x_train is not None and y_train is not None:
            self._logger.info("Computing PCA for 2D decision boundary visualization...")
            pca = PCA(n_components=2)
            x_train_pca = pca.fit_transform(x_train)
            x_test_pca = pca.transform(x_test)

            # Fit a 2D SVM using the same hyperparams
            model_2d = SVC(
                C=self._model_config.C,
                kernel=self._model_config.kernel,
                degree=self._model_config.degree,
                gamma=self._model_config.gamma,
                random_state=self._model_config.random_state,
            )
            model_2d.fit(x_train_pca, y_train)

            # Determine grid range
            x_min, x_max = x_train_pca[:, 0].min() - 0.5, x_train_pca[:, 0].max() + 0.5
            y_min, y_max = x_train_pca[:, 1].min() - 0.5, x_train_pca[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.02),
                np.arange(y_min, y_max, 0.02)
            )

            Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(10, 8))
            
            # Map contour levels and colors
            colors = ["#ff9999", "#99ff99", "#9999ff"]
            contour_color_map = matplotlib.colors.ListedColormap(colors[:len(self._label_names)])
            
            # Plot decision regions
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=contour_color_map)
            
            # Scatter train points
            scatter_train = plt.scatter(
                x_train_pca[:, 0],
                x_train_pca[:, 1],
                c=y_train,
                cmap=contour_color_map,
                edgecolors="k",
                marker="o",
                s=50,
                label="Train Data",
            )
            
            # Scatter test points
            scatter_test = plt.scatter(
                x_test_pca[:, 0],
                x_test_pca[:, 1],
                c=y_test,
                cmap=contour_color_map,
                edgecolors="y",
                marker="s",
                s=80,
                label="Test Data",
            )

            # Highlight Support Vectors
            sv = model_2d.support_vectors_
            plt.scatter(
                sv[:, 0],
                sv[:, 1],
                s=150,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Support Vectors (2D)",
            )

            # Create a legend
            handles, labels = plt.gca().get_legend_handles_labels()
            # Add custom handles for each class
            for i, name in enumerate(self._label_names):
                handles.append(
                    plt.Line2D(
                        [0], [0], marker="o", color="w",
                        markerfacecolor=colors[i], markeredgecolor="k", markersize=10
                    )
                )
                labels.append(name)
            
            plt.legend(handles=handles, labels=labels, loc="best")
            
            # PCA variance explained ratio
            var_exp = pca.explained_variance_ratio_
            plt.xlabel(f"PC 1 (Variance Explained: {var_exp[0]*100:.1f}%)")
            plt.ylabel(f"PC 2 (Variance Explained: {var_exp[1]*100:.1f}%)")
            plt.title(
                f"SVM Decision Boundary (2D PCA projection)\n"
                f"kernel={self._model_config.kernel} | C={self._model_config.C}"
            )
            plt.tight_layout()
            
            db_path = os.path.join(
                self._path_config.output_dir, "svm_decision_boundary.png"
            )
            plt.savefig(db_path, dpi=300)
            plt.close()
            self._logger.info("2D PCA Decision boundary plot saved to: %s", db_path)

    def _save_analysis(self, results: dict, y_test: np.ndarray) -> None:
        """Write a markdown analysis report to the output folder.

        Args:
            results: Dictionary containing performance metrics.
            y_test:  True test labels.
        """
        self._logger.info("Writing classification explanation report...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
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

        importances = results["feature_importances"]
        sorted_idx = np.argsort(importances)[::-1]

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Support Vector Machine (SVM) Classification Analysis Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                "This report provides a formal explanation of the SVM "
                "model's performance on predicting Iris species. "
            )
            fh.write(
                f"The model achieved an overall accuracy of "
                f"**{results['accuracy'] * 100:.2f}%** on the test dataset. "
            )
            fh.write(
                "Below is a detailed analysis of the performance metrics, "
                "confusion matrix, support vector distribution, and feature "
                "permutation importances.\n\n"
            )

            fh.write("## 2. Model Configuration\n\n")
            fh.write("| Hyperparameter | Value |\n")
            fh.write("|----------------|-------|\n")
            fh.write(f"| `C` | `{self._model_config.C}` |\n")
            fh.write(f"| `kernel` | `{self._model_config.kernel}` |\n")
            fh.write(f"| `degree` | `{self._model_config.degree}` |\n")
            fh.write(f"| `gamma` | `{self._model_config.gamma}` |\n")
            fh.write(f"| `probability` | `{self._model_config.probability}` |\n\n")

            fh.write("### Fitted SVM Properties\n\n")
            fh.write(f"- **Total support vectors**: {self._model.support_.shape[0]}\n")
            for idx, count in enumerate(self._model.n_support_):
                fh.write(f"- **Class '{self._label_names[idx]}' support vectors**: {count}\n")
            fh.write("\n")

            fh.write("## 3. Evaluation Metrics Breakdown\n")
            fh.write(
                "The classification metrics are analyzed below for each target class:\n\n"
            )
            fh.write("| Class | Precision | Recall | F1-Score |\n")
            fh.write("|-------|-----------|--------|----------|\n")
            for idx, name in enumerate(self._label_names):
                fh.write(
                    f"| {name} | {precisions[idx]:.4f} | "
                    f"{recalls[idx]:.4f} | {f1_scores[idx]:.4f} |\n"
                )
            fh.write("\n")

            fh.write("### Theoretical Interpretation of Metrics\n")
            fh.write(
                "- **Precision**: Shows the proportion of positive identifications that were actually correct. "
                "A model with high precision is selective about the class predictions it makes.\n"
            )
            fh.write(
                "- **Recall**: Shows the proportion of actual positives that were identified correctly. "
                "High recall is critical when missing a class has a high penalty.\n"
            )
            fh.write(
                "- **F1-Score**: The harmonic mean of precision and recall. It balances both metrics, especially "
                "when class frequencies are unequal.\n\n"
            )

            fh.write("## 4. Confusion Matrix Analysis\n\n")
            fh.write(
                "The confusion matrix (visualized in `confusion_matrix.png`) "
                "displays the relationship between actual and predicted classes:\n\n"
            )
            fh.write("| Actual \\ Predicted |")
            for name in self._label_names:
                fh.write(f" {name} |")
            fh.write("\n| --- |")
            for _ in self._label_names:
                fh.write(" --- |")
            fh.write("\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                fh.write(f"| **{self._label_names[idx]}** |")
                for val in row:
                    fh.write(f" {val} |")
                fh.write("\n")
            fh.write("\n")

            fh.write("### Mathematical Interpretation of SVM Boundaries & Support Vectors\n")
            fh.write(
                "Support vectors are the critical training samples that lie closest to the decision hyperplane. "
                "The position of the decision boundary is determined solely by these points. Changing other training "
                "points has no effect on the boundary. The regularization parameter `C` regulates the trade-off "
                "between maximizing the margin width and minimizing training error. A high `C` creates a narrower margin "
                "with fewer classification errors on the training set, which can lead to overfitting. A smaller `C` "
                "promotes a wider margin and allows some training misclassifications, improving generalization.\n\n"
            )

            fh.write("## 5. Permutation Feature Importances\n")
            fh.write(
                "Since SVMs (especially non-linear ones) do not provide a direct measure of feature importance, "
                "we use Permutation Importance. This evaluates the decrease in test accuracy when the values of "
                "a single feature are shuffled. A higher decrease indicates that the model relies heavily on that feature "
                "for classification.\n\n"
            )
            fh.write("| Rank | Feature | Importance (Accuracy drop) |\n")
            fh.write("|------|---------|---------------------------|\n")
            for rank, idx in enumerate(sorted_idx, start=1):
                fh.write(
                    f"| {rank} | {self._feature_names[idx]} | "
                    f"{importances[idx]:.4f} |\n"
                )
            fh.write("\n")

            fh.write("## 6. Output Artifacts\n\n")
            fh.write("| File | Description |\n")
            fh.write("|------|-------------|\n")
            fh.write("| `classification_results.txt` | Complete metrics, confusion matrix, hyperparameters, and feature importance |\n")
            fh.write("| `classification_analysis.md` | This technical analysis report |\n")
            fh.write("| `svm_model_details.txt` | Support vector indices, intercepts, and dual coefficients |\n")
            fh.write("| `confusion_matrix.png` | Confusion matrix heatmap |\n")
            fh.write("| `classification_report.png` | Precision/Recall/F1 grouped bar chart |\n")
            fh.write("| `feature_importance.png` | Permutation feature importance bar chart |\n")
            fh.write("| `svm_decision_boundary.png` | 2D PCA projected decision boundary contour and support vector plot |\n")
            fh.write("| `svm_iris.log` | Pipeline log with full execution outputs |\n\n")

            fh.write("## 7. Kernel Hyperparameter Explanations\n")
            fh.write(
                f"- **`C={self._model_config.C}`**: Regularization parameter. Adjusts the margin width vs. training errors.\n"
            )
            fh.write(
                f"- **`kernel={self._model_config.kernel}`**: Kernel function mapping data to high-dimensional space. "
                "RBF (Radial Basis Function) uses radial distance similarity to handle complex non-linear boundaries.\n"
            )
            fh.write(
                f"- **`gamma={self._model_config.gamma}`**: Kernel coefficient for RBF. It determines the radius of influence "
                "of individual support vectors. 'scale' uses `1 / (n_features * X.var())` as value.\n"
            )

        self._logger.info("Explanation report saved to: %s", report_path)
