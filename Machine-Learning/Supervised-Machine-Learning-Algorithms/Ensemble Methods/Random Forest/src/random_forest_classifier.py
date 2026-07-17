# ============================================================================
# Random Forest Classifier Service for Classification Pipeline
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
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from config import ModelConfig, PathConfig


class RandomForestClassifierService:
    """Service encapsulating Random Forest classification.

    Responsibilities:
        1. Train a scikit-learn RandomForestClassifier with hyperparameters.
        2. Generate predictions on unseen test data.
        3. Compute and log a comprehensive evaluation report.
        4. Plot the confusion matrix, performance metrics, MDI feature importance,
           and a representative single estimator decision tree structure.
        5. Persist all output artifacts (reports, tree rules, and graphs).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the Random Forest classifier service.

        Args:
            model_config:  Random Forest hyperparameters.
            path_config:   Path settings for saving output artifacts.
            label_names:   Human-readable class labels.
            feature_names: Feature column names.
            logger:        Logger instance.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger
        self._model: RandomForestClassifier = self._build_model()

    # -- Public workflow methods ---------------------------------------------

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Random Forest classifier on the training data.

        Args:
            x_train: Training feature matrix (n_samples, n_features).
            y_train: Training target vector (n_samples,).
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        self._logger.info(
            "Training Random Forest on %d samples with %d features...",
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info("Training completed in %.3f seconds.", elapsed)
        
        # Log out-of-bag validation score if enabled
        if self._model_config.oob_score and self._model_config.bootstrap:
            self._logger.info("Out-of-Bag (OOB) Generalization Score: %.4f", self._model.oob_score_)

        # Log some statistics about the forest estimators
        depths = [estimator.get_depth() for estimator in self._model.estimators_]
        leaves = [estimator.get_n_leaves() for estimator in self._model.estimators_]
        self._logger.info(
            "Forest Estimators Summary: "
            "Mean Depth: %.1f (Min: %d, Max: %d) | Mean Leaves: %.1f (Min: %d, Max: %d)",
            np.mean(depths), min(depths), max(depths),
            np.mean(leaves), min(leaves), max(leaves)
        )

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Predict on the test set and compute evaluation metrics.

        Args:
            x_test: Test feature matrix.
            y_test: True test labels.

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

        results = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "predictions": predictions,
            "feature_importances": self._model.feature_importances_,
        }

        self._log_evaluation(results)
        self._save_results(results)
        self._save_forest_details()
        self._generate_plots(results, y_test)
        self._save_analysis(results, y_test)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self) -> RandomForestClassifier:
        """Construct the RandomForestClassifier estimator from configuration.

        Returns:
            An unfitted RandomForestClassifier.
        """
        self._logger.info("Instantiating RandomForestClassifier (scikit-learn).")
        return RandomForestClassifier(
            n_estimators=self._model_config.n_estimators,
            criterion=self._model_config.criterion,
            max_depth=self._model_config.max_depth,
            min_samples_split=self._model_config.min_samples_split,
            min_samples_leaf=self._model_config.min_samples_leaf,
            max_features=self._model_config.max_features,
            bootstrap=self._model_config.bootstrap,
            oob_score=self._model_config.oob_score,
            random_state=self._model_config.random_state,
            n_jobs=-1,  # Use all CPU cores for parallel forest construction
        )

    def _log_hyperparameters(self) -> None:
        """Log model hyperparameters."""
        self._logger.info("Random Forest Hyperparameters:")
        self._logger.info("  n_estimators      : %d", self._model_config.n_estimators)
        self._logger.info("  criterion         : %s", self._model_config.criterion)
        self._logger.info("  max_depth         : %s", self._model_config.max_depth)
        self._logger.info("  min_samples_split : %d", self._model_config.min_samples_split)
        self._logger.info("  min_samples_leaf  : %d", self._model_config.min_samples_leaf)
        self._logger.info("  max_features      : %s", self._model_config.max_features)
        self._logger.info("  bootstrap         : %s", self._model_config.bootstrap)
        self._logger.info("  oob_score         : %s", self._model_config.oob_score)
        self._logger.info("  random_state      : %d", self._model_config.random_state)

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
            header += "  %-15s" % name
        self._logger.info(header)

        for idx, row in enumerate(results["confusion_matrix"]):
            row_str = "%-15s" % self._label_names[idx]
            for val in row:
                row_str += "  %-15d" % val
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
            fh.write("RANDOM FOREST CLASSIFICATION RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  n_estimators      : {self._model_config.n_estimators}\n")
            fh.write(f"  criterion         : {self._model_config.criterion}\n")
            fh.write(f"  max_depth         : {self._model_config.max_depth}\n")
            fh.write(f"  min_samples_split : {self._model_config.min_samples_split}\n")
            fh.write(f"  min_samples_leaf  : {self._model_config.min_samples_leaf}\n")
            fh.write(f"  max_features      : {self._model_config.max_features}\n")
            fh.write(f"  bootstrap         : {self._model_config.bootstrap}\n")
            fh.write(f"  oob_score         : {self._model_config.oob_score}\n")
            fh.write(f"  random_state      : {self._model_config.random_state}\n\n")

            fh.write("FOREST SUMMARY\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Total Estimators  : {len(self._model.estimators_)}\n")
            if self._model_config.oob_score and self._model_config.bootstrap:
                fh.write(f"  OOB Score         : {self._model.oob_score_:.4f}\n")
            fh.write("\n")

            fh.write("FEATURES USED\n")
            fh.write("-" * 40 + "\n")
            for i, name in enumerate(self._feature_names, start=1):
                fh.write(f"  {i:2d}. {name}\n")
            fh.write("\n")

            # Mean Decrease Impurity Feature importances
            fh.write("FEATURE IMPORTANCES (Mean Decrease in Impurity)\n")
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
                header += f"  {name:<15}"
            fh.write(header + "\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                row_str = f"{self._label_names[idx]:<15}"
                for val in row:
                    row_str += f"  {val:<15}"
                fh.write(row_str + "\n")
            fh.write("\n")

            fh.write("PER-CLASS CLASSIFICATION REPORT\n")
            fh.write("-" * 40 + "\n")
            fh.write(results["classification_report"])
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Results saved to: %s", output_path)

    def _save_forest_details(self) -> None:
        """Export details of individual trees in the forest.

        This lists tree depths and leaf counts for each estimator in the ensemble.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        details_path = os.path.join(
            self._path_config.output_dir, "random_forest_details.txt"
        )

        depths = [estimator.get_depth() for estimator in self._model.estimators_]
        leaves = [estimator.get_n_leaves() for estimator in self._model.estimators_]

        with open(details_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("RANDOM FOREST ESTIMATOR DETAILED METRICS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("FOREST-WIDE METRICS:\n")
            fh.write(f"  Estimator Count           : {len(self._model.estimators_)}\n")
            fh.write(f"  Mean Estimator Depth      : {np.mean(depths):.2f}\n")
            fh.write(f"  Max Estimator Depth       : {max(depths)}\n")
            fh.write(f"  Min Estimator Depth       : {min(depths)}\n")
            fh.write(f"  Mean Estimator Leaves     : {np.mean(leaves):.2f}\n")
            fh.write(f"  Max Estimator Leaves      : {max(leaves)}\n")
            fh.write(f"  Min Estimator Leaves      : {min(leaves)}\n\n")

            fh.write("ESTIMATOR-BY-ESTIMATOR METRICS:\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  {'Estimator ID':<15} | {'Depth':<8} | {'Leaves Count':<12}\n")
            fh.write("-" * 40 + "\n")
            for idx, (depth, leaf) in enumerate(zip(depths, leaves)):
                fh.write(f"  {idx:<15d} | {depth:<8d} | {leaf:<12d}\n")
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Forest detailed metrics saved to: %s", details_path)

    def _generate_plots(self, results: dict, y_test: np.ndarray) -> None:
        """Generate evaluation plots.

        Plots:
            1. Confusion Matrix Heatmap
            2. Per-Class Metrics Bar Chart
            3. Mean Decrease in Impurity (MDI) Feature Importance
            4. Tree Diagram representing the first decision tree estimator.
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
            cmap="Oranges",
            xticklabels=self._label_names,
            yticklabels=self._label_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Random Forest -- Confusion Matrix Heatmap")
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
        sns.barplot(x="Class", y="Value", hue="Metric", data=df_metrics, palette="Oranges")
        plt.ylim(0, 1.05)
        plt.title("Random Forest -- Per-Class Classification Metrics")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(loc="lower right")
        plt.tight_layout()
        report_fig_path = os.path.join(
            self._path_config.output_dir, "classification_report.png"
        )
        plt.savefig(report_fig_path, dpi=300)
        plt.close()

        # 3. MDI Feature Importance Plot
        importances = results["feature_importances"]
        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [self._feature_names[i] for i in sorted_idx]
        sorted_imp = importances[sorted_idx]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=sorted_imp, y=sorted_names, hue=sorted_names, palette="plasma", legend=False)
        plt.xlabel("Feature Importance (Mean Decrease in Impurity)")
        plt.ylabel("Feature")
        plt.title("Random Forest -- Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(
            self._path_config.output_dir, "feature_importance.png"
        )
        plt.savefig(fi_path, dpi=300)
        plt.close()

        # 4. First Tree Estimator Structure Diagram
        plt.figure(figsize=(24, 12))
        plot_tree(
            self._model.estimators_[0],
            feature_names=self._feature_names,
            class_names=self._label_names,
            filled=True,
            rounded=True,
            max_depth=3,  # Capped at depth 3 for layout legibility
            fontsize=9,
            impurity=True,
            proportion=False,
        )
        plt.title(
            f"Random Forest -- Representative Tree Structure (Estimator [0], capped at Depth 3 of {self._model.estimators_[0].get_depth()} total)",
            fontsize=14,
        )
        plt.tight_layout()
        tree_fig_path = os.path.join(
            self._path_config.output_dir, "random_forest_tree_structure.png"
        )
        plt.savefig(tree_fig_path, dpi=150)
        plt.close()
        self._logger.info("Forest tree estimator structure saved to: %s", tree_fig_path)

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
            fh.write("# Random Forest Classification Analysis Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                "This report provides a formal explanation of the Random Forest "
                "model's performance on predicting product categories based on sales transactions. "
            )
            fh.write(
                f"The model achieved an overall accuracy of "
                f"**{results['accuracy'] * 100:.2f}%** on the test dataset. "
            )
            fh.write(
                "Below is a detailed analysis of the performance metrics, "
                "confusion matrix, and Gini-based feature importances from a classification perspective.\n\n"
            )

            fh.write("## 2. Model Configuration\n\n")
            fh.write("| Hyperparameter | Value |\n")
            fh.write("|----------------|-------|\n")
            fh.write(f"| `n_estimators` | `{self._model_config.n_estimators}` |\n")
            fh.write(f"| `criterion` | `{self._model_config.criterion}` |\n")
            fh.write(f"| `max_depth` | `{self._model_config.max_depth}` |\n")
            fh.write(f"| `min_samples_split` | `{self._model_config.min_samples_split}` |\n")
            fh.write(f"| `min_samples_leaf` | `{self._model_config.min_samples_leaf}` |\n")
            fh.write(f"| `max_features` | `{self._model_config.max_features}` |\n")
            fh.write(f"| `bootstrap` | `{self._model_config.bootstrap}` |\n")
            fh.write(f"| `oob_score` | `{self._model_config.oob_score}` |\n\n")

            fh.write("### Fitted Ensemble Properties\n\n")
            depths = [estimator.get_depth() for estimator in self._model.estimators_]
            leaves = [estimator.get_n_leaves() for estimator in self._model.estimators_]
            fh.write(f"- **Total trees (estimators)**: {len(self._model.estimators_)}\n")
            fh.write(f"- **Mean tree depth**: {np.mean(depths):.2f} (Range: {min(depths)} - {max(depths)})\n")
            fh.write(f"- **Mean leaf count**: {np.mean(leaves):.2f} (Range: {min(leaves)} - {max(leaves)})\n")
            if self._model_config.oob_score and self._model_config.bootstrap:
                fh.write(f"- **Out-of-Bag (OOB) Generalization Score**: {self._model.oob_score_:.4f}\n")
            fh.write("\n")

            fh.write("## 3. Evaluation Metrics Breakdown\n")
            fh.write(
                "Precision, Recall, and F1-Score are analysed per class:\n\n"
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
                "- **Precision**: Proportion of correct positive predictions. "
                "High precision indicates that the class was rarely mispredicted.\n"
            )
            fh.write(
                "- **Recall**: Proportion of actual positive samples identified. "
                "High recall indicates that most samples of the class were caught.\n"
            )
            fh.write(
                "- **F1-Score**: Harmonic mean of precision and recall. Useful for overall diagnostics "
                "balancing precision and recall bounds.\n\n"
            )

            fh.write("## 4. Confusion Matrix Analysis\n\n")
            fh.write(
                "The confusion matrix (visualized in `confusion_matrix.png`) "
                "displays actual vs predicted counts:\n\n"
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

            fh.write("### Why Does Random Forest Generalize Better than Decision Trees?\n")
            fh.write(
                "A single Decision Tree splits on features to minimize node impurity. However, "
                "a single tree is prone to high variance and overfitting because it can capture local "
                "noise. Random Forest reduces this variance through **Bagging** (Bootstrap Aggregation) "
                "and **Feature Subspace Sampling** (choosing random features at each split node). "
                "By training many independent trees and averaging their predictions (voting), "
                "individual high-variance errors cancel out, resulting in a robust model that generalizes "
                "substantially better on unseen data.\n\n"
            )

            fh.write("## 5. Feature Importances\n")
            fh.write(
                "Feature importance in Random Forest measures the Mean Decrease in Impurity (MDI). "
                "For each feature, it represents the total reduction in criterion (Gini or Entropy) "
                "brought by splits on that feature, averaged across all trees in the forest. "
                "Higher values indicate that the feature was selected frequently and produced pure splits.\n\n"
            )
            fh.write("| Rank | Feature | Importance (MDI) |\n")
            fh.write("|------|---------|------------------|\n")
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
            fh.write("| `random_forest_details.txt` | Detailed estimator stats (leaf counts, individual tree depths) |\n")
            fh.write("| `confusion_matrix.png` | Confusion matrix heatmap |\n")
            fh.write("| `classification_report.png` | Precision/Recall/F1 grouped bar chart |\n")
            fh.write("| `feature_importance.png` | Gini MDI feature importance bar chart |\n")
            fh.write("| `random_forest_tree_structure.png` | Visual diagram of the first decision tree in the ensemble |\n")
            fh.write("| `random_forest_sales.log` | Pipeline log file |\n\n")

        self._logger.info("Explanation report saved to: %s", report_path)
