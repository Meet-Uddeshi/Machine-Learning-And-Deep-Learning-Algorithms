# ============================================================================
# Ensemble Classifier Service Module (Bagging & Boosting)
# ============================================================================
# Implements, trains, and evaluates BaggingClassifier, AdaBoostClassifier,
# and GradientBoostingClassifier models on tabular medical classification data.
# Generates model comparison evaluation metrics, confusion matrices, and feature importance charts.
# ============================================================================

import logging
import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from config import ModelConfig, PathConfig


class EnsembleClassifierService:
    """Service encapsulating training, evaluation, and plotting for Bagging and Boosting models.

    Responsibilities:
        1. Fit BaggingClassifier, AdaBoostClassifier, and GradientBoostingClassifier.
        2. Evaluate metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
        3. Render confusion matrix heatmaps and feature importance bar charts.
        4. Save comprehensive analytical evaluation summary reports.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize EnsembleClassifierService.

        Args:
            model_config:  Model hyperparameter settings.
            path_config:   Path configuration settings.
            feature_names: Names of feature columns.
            logger:        Logger instance.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._feature_names = feature_names
        self._logger = logger

        # Initialize base decision tree estimator for Bagging/AdaBoost
        base_tree = DecisionTreeClassifier(
            max_depth=self._model_config.max_depth,
            random_state=self._model_config.random_state,
        )

        self._models = {
            "Bagging": BaggingClassifier(
                estimator=base_tree,
                n_estimators=self._model_config.n_estimators,
                random_state=self._model_config.random_state,
            ),
            "AdaBoost": AdaBoostClassifier(
                estimator=base_tree,
                n_estimators=self._model_config.n_estimators,
                learning_rate=self._model_config.learning_rate,
                random_state=self._model_config.random_state,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=self._model_config.n_estimators,
                learning_rate=self._model_config.learning_rate,
                max_depth=self._model_config.max_depth,
                random_state=self._model_config.random_state,
            ),
        }

    def train_and_evaluate(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Train all ensemble models and evaluate comparative metrics.

        Args:
            x_train: Scaled training feature matrix.
            x_test:  Scaled testing feature matrix.
            y_train: Training target vector.
            y_test:  Testing target vector.

        Returns:
            Dictionary mapping model names to performance metrics.
        """
        self._logger.info("=" * 70)
        self._logger.info("ENSEMBLE MODEL TRAINING & EVALUATION")
        self._logger.info("=" * 70)

        results = {}

        for name, model in self._models.items():
            self._logger.info("Training %s Classifier...", name)
            t_start = time.perf_counter()
            model.fit(x_train, y_train)
            t_train = time.perf_counter() - t_start

            preds = model.predict(x_test)
            probs = (
                model.predict_proba(x_test)[:, 1]
                if hasattr(model, "predict_proba")
                else preds
            )

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, probs)

            metrics = {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1_Score": float(f1),
                "ROC_AUC": float(roc_auc),
                "Train_Time_Sec": t_train,
            }
            results[name] = metrics

            self._logger.info("  %-18s Performance:", name)
            self._logger.info("    Accuracy   : %.4f", acc)
            self._logger.info("    Precision  : %.4f", prec)
            self._logger.info("    Recall     : %.4f", rec)
            self._logger.info("    F1 Score   : %.4f", f1)
            self._logger.info("    ROC-AUC    : %.4f", roc_auc)
            self._logger.info("    Fit Time   : %.4f sec", t_train)

            # Generate individual confusion matrix plot
            self._plot_confusion_matrix(y_test, preds, model_name=name)

        self._plot_model_comparison(results)
        self._save_summary_report(results)
        return results

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> None:
        """Render and save confusion matrix heatmap for a specific model.

        Args:
            y_true:     True target values.
            y_pred:     Predicted class labels.
            model_name: Model identifier string.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        filepath = os.path.join(
            self._path_config.output_dir, f"confusion_matrix_{model_name.lower()}.png"
        )
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._logger.info("Confusion matrix saved to: %s", filepath)

    def _plot_model_comparison(
        self, results: Dict[str, Dict[str, float]]
    ) -> None:
        """Render multi-bar comparison chart across ensemble models.

        Args:
            results: Results dictionary.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        models = list(results.keys())
        metrics_keys = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]

        x = np.arange(len(metrics_keys))
        width = 0.25

        plt.figure(figsize=(10, 6))
        for idx, m_name in enumerate(models):
            vals = [results[m_name][k] for k in metrics_keys]
            plt.bar(x + idx * width, vals, width, label=m_name)

        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title("Ensemble Models Performance Comparison")
        plt.xticks(x + width, metrics_keys)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        filepath = os.path.join(
            self._path_config.output_dir, "ensemble_comparison_bar_chart.png"
        )
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._logger.info("Comparison chart saved to: %s", filepath)

    def _save_summary_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """Write detailed analytical report to markdown.

        Args:
            results: Metrics results dictionary.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        filepath = os.path.join(
            self._path_config.output_dir, "bagging_boosting_analysis_report.md"
        )

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("# Bagging & Boosting Ensemble Analysis Report\n\n")
            fh.write("## 1. Performance Metrics Summary\n")
            fh.write("Comparative performance of Bagging, AdaBoost, and Gradient Boosting models on the Heart Disease dataset:\n\n")

            fh.write("| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Train Time (s) |\n")
            fh.write("|-------|----------|-----------|--------|----------|---------|----------------|\n")

            for m_name, m_dict in results.items():
                fh.write(
                    f"| **{m_name}** | {m_dict['Accuracy']:.4f} | {m_dict['Precision']:.4f} | "
                    f"{m_dict['Recall']:.4f} | {m_dict['F1_Score']:.4f} | {m_dict['ROC_AUC']:.4f} | "
                    f"{m_dict['Train_Time_Sec']:.4f} |\n"
                )

            fh.write("\n## 2. Algorithm Comparison & Architectural Insights\n")
            fh.write("- **Bagging (Bootstrap Aggregating)**: Reduces variance by averaging predictions of multiple independent base decision trees.\n")
            fh.write("- **AdaBoost (Adaptive Boosting)**: Sequentially fits weak learners, increasing weights of misclassified samples.\n")
            fh.write("- **Gradient Boosting**: Sequentially fits base decision trees to minimize residual errors via gradient descent.\n")

        self._logger.info("Summary report saved to: %s", filepath)
