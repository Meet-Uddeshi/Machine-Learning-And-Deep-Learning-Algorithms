# ============================================================================
# PCA Service for Unsupervised Learning Pipeline
# ============================================================================
# Owns the PCA model lifecycle: fitting, explained variance calculation,
# scree plot generation, feature loadings analysis, 2D projections, and
# technical report generation.
# ============================================================================

import logging
import os
import time
from typing import List, Optional

# Non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from config import ModelConfig, PathConfig


class PCAService:
    """Service encapsulating Principal Component Analysis.

    Responsibilities:
        1. Fit PCA model on standardised features.
        2. Compute explained variance ratios, cumulative variance, and singular values.
        3. Extract feature loadings (eigenvectors) for interpretability.
        4. Project high-dimensional data onto lower-dimensional subspaces.
        5. Visualise scree plots, loadings heatmaps, and 2D class projections.
        6. Write structured text and markdown analysis reports.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the PCA service.

        Args:
            model_config:  PCA hyperparameters.
            path_config:   Path settings for saving outputs.
            label_names:   Vehicle class label names.
            feature_names: Feature column names.
            logger:        Logger instance.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger
        self._model = self._build_model()

    # -- Public workflow methods ---------------------------------------------

    def train_and_evaluate(
        self,
        x_train_scaled: np.ndarray,
        y_train: np.ndarray,
        x_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Fit PCA on training data, transform test set, and generate reports.

        Args:
            x_train_scaled: Standardised training feature matrix.
            y_train:        Training ground-truth target vector.
            x_test_scaled:  Standardised test feature matrix.
            y_test:         Test ground-truth target vector.

        Returns:
            Dictionary containing PCA results and transformed features.
        """
        self._logger.info("=" * 70)
        self._logger.info("PCA FIT & DECOMPOSITION")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        start_time = time.perf_counter()
        x_train_pca = self._model.fit_transform(x_train_scaled)
        x_test_pca = self._model.transform(x_test_scaled)
        elapsed = time.perf_counter() - start_time

        self._logger.info("PCA decomposition completed in %.3f seconds.", elapsed)
        self._logger.info("Number of components retained: %d", self._model.n_components_)

        # Calculate variance metrics
        exp_var_ratio = self._model.explained_variance_ratio_
        cum_exp_var = np.cumsum(exp_var_ratio)
        eigenvalues = self._model.explained_variance_
        singular_values = self._model.singular_values_
        loadings = self._model.components_

        results = {
            "n_components": self._model.n_components_,
            "explained_variance_ratio": exp_var_ratio,
            "cumulative_variance_ratio": cum_exp_var,
            "eigenvalues": eigenvalues,
            "singular_values": singular_values,
            "loadings": loadings,
            "x_train_pca": x_train_pca,
            "x_test_pca": x_test_pca,
        }

        self._log_evaluation(results)
        self._save_results(results)
        self._generate_plots(results, y_train)
        self._save_analysis(results)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self) -> PCA:
        """Construct PCA estimator from configuration.

        Returns:
            An unfitted PCA estimator.
        """
        self._logger.info("Instantiating PCA Estimator (scikit-learn).")
        return PCA(
            n_components=self._model_config.n_components,
            svd_solver=self._model_config.svd_solver,
            whiten=self._model_config.whiten,
            random_state=self._model_config.random_state,
        )

    def _log_hyperparameters(self) -> None:
        """Log model configuration settings."""
        self._logger.info("PCA Hyperparameters:")
        self._logger.info("  n_components: %s", self._model_config.n_components)
        self._logger.info("  svd_solver  : %s", self._model_config.svd_solver)
        self._logger.info("  whiten      : %s", self._model_config.whiten)
        self._logger.info("  random_state: %d", self._model_config.random_state)

    def _log_evaluation(self, results: dict) -> None:
        """Log key PCA metrics to console."""
        self._logger.info("-" * 70)
        self._logger.info("PCA EXPLAINED VARIANCE SUMMARY:")
        self._logger.info("%-10s  %-20s  %-20s", "Component", "Explained Var Ratio", "Cumulative Var Ratio")
        self._logger.info("-" * 70)
        for idx in range(min(10, results["n_components"])):
            var_r = results["explained_variance_ratio"][idx]
            cum_r = results["cumulative_variance_ratio"][idx]
            self._logger.info(
                "PC%-8d  %-20.4f (%.2f%%)  %-20.4f (%.2f%%)",
                idx + 1, var_r, var_r * 100, cum_r, cum_r * 100
            )
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Save detailed PCA stats and loadings to pca_results.txt."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        results_path = os.path.join(self._path_config.output_dir, "pca_results.txt")

        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("PRINCIPAL COMPONENT ANALYSIS (PCA) RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  n_components: {self._model_config.n_components}\n")
            fh.write(f"  svd_solver  : {self._model_config.svd_solver}\n")
            fh.write(f"  whiten      : {self._model_config.whiten}\n")
            fh.write(f"  random_state: {self._model_config.random_state}\n\n")

            fh.write("EXPLAINED VARIANCE BREAKDOWN\n")
            fh.write("-" * 65 + "\n")
            fh.write(f"{'PC':<5} | {'Eigenvalue':<12} | {'Singular Value':<15} | {'Var Ratio':<12} | {'Cum Var Ratio':<14}\n")
            fh.write("-" * 65 + "\n")
            for idx in range(results["n_components"]):
                eig = results["eigenvalues"][idx]
                sing = results["singular_values"][idx]
                v_ratio = results["explained_variance_ratio"][idx]
                c_ratio = results["cumulative_variance_ratio"][idx]
                fh.write(f"PC{idx+1:<3} | {eig:<12.4f} | {sing:<15.4f} | {v_ratio:<12.4f} | {c_ratio:<14.4f}\n")
            fh.write("\n")

            fh.write("FEATURE LOADINGS MATRIX (EIGENVECTORS)\n")
            fh.write("-" * 40 + "\n")
            loadings = results["loadings"]
            # Show first 5 components or n_components
            n_show = min(5, results["n_components"])
            header = f"{'Feature':<30}"
            for c in range(n_show):
                header += f"  PC{c+1:<8}"
            fh.write(header + "\n")
            fh.write("-" * len(header) + "\n")

            for f_idx, feat in enumerate(self._feature_names):
                row_str = f"{feat:<30}"
                for c in range(n_show):
                    row_str += f"  {loadings[c, f_idx]:<10.4f}"
                fh.write(row_str + "\n")
            fh.write("=" * 70 + "\n")

        self._logger.info("PCA results saved to: %s", results_path)

    def _generate_plots(self, results: dict, y_train: np.ndarray) -> None:
        """Generate scree plot, 2D PCA projection scatter, and loadings heatmap."""
        self._logger.info("Generating PCA visualization plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        n_comp = results["n_components"]
        comp_indices = np.arange(1, n_comp + 1)
        exp_var = results["explained_variance_ratio"] * 100
        cum_var = results["cumulative_variance_ratio"] * 100

        # Plot 1: Scree Plot (Individual & Cumulative Explained Variance)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Individual Explained Variance (%)', color=color)
        bars = ax1.bar(comp_indices, exp_var, color=color, alpha=0.6, label='Individual Variance')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(comp_indices)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Cumulative Explained Variance (%)', color=color)
        line = ax2.plot(comp_indices, cum_var, color=color, marker='o', linewidth=2, label='Cumulative Variance')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90% Variance Threshold')

        plt.title('PCA Scree Plot -- Individual & Cumulative Explained Variance')
        fig.tight_layout()
        scree_path = os.path.join(self._path_config.output_dir, "scree_plot.png")
        plt.savefig(scree_path, dpi=300)
        plt.close()

        # Plot 2: 2D PCA Projection colored by Vehicle Class
        x_pca = results["x_train_pca"]
        plt.figure(figsize=(10, 7))

        unique_labels = np.unique(y_train)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for idx, lbl in enumerate(unique_labels):
            mask = y_train == lbl
            class_name = self._label_names[lbl] if lbl < len(self._label_names) else str(lbl)
            plt.scatter(
                x_pca[mask, 0],
                x_pca[mask, 1],
                color=colors[idx],
                label=f"Class: {class_name}",
                alpha=0.7,
                edgecolors="w",
                linewidth=0.5
            )

        pc1_var = exp_var[0]
        pc2_var = exp_var[1] if n_comp > 1 else 0.0
        plt.xlabel(f"Principal Component 1 ({pc1_var:.2f}% Variance)")
        plt.ylabel(f"Principal Component 2 ({pc2_var:.2f}% Variance)")
        plt.title("Vehicle Silhouettes Dataset -- 2D PCA Projection (PC1 vs PC2)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        proj_path = os.path.join(self._path_config.output_dir, "pca_2d_projection.png")
        plt.savefig(proj_path, dpi=300)
        plt.close()

        # Plot 3: Loadings Heatmap
        n_show_comp = min(8, n_comp)
        loadings_df = pd.DataFrame(
            results["loadings"][:n_show_comp, :],
            index=[f"PC{i+1}" for i in range(n_show_comp)],
            columns=self._feature_names
        )

        plt.figure(figsize=(14, 6))
        sns.heatmap(
            loadings_df,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar_kws={'label': 'Loading Coefficient'}
        )
        plt.title("PCA Feature Loadings Heatmap (Eigenvectors)")
        plt.xlabel("Original Features")
        plt.ylabel("Principal Components")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        heatmap_path = os.path.join(self._path_config.output_dir, "loadings_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        self._logger.info("Evaluation plots saved successfully.")

    def _save_analysis(self, results: dict) -> None:
        """Write technical markdown explanation report pca_analysis.md."""
        report_path = os.path.join(self._path_config.output_dir, "pca_analysis.md")

        exp_var = results["explained_variance_ratio"] * 100
        cum_var = results["cumulative_variance_ratio"] * 100

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Principal Component Analysis (PCA) Technical Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                "This report details the Principal Component Analysis (PCA) dimensionality reduction "
                "performed on the Vehicle Silhouettes dataset (`pca.csv`). "
            )
            fh.write(
                f"The original feature space of **{len(self._feature_names)}** continuous physical attributes "
                f"was decomposed into orthogonal principal components. "
                f"The top 2 principal components explain **{cum_var[1]:.2f}%** of the total variance, "
                f"while retaining **{cum_var[min(5, len(cum_var)-1)]:.2f}%** across the first 6 components.\n\n"
            )

            fh.write("## 2. Model Configuration\n\n")
            fh.write("| Hyperparameter | Value |\n")
            fh.write("|----------------|-------|\n")
            fh.write(f"| `n_components` | `{self._model_config.n_components}` |\n")
            fh.write(f"| `svd_solver` | `{self._model_config.svd_solver}` |\n")
            fh.write(f"| `whiten` | `{self._model_config.whiten}` |\n")
            fh.write(f"| `random_state` | `{self._model_config.random_state}` |\n\n")

            fh.write("## 3. Explained Variance Spectrum\n\n")
            fh.write("| Component | Eigenvalue | Variance Explained (%) | Cumulative Variance (%) |\n")
            fh.write("|-----------|------------|------------------------|-------------------------|\n")
            for idx in range(min(10, results["n_components"])):
                eig = results["eigenvalues"][idx]
                v_r = exp_var[idx]
                c_r = cum_var[idx]
                fh.write(f"| PC{idx+1} | {eig:.4f} | {v_r:.2f}% | {c_r:.2f}% |\n")
            fh.write("\n")

            fh.write("### Theoretical Interpretation of PCA Metrics\n")
            fh.write(
                "- **Eigenvalues**: Represent the total variance captured along each orthogonal eigenvector direction.\n"
                "- **Explained Variance Ratio**: Proportion of the dataset's total variance accounted for by a specific principal component.\n"
                "- **Scree Plot Curve**: Identifies the 'elbow point' where additional components yield diminishing returns in variance retention.\n\n"
            )

            fh.write("## 4. Top Feature Loadings Analysis\n\n")
            fh.write(
                "Feature loadings represent the coefficients of the linear combination of original features that form each component:\n\n"
            )
            loadings = results["loadings"]
            fh.write("| Feature | PC1 Loading | PC2 Loading | PC3 Loading |\n")
            fh.write("|---------|-------------|-------------|-------------|\n")
            for f_idx, feat in enumerate(self._feature_names):
                fh.write(
                    f"| {feat} | {loadings[0, f_idx]:.4f} | "
                    f"{loadings[1, f_idx]:.4f} | {loadings[2, f_idx]:.4f} |\n"
                )
            fh.write("\n")

            fh.write("## 5. Output Artifacts Summary\n\n")
            fh.write("| File | Description |\n")
            fh.write("|------|-------------|\n")
            fh.write("| `pca_results.txt` | Numerical breakdown of eigenvalues, variance ratios, and full loadings matrix |\n")
            fh.write("| `pca_analysis.md` | Technical markdown report on dimensionality reduction |\n")
            fh.write("| `scree_plot.png` | Individual and cumulative variance scree plot curve |\n")
            fh.write("| `pca_2d_projection.png` | 2D scatter plot of PC1 vs PC2 colored by vehicle class |\n")
            fh.write("| `loadings_heatmap.png` | Heatmap visualization of feature loadings across components |\n")

        self._logger.info("PCA technical report saved to: %s", report_path)
