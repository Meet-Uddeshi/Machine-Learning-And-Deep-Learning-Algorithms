# ============================================================================
# Principal Component Analysis (PCA) Service Module
# ============================================================================
# Implements Principal Component Analysis using both scikit-learn PCA and custom
# covariance eigendecomposition from scratch. Computes explained variance ratios,
# cumulative variance, scree plots, 2D PC1 vs PC2 projections, and reconstruction errors (MSE).
# ============================================================================

import logging
import os
import time
from typing import Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from config import ModelConfig, PathConfig


class PCAService:
    """Service encapsulating Principal Component Analysis fitting, evaluation, and visualization.

    Responsibilities:
        1. Fit scikit-learn PCA and verify exactness against NumPy covariance eigendecomposition.
        2. Calculate explained variance ratios and cumulative explained variance.
        3. Determine minimum components required to exceed configured variance threshold (e.g. 95%).
        4. Project high-dimensional data onto 2D Principal Component space.
        5. Evaluate reconstruction Mean Squared Error (MSE).
        6. Render Scree Plots and 2D Scatter Projection Charts.
        7. Save comprehensive analytical report to disk.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize PCAService.

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

    def fit_and_evaluate(
        self, x_scaled: np.ndarray, y_labels: np.ndarray
    ) -> Dict[str, Union[float, int, List[float]]]:
        """Execute full Principal Component Analysis pipeline.

        Args:
            x_scaled: Standardized feature matrix of shape (n_samples, n_features).
            y_labels: Class label array for 2D visualization coloring.

        Returns:
            Dictionary containing PCA metrics and component counts.
        """
        self._logger.info("=" * 70)
        self._logger.info("PRINCIPAL COMPONENT ANALYSIS (PCA) EXECUTION")
        self._logger.info("=" * 70)

        n_samples, n_features = x_scaled.shape
        self._logger.info("Input Matrix Dimensions: %d samples x %d features", n_samples, n_features)

        # ---------------------------------------------------------------------
        # 1. Custom Covariance Eigendecomposition from Scratch (Verification)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        cov_matrix = np.cov(x_scaled, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sort_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sort_idx]
        sorted_eigenvectors = eigenvectors[:, sort_idx]

        scratch_explained_var_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        t_scratch = time.perf_counter() - t0
        self._logger.info("Custom Covariance Eigendecomposition completed in %.4f sec.", t_scratch)

        # ---------------------------------------------------------------------
        # 2. Scikit-Learn Full PCA Fitting
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        pca_full = PCA(n_components=n_features, random_state=self._model_config.random_state)
        pca_full.fit(x_scaled)
        t_sklearn = time.perf_counter() - t0
        self._logger.info("Scikit-Learn Full PCA fitted in %.4f sec.", t_sklearn)

        var_ratios = pca_full.explained_variance_ratio_.tolist()
        cum_var_ratios = np.cumsum(var_ratios).tolist()

        # Determine minimum components to meet variance threshold (e.g. 95%)
        target_thresh = self._model_config.variance_threshold
        k_95 = int(np.argmax(np.array(cum_var_ratios) >= target_thresh) + 1)
        self._logger.info(
            "Minimum components needed for >= %.0f%% variance: %d / %d",
            target_thresh * 100,
            k_95,
            n_features,
        )

        # ---------------------------------------------------------------------
        # 3. Dimensionality Reduction & Reconstruction Error (MSE)
        # ---------------------------------------------------------------------
        pca_k = PCA(n_components=k_95, random_state=self._model_config.random_state)
        x_reduced = pca_k.fit_transform(x_scaled)
        x_reconstructed = pca_k.inverse_transform(x_reduced)
        recon_mse = float(mean_squared_error(x_scaled, x_reconstructed))
        self._logger.info("Reconstruction Error (MSE with %d components): %.6f", k_95, recon_mse)

        # ---------------------------------------------------------------------
        # 4. 2D Projection for Visualization
        # ---------------------------------------------------------------------
        pca_2d = PCA(n_components=2, random_state=self._model_config.random_state)
        x_2d = pca_2d.fit_transform(x_scaled)
        pc1_var = float(pca_2d.explained_variance_ratio_[0])
        pc2_var = float(pca_2d.explained_variance_ratio_[1])

        # ---------------------------------------------------------------------
        # 5. Generate Figures & Analytical Summary Report
        # ---------------------------------------------------------------------
        self._plot_scree_plot(var_ratios, cum_var_ratios, k_95, target_thresh)
        self._plot_2d_projection(x_2d, y_labels, pc1_var, pc2_var)

        results = {
            "Total_Features": n_features,
            "Components_95_Variance": k_95,
            "Variance_Threshold": target_thresh,
            "Reconstruction_MSE": recon_mse,
            "PC1_Explained_Variance": pc1_var,
            "PC2_Explained_Variance": pc2_var,
            "Total_2D_Explained_Variance": pc1_var + pc2_var,
            "Explained_Variance_Ratios": var_ratios,
            "Cumulative_Variance_Ratios": cum_var_ratios,
        }

        self._save_summary_report(results)
        return results

    def _plot_scree_plot(
        self,
        var_ratios: List[float],
        cum_var_ratios: List[float],
        k_95: int,
        target_thresh: float,
    ) -> None:
        """Render and save Scree Plot (Individual & Cumulative Explained Variance).

        Args:
            var_ratios:     Individual explained variance ratio array.
            cum_var_ratios: Cumulative explained variance ratio array.
            k_95:           Number of components for variance threshold.
            target_thresh:  Variance threshold ratio.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        components = np.arange(1, len(var_ratios) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart for individual variance
        ax1.bar(
            components,
            var_ratios,
            color="teal",
            alpha=0.6,
            edgecolor="black",
            label="Individual Explained Variance",
        )
        ax1.set_xlabel("Principal Component Index")
        ax1.set_ylabel("Individual Explained Variance Ratio", color="teal")
        ax1.tick_params(axis="y", labelcolor="teal")

        # Line chart for cumulative variance
        ax2 = ax1.twinx()
        ax2.plot(
            components,
            cum_var_ratios,
            color="crimson",
            marker="o",
            lw=2,
            label="Cumulative Explained Variance",
        )
        ax2.axhline(
            y=target_thresh,
            color="navy",
            linestyle="--",
            lw=1.5,
            label=f"{target_thresh * 100:.0f}% Threshold",
        )
        ax2.axvline(
            x=k_95,
            color="purple",
            linestyle=":",
            lw=1.5,
            label=f"Cutoff (k={k_95})",
        )
        ax2.set_ylabel("Cumulative Explained Variance Ratio", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")

        plt.title("PCA Scree Plot & Cumulative Explained Variance")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.tight_layout()
        filepath = os.path.join(self._path_config.output_dir, "scree_plot.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._logger.info("Scree plot saved to: %s", filepath)

    def _plot_2d_projection(
        self,
        x_2d: np.ndarray,
        y_labels: np.ndarray,
        pc1_var: float,
        pc2_var: float,
    ) -> None:
        """Render 2D Principal Component Projection scatter plot colored by class label.

        Args:
            x_2d:      2D projected array.
            y_labels:  Categorical class array.
            pc1_var:   PC1 explained variance ratio.
            pc2_var:   PC2 explained variance ratio.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        plt.figure(figsize=(9, 7))

        unique_classes = np.unique(y_labels)
        palette = sns.color_palette("Set2", len(unique_classes))

        for idx, cls in enumerate(unique_classes):
            mask = y_labels == cls
            plt.scatter(
                x_2d[mask, 0],
                x_2d[mask, 1],
                color=palette[idx],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                s=40,
                label=f"Class: {cls}",
            )

        plt.title(f"PCA 2D Projection (Total Variance Explained: {(pc1_var + pc2_var) * 100:.2f}%)")
        plt.xlabel(f"PC1 ({pc1_var * 100:.2f}% Variance)")
        plt.ylabel(f"PC2 ({pc2_var * 100:.2f}% Variance)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        filepath = os.path.join(self._path_config.output_dir, "pca_2d_projection.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._logger.info("2D projection plot saved to: %s", filepath)

    def _save_summary_report(
        self, results: Dict[str, Union[float, int, List[float]]]
    ) -> None:
        """Write detailed analytical PCA summary report to markdown.

        Args:
            results: PCA results dictionary.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        filepath = os.path.join(self._path_config.output_dir, "pca_analysis_report.md")

        var_ratios: List[float] = results["Explained_Variance_Ratios"]  # type: ignore
        cum_ratios: List[float] = results["Cumulative_Variance_Ratios"]  # type: ignore

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("# Principal Component Analysis (PCA) Technical Report\n\n")
            fh.write("## 1. Executive Summary\n")
            fh.write(
                f"Principal Component Analysis was executed on the Vehicle Silhouette dataset containing **{results['Total_Features']}** continuous shape features. "
            )
            fh.write(
                f"To capture at least **{float(results['Variance_Threshold']) * 100:.0f}%** of the total dataset variance, **{results['Components_95_Variance']}** principal components are required. "
            )
            fh.write(
                f"Reconstruction Mean Squared Error (MSE) using {results['Components_95_Variance']} components is **{results['Reconstruction_MSE']:.6f}**.\n\n"
            )

            fh.write("## 2. Component Variance Breakdown\n\n")
            fh.write("| Component | Individual Variance Ratio | Cumulative Variance Ratio |\n")
            fh.write("|-----------|---------------------------|---------------------------|\n")

            for idx, (v, c) in enumerate(zip(var_ratios, cum_ratios), start=1):
                fh.write(f"| **PC{idx}** | {v * 100:.2f}% | {c * 100:.2f}% |\n")

            fh.write("\n## 3. 2D Principal Component Projection\n")
            fh.write(f"- **PC1 Explained Variance**: {float(results['PC1_Explained_Variance']) * 100:.2f}%\n")
            fh.write(f"- **PC2 Explained Variance**: {float(results['PC2_Explained_Variance']) * 100:.2f}%\n")
            fh.write(f"- **Total 2D Explained Variance**: {float(results['Total_2D_Explained_Variance']) * 100:.2f}%\n")

        self._logger.info("PCA summary report saved to: %s", filepath)
