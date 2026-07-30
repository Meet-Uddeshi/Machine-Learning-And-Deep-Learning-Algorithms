# ============================================================================
# Clustering Algorithms Service Module
# ============================================================================
# Implements K-Means, DBSCAN, and Agglomerative Hierarchical Clustering algorithms.
# Evaluates unsupervised clustering validation metrics (Silhouette Score, Calinski-Harabasz Index,
# Davies-Bouldin Index), plots elbow curves and 2D PCA cluster visualizations.
# ============================================================================

import logging
import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from config import ModelConfig, PathConfig


class ClusteringService:
    """Service encapsulating unsupervised clustering algorithms and evaluation metrics.

    Responsibilities:
        1. Fit K-Means, DBSCAN, and Agglomerative Clustering algorithms.
        2. Compute validation metrics: Silhouette Score, Calinski-Harabasz, Davies-Bouldin.
        3. Evaluate K-Means Elbow Curve over a range of cluster values k.
        4. Render 2D PCA cluster scatter plots for cluster visualization.
        5. Write comprehensive analytical clustering summary reports.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize ClusteringService.

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

    def fit_and_evaluate_all(
        self, x_scaled: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Fit K-Means, DBSCAN, and Agglomerative Clustering algorithms and evaluate metrics.

        Args:
            x_scaled: Standardized feature matrix.

        Returns:
            Dictionary mapping algorithm names to evaluation metrics.
        """
        self._logger.info("=" * 70)
        self._logger.info("UNSUPERVISED CLUSTERING TRAINING & EVALUATION")
        self._logger.info("=" * 70)

        # Plot K-Means Elbow Curve first
        self._plot_elbow_curve(x_scaled)

        # Reduce dimensionality to 2D for plotting
        pca = PCA(n_components=2, random_state=self._model_config.random_state)
        x_2d = pca.fit_transform(x_scaled)

        models = {
            "K-Means": KMeans(
                n_clusters=self._model_config.n_clusters,
                random_state=self._model_config.random_state,
                n_init=10,
            ),
            "Agglomerative": AgglomerativeClustering(
                n_clusters=self._model_config.n_clusters
            ),
            "DBSCAN": DBSCAN(
                eps=self._model_config.dbscan_eps,
                min_samples=self._model_config.dbscan_min_samples,
            ),
        }

        results = {}

        for name, model in models.items():
            self._logger.info("Fitting %s...", name)
            t_start = time.perf_counter()
            labels = model.fit_predict(x_scaled)
            t_elapsed = time.perf_counter() - t_start

            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            self._logger.info("  %s found %d clusters.", name, n_clusters_found)

            if n_clusters_found > 1:
                sil = float(silhouette_score(x_scaled, labels))
                ch = float(calinski_harabasz_score(x_scaled, labels))
                db = float(davies_bouldin_score(x_scaled, labels))
            else:
                sil, ch, db = -1.0, 0.0, float("inf")

            metrics = {
                "Clusters_Found": float(n_clusters_found),
                "Silhouette_Score": sil,
                "Calinski_Harabasz": ch,
                "Davies_Bouldin": db,
                "Fit_Time_Sec": t_elapsed,
            }
            results[name] = metrics

            self._logger.info("  %-18s Validation Metrics:", name)
            self._logger.info("    Silhouette Score  : %.4f", sil)
            self._logger.info("    Calinski-Harabasz : %.2f", ch)
            self._logger.info("    Davies-Bouldin    : %.4f", db)
            self._logger.info("    Fit Time          : %.4f sec", t_elapsed)

            # Plot 2D Cluster Scatter Plot
            self._plot_cluster_scatter(x_2d, labels, algorithm_name=name)

        self._save_summary_report(results)
        return results

    def _plot_elbow_curve(self, x_scaled: np.ndarray, k_range: range = range(2, 10)) -> None:
        """Calculate and plot K-Means Inertia (Elbow curve) across k range.

        Args:
            x_scaled: Standardized feature matrix.
            k_range:  Range of k cluster values to test.
        """
        self._logger.info("Evaluating K-Means Elbow Curve...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        inertias = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self._model_config.random_state, n_init=5)
            km.fit(x_scaled)
            inertias.append(km.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), inertias, marker="o", color="teal", lw=2)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
        plt.title("K-Means Elbow Method Curve")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        filepath = os.path.join(self._path_config.output_dir, "kmeans_elbow_curve.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._logger.info("Elbow curve saved to: %s", filepath)

    def _plot_cluster_scatter(
        self, x_2d: np.ndarray, labels: np.ndarray, algorithm_name: str
    ) -> None:
        """Render 2D PCA scatter plot colored by cluster assignments.

        Args:
            x_2d:           2D PCA projected features.
            labels:         Cluster assignment vector.
            algorithm_name: Algorithm identifier string.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]  # Black for noise points in DBSCAN
                label_text = "Noise"
            else:
                label_text = f"Cluster {k}"

            class_mask = labels == k
            plt.scatter(
                x_2d[class_mask, 0],
                x_2d[class_mask, 1],
                color=tuple(col),
                alpha=0.6,
                edgecolor="none",
                s=20,
                label=label_text,
            )

        plt.title(f"2D PCA Projection - {algorithm_name} Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        filepath = os.path.join(
            self._path_config.output_dir, f"cluster_scatter_{algorithm_name.lower().replace('-', '_')}.png"
        )
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._logger.info("Cluster scatter plot saved to: %s", filepath)

    def _save_summary_report(self, results: Dict[str, Dict[str, float]]) -> None:
        """Write analytical markdown report summarizing clustering validation metrics.

        Args:
            results: Results dictionary.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        filepath = os.path.join(self._path_config.output_dir, "clustering_analysis_report.md")

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("# Unsupervised Clustering Analysis Report\n\n")
            fh.write("## 1. Validation Metrics Summary\n")
            fh.write("Evaluation metrics for K-Means, Agglomerative, and DBSCAN clustering on Credit Card transactions:\n\n")

            fh.write("| Algorithm | Clusters Found | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Fit Time (s) |\n")
            fh.write("|-----------|----------------|------------------|-------------------|----------------|--------------|\n")

            for algo, m_dict in results.items():
                fh.write(
                    f"| **{algo}** | {int(m_dict['Clusters_Found'])} | {m_dict['Silhouette_Score']:.4f} | "
                    f"{m_dict['Calinski_Harabasz']:.2f} | {m_dict['Davies_Bouldin']:.4f} | "
                    f"{m_dict['Fit_Time_Sec']:.4f} |\n"
                )

            fh.write("\n## 2. Validation Metrics Definitions\n")
            fh.write("- **Silhouette Score**: Measures cluster cohesion vs separation (-1 to +1; higher is better).\n")
            fh.write("- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion (higher is better).\n")
            fh.write("- **Davies-Bouldin Index**: Average similarity between each cluster and its most similar cluster (lower is better).\n")

        self._logger.info("Clustering analysis report saved to: %s", filepath)
