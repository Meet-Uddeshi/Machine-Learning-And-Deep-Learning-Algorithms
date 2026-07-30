# ============================================================================
# K-Means Clustering Service for Clustering Pipeline
# ============================================================================
# Owns the entire model lifecycle: training, elbow/silhouette parameter search,
# cluster assignment, evaluation metrics, PCA visualisation, and report writing.
# ============================================================================

import logging
import os
import time
from typing import List

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from config import ModelConfig, PathConfig


class KMeansClusteringService:
    """Service encapsulating K-Means clustering logic.

    Responsibilities:
        1. Run elbow and silhouette analysis to guide selection of K.
        2. Fit a KMeans model with configured parameters.
        3. Predict cluster assignments on training and test datasets.
        4. Compute intrinsic and extrinsic clustering evaluation metrics.
        5. Visualise results via PCA dimensionality reduction and plot centroids.
        6. Persist text reports, markdown analysis, and image plots.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the clustering service.

        Args:
            model_config:  K-Means hyperparameters.
            path_config:   Path settings for output saving.
            label_names:   Actual labels of the validation target (Outcome).
            feature_names: Feature names for documentation.
            logger:        Logger instance.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger
        self._model = self._build_model()

    # -- Public workflow methods ---------------------------------------------

    def run_cluster_analysis(self, x_scaled: np.ndarray, max_k: int = 10) -> None:
        """Compute and plot WCSS (Elbow) and Silhouette scores for K in [2, max_k].

        Args:
            x_scaled: Scaled feature matrix.
            max_k:    Maximum number of clusters to test.
        """
        self._logger.info("=" * 70)
        self._logger.info("RUNNING CLUSTER PARAMETER PARAMETRIC SEARCH (K-selection)")
        self._logger.info("=" * 70)
        
        k_values = list(range(2, max_k + 1))
        wcss_scores = []
        silhouette_scores = []

        for k in k_values:
            kmeans_test = KMeans(
                n_clusters=k,
                init=self._model_config.init,
                max_iter=self._model_config.max_iter,
                n_init=self._model_config.n_init,
                random_state=self._model_config.random_state,
            )
            labels = kmeans_test.fit_predict(x_scaled)
            wcss_scores.append(kmeans_test.inertia_)
            
            sil_coef = silhouette_score(x_scaled, labels)
            silhouette_scores.append(sil_coef)
            self._logger.info(f"  K = {k:2d} | WCSS (Inertia): {kmeans_test.inertia_:.2f} | Silhouette Score: {sil_coef:.4f}")

        self._plot_parameter_searches(k_values, wcss_scores, silhouette_scores)

    def train(self, x_train_scaled: np.ndarray) -> None:
        """Fit the configured K-Means model on the training data.

        Args:
            x_train_scaled: Standardised training feature matrix.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._logger.info(f"Fitting K-Means model with K = {self._model_config.n_clusters} clusters...")
        self._logger.info(f"  Initialisation method : {self._model_config.init}")
        self._logger.info(f"  Max iterations        : {self._model_config.max_iter}")

        start_time = time.perf_counter()
        self._model.fit(x_train_scaled)
        elapsed = time.perf_counter() - start_time

        self._logger.info("K-Means fit completed in %.3f seconds.", elapsed)
        self._logger.info("Final converged WCSS (Inertia): %.4f", self._model.inertia_)
        self._logger.info("Number of iterations run to converge: %d", self._model.n_iter_)

    def evaluate(
        self,
        x_train_scaled: np.ndarray,
        y_train: np.ndarray,
        x_test_scaled: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Predict assignments and compute clustering performance metrics.

        Args:
            x_train_scaled: Standardised training features.
            y_train:        Ground-truth training labels (Outcome).
            x_test_scaled:  Standardised test features.
            y_test:         Ground-truth test labels.

        Returns:
            Dictionary containing clustering metrics.
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL EVALUATION")
        self._logger.info("=" * 70)

        train_clusters = self._model.labels_
        test_clusters = self._model.predict(x_test_scaled)

        # Intrinsic metrics (doesn't use true labels)
        train_silhouette = silhouette_score(x_train_scaled, train_clusters)
        test_silhouette = silhouette_score(x_test_scaled, test_clusters)

        # Extrinsic metrics (compares clusters to true Outcome classes)
        train_ari = adjusted_rand_score(y_train, train_clusters)
        test_ari = adjusted_rand_score(y_test, test_clusters)
        
        train_nmi = normalized_mutual_info_score(y_train, train_clusters)
        test_nmi = normalized_mutual_info_score(y_test, test_clusters)

        # Contingency Matrix
        contingency_train = pd.crosstab(
            index=y_train,
            columns=train_clusters,
            rownames=["Actual Outcome"],
            colnames=["Assigned Cluster"]
        )

        results = {
            "train_inertia": self._model.inertia_,
            "train_silhouette": train_silhouette,
            "test_silhouette": test_silhouette,
            "train_ari": train_ari,
            "test_ari": test_ari,
            "train_nmi": train_nmi,
            "test_nmi": test_nmi,
            "contingency_matrix_train": contingency_train,
            "train_clusters": train_clusters,
            "test_clusters": test_clusters,
        }

        self._log_evaluation(results)
        self._save_results(results)
        self._visualise_clusters_2d(x_train_scaled, train_clusters)
        self._save_analysis(results)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self) -> KMeans:
        """Construct KMeans estimator from configuration.

        Returns:
            An unfitted KMeans estimator.
        """
        self._logger.info("Instantiating K-Means Estimator (scikit-learn).")
        return KMeans(
            n_clusters=self._model_config.n_clusters,
            init=self._model_config.init,
            max_iter=self._model_config.max_iter,
            n_init=self._model_config.n_init,
            random_state=self._model_config.random_state,
        )

    def _plot_parameter_searches(self, k_values: list, wcss: list, sil: list) -> None:
        """Generate and save Elbow and Silhouette plots."""
        self._logger.info("Generating parameter search plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # Elbow plot
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, wcss, marker="o", linestyle="-", color="darkblue", linewidth=2)
        plt.title("Elbow Method Analysis (K Selection)", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Within-Cluster Sum of Squares (WCSS / Inertia)")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        elbow_path = os.path.join(self._path_config.output_dir, "elbow_method.png")
        plt.savefig(elbow_path, dpi=300)
        plt.close()

        # Silhouette score plot
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, sil, marker="s", linestyle="--", color="crimson", linewidth=2)
        plt.title("Silhouette Analysis (Clustering Quality)", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Average Silhouette Coefficient")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        sil_path = os.path.join(self._path_config.output_dir, "silhouette_analysis.png")
        plt.savefig(sil_path, dpi=300)
        plt.close()
        
        self._logger.info(f"Elbow plot saved to: {elbow_path}")
        self._logger.info(f"Silhouette analysis plot saved to: {sil_path}")

    def _log_evaluation(self, results: dict) -> None:
        """Print results metrics to console logs."""
        self._logger.info("-" * 70)
        self._logger.info("CLUSTERING METRICS:")
        self._logger.info("  Training Inertia (WCSS)        : %.4f", results["train_inertia"])
        self._logger.info("  Silhouette Score (Train Set)   : %.4f", results["train_silhouette"])
        self._logger.info("  Silhouette Score (Test Set)    : %.4f", results["test_silhouette"])
        self._logger.info("  Adjusted Rand Index (Train Set): %.4f (ARI matches random=0, perfect=1)", results["train_ari"])
        self._logger.info("  Adjusted Rand Index (Test Set) : %.4f", results["test_ari"])
        self._logger.info("  Normalized Mutual Info (Train) : %.4f (NMI matches independence=0, perfect=1)", results["train_nmi"])
        self._logger.info("  Normalized Mutual Info (Test)  : %.4f", results["test_nmi"])
        
        self._logger.info("-" * 70)
        self._logger.info("CLUSTER TO ACTUAL OUTCOME CONTINGENCY MATRIX (TRAINING):")
        self._logger.info("\n" + str(results["contingency_matrix_train"]))
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Write numeric metrics to clustering_results.txt."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        results_path = os.path.join(self._path_config.output_dir, "clustering_results.txt")

        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("K-MEANS CLUSTERING RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("MODEL CONFIGURATION\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  n_clusters       : {self._model_config.n_clusters}\n")
            fh.write(f"  init method      : {self._model_config.init}\n")
            fh.write(f"  max_iter         : {self._model_config.max_iter}\n")
            fh.write(f"  n_init           : {self._model_config.n_init}\n")
            fh.write(f"  random_state     : {self._model_config.random_state}\n\n")

            fh.write("FEATURES INCLUDED IN CLUSTERING\n")
            fh.write("-" * 40 + "\n")
            for i, name in enumerate(self._feature_names, start=1):
                fh.write(f"  {i:2d}. {name}\n")
            fh.write("\n")

            fh.write("EVALUATION METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  WCSS (Inertia)          : {results['train_inertia']:.4f}\n")
            fh.write(f"  Silhouette Score (Train): {results['train_silhouette']:.4f}\n")
            fh.write(f"  Silhouette Score (Test) : {results['test_silhouette']:.4f}\n")
            fh.write(f"  Adjusted Rand Index (Tr): {results['train_ari']:.4f}\n")
            fh.write(f"  Adjusted Rand Index (Te): {results['test_ari']:.4f}\n")
            fh.write(f"  Normalized Mutual Info (Tr): {results['train_nmi']:.4f}\n")
            fh.write(f"  Normalized Mutual Info (Te): {results['test_nmi']:.4f}\n\n")

            fh.write("CLUSTER CENTROIDS (STANDARDISED FEATURES)\n")
            fh.write("-" * 40 + "\n")
            centroids = self._model.cluster_centers_
            for c_idx in range(self._model_config.n_clusters):
                fh.write(f"  Cluster {c_idx} Centroid:\n")
                for f_idx, feat in enumerate(self._feature_names):
                    fh.write(f"    {feat:<25s}: {centroids[c_idx, f_idx]:.4f}\n")
                fh.write("\n")

            fh.write("CLUSTER CONGRUENCE CONTINGENCY TABLE\n")
            fh.write("-" * 40 + "\n")
            con_df = results["contingency_matrix_train"]
            fh.write(con_df.to_string() + "\n")
            fh.write("=" * 70 + "\n")

        self._logger.info(f"Clustering results saved to: {results_path}")

    def _visualise_clusters_2d(self, x_scaled: np.ndarray, clusters: np.ndarray) -> None:
        """Run PCA to project features to 2D and plot cluster distribution."""
        self._logger.info("Running Principal Component Analysis (PCA) for cluster visualisation...")
        
        pca = PCA(n_components=2, random_state=self._model_config.random_state)
        x_pca = pca.fit_transform(x_scaled)
        centroids_pca = pca.transform(self._model.cluster_centers_)

        plt.figure(figsize=(10, 8))
        
        # Plot data points colored by assigned cluster
        scatter = plt.scatter(
            x_pca[:, 0],
            x_pca[:, 1],
            c=clusters,
            cmap="viridis",
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5,
            label="Patients"
        )
        
        # Plot cluster centroids as prominent red stars
        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c="red",
            marker="*",
            s=250,
            edgecolors="black",
            linewidth=1.5,
            label="Centroids"
        )

        plt.title(
            f"K-Means Clustering -- 2D PCA Projection (K = {self._model_config.n_clusters})",
            fontsize=12,
            fontweight="bold"
        )
        plt.xlabel(f"Principal Component 1 (Variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"Principal Component 2 (Variance explained: {pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        
        vis_path = os.path.join(self._path_config.output_dir, "cluster_visualization_2d.png")
        plt.savefig(vis_path, dpi=300)
        plt.close()
        
        self._logger.info(f"PCA 2D Cluster visualisation saved to: {vis_path}")

    def _save_analysis(self, results: dict) -> None:
        """Generate and save technical markdown report classification_analysis.md."""
        report_path = os.path.join(self._path_config.output_dir, "clustering_analysis.md")

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(f"# K-Means Clustering Analysis Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                f"This report presents a formal unsupervised clustering analysis of the Pima Indians Diabetes Dataset "
                f"using K-Means. By dropping target labels, the data features were grouped solely on geographical distance. "
                f"Setting $K = 2$, the algorithm grouped patients into two distinct clusters. "
                f"The alignment with ground-truth diabetes outcomes shows an Adjusted Rand Index (ARI) of "
                f"**{results['train_ari']:.4f}** and Normalized Mutual Information (NMI) of **{results['train_nmi']:.4f}**.\n\n"
            )

            fh.write("## 2. Clustering Configuration\n\n")
            fh.write("| Parameter | Value |\n")
            fh.write("|-----------|-------|\n")
            fh.write(f"| `n_clusters` | `{self._model_config.n_clusters}` |\n")
            fh.write(f"| `init` | `{self._model_config.init}` |\n")
            fh.write(f"| `max_iter` | `{self._model_config.max_iter}` |\n")
            fh.write(f"| `n_init` | `{self._model_config.n_init}` |\n")
            fh.write(f"| `random_state` | `{self._model_config.random_state}` |\n\n")

            fh.write("## 3. Evaluation Metrics Breakdown\n\n")
            fh.write("| Metric | Training Set | Test Set |\n")
            fh.write("|--------|--------------|----------|\n")
            fh.write(f"| **Silhouette Score** | {results['train_silhouette']:.4f} | {results['test_silhouette']:.4f} |\n")
            fh.write(f"| **Adjusted Rand Index (ARI)** | {results['train_ari']:.4f} | {results['test_ari']:.4f} |\n")
            fh.write(f"| **Normalized Mutual Info (NMI)** | {results['train_nmi']:.4f} | {results['test_nmi']:.4f} |\n")
            fh.write(f"| **WCSS (Inertia)** | {results['train_inertia']:.4f} | N/A |\n\n")

            fh.write("### Theoretical Interpretation of Metrics\n")
            fh.write(
                "- **WCSS (Within-Cluster Sum of Squares)**: Measures the compactness of the clusters. Decreases as $K$ increases.\n"
                "- **Silhouette Score**: Evaluates cluster cohesion and separation. A score near +1 denotes well-separated clusters; a score close to 0 denotes overlapping clusters.\n"
                "- **Adjusted Rand Index (ARI)**: Corrects for chance agreements and measures the similarity of assigned clusters to actual clinical outcome labels. ARI ranges from -1 to +1 (0 indicates random alignment).\n"
                "- **Normalized Mutual Information (NMI)**: Information theoretic measure of alignment between partitions. Normalised to [0, 1] range.\n\n"
            )

            fh.write("## 4. Contingency Table Analysis (Cluster to Outcome Matching)\n\n")
            fh.write(
                "Cross-tabulation comparing cluster assignments with ground-truth clinical Outcomes (0: Non-Diabetic, 1: Diabetic):\n\n"
            )
            con_df = results["contingency_matrix_train"]
            fh.write("| Outcome \\ Cluster |")
            for c_idx in con_df.columns:
                fh.write(f" Cluster {c_idx} |")
            fh.write("\n| --- |")
            for _ in con_df.columns:
                fh.write(" --- |")
            fh.write("\n")
            for out_val in con_df.index:
                fh.write(f"| **Outcome {out_val}** |")
                for c_idx in con_df.columns:
                    fh.write(f" {con_df.loc[out_val, c_idx]} |")
                fh.write("\n")
            fh.write("\n")

            fh.write("### Analysis of Overlaps\n")
            fh.write(
                "Because K-Means partitions the feature space into spherical clusters using standard Euclidean distance, "
                "overlapping regions in clinical dimensions (e.g. patients with similar high BMI or intermediate Glucose levels) "
                "cross the partition boundary, creating cluster impurity. In particular, diabetics (Outcome 1) show broad statistical dispersion, "
                "leading to multi-cluster distribution.\n\n"
            )

            fh.write("## 5. Centroid Interpretations (Standardised Feature Z-Scores)\n\n")
            fh.write(
                "Centroids define the center of each cluster. Since the variables are standardised, values represent deviations "
                "from the dataset mean (in units of standard deviation):\n\n"
            )
            fh.write("| Feature |")
            for c_idx in range(self._model_config.n_clusters):
                fh.write(f" Cluster {c_idx} |")
            fh.write("\n| --- |")
            for _ in range(self._model_config.n_clusters):
                fh.write(" --- |")
            fh.write("\n")
            centroids = self._model.cluster_centers_
            for f_idx, feat in enumerate(self._feature_names):
                fh.write(f"| **{feat}** |")
                for c_idx in range(self._model_config.n_clusters):
                    fh.write(f" {centroids[c_idx, f_idx]:.4f} |")
                fh.write("\n")
            fh.write("\n")

            fh.write("## 6. Output Artifacts\n\n")
            fh.write("| File | Description |\n")
            fh.write("|------|-------------|\n")
            fh.write("| `clustering_results.txt` | Hyperparameters, centroids, cross-tabulation table, and metrics |\n")
            fh.write("| `clustering_analysis.md` | This technical explanation markdown report |\n")
            fh.write("| `elbow_method.png` | Line plot of WCSS for K parameter search |\n")
            fh.write("| `silhouette_analysis.png` | Plot of Average Silhouette Score across K |\n")
            fh.write("| `cluster_visualization_2d.png` | PCA 2D scatter plot projection with marked centroids |\n")

        self._logger.info(f"Clustering markdown analysis report saved to: {report_path}")
