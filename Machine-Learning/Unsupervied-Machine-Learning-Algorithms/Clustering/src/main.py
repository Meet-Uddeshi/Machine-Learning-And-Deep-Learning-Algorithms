# ============================================================================
# Main Entry Point -- Unsupervised Clustering Pipeline
# ============================================================================
# Thin orchestration script that wires configuration, logging, data loading,
# and clustering model fitting (K-Means, DBSCAN, Agglomerative) for Credit Card transaction features.
# Stores all figures and markdown report in the output/ directory.
# ============================================================================

import os
import sys
import time

# Ensure src directory is in Python path for flexible invocation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clustering_service import ClusteringService
from config import PipelineConfig
from data_loader import DataLoaderService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Unsupervised Clustering pipeline.

    Execution Steps:
        1. Initialize configuration parameters.
        2. Set up stdout logger.
        3. Load, sample, and standardize `creditcard.csv` dataset.
        4. Fit K-Means, DBSCAN, and Agglomerative Clustering algorithms.
        5. Calculate validation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin).
        6. Render Elbow curve and 2D PCA cluster scatter plots.
        7. Export analytical report (`output/clustering_analysis_report.md`).
    """
    pipeline_start = time.perf_counter()

    config = PipelineConfig()
    logger = LoggerFactory.create(
        name="Clustering-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("UNSUPERVISED CLUSTERING PIPELINE (CREDIT CARD TRANSACTIONS)")
    logger.info("=" * 70)
    logger.info("Dataset File   : %s", config.paths.dataset_file)
    logger.info("Output Dir     : %s", config.paths.output_dir)
    logger.info("Sample Size    : %d", config.data.sample_size)

    try:
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        x_scaled, _ = data_service.load_and_prepare()

        clustering_service = ClusteringService(
            model_config=config.model,
            path_config=config.paths,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        clustering_service.fit_and_evaluate_all(x_scaled)

    except Exception as exc:
        logger.exception("Pipeline execution failed with error: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully in %.4f seconds.", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
