# ============================================================================
# Main Entry Point -- K-Means Clustering Pipeline
# ============================================================================
# Orchestration script that wires together configuration, logging, data loading,
# cluster number parameter search, model fitting, and evaluation reporting.
# Follows the Service Layer pattern (SRP).
# ============================================================================

import sys
import time
import numpy as np
from config import PipelineConfig
from data_loader import DataLoaderService
from kmeans_clustering import KMeansClusteringService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end K-Means clustering pipeline."""
    # ----------------------------------------------------------------
    # Step 1: Configuration
    # ----------------------------------------------------------------
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="KMeans-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("K-MEANS CLUSTERING PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset          : %s", config.paths.dataset_file)
    logger.info("  Validation target: %s", config.data.target_column)
    logger.info("  Scale features   : %s", config.data.scale_features)
    logger.info("  Target K clusters: %d", config.model.n_clusters)
    logger.info("  Init method      : %s", config.model.init)

    pipeline_start = time.perf_counter()

    try:
        # ----------------------------------------------------------------
        # Step 3: Data Loading & Preprocessing
        # ----------------------------------------------------------------
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        x_train, x_test, y_train, y_test = data_service.load_and_prepare()

        # Combine sets for global elbow/silhouette parameter analysis
        x_full = np.vstack((x_train, x_test))

        # ----------------------------------------------------------------
        # Step 4: Clustering Model & Parameter Search
        # ----------------------------------------------------------------
        clustering_service = KMeansClusteringService(
            model_config=config.model,
            path_config=config.paths,
            label_names=data_service.label_names,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        
        # Run WCSS Elbow & Silhouette score calculation across K=[2..10]
        clustering_service.run_cluster_analysis(x_full, max_k=10)

        # Train model with configured target cluster count K
        clustering_service.train(x_train)

        # ----------------------------------------------------------------
        # Step 5: Evaluation & Reporting
        # ----------------------------------------------------------------
        clustering_service.evaluate(
            x_train_scaled=x_train,
            y_train=y_train,
            x_test_scaled=x_test,
            y_test=y_test,
        )

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        logger.error("Schema error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed with unexpected error: %s", exc)
        sys.exit(1)

    pipeline_elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info(
        "Clustering pipeline completed successfully in %.3f seconds.", pipeline_elapsed
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
