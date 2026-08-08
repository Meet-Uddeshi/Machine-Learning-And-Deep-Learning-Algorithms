# ============================================================================
# Main Entry Point -- Principal Component Analysis (PCA) Pipeline
# ============================================================================
# Thin orchestration script that wires configuration, logging, data loading,
# and PCA execution for the Vehicle Silhouette dataset (`pca.csv`).
# Stores all figures and markdown report in the output/ directory.
# ============================================================================

import os
import sys
import time

# Ensure src directory is in Python path for flexible invocation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from data_loader import DataLoaderService
from logger import LoggerFactory
from pca_service import PCAService


def main() -> None:
    """Orchestrate the end-to-end Principal Component Analysis (PCA) pipeline.

    Execution Steps:
        1. Initialize configuration parameters.
        2. Set up stdout logger.
        3. Load, clean, and standardize `pca.csv` dataset.
        4. Fit PCA models (custom matrix eigendecomposition and scikit-learn PCA).
        5. Evaluate explained variance ratios, cumulative variance, and reconstruction MSE.
        6. Render Scree Plot and 2D Projection scatter plot.
        7. Export analytical markdown report (`output/pca_analysis_report.md`).
    """
    pipeline_start = time.perf_counter()

    config = PipelineConfig()
    logger = LoggerFactory.create(
        name="PCA-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("PRINCIPAL COMPONENT ANALYSIS (PCA) PIPELINE")
    logger.info("=" * 70)
    logger.info("Dataset File   : %s", config.paths.dataset_file)
    logger.info("Output Dir     : %s", config.paths.output_dir)
    logger.info("Target Column  : %s", config.data.target_column)

    try:
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        x_scaled, y_labels, _ = data_service.load_and_prepare()

        pca_service = PCAService(
            model_config=config.model,
            path_config=config.paths,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        pca_service.fit_and_evaluate(x_scaled, y_labels)

    except Exception as exc:
        logger.exception("Pipeline execution failed with error: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully in %.4f seconds.", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
