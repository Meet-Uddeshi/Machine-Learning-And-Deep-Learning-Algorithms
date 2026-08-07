# ============================================================================
# Main Entry Point -- PCA Unsupervised Learning Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# data loading, PCA decomposition, and report generation (SRP).
# ============================================================================

import sys
import time
from config import PipelineConfig
from data_loader import DataLoaderService
from pca_service import PCAService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end PCA unsupervised learning pipeline.

    Steps:
        1. Initialize configuration (all defaults; edit config.py to adjust).
        2. Create a pipeline logger with console-only output.
        3. Load, validate, impute nulls, preprocess, and scale the dataset.
        4. Perform PCA decomposition on training set features.
        5. Evaluate variance ratios, plot scree/loadings, and generate reports.
    """
    # ----------------------------------------------------------------
    # Step 1: Configuration
    # ----------------------------------------------------------------
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="PCA-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("PRINCIPAL COMPONENT ANALYSIS (PCA) PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset          : %s", config.paths.dataset_file)
    logger.info("  Target column    : %s", config.data.target_column)
    logger.info("  Test size        : %.0f%%", config.data.test_size * 100)
    logger.info("  Components       : %s", str(config.model.n_components))
    logger.info("  SVD Solver       : %s", config.model.svd_solver)

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

        # ----------------------------------------------------------------
        # Step 4: PCA Decomposition & Evaluation
        # ----------------------------------------------------------------
        pca_service = PCAService(
            model_config=config.model,
            path_config=config.paths,
            label_names=data_service.label_names,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        pca_service.train_and_evaluate(x_train, y_train, x_test, y_test)

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
        "PCA Pipeline completed successfully in %.3f seconds.", pipeline_elapsed
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
