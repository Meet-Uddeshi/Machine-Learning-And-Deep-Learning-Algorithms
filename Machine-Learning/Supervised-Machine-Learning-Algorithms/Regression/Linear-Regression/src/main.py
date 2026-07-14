# ============================================================================
# Main Entry Point -- Linear Regression Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# data loading, and model evaluation for the Walmart Sales regression pipeline.
# Contains no business logic itself; all work is delegated to service classes.
# ============================================================================

import sys
import time

from config import PipelineConfig
from data_loader import DataLoaderService
from linear_regressor import LinearRegressorService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Linear Regression pipeline.

    Steps:
        1. Initialize configuration (from config.py).
        2. Create a pipeline logger with console + file output.
        3. Load, validate, preprocess, and split the dataset.
        4. Train the Ordinary Least Squares (OLS) regressor.
        5. Evaluate on the held-out test set and write report summaries.
    """
    # ----------------------------------------------------------------
    # Step 1: Configuration Ingestion
    # ----------------------------------------------------------------
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger Setup
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="LinearRegression-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("LINEAR REGRESSION PIPELINE (WALMART WEEKLY SALES)")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset       : %s", config.paths.dataset_file)
    logger.info("  Target column : %s", config.data.target_column)
    logger.info("  Test size     : %.0f%%", config.data.test_size * 100)
    logger.info("  Fit Intercept : %s", config.model.fit_intercept)

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
        # Step 4: Model Training
        # ----------------------------------------------------------------
        regressor_service = LinearRegressorService(
            model_config=config.model,
            path_config=config.paths,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        regressor_service.train(x_train, y_train)

        # ----------------------------------------------------------------
        # Step 5: Evaluation & Report Generation
        # ----------------------------------------------------------------
        regressor_service.evaluate(x_test, y_test)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        logger.error("Schema validation error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        # Catch unexpected errors; log full traceback to help developers debug
        logger.exception("Pipeline failed with unexpected error: %s", exc)
        sys.exit(1)

    pipeline_elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info(
        "Pipeline completed successfully in %.4f seconds.", pipeline_elapsed
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
