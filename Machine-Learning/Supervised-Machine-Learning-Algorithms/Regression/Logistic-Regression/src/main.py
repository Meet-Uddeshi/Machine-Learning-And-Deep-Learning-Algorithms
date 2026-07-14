# ============================================================================
# Main Entry Point -- Logistic Regression Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# data loading, and model evaluation for the Supply Chain Risk classification pipeline.
# Contains no business logic itself; all work is delegated to service classes.
# ============================================================================

import sys
import time

from config import PipelineConfig
from data_loader import DataLoaderService
from logger import LoggerFactory
from logistic_regressor import LogisticRegressorService


def main() -> None:
    """Orchestrate the end-to-end Logistic Regression classification pipeline.

    Steps:
        1. Initialize configurations.
        2. Create a pipeline logger with console + file output.
        3. Load, validate, preprocess, scale, and split the dataset.
        4. Train the regularized Logistic Regression classifier.
        5. Evaluate on the test set, generate plots, and write report summaries.
    """
    # ----------------------------------------------------------------
    # Step 1: Configuration Ingestion
    # ----------------------------------------------------------------
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger Setup
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="LogisticRegression-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("LOGISTIC REGRESSION PIPELINE (SUPPLY CHAIN RISK)")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset       : %s", config.paths.dataset_file)
    logger.info("  Target column : %s", config.data.target_column)
    logger.info("  Test size     : %.0f%%", config.data.test_size * 100)
    logger.info("  Regularization: %s (C=%.4f)", config.model.penalty, config.model.C)
    logger.info("  Solver        : %s", config.model.solver)

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
        classifier_service = LogisticRegressorService(
            model_config=config.model,
            path_config=config.paths,
            label_names=data_service.label_names,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        classifier_service.train(x_train, y_train)

        # ----------------------------------------------------------------
        # Step 5: Evaluation & Report Generation
        # ----------------------------------------------------------------
        classifier_service.evaluate(x_test, y_test)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        logger.error("Schema validation error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        # Catch unexpected errors and log full traceback to help developers debug
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
