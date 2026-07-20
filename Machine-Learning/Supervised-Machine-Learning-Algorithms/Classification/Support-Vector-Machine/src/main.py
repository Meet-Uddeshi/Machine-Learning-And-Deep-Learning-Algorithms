# ============================================================================
# Main Entry Point -- Support Vector Machine Classification Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# data loading, and model evaluation. Contains NO business logic itself;
# all work is delegated to dedicated service classes (SRP).
# ============================================================================

import sys
import time
from config import PipelineConfig
from data_loader import DataLoaderService
from logger import LoggerFactory
from svm_classifier import SVMClassifierService


def main() -> None:
    """Orchestrate the end-to-end SVM classification pipeline.

    Steps:
        1. Initialize configuration (all defaults; edit config.py to adjust).
        2. Create a pipeline logger with console + file output.
        3. Load, validate, preprocess, and split the dataset.
        4. Train the Support Vector Machine classifier.
        5. Evaluate on the held-out test set and report results.
    """
    config = PipelineConfig()

    logger = LoggerFactory.create(
        name="SVM-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("SUPPORT VECTOR MACHINE CLASSIFICATION PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset       : %s", config.paths.dataset_file)
    logger.info("  Target column : %s", config.data.target_column)
    logger.info("  Test size     : %.0f%%", config.data.test_size * 100)
    logger.info("  Kernel        : %s", config.model.kernel)
    logger.info("  C (Penalty)   : %.4f", config.model.C)
    logger.info("  Gamma         : %s", config.model.gamma)

    pipeline_start = time.perf_counter()

    try:
        # Step 3: Data Loading & Preprocessing
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        x_train, x_test, y_train, y_test = data_service.load_and_prepare()

        # Step 4: Model Training
        classifier_service = SVMClassifierService(
            model_config=config.model,
            path_config=config.paths,
            label_names=data_service.label_names,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        classifier_service.train(x_train, y_train)

        # Step 5: Evaluation
        classifier_service.evaluate(x_test, y_test)

    except FileNotFoundError as exc:
        logger.error("File not found error: %s", exc)
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
        "Pipeline completed successfully in %.3f seconds.", pipeline_elapsed
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
