# ============================================================================
# Main Entry Point -- Decision Tree Classification Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# data loading, and model evaluation.  Contains NO business logic itself;
# all work is delegated to dedicated service classes (SRP).
# ============================================================================

import sys
import time
from config import PipelineConfig
from data_loader import DataLoaderService
from decision_tree_classifier import DecisionTreeClassifierService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Decision Tree classification pipeline.

    Steps:
        1. Initialize configuration (all defaults; edit config.py to adjust).
        2. Create a pipeline logger with console + file output.
        3. Load, validate, preprocess, and split the dataset.
        4. Train the Decision Tree classifier.
        5. Evaluate on the held-out test set and report results.
    """
    # ----------------------------------------------------------------
    # Step 1: Configuration
    # ----------------------------------------------------------------
    # To change any parameter, modify the dataclass defaults in config.py
    # or pass overrides here, e.g.:
    #   config = PipelineConfig(model=ModelConfig(max_depth=5, criterion='entropy'))
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="DT-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("DECISION TREE CLASSIFICATION PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset          : %s", config.paths.dataset_file)
    logger.info("  Target column    : %s", config.data.target_column)
    logger.info("  Test size        : %.0f%%", config.data.test_size * 100)
    logger.info("  Criterion        : %s", config.model.criterion)
    logger.info("  Max depth        : %s", config.model.max_depth)
    logger.info("  Min samples split: %d", config.model.min_samples_split)
    logger.info("  Min samples leaf : %d", config.model.min_samples_leaf)
    logger.info("  Class weight     : %s", config.model.class_weight)
    logger.info("  CCP alpha        : %.4f", config.model.ccp_alpha)

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
        classifier_service = DecisionTreeClassifierService(
            model_config=config.model,
            path_config=config.paths,
            label_names=data_service.label_names,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        classifier_service.train(x_train, y_train)

        # ----------------------------------------------------------------
        # Step 5: Evaluation
        # ----------------------------------------------------------------
        classifier_service.evaluate(x_test, y_test)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        logger.error("Schema error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        # Catch-all for unexpected errors; log full traceback for debugging
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
