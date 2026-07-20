# ============================================================================
# Main Entry Point -- Decision Tree Classification Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# data loading, and model evaluation. Contains NO business logic itself.
# ============================================================================

import sys
import time
from config import PipelineConfig
from data_loader import DataLoaderService
from decision_tree_classifier import DecisionTreeClassifierService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Decision Tree classification pipeline."""
    config = PipelineConfig()

    logger = LoggerFactory.create(
        name="DecisionTree-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("DECISION TREE CLASSIFICATION PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Dataset           : %s", config.paths.dataset_file)
    logger.info("  Target column     : %s", config.data.target_column)
    logger.info("  Max Samples       : %s", str(config.data.max_samples))
    logger.info("  Criterion         : %s", config.model.criterion)
    logger.info("  Max Depth         : %s", str(config.model.max_depth))
    logger.info("  Min Samples Split : %d", config.model.min_samples_split)

    pipeline_start = time.perf_counter()

    try:
        # Data loading & preprocessing
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        x_train, x_test, y_train, y_test = data_service.load_and_prepare()

        # Model training
        classifier_service = DecisionTreeClassifierService(
            model_config=config.model,
            path_config=config.paths,
            label_names=data_service.label_names,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        classifier_service.train(x_train, y_train)

        # Evaluation
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
