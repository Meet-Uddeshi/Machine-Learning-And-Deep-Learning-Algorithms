# ============================================================================
# Main Entry Point -- Bagging and Boosting Ensemble Pipeline
# ============================================================================
# Thin orchestration script that wires configuration, logging, data loading,
# and ensemble model training (Bagging, AdaBoost, Gradient Boosting) for heart disease classification.
# Stores all visual plots and markdown report in the output/ directory.
# ============================================================================

import os
import sys
import time

# Ensure src directory is in Python path for flexible invocation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from data_loader import DataLoaderService
from ensemble_classifier import EnsembleClassifierService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Bagging and Boosting classification pipeline.

    Execution Steps:
        1. Initialize configuration parameters.
        2. Set up stdout logger.
        3. Load, validate, scale, and split `heart.csv` dataset.
        4. Train Bagging, AdaBoost, and Gradient Boosting classifiers.
        5. Evaluate models, render confusion matrices & comparison charts.
        6. Export analytical markdown report (`output/bagging_boosting_analysis_report.md`).
    """
    pipeline_start = time.perf_counter()

    config = PipelineConfig()
    logger = LoggerFactory.create(
        name="Bagging-And-Boosting-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("BAGGING AND BOOSTING ENSEMBLE PIPELINE (HEART DISEASE)")
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
        x_train, x_test, y_train, y_test = data_service.load_and_prepare()

        classifier_service = EnsembleClassifierService(
            model_config=config.model,
            path_config=config.paths,
            feature_names=data_service.feature_names,
            logger=logger,
        )
        classifier_service.train_and_evaluate(x_train, x_test, y_train, y_test)

    except Exception as exc:
        logger.exception("Pipeline execution failed with error: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully in %.4f seconds.", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
