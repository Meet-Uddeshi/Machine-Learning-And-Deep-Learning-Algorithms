# ============================================================================
# Main Entry Point -- Market Basket Analysis & Apriori Pipeline
# ============================================================================
# Orchestration script that wires together configuration, logging, data loading,
# basket extraction, Apriori candidate mining, association rule evaluation,
# visualisations, and report generation (SRP).
# ============================================================================

import sys
import time
from config import PipelineConfig
from data_loader import DataLoaderService
from apriori_service import AprioriMBAService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Market Basket Analysis & Apriori pipeline."""
    # ----------------------------------------------------------------
    # Step 1: Configuration
    # ----------------------------------------------------------------
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="MBA-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("MARKET BASKET ANALYSIS & APRIORI ALGORITHM PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Zip File         : %s", config.paths.zip_file)
    logger.info("  CSV File         : %s", config.data.csv_filename)
    logger.info("  Country Filter   : %s", str(config.data.country_filter))
    logger.info("  Min Support      : %.3f", config.model.min_support)
    logger.info("  Min Confidence   : %.2f", config.model.min_confidence)
    logger.info("  Min Lift         : %.2f", config.model.min_lift)
    logger.info("  Max Itemset Size : %d", config.model.max_itemset_length)

    pipeline_start = time.perf_counter()

    try:
        # ----------------------------------------------------------------
        # Step 3: Data Loading & Transaction Basket Construction
        # ----------------------------------------------------------------
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        baskets, cleaned_df = data_service.load_and_prepare_baskets()

        # ----------------------------------------------------------------
        # Step 4: Apriori Mining & Rule Evaluation
        # ----------------------------------------------------------------
        apriori_service = AprioriMBAService(
            model_config=config.model,
            path_config=config.paths,
            logger=logger,
        )
        apriori_service.run_pipeline(baskets)

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
        "MBA Pipeline completed successfully in %.3f seconds.", pipeline_elapsed
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
