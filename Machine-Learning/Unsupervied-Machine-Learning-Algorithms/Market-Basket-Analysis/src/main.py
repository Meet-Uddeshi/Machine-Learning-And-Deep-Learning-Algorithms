# ============================================================================
# Main Entry Point -- Market Basket Analysis Pipeline
# ============================================================================
# Thin orchestration script that wires configuration, logging, data loading,
# transaction matrix binarization, and Apriori algorithm fitting for retail transactions.
# Exports figures and markdown analysis report into output/ directory.
# ============================================================================

import os
import sys
import time

# Ensure src directory is in Python path for flexible invocation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apriori_service import AprioriService
from config import PipelineConfig
from data_loader import DataLoaderService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end Market Basket Analysis pipeline.

    Execution Steps:
        1. Initialize configuration parameters.
        2. Set up stdout logger.
        3. Load, clean, and binarize dataset into transaction itemsets.
        4. Execute custom Apriori algorithm from scratch to mine frequent itemsets.
        5. Generate Association Rules and compute metrics (Support, Confidence, Lift, etc.).
        6. Render visualizations (Top Itemsets, Support vs Confidence, Rule Matrix).
        7. Export analytical report (`output/market_basket_analysis_report.md`).
    """
    pipeline_start = time.perf_counter()

    config = PipelineConfig()
    logger = LoggerFactory.create(
        name="Market-Basket-Analysis-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("MARKET BASKET ANALYSIS PIPELINE (APRIORI ALGORITHM)")
    logger.info("=" * 70)
    logger.info("Dataset File   : %s", config.paths.dataset_file)
    logger.info("Output Dir     : %s", config.paths.output_dir)
    logger.info("Target Country : %s", config.data.target_country)
    logger.info("Min Support    : %.4f", config.apriori.min_support)
    logger.info("Min Confidence : %.4f", config.apriori.min_confidence)
    logger.info("Min Lift       : %.4f", config.apriori.min_lift)

    try:
        data_service = DataLoaderService(
            path_config=config.paths,
            data_config=config.data,
            logger=logger,
        )
        basket_df, transactions = data_service.load_and_prepare()

        apriori_service = AprioriService(
            apriori_config=config.apriori,
            path_config=config.paths,
            logger=logger,
        )
        apriori_service.fit_and_evaluate_all(basket_df, transactions)

    except Exception as exc:
        logger.exception("Pipeline execution failed with error: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully in %.4f seconds.", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
