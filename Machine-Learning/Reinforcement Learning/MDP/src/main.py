# ============================================================================
# Main Entry Point -- Reinforcement Learning MDP Pipeline
# ============================================================================
# Thin orchestration script that wires together configuration, logging,
# Gridworld environment, dynamic programming solvers, and reports (SRP).
# ============================================================================

import sys
import time
from config import PipelineConfig
from mdp_environment import GridworldMDP
from mdp_solver import MDPSolverService
from logger import LoggerFactory


def main() -> None:
    """Orchestrate the end-to-end RL MDP Gridworld pipeline.

    Steps:
        1. Initialize configuration (all defaults; edit config.py to adjust).
        2. Create a pipeline logger with console-only output.
        3. Instantiate stochastic Gridworld MDP Environment.
        4. Solve MDP using Value Iteration or Policy Iteration.
        5. Extract optimal policy, plot heatmaps/grids, and write reports.
    """
    # ----------------------------------------------------------------
    # Step 1: Configuration
    # ----------------------------------------------------------------
    config = PipelineConfig()

    # ----------------------------------------------------------------
    # Step 2: Logger
    # ----------------------------------------------------------------
    logger = LoggerFactory.create(
        name="MDP-Pipeline",
        logging_config=config.logging,
        path_config=config.paths,
    )

    logger.info("=" * 70)
    logger.info("REINFORCEMENT LEARNING MARKOV DECISION PROCESS (MDP) PIPELINE")
    logger.info("=" * 70)
    logger.info("Configuration loaded successfully.")
    logger.info("  Grid Dimensions  : %dx%d", config.env.grid_rows, config.env.grid_cols)
    logger.info("  Goal State       : %s", str(config.env.goal_state))
    logger.info("  Trap States      : %s", str(config.env.trap_states))
    logger.info("  Wall States      : %s", str(config.env.wall_states))
    logger.info("  Algorithm Type   : %s", config.model.algorithm_type)
    logger.info("  Discount (gamma) : %.4f", config.model.gamma)
    logger.info("  Tolerance (theta): %.1e", config.model.theta)

    pipeline_start = time.perf_counter()

    try:
        # ----------------------------------------------------------------
        # Step 3: MDP Environment Setup
        # ----------------------------------------------------------------
        env = GridworldMDP(env_config=config.env, logger=logger)

        # ----------------------------------------------------------------
        # Step 4: DP Solver Execution & Evaluation
        # ----------------------------------------------------------------
        solver_service = MDPSolverService(
            env=env,
            model_config=config.model,
            path_config=config.paths,
            logger=logger,
        )
        solver_service.solve_and_evaluate()

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
        "MDP Pipeline completed successfully in %.3f seconds.", pipeline_elapsed
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
