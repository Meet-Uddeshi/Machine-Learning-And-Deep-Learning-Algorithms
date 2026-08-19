# ============================================================================
# Configuration Module for RL MDP Pipeline
# ============================================================================
# Centralizes environment parameters, reward specifications, transition noise,
# solver parameters, and path specifications.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the MDP directory.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results and plots are written.
    """

    # Resolve project root as parent of 'src/'
    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    src_dir: str = ""
    output_dir: str = ""

    def __post_init__(self) -> None:
        """Derive dependent paths from project_root after init."""
        object.__setattr__(
            self, "src_dir", os.path.join(self.project_root, "src")
        )
        object.__setattr__(
            self, "output_dir", os.path.join(self.project_root, "output")
        )


@dataclass(frozen=True)
class EnvConfig:
    """Defines the Gridworld MDP Environment specification.

    Attributes:
        grid_rows:    Number of rows in the gridworld.
        grid_cols:    Number of columns in the gridworld.
        start_state:  Starting (row, col) coordinate for the agent.
        goal_state:   Terminal goal state (row, col) with positive reward.
        trap_states:  List of terminal hazard trap states (row, col) with negative reward.
        wall_states:  List of impassable wall/obstacle states (row, col).
        step_reward:  Cost/reward per transition step (living reward).
        goal_reward:  Reward for reaching the goal state.
        trap_reward:  Penalty/reward for falling into a trap state.
        success_prob: Probability of moving in the intended action direction.
        slip_prob:    Probability of slipping perpendicularly to left or right.
    """

    grid_rows: int = 5
    grid_cols: int = 5
    start_state: Tuple[int, int] = (4, 0)
    goal_state: Tuple[int, int] = (0, 4)
    trap_states: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 3), (3, 1)]
    )
    wall_states: List[Tuple[int, int]] = field(
        default_factory=lambda: [(2, 2)]
    )
    step_reward: float = -0.04
    goal_reward: float = 1.0
    trap_reward: float = -1.0
    success_prob: float = 0.8
    slip_prob: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for MDP Dynamic Programming Solvers.

    Attributes:
        algorithm_type: Choice of solver algorithm ('value_iteration' or 'policy_iteration').
        gamma:          Discount factor for future rewards (0 <= gamma < 1).
        theta:          Convergence threshold for Bellman residual error.
        max_iterations: Maximum iteration limit for Value/Policy iteration.
    """

    algorithm_type: str = "value_iteration"  # 'value_iteration' or 'policy_iteration'
    gamma: float = 0.95
    theta: float = 1e-6
    max_iterations: int = 1000


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:     Minimum severity level.
        log_to_file:   Must be False to prevent logs being saved locally.
        log_filename:  Disabled, set to empty string.
    """

    log_level: str = "INFO"
    log_to_file: bool = False
    log_filename: str = ""


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations."""

    paths: PathConfig = field(default_factory=PathConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
