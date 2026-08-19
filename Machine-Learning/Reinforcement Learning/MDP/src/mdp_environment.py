# ============================================================================
# Gridworld MDP Environment Simulation
# ============================================================================
# Formulates the Markov Decision Process 5-tuple (S, A, P, R, gamma) for a
# 2D Gridworld environment with stochastic transition dynamics.
# ============================================================================

import logging
from typing import Dict, List, Set, Tuple

from config import EnvConfig


class GridworldMDP:
    """Class representing a stochastic Gridworld Markov Decision Process.

    Attributes:
        states:        List of all valid state coordinates (row, col).
        actions:       List of action identifiers (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT).
        action_names:  Human-readable action string labels.
        terminal_states: Set of goal and trap states.
    """

    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    # Direction vectors (delta_row, delta_col)
    OFFSETS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1),  # LEFT
    }

    ACTION_NAMES = {
        0: "UP",
        1: "RIGHT",
        2: "DOWN",
        3: "LEFT",
    }

    ACTION_ARROWS = {
        0: "^",
        1: ">",
        2: "v",
        3: "<",
    }

    def __init__(self, env_config: EnvConfig, logger: logging.Logger) -> None:
        """Initialize the MDP Gridworld Environment.

        Args:
            env_config: Environment specifications (grid size, rewards, noise).
            logger:     Logger instance.
        """
        self._config = env_config
        self._logger = logger

        self.rows = env_config.grid_rows
        self.cols = env_config.grid_cols
        self.goal_state = env_config.goal_state
        self.trap_states = set(env_config.trap_states)
        self.wall_states = set(env_config.wall_states)
        self.terminal_states = set([self.goal_state]).union(self.trap_states)

        self.actions = [self.ACTION_UP, self.ACTION_RIGHT, self.ACTION_DOWN, self.ACTION_LEFT]

        # Construct all valid states (exclude walls)
        self.states: List[Tuple[int, int]] = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.wall_states
        ]

        self._logger.info(
            "Gridworld MDP Initialized -- Grid: %dx%d | Total States: %d | Walls: %d",
            self.rows, self.cols, len(self.states), len(self.wall_states)
        )

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if a state is a terminal goal or trap state."""
        return state in self.terminal_states

    def get_transitions(
        self, state: Tuple[int, int], action: int
    ) -> List[Tuple[Tuple[int, int], float, float]]:
        """Compute state transition dynamics P(s' | s, a) and rewards R(s, a, s').

        Stochastic dynamics:
            - Intended direction action with probability `success_prob` (0.8).
            - Left perpendicular slip with probability `slip_prob` (0.1).
            - Right perpendicular slip with probability `slip_prob` (0.1).
            - If moving hits a boundary or wall, the agent stays in the current state.

        Args:
            state:  Current state (row, col).
            action: Action taken (0..3).

        Returns:
            List of tuples: `[(next_state, reward, probability)]`.
        """
        if self.is_terminal(state):
            # Terminal states transition to self with 0 reward
            return [(state, 0.0, 1.0)]

        # Perpendicular actions for slip noise
        # UP (0) -> slip LEFT (3) or RIGHT (1)
        # RIGHT (1) -> slip UP (0) or DOWN (2)
        # DOWN (2) -> slip RIGHT (1) or LEFT (3)
        # LEFT (3) -> slip DOWN (2) or UP (0)
        left_slip_action = (action - 1) % 4
        right_slip_action = (action + 1) % 4

        possible_moves = [
            (action, self._config.success_prob),
            (left_slip_action, self._config.slip_prob),
            (right_slip_action, self._config.slip_prob),
        ]

        transitions_dict: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for act, prob in possible_moves:
            dr, dc = self.OFFSETS[act]
            next_r = state[0] + dr
            next_c = state[1] + dc

            # Boundary or wall collision check
            if (
                next_r < 0
                or next_r >= self.rows
                or next_c < 0
                or next_c >= self.cols
                or (next_r, next_c) in self.wall_states
            ):
                next_state = state
            else:
                next_state = (next_r, next_c)

            # Determine reward
            if next_state == self.goal_state:
                reward = self._config.goal_reward
            elif next_state in self.trap_states:
                reward = self._config.trap_reward
            else:
                reward = self._config.step_reward

            # Accumulate probabilities if different moves land in same next_state
            if next_state in transitions_dict:
                existing_reward, existing_prob = transitions_dict[next_state]
                transitions_dict[next_state] = (reward, existing_prob + prob)
            else:
                transitions_dict[next_state] = (reward, prob)

        return [
            (next_s, r, p) for next_s, (r, p) in transitions_dict.items()
        ]
