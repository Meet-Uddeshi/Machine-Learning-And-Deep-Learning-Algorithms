# ============================================================================
# MDP Dynamic Programming Solvers & Visualization Service
# ============================================================================
# Implements Value Iteration and Policy Iteration dynamic programming algorithms,
# extracts optimal policy pi*(s), generates grid heatmaps, policy arrow grids,
# convergence curves, and formats reports.
# ============================================================================

import logging
import os
import time
from typing import Dict, List, Tuple

# Non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import ModelConfig, PathConfig
from mdp_environment import GridworldMDP


class MDPSolverService:
    """Service encapsulating Dynamic Programming algorithms for solving MDPs.

    Responsibilities:
        1. Solve MDP via Value Iteration or Policy Iteration.
        2. Extract optimal policy pi*(s).
        3. Track Bellman error convergence history.
        4. Visualise state value heatmaps, policy direction grids, and error curves.
        5. Write text and markdown analysis reports.
    """

    def __init__(
        self,
        env: GridworldMDP,
        model_config: ModelConfig,
        path_config: PathConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the MDP Solver service.

        Args:
            env:          GridworldMDP environment instance.
            model_config: Model hyperparameters (gamma, theta, algorithm choice).
            path_config:  Path settings for output saving.
            logger:       Logger instance.
        """
        self._env = env
        self._model_config = model_config
        self._path_config = path_config
        self._logger = logger

    # -- Public workflow methods ---------------------------------------------

    def solve_and_evaluate(self) -> dict:
        """Execute the configured DP algorithm (Value Iteration or Policy Iteration).

        Returns:
            Dictionary containing state values V*(s), optimal policy pi*(s), and metrics.
        """
        self._logger.info("=" * 70)
        self._logger.info("SOLVING MDP (%s)", self._model_config.algorithm_type.upper())
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        start_time = time.perf_counter()

        if self._model_config.algorithm_type == "value_iteration":
            v_star, delta_history = self._value_iteration()
            pi_star = self._extract_optimal_policy(v_star)
            num_iterations = len(delta_history)
        elif self._model_config.algorithm_type == "policy_iteration":
            v_star, pi_star, delta_history = self._policy_iteration()
            num_iterations = len(delta_history)
        else:
            raise ValueError(f"Unknown algorithm: {self._model_config.algorithm_type}")

        elapsed = time.perf_counter() - start_time

        self._logger.info("DP Solver completed in %.3f seconds.", elapsed)
        self._logger.info("Total iterations to converge: %d", num_iterations)
        self._logger.info("Final Bellman Residual Error: %.8f", delta_history[-1])

        results = {
            "v_star": v_star,
            "pi_star": pi_star,
            "delta_history": delta_history,
            "num_iterations": num_iterations,
            "elapsed_time": elapsed,
        }

        self._log_results(results)
        self._save_results(results)
        self._generate_plots(results)
        self._save_analysis(results)

        return results

    # -- Private DP Solvers --------------------------------------------------

    def _value_iteration(self) -> Tuple[Dict[Tuple[int, int], float], List[float]]:
        """Run Value Iteration algorithm using Bellman Optimality Operator.

        Returns:
            Tuple of (V_star, delta_history).
        """
        v: Dict[Tuple[int, int], float] = {s: 0.0 for s in self._env.states}
        
        # Set terminal rewards
        v[self._env.goal_state] = self._env._config.goal_reward
        for trap in self._env.trap_states:
            v[trap] = self._env._config.trap_reward

        delta_history: List[float] = []

        for it in range(1, self._model_config.max_iterations + 1):
            delta = 0.0
            new_v = v.copy()

            for s in self._env.states:
                if self._env.is_terminal(s):
                    continue

                # Compute Q(s, a) for all actions
                action_values = []
                for a in self._env.actions:
                    q_val = 0.0
                    for next_s, r, p in self._env.get_transitions(s, a):
                        q_val += p * (r + self._model_config.gamma * v[next_s])
                    action_values.append(q_val)

                max_q = max(action_values)
                delta = max(delta, abs(max_q - v[s]))
                new_v[s] = max_q

            v = new_v
            delta_history.append(delta)

            if it % 10 == 0 or delta < self._model_config.theta:
                self._logger.info("Iter %3d | Max Bellman Error Delta: %.8f", it, delta)

            if delta < self._model_config.theta:
                self._logger.info("Value Iteration converged at iteration %d.", it)
                break

        return v, delta_history

    def _policy_iteration(
        self,
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int], List[float]]:
        """Run Policy Iteration algorithm (Evaluation + Improvement).

        Returns:
            Tuple of (V_star, pi_star, delta_history).
        """
        v: Dict[Tuple[int, int], float] = {s: 0.0 for s in self._env.states}
        pi: Dict[Tuple[int, int], int] = {
            s: self._env.ACTION_UP for s in self._env.states if not self._env.is_terminal(s)
        }

        # Terminal rewards
        v[self._env.goal_state] = self._env._config.goal_reward
        for trap in self._env.trap_states:
            v[trap] = self._env._config.trap_reward

        delta_history: List[float] = []

        for outer_it in range(1, self._model_config.max_iterations + 1):
            # 1. Policy Evaluation
            for eval_it in range(self._model_config.max_iterations):
                delta = 0.0
                new_v = v.copy()

                for s in self._env.states:
                    if self._env.is_terminal(s):
                        continue

                    a = pi[s]
                    q_val = 0.0
                    for next_s, r, p in self._env.get_transitions(s, a):
                        q_val += p * (r + self._model_config.gamma * v[next_s])

                    delta = max(delta, abs(q_val - v[s]))
                    new_v[s] = q_val

                v = new_v
                if delta < self._model_config.theta:
                    break

            # 2. Policy Improvement
            policy_stable = True
            max_improve_delta = 0.0

            for s in self._env.states:
                if self._env.is_terminal(s):
                    continue

                old_action = pi[s]

                # Find greedy action
                action_values = []
                for a in self._env.actions:
                    q_val = 0.0
                    for next_s, r, p in self._env.get_transitions(s, a):
                        q_val += p * (r + self._model_config.gamma * v[next_s])
                    action_values.append(q_val)

                best_action = self._env.actions[int(np.argmax(action_values))]
                max_improve_delta = max(max_improve_delta, abs(max(action_values) - v[s]))

                if best_action != old_action:
                    policy_stable = False
                    pi[s] = best_action

            delta_history.append(max_improve_delta)
            self._logger.info("Policy Iteration Outer Iter %2d | Policy Stable: %s | Delta: %.8f", outer_it, policy_stable, max_improve_delta)

            if policy_stable:
                self._logger.info("Policy Iteration converged at outer iteration %d.", outer_it)
                break

        return v, pi, delta_history

    def _extract_optimal_policy(
        self, v_star: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], int]:
        """Extract greedy policy pi*(s) from optimal state values V*(s)."""
        pi_star: Dict[Tuple[int, int], int] = {}

        for s in self._env.states:
            if self._env.is_terminal(s):
                continue

            action_values = []
            for a in self._env.actions:
                q_val = 0.0
                for next_s, r, p in self._env.get_transitions(s, a):
                    q_val += p * (r + self._model_config.gamma * v_star[next_s])
                action_values.append(q_val)

            best_action = self._env.actions[int(np.argmax(action_values))]
            pi_star[s] = best_action

        return pi_star

    # -- Helpers & Output Persistence ----------------------------------------

    def _log_hyperparameters(self) -> None:
        """Log configuration parameters."""
        self._logger.info("MDP Solvers Hyperparameters:")
        self._logger.info("  algorithm_type: %s", self._model_config.algorithm_type)
        self._logger.info("  gamma          : %.4f", self._model_config.gamma)
        self._logger.info("  theta (epsilon): %.1e", self._model_config.theta)
        self._logger.info("  max_iterations : %d", self._model_config.max_iterations)

    def _log_results(self, results: dict) -> None:
        """Log state values and optimal policy to console."""
        v_star = results["v_star"]
        pi_star = results["pi_star"]

        self._logger.info("-" * 70)
        self._logger.info("OPTIMAL STATE VALUES V*(s):")
        self._logger.info("-" * 70)
        
        # Grid representation of values
        v_matrix = np.full((self._env.rows, self._env.cols), np.nan)
        for (r, c), val in v_star.items():
            v_matrix[r, c] = val

        for r in range(self._env.rows):
            row_str = "  "
            for c in range(self._env.cols):
                if (r, c) in self._env.wall_states:
                    row_str += " [ WALL ] "
                else:
                    row_str += f" [{v_matrix[r, c]:7.3f}] "
            self._logger.info(row_str)

        self._logger.info("-" * 70)
        self._logger.info("OPTIMAL POLICY DIRECTION GRID pi*(s):")
        self._logger.info("-" * 70)
        for r in range(self._env.rows):
            row_str = "  "
            for c in range(self._env.cols):
                if (r, c) in self._env.wall_states:
                    row_str += "  WALL  "
                elif (r, c) == self._env.goal_state:
                    row_str += "  GOAL  "
                elif (r, c) in self._env.trap_states:
                    row_str += "  TRAP  "
                else:
                    a = pi_star[(r, c)]
                    row_str += f"   {self._env.ACTION_ARROWS[a]}    "
            self._logger.info(row_str)
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Save results summary to mdp_results.txt."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        results_path = os.path.join(self._path_config.output_dir, "mdp_results.txt")

        v_star = results["v_star"]
        pi_star = results["pi_star"]

        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write(f"MARKOV DECISION PROCESS (MDP) RESULTS ({self._model_config.algorithm_type.upper()})\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("ENVIRONMENT & SOLVER CONFIGURATION\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Grid Size        : {self._env.rows} x {self._env.cols}\n")
            fh.write(f"  Algorithm        : {self._model_config.algorithm_type}\n")
            fh.write(f"  Discount (gamma) : {self._model_config.gamma}\n")
            fh.write(f"  Tolerance (theta): {self._model_config.theta}\n")
            fh.write(f"  Iterations Run   : {results['num_iterations']}\n")
            fh.write(f"  Execution Time   : {results['elapsed_time']:.4f} seconds\n\n")

            fh.write("OPTIMAL STATE VALUES V*(s)\n")
            fh.write("-" * 40 + "\n")
            for r in range(self._env.rows):
                row_str = ""
                for c in range(self._env.cols):
                    if (r, c) in self._env.wall_states:
                        row_str += f"{'WALL':>8}"
                    else:
                        row_str += f"{v_star[(r, c)]:8.3f}"
                fh.write(row_str + "\n")
            fh.write("\n")

            fh.write("OPTIMAL POLICY GRID pi*(s)\n")
            fh.write("-" * 40 + "\n")
            for r in range(self._env.rows):
                row_str = ""
                for c in range(self._env.cols):
                    if (r, c) in self._env.wall_states:
                        row_str += f"{'WALL':>8}"
                    elif (r, c) == self._env.goal_state:
                        row_str += f"{'GOAL':>8}"
                    elif (r, c) in self._env.trap_states:
                        row_str += f"{'TRAP':>8}"
                    else:
                        a = pi_star[(r, c)]
                        arrow_name = f"{self._env.ACTION_ARROWS[a]} ({self._env.ACTION_NAMES[a]})"
                        row_str += f"{arrow_name:>10}"
                fh.write(row_str + "\n")
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("MDP results saved to: %s", results_path)

    def _generate_plots(self, results: dict) -> None:
        """Generate state value heatmap, policy grid plot, and convergence curve."""
        self._logger.info("Generating MDP visualization plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        v_star = results["v_star"]
        pi_star = results["pi_star"]

        # Plot 1: State Value Matrix Heatmap
        v_matrix = np.full((self._env.rows, self._env.cols), np.nan)
        for (r, c), val in v_star.items():
            v_matrix[r, c] = val

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            v_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={'label': 'Optimal Value V*(s)'},
            linewidths=1,
            linecolor="black"
        )
        plt.title(f"MDP Optimal State Values V*(s) ({self._model_config.algorithm_type.upper()})")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.tight_layout()
        heatmap_path = os.path.join(self._path_config.output_dir, "value_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        # Plot 2: Optimal Policy Arrow Grid
        plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid cells
        for r in range(self._env.rows):
            for c in range(self._env.cols):
                rect = plt.Rectangle((c, self._env.rows - 1 - r), 1, 1, facecolor="white", edgecolor="black")
                if (r, c) in self._env.wall_states:
                    rect.set_facecolor("gray")
                    ax.add_patch(rect)
                    ax.text(c + 0.5, self._env.rows - 1 - r + 0.5, "WALL", ha="center", va="center", fontweight="bold")
                elif (r, c) == self._env.goal_state:
                    rect.set_facecolor("lightgreen")
                    ax.add_patch(rect)
                    ax.text(c + 0.5, self._env.rows - 1 - r + 0.5, "GOAL\n+1.0", ha="center", va="center", fontweight="bold", color="darkgreen")
                elif (r, c) in self._env.trap_states:
                    rect.set_facecolor("salmon")
                    ax.add_patch(rect)
                    ax.text(c + 0.5, self._env.rows - 1 - r + 0.5, "TRAP\n-1.0", ha="center", va="center", fontweight="bold", color="darkred")
                else:
                    ax.add_patch(rect)
                    a = pi_star[(r, c)]
                    arrow = self._env.ACTION_ARROWS[a]
                    val_text = f"v={v_star[(r, c)]:.2f}"
                    ax.text(c + 0.5, self._env.rows - 1 - r + 0.6, arrow, ha="center", va="center", fontsize=22, fontweight="bold", color="navy")
                    ax.text(c + 0.5, self._env.rows - 1 - r + 0.25, val_text, ha="center", va="center", fontsize=9, color="black")

        ax.set_xlim(0, self._env.cols)
        ax.set_ylim(0, self._env.rows)
        ax.set_xticks(np.arange(0.5, self._env.cols, 1))
        ax.set_yticks(np.arange(0.5, self._env.rows, 1))
        ax.set_xticklabels([str(i) for i in range(self._env.cols)])
        ax.set_yticklabels([str(self._env.rows - 1 - i) for i in range(self._env.rows)])
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        plt.title(f"Optimal Policy Grid pi*(s) ({self._model_config.algorithm_type.upper()})")
        plt.tight_layout()
        policy_path = os.path.join(self._path_config.output_dir, "policy_grid.png")
        plt.savefig(policy_path, dpi=300)
        plt.close()

        # Plot 3: Convergence Error Curve
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(results["delta_history"]) + 1), results["delta_history"], marker="o", color="crimson", linewidth=2)
        plt.yscale("log")
        plt.title(f"Bellman Residual Error Convergence ({self._model_config.algorithm_type.upper()})")
        plt.xlabel("Iteration Step")
        plt.ylabel("Max Value Delta Error (log scale)")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        conv_path = os.path.join(self._path_config.output_dir, "convergence_curve.png")
        plt.savefig(conv_path, dpi=300)
        plt.close()

        self._logger.info("Evaluation plots saved successfully.")

    def _save_analysis(self, results: dict) -> None:
        """Write technical markdown explanation report mdp_analysis.md."""
        report_path = os.path.join(self._path_config.output_dir, "mdp_analysis.md")

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Markov Decision Process (MDP) Technical Analysis Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                f"This report presents the mathematical and empirical results of solving a stochastic Gridworld "
                f"Markov Decision Process (MDP) using **{self._model_config.algorithm_type.upper()}**. "
            )
            fh.write(
                f"The algorithm converged in **{results['num_iterations']}** iterations in **{results['elapsed_time']:.4f}** seconds "
                f"with a final Bellman residual error threshold of **{results['delta_history'][-1]:.2e}**.\n\n"
            )

            fh.write("## 2. Environment & Model Configuration\n\n")
            fh.write("| Parameter | Value |\n")
            fh.write("|-----------|-------|\n")
            fh.write(f"| `grid_size` | `{self._env.rows} x {self._env.cols}` |\n")
            fh.write(f"| `algorithm_type` | `{self._model_config.algorithm_type}` |\n")
            fh.write(f"| `discount_factor (gamma)` | `{self._model_config.gamma}` |\n")
            fh.write(f"| `convergence_threshold (theta)` | `{self._model_config.theta}` |\n")
            fh.write(f"| `success_prob (intended action)` | `{self._env._config.success_prob}` |\n")
            fh.write(f"| `slip_prob (lateral noise)` | `{self._env._config.slip_prob}` |\n")
            fh.write(f"| `step_reward` | `{self._env._config.step_reward}` |\n\n")

            fh.write("## 3. Theoretical Framework & Equations\n\n")
            fh.write(
                "- **Bellman Optimality Equation for V*(s)**:\n"
                "  $$V^*(s) = \\max_{a \\in A} \\sum_{s' \\in S} P(s' \\mid s, a) \\left[ R(s, a, s') + \\gamma V^*(s') \\right]$$\n"
                "- **Bellman Optimality Equation for Q*(s, a)**:\n"
                "  $$Q^*(s, a) = \\sum_{s' \\in S} P(s' \\mid s, a) \\left[ R(s, a, s') + \\gamma \\max_{a'} Q^*(s', a') \\right]$$\n"
                "- **Optimal Policy Extraction \\pi*(s)**:\n"
                "  $$\\pi^*(s) = \\arg\\max_{a \\in A} \\sum_{s' \\in S} P(s' \\mid s, a) \\left[ R(s, a, s') + \\gamma V^*(s') \\right]$$\n\n"
            )

            fh.write("## 4. Optimal Policy Interpretation\n\n")
            fh.write(
                "1. **Shortest Path & Risk Avoidance**: Because a step cost of `-0.04` is enforced, the optimal policy "
                "steers the agent along the shortest path toward the goal while actively routing around the hazard traps.\n"
                "2. **Stochastic Slipping Buffer**: Due to lateral slipping probability (`0.1`), states adjacent to traps "
                "direct the agent away from hazard edges to minimize accidental entry into negative reward states.\n\n"
            )

            fh.write("## 5. Output Artifacts Summary\n\n")
            fh.write("| File | Description |\n")
            fh.write("|------|-------------|\n")
            fh.write("| `mdp_results.txt` | Text report of optimal values V*(s), policy grid, and convergence history |\n")
            fh.write("| `mdp_analysis.md` | Technical report on MDP mathematics and gridworld policy |\n")
            fh.write("| `value_heatmap.png` | 2D color heatmap of optimal state values V*(s) |\n")
            fh.write("| `policy_grid.png` | Visual grid displaying optimal policy arrows pi*(s) |\n")
            fh.write("| `convergence_curve.png` | Convergence log plot of Bellman residual error over iterations |\n")

        self._logger.info("MDP technical report saved to: %s", report_path)
