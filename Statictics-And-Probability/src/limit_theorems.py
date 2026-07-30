# ============================================================================
# Limit Theorems Service Module
# ============================================================================
# Demonstrates the Law of Large Numbers (LLN) sample mean convergence
# and the Central Limit Theorem (CLT) Gaussian distribution convergence,
# along with Z-score standardization calculations per Probability infographic specs.
# ============================================================================

import logging
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import PathConfig


class LimitTheoremsService:
    """Service class demonstrating Law of Large Numbers (LLN) and Central Limit Theorem (CLT).

    Responsibilities:
        1. Simulate Law of Large Numbers sample mean trajectory across sample sizes n.
        2. Simulate Central Limit Theorem sampling distribution of means from skewed distributions.
        3. Compute Z-score standardization Z = (X - mu) / sigma.
        4. Save diagnostic convergence plots to disk.
    """

    def __init__(self, path_config: PathConfig, logger: logging.Logger) -> None:
        """Initialize LimitTheoremsService.

        Args:
            path_config: File output path settings.
            logger:      Logger instance.
        """
        self._path_config = path_config
        self._logger = logger

    def compute_z_score(self, x: float, mean: float, std_dev: float) -> float:
        """Compute Z-score standardization Z = (X - mu) / sigma.

        Args:
            x:        Observed value.
            mean:     Population mean mu.
            std_dev:  Population standard deviation sigma.

        Returns:
            Standardized Z-score value.
        """
        if std_dev <= 0:
            raise ValueError("Standard deviation must be strictly positive for Z-score.")
        z = (x - mean) / std_dev
        self._logger.info("Z-Score for x=%.2f (mu=%.2f, sigma=%.2f) -> Z = %.4f", x, mean, std_dev, z)
        return z

    def simulate_lln_and_clt(
        self, num_simulations: int = 1000, sample_sizes: Tuple[int, ...] = (5, 30, 100)
    ) -> str:
        """Simulate LLN and CLT convergence from a non-Normal Exponential population.

        Simulations:
            - Left: LLN path showing sample mean approaching true population mean as n grows.
            - Right: CLT histograms of sample means for n=5, n=30, n=100 demonstrating bell curve transition.

        Args:
            num_simulations: Number of Monte Carlo iterations for CLT.
            sample_sizes:    Tuple of sample sizes n to compare.

        Returns:
            File path of saved output plot.
        """
        self._logger.info("Running LLN and CLT Monte Carlo simulations...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # Base Non-Normal Population: Exponential(scale=10.0) -> True mu = 10.0
        true_mu = 10.0
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ---------------------------------------------------------------------
        # 1. Law of Large Numbers (LLN) Simulation
        # ---------------------------------------------------------------------
        n_max = 2000
        single_sample = np.random.exponential(scale=true_mu, size=n_max)
        running_means = np.cumsum(single_sample) / np.arange(1, n_max + 1)

        axes[0].plot(range(1, n_max + 1), running_means, color="navy", label="Sample Mean X_bar_n")
        axes[0].axhline(y=true_mu, color="crimson", linestyle="--", lw=2, label=f"True Mean mu = {true_mu}")
        axes[0].set_title("Law of Large Numbers (LLN) Convergence")
        axes[0].set_xlabel("Sample Size (n)")
        axes[0].set_ylabel("Sample Mean")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # ---------------------------------------------------------------------
        # 2. Central Limit Theorem (CLT) Simulation
        # ---------------------------------------------------------------------
        colors = ["teal", "darkorange", "purple"]
        for idx, n_samples in enumerate(sample_sizes):
            # Draw num_simulations samples of size n_samples
            sample_matrix = np.random.exponential(scale=true_mu, size=(num_simulations, n_samples))
            sample_means = sample_matrix.mean(axis=1)

            axes[1].hist(
                sample_means,
                bins=30,
                density=True,
                alpha=0.4,
                color=colors[idx % len(colors)],
                label=f"n = {n_samples}",
            )

        axes[1].axhline(y=0, color="black", lw=1)
        axes[1].set_title("Central Limit Theorem (CLT) Distribution of Means")
        axes[1].set_xlabel("Sample Mean")
        axes[1].set_ylabel("Density")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        output_path = os.path.join(self._path_config.output_dir, "clt_and_lln_demonstration.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        self._logger.info("LLN and CLT plot saved to: %s", output_path)
        return output_path
