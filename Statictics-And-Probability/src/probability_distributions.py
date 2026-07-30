# ============================================================================
# Probability Distributions Service Module
# ============================================================================
# Implements Discrete (Bernoulli, Binomial, Geometric, Poisson) and Continuous
# (Uniform, Normal, Exponential) distributions with exact PMF/PDF formulas,
# Expected Value E[X], Variance Var(X), CDF, random sampling, and plotting utilities.
# ============================================================================

import logging
import os
from typing import Dict, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import AnalysisConfig, PathConfig


class DistributionsService:
    """Service encapsulating probability distributions computations and plots.

    Responsibilities:
        1. Evaluate PMF/PDF, E[X], and Var(X) for Discrete Distributions:
           - Bernoulli(p)
           - Binomial(n, p)
           - Geometric(p)
           - Poisson(lambda)
        2. Evaluate PDF, E[X], and Var(X) for Continuous Distributions:
           - Uniform(a, b)
           - Normal(mu, sigma)
           - Exponential(lambda)
        3. Render distribution comparison plots and save to disk.
    """

    def __init__(
        self,
        analysis_config: AnalysisConfig,
        path_config: PathConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize DistributionsService.

        Args:
            analysis_config: Distribution parameter settings.
            path_config:     Output file path settings.
            logger:          Logging instance.
        """
        self._config = analysis_config
        self._path_config = path_config
        self._logger = logger

    def evaluate_binomial(self, k: int) -> Dict[str, Union[float, int, str]]:
        """Compute Binomial(n, p) distribution metrics and PMF for x=k.

        Formulae:
            - PMF: P(X=k) = (n choose k) * p^k * (1-p)^(n-k)
            - E[X] = n * p
            - Var(X) = n * p * (1 - p)

        Args:
            k: Target number of success outcomes.

        Returns:
            Dictionary with PMF value, mean, and variance.
        """
        n = self._config.binomial_n
        p = self._config.binomial_p
        pmf_val = float(stats.binom.pmf(k, n, p))
        mean_val = n * p
        var_val = n * p * (1.0 - p)

        return {
            "Distribution": f"Binomial(n={n}, p={p})",
            "k": k,
            "PMF_P(X=k)": pmf_val,
            "Expected_Value_E[X]": mean_val,
            "Variance_Var(X)": var_val,
        }

    def evaluate_poisson(self, k: int) -> Dict[str, Union[float, int, str]]:
        """Compute Poisson(lambda) distribution metrics and PMF for x=k.

        Formulae:
            - PMF: P(X=k) = (lambda^k * e^-lambda) / k!
            - E[X] = lambda
            - Var(X) = lambda

        Args:
            k: Target number of occurrences.

        Returns:
            Dictionary with PMF value, mean, and variance.
        """
        lam = self._config.poisson_lambda
        pmf_val = float(stats.poisson.pmf(k, lam))

        return {
            "Distribution": f"Poisson(lambda={lam})",
            "k": k,
            "PMF_P(X=k)": pmf_val,
            "Expected_Value_E[X]": lam,
            "Variance_Var(X)": lam,
        }

    def evaluate_normal(self, x: float) -> Dict[str, Union[float, int, str]]:
        """Compute Normal(mu, sigma^2) distribution PDF and parameters.

        Formulae:
            - PDF: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
            - E[X] = mu
            - Var(X) = sigma^2

        Args:
            x: Evaluation point.

        Returns:
            Dictionary with PDF value, mean, and variance.
        """
        mu = self._config.normal_mean
        sigma = self._config.normal_std
        pdf_val = float(stats.norm.pdf(x, loc=mu, scale=sigma))

        return {
            "Distribution": f"Normal(mu={mu}, sigma={sigma})",
            "x": x,
            "PDF_f(x)": pdf_val,
            "Expected_Value_E[X]": mu,
            "Variance_Var(X)": sigma ** 2,
        }

    def generate_distribution_plots(self) -> str:
        """Render multi-panel figure displaying Discrete & Continuous probability distributions.

        Panels:
            - Top-Left: Binomial PMF
            - Top-Right: Poisson PMF
            - Bottom-Left: Normal PDF
            - Bottom-Right: Exponential PDF

        Returns:
            Saved output image file path.
        """
        self._logger.info("Generating probability distribution curves figure...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Binomial
        n, p = self._config.binomial_n, self._config.binomial_p
        x_binom = np.arange(0, n + 1)
        y_binom = stats.binom.pmf(x_binom, n, p)
        axes[0, 0].bar(x_binom, y_binom, color="teal", alpha=0.8, edgecolor="black")
        axes[0, 0].set_title(f"Binomial Distribution (n={n}, p={p})")
        axes[0, 0].set_xlabel("x (Successes)")
        axes[0, 0].set_ylabel("P(X = x)")
        axes[0, 0].grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Poisson
        lam = self._config.poisson_lambda
        x_poisson = np.arange(0, 15)
        y_poisson = stats.poisson.pmf(x_poisson, lam)
        markerline, stemlines, _ = axes[0, 1].stem(x_poisson, y_poisson)
        plt.setp(stemlines, color="navy")
        plt.setp(markerline, color="navy")
        axes[0, 1].set_title(f"Poisson Distribution (lambda={lam})")
        axes[0, 1].set_xlabel("x (Events)")
        axes[0, 1].set_ylabel("P(X = x)")
        axes[0, 1].grid(True, linestyle=":", alpha=0.6)

        # Panel 3: Normal
        mu, sigma = self._config.normal_mean, self._config.normal_std
        x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        y_norm = stats.norm.pdf(x_norm, loc=mu, scale=sigma)
        axes[1, 0].plot(x_norm, y_norm, color="crimson", lw=2)
        axes[1, 0].fill_between(x_norm, y_norm, color="crimson", alpha=0.2)
        axes[1, 0].set_title(f"Normal Distribution (mu={mu}, sigma={sigma})")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("f(x)")
        axes[1, 0].grid(True, linestyle=":", alpha=0.6)

        # Panel 4: Exponential
        exp_lam = self._config.exp_lambda
        x_exp = np.linspace(0, 20, 200)
        y_exp = stats.expon.pdf(x_exp, scale=1.0 / exp_lam)
        axes[1, 1].plot(x_exp, y_exp, color="darkorange", lw=2)
        axes[1, 1].fill_between(x_exp, y_exp, color="darkorange", alpha=0.2)
        axes[1, 1].set_title(f"Exponential Distribution (lambda={exp_lam})")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("f(x)")
        axes[1, 1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        output_path = os.path.join(self._path_config.output_dir, "probability_distributions.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        self._logger.info("Distribution plots saved to: %s", output_path)
        return output_path
