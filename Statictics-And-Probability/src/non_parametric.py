# ============================================================================
# Non-Parametric Statistical Methods Service Module
# ============================================================================
# Implements non-parametric statistical hypothesis tests (Mann-Whitney U Test,
# Wilcoxon Signed-Rank Test, Kruskal-Wallis Test, Spearman Rank Correlation)
# used when normality distribution assumptions are violated per Statistics infographic.
# ============================================================================

import logging
from typing import Dict, Union

import numpy as np
from scipy import stats

from config import DataConfig


class NonParametricService:
    """Service encapsulating non-parametric statistical tests when normality is violated.

    Responsibilities:
        1. Mann-Whitney U Test: Non-parametric comparison of 2 independent groups.
        2. Wilcoxon Signed-Rank Test: Non-parametric comparison of paired/related samples.
        3. Kruskal-Wallis H Test: Non-parametric comparison across 3+ independent groups.
        4. Spearman Rank Correlation (rho): Monotonic association measure between variables.
    """

    def __init__(self, data_config: DataConfig, logger: logging.Logger) -> None:
        """Initialize NonParametricService.

        Args:
            data_config: Dataset and testing parameter configurations.
            logger:      Logging instance.
        """
        self._config = data_config
        self._logger = logger

    def mann_whitney_u_test(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> Dict[str, Union[float, str]]:
        """Perform Mann-Whitney U Test for two independent groups.

        H0: Distribution of both groups is equal.
        Ha: Distribution of one group is stochastically larger than the other.

        Args:
            group1: Observations for group 1.
            group2: Observations for group 2.

        Returns:
            Results dictionary with U statistic, p-value, decision.
        """
        self._logger.info("Executing Mann-Whitney U Test...")
        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "Test_Type": "Mann-Whitney U Test",
            "U_Statistic": float(u_stat),
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }

    def wilcoxon_signed_rank_test(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> Dict[str, Union[float, str]]:
        """Perform Wilcoxon Signed-Rank Test for paired observations.

        H0: Median difference between pairs is zero.
        Ha: Median difference between pairs is non-zero.

        Args:
            group1: Paired observations sample 1.
            group2: Paired observations sample 2.

        Returns:
            Results dictionary with W statistic, p-value, decision.
        """
        self._logger.info("Executing Wilcoxon Signed-Rank Test...")
        stat_val, p_val = stats.wilcoxon(group1, group2)
        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "Test_Type": "Wilcoxon Signed-Rank Test",
            "W_Statistic": float(stat_val),
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }

    def kruskal_wallis_test(
        self, *groups: np.ndarray
    ) -> Dict[str, Union[float, str]]:
        """Perform Kruskal-Wallis H Test across 3+ independent groups.

        H0: All population medians are equal.
        Ha: At least one population median differs.

        Args:
            *groups: Tuple of numerical group arrays.

        Returns:
            Results dictionary with H statistic, p-value, decision.
        """
        self._logger.info("Executing Kruskal-Wallis Test across %d groups...", len(groups))
        h_stat, p_val = stats.kruskal(*groups)
        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "Test_Type": "Kruskal-Wallis Test",
            "H_Statistic": float(h_stat),
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }

    def spearman_rank_correlation(
        self, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Compute Spearman Rank Correlation Coefficient rho.

        Monotonic correlation measure calculated on ranks of the variables.

        Args:
            x: Input feature array.
            y: Input feature array.

        Returns:
            Dictionary containing Spearman rho and p-value.
        """
        self._logger.info("Computing Spearman Rank Correlation...")
        rho_val, p_val = stats.spearmanr(x, y)
        self._logger.info("  Spearman rho = %.4f (p-value = %.4e)", rho_val, p_val)

        return {
            "Spearman_Rho": float(rho_val),
            "P_Value": float(p_val),
        }
