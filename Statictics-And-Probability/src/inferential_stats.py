# ============================================================================
# Inferential Statistics & Hypothesis Testing Service Module
# ============================================================================
# Implements Point Estimation, Confidence Intervals (Z and T intervals),
# and Hypothesis Testing algorithms (Z-test, T-tests, Chi-Square test,
# F-test of variance, One-Way ANOVA) per Statistics infographic specifications.
# ============================================================================

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from config import DataConfig


class InferentialStatisticsService:
    """Service encapsulating point estimation, confidence intervals, and hypothesis testing.

    Responsibilities:
        1. Calculate Z and T Confidence Intervals for population means.
        2. Perform 1-Sample and 2-Sample (Independent & Paired) T-tests.
        3. Perform Chi-Square Test of Independence on contingency tables.
        4. Perform F-test of Variance Equality.
        5. Perform One-Way ANOVA for multi-group mean comparisons.
    """

    def __init__(self, data_config: DataConfig, logger: logging.Logger) -> None:
        """Initialize InferentialStatisticsService.

        Args:
            data_config: Dataset and testing parameters (alpha, confidence level).
            logger:      Logging instance.
        """
        self._config = data_config
        self._logger = logger

    def confidence_interval_mean(
        self, data: np.ndarray, known_std: float = None
    ) -> Dict[str, float]:
        """Compute Confidence Interval for Population Mean (Z or T distribution).

        Formulae:
            - Known sigma (Z-interval): x_bar +/- z_(alpha/2) * (sigma / sqrt(n))
            - Unknown sigma (T-interval): x_bar +/- t_(alpha/2, n-1) * (s / sqrt(n))

        Args:
            data:      Sample observations array.
            known_std: Known population standard deviation sigma (Optional).

        Returns:
            Dictionary containing point estimate, margin of error, lower and upper bounds.
        """
        arr = np.asarray(data, dtype=float)
        n = len(arr)
        mean_val = float(np.mean(arr))
        confidence = self._config.confidence_level
        alpha = 1.0 - confidence

        if known_std is not None:
            # Z-interval
            z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
            margin_error = z_crit * (known_std / np.sqrt(n))
            method = "Z-Interval (known sigma)"
        else:
            # T-interval
            s = float(np.std(arr, ddof=1))
            t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
            margin_error = t_crit * (s / np.sqrt(n))
            method = "T-Interval (unknown sigma)"

        lower_bound = mean_val - margin_error
        upper_bound = mean_val + margin_error

        results = {
            "Method": method,
            "Point_Estimate_Mean": mean_val,
            "Margin_of_Error": margin_error,
            "Lower_CI": lower_bound,
            "Upper_CI": upper_bound,
            "Confidence_Level": confidence,
        }

        self._logger.info("%s (%.0f%% CI):", method, confidence * 100)
        self._logger.info("  Mean : %.4f +/- %.4f", mean_val, margin_error)
        self._logger.info("  Bounds : [%.4f, %.4f]", lower_bound, upper_bound)

        return results

    def one_sample_t_test(
        self, data: np.ndarray, pop_mean_h0: float
    ) -> Dict[str, Union[float, str]]:
        """Perform One-Sample Student's T-Test.

        H0: mu = pop_mean_h0
        Ha: mu != pop_mean_h0

        Args:
            data:        Sample data array.
            pop_mean_h0: Hypothesized population mean under H0.

        Returns:
            Results dictionary with t-statistic, df, p-value, decision.
        """
        self._logger.info("Executing One-Sample T-Test against H0: mu = %.2f...", pop_mean_h0)
        arr = np.asarray(data, dtype=float)
        t_stat, p_val = stats.ttest_1samp(arr, popmean=pop_mean_h0)
        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "Test_Statistic": float(t_stat),
            "Degrees_of_Freedom": len(arr) - 1,
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }

    def two_sample_t_test(
        self, group1: np.ndarray, group2: np.ndarray, paired: bool = False
    ) -> Dict[str, Union[float, str]]:
        """Perform Two-Sample T-Test (Independent or Paired).

        H0: mu1 = mu2
        Ha: mu1 != mu2

        Args:
            group1: Array for sample 1.
            group2: Array for sample 2.
            paired: If True, performs Paired T-Test; else Independent T-Test.

        Returns:
            Results dictionary.
        """
        test_type = "Paired T-Test" if paired else "Independent 2-Sample T-Test"
        self._logger.info("Executing %s...", test_type)

        if paired:
            t_stat, p_val = stats.ttest_rel(group1, group2)
            df_val = len(group1) - 1
        else:
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)
            df_val = len(group1) + len(group2) - 2

        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "Test_Type": test_type,
            "Test_Statistic": float(t_stat),
            "Degrees_of_Freedom": df_val,
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }

    def chi_square_independence_test(
        self, contingency_table: pd.DataFrame
    ) -> Dict[str, Union[float, str]]:
        """Perform Chi-Square Test of Independence on a 2D contingency table.

        H0: Categorical variables are independent.
        Ha: Categorical variables are dependent / associated.

        Args:
            contingency_table: 2D frequency cross-tabulation table.

        Returns:
            Results dictionary with Chi2 statistic, df, p-value, decision.
        """
        self._logger.info("Executing Chi-Square Test of Independence...")
        chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency_table)
        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "Chi2_Statistic": float(chi2_stat),
            "Degrees_of_Freedom": dof,
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }

    def one_way_anova(
        self, *groups: np.ndarray
    ) -> Dict[str, Union[float, str]]:
        """Perform One-Way Analysis of Variance (ANOVA) across multiple groups.

        Formula: F = MSB / MSW

        H0: mu1 = mu2 = ... = muk
        Ha: At least one group mean is different.

        Args:
            *groups: Tuple of numerical group arrays.

        Returns:
            Results dictionary with F-statistic, p-value, decision.
        """
        self._logger.info("Executing One-Way ANOVA across %d groups...", len(groups))
        f_stat, p_val = stats.f_oneway(*groups)
        alpha = self._config.alpha_significance
        decision = "Reject H0" if p_val <= alpha else "Fail to Reject H0"

        return {
            "F_Statistic": float(f_stat),
            "P_Value": float(p_val),
            "Alpha": alpha,
            "Decision": decision,
        }
