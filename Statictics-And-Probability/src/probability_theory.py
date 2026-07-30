# ============================================================================
# Probability Theory Service Module
# ============================================================================
# Implements fundamental probability concepts, empirical sample probabilities,
# rules (Addition, Complement, Conditional, Multiplication, Total Probability),
# Bayes' Theorem solver, independence testing, and common probability inequalities
# (Markov, Chebyshev, Union Bound) applied to the GPU database.
# ============================================================================

import logging
from typing import Dict, List, Tuple

import pandas as pd


class ProbabilityService:
    """Service encapsulating probability calculations, empirical rules, Bayes' theorem, and inequalities.

    Responsibilities:
        1. Evaluate classical and empirical event probabilities on GPU data.
        2. Apply addition rule for mutually exclusive or overlapping events.
        3. Compute conditional probability P(A|B) and check event independence.
        4. Solve Bayes' theorem problems (e.g. GPU high-performance power classification).
        5. Calculate upper bounds using Markov's, Chebyshev's, and Union Bound inequalities.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize ProbabilityService.

        Args:
            logger: Configured logging instance.
        """
        self._logger = logger

    def compute_empirical_gpu_probabilities(
        self, df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute empirical probabilities directly from GPU dataset observations.

        Events Evaluated:
            - Event A: GPU TDP > 150 Watts (High Power GPU)
            - Event B: Manufacturer is Nvidia
            - Event A and B: High Power Nvidia GPU

        Returns:
            Dictionary with P(A), P(B), P(A and B), and P(A|B).
        """
        self._logger.info("Computing empirical probabilities on GPU dataset...")
        total_gpus = len(df)
        if total_gpus == 0:
            raise ValueError("GPU DataFrame is empty.")

        high_tdp_count = len(df[df["tdp_watts"] > 150])
        nvidia_count = len(df[df["manufacturer_clean"] == "Nvidia"])
        high_tdp_nvidia_count = len(
            df[(df["tdp_watts"] > 150) & (df["manufacturer_clean"] == "Nvidia")]
        )

        p_high_tdp = high_tdp_count / total_gpus
        p_nvidia = nvidia_count / total_gpus
        p_high_tdp_and_nvidia = high_tdp_nvidia_count / total_gpus

        p_high_tdp_given_nvidia = (
            p_high_tdp_and_nvidia / p_nvidia if p_nvidia > 0 else 0.0
        )

        results = {
            "Total_GPUs": total_gpus,
            "P(TDP > 150W)": p_high_tdp,
            "P(Nvidia)": p_nvidia,
            "P(TDP > 150W and Nvidia)": p_high_tdp_and_nvidia,
            "P(TDP > 150W | Nvidia)": p_high_tdp_given_nvidia,
        }

        self._logger.info("Empirical Probability Results:")
        for k, v in results.items():
            self._logger.info("  %-30s : %.4f", k, float(v))

        return results

    def compute_classical_probability(
        self, favorable_outcomes: int, total_outcomes: int
    ) -> float:
        """Compute classical probability P(A) = Favorable Outcomes / Total Outcomes.

        Args:
            favorable_outcomes: Number of outcomes satisfying event A.
            total_outcomes:     Total sample space size S.

        Returns:
            Probability value between 0.0 and 1.0.
        """
        if total_outcomes <= 0:
            raise ValueError("Total outcomes in sample space must be strictly positive.")
        if favorable_outcomes < 0 or favorable_outcomes > total_outcomes:
            raise ValueError(
                "Favorable outcomes must be non-negative and <= total outcomes."
            )

        prob = favorable_outcomes / total_outcomes
        self._logger.info(
            "Classical Probability P(A) = %d / %d = %.4f",
            favorable_outcomes,
            total_outcomes,
            prob,
        )
        return prob

    def addition_rule(
        self, p_a: float, p_b: float, p_a_and_b: float = 0.0
    ) -> float:
        """Calculate P(A U B) = P(A) + P(B) - P(A intersect B).

        Args:
            p_a:       Probability of event A.
            p_b:       Probability of event B.
            p_a_and_b: Probability of joint event (A and B). Default 0.0 (mutually exclusive).

        Returns:
            P(A U B) value.
        """
        p_union = p_a + p_b - p_a_and_b
        self._logger.info(
            "Addition Rule: P(A U B) = %.4f + %.4f - %.4f = %.4f",
            p_a,
            p_b,
            p_a_and_b,
            p_union,
        )
        return p_union

    def conditional_probability(self, p_a_and_b: float, p_b: float) -> float:
        """Calculate P(A | B) = P(A intersect B) / P(B).

        Args:
            p_a_and_b: Joint probability P(A intersect B).
            p_b:       Prior probability P(B) > 0.

        Returns:
            Conditional probability P(A | B).
        """
        if p_b <= 0.0:
            raise ValueError("P(B) must be strictly greater than 0 for conditional probability.")

        p_a_given_b = p_a_and_b / p_b
        self._logger.info(
            "Conditional Probability P(A|B) = %.4f / %.4f = %.4f",
            p_a_and_b,
            p_b,
            p_a_given_b,
        )
        return p_a_given_b

    def check_independence(
        self, p_a: float, p_b: float, p_a_and_b: float
    ) -> bool:
        """Verify whether events A and B are independent: P(A intersect B) == P(A) * P(B).

        Args:
            p_a:       P(A).
            p_b:       P(B).
            p_a_and_b: Joint probability P(A intersect B).

        Returns:
            True if independent within numerical tolerance, False otherwise.
        """
        expected_joint = p_a * p_b
        is_indep = abs(p_a_and_b - expected_joint) < 1e-6
        self._logger.info(
            "Independence Test: P(A intersect B) = %.4f, P(A)*P(B) = %.4f -> Independent: %s",
            p_a_and_b,
            expected_joint,
            is_indep,
        )
        return is_indep

    def bayes_theorem(
        self, prior_p_b: float, sensitivity_p_a_given_b: float, false_positive_p_a_given_b_c: float
    ) -> Dict[str, float]:
        """Solve Bayes' Theorem for binary classification screening.

        Formula:
            P(B | A) = P(A | B) * P(B) / [ P(A | B)*P(B) + P(A | B^c)*P(B^c) ]

        Args:
            prior_p_b:                  Prior prevalence P(B).
            sensitivity_p_a_given_b:     Sensitivity / True positive rate P(A | B).
            false_positive_p_a_given_b_c: False positive rate P(A | B^c).

        Returns:
            Dictionary containing Total Probability P(A) and Posterior Probability P(B | A).
        """
        prior_p_b_c = 1.0 - prior_p_b
        p_a = (sensitivity_p_a_given_b * prior_p_b) + (
            false_positive_p_a_given_b_c * prior_p_b_c
        )
        posterior_p_b_given_a = (sensitivity_p_a_given_b * prior_p_b) / p_a

        results = {
            "Prior_P(B)": prior_p_b,
            "Total_Probability_P(A)": p_a,
            "Posterior_P(B|A)": posterior_p_b_given_a,
        }

        self._logger.info("Bayes' Theorem Evaluation:")
        self._logger.info("  Prior P(B)            : %.4f", prior_p_b)
        self._logger.info("  Total Prob P(A)       : %.4f", p_a)
        self._logger.info("  Posterior P(B|A)      : %.4f", posterior_p_b_given_a)

        return results

    def evaluate_inequalities(
        self, mean: float, variance: float, a: float, k: float, probabilities: List[float]
    ) -> Dict[str, float]:
        """Calculate upper bounds for Markov, Chebyshev, and Union Bound inequalities.

        Inequalities:
            - Markov's Inequality: P(X >= a) <= E[X] / a  (for non-negative X, a > 0)
            - Chebyshev's Inequality: P(|X - mu| >= k*sigma) <= 1 / k^2
            - Union Bound: P(U A_i) <= sum(P(A_i))

        Args:
            mean:          Expected value E[X].
            variance:      Variance Var(X).
            a:             Threshold parameter for Markov's inequality (> 0).
            k:             Standard deviation multiplier for Chebyshev's inequality (> 0).
            probabilities: List of probabilities for individual events A_i.

        Returns:
            Dictionary with bound results.
        """
        markov_bound = mean / a if a > 0 else float("inf")
        chebyshev_bound = 1.0 / (k ** 2) if k > 0 else float("inf")
        union_bound = min(sum(probabilities), 1.0)

        results = {
            "Markov_Bound": min(markov_bound, 1.0),
            "Chebyshev_Bound": min(chebyshev_bound, 1.0),
            "Union_Bound": union_bound,
        }

        self._logger.info("Probability Inequalities Upper Bounds:")
        self._logger.info("  Markov Bound (P(X >= %.1f))       : <= %.4f", a, results["Markov_Bound"])
        self._logger.info("  Chebyshev Bound (k=%.1f std dev)  : <= %.4f", k, results["Chebyshev_Bound"])
        self._logger.info("  Union Bound                       : <= %.4f", results["Union_Bound"])

        return results
