# ============================================================================
# Apriori & Market Basket Analysis Service
# ============================================================================
# Pure Python/Pandas implementation of the Apriori Algorithm and Association
# Rule Mining Engine. Computes Support, Confidence, Lift, Leverage, and
# Conviction metrics, generates plots, and creates structured reports.
# ============================================================================

import itertools
import logging
import os
import time
from typing import Dict, List, Set, Tuple

# Non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ModelConfig, PathConfig


class AprioriMBAService:
    """Service encapsulating the Apriori algorithm and Association Rule Mining.

    Responsibilities:
        1. Find frequent itemsets using Apriori candidate generation and anti-monotonicity pruning.
        2. Mine association rules (A -> B) from frequent itemsets.
        3. Compute rule evaluation metrics: Support, Confidence, Lift, Leverage, Conviction.
        4. Visualise top itemsets, rule scatter plots, and top rules by lift.
        5. Write text and markdown analysis reports.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the Apriori service.

        Args:
            model_config: Apriori and Rule mining hyperparameters.
            path_config:  Path settings for saving outputs.
            logger:       Logger instance.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._logger = logger

    # -- Public workflow methods ---------------------------------------------

    def run_pipeline(
        self, baskets: List[Set[str]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full Apriori and Rule Mining workflow.

        Args:
            baskets: List of sets representing customer transaction itemsets.

        Returns:
            Tuple of (frequent_itemsets_df, association_rules_df).
        """
        self._logger.info("=" * 70)
        self._logger.info("APRIORI FREQUENT ITEMSET MINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        start_time = time.perf_counter()
        frequent_itemsets_dict = self._find_frequent_itemsets(baskets)
        apriori_elapsed = time.perf_counter() - start_time
        self._logger.info(
            "Frequent itemsets mining completed in %.3f seconds.", apriori_elapsed
        )

        # Convert itemsets dict to DataFrame
        itemsets_data = [
            {"itemset": tuple(sorted(itemset)), "support": supp, "length": len(itemset)}
            for itemset, supp in frequent_itemsets_dict.items()
        ]
        df_itemsets = pd.DataFrame(itemsets_data).sort_values(
            by="support", ascending=False
        ).reset_index(drop=True)

        self._logger.info(
            "Discovered %d frequent itemsets (min_support=%.3f).",
            len(df_itemsets), self._model_config.min_support
        )

        self._logger.info("=" * 70)
        self._logger.info("ASSOCIATION RULE MINING")
        self._logger.info("=" * 70)

        start_time = time.perf_counter()
        df_rules = self._generate_association_rules(frequent_itemsets_dict, len(baskets))
        rule_elapsed = time.perf_counter() - start_time
        self._logger.info(
            "Association rule generation completed in %.3f seconds.", rule_elapsed
        )

        self._logger.info(
            "Generated %d association rules (min_confidence=%.2f, min_lift=%.2f).",
            len(df_rules), self._model_config.min_confidence, self._model_config.min_lift
        )

        self._log_top_rules(df_rules)
        self._save_results(df_itemsets, df_rules, len(baskets))
        self._generate_plots(df_itemsets, df_rules)
        self._save_analysis(df_itemsets, df_rules, len(baskets))

        return df_itemsets, df_rules

    # -- Private Apriori Implementation ---------------------------------------

    def _find_frequent_itemsets(
        self, baskets: List[Set[str]]
    ) -> Dict[Tuple[str, ...], float]:
        """Core Apriori Algorithm implementation with candidate generation & pruning.

        Args:
            baskets: Transaction baskets.

        Returns:
            Dictionary mapping itemsets (as sorted tuples) to their support values.
        """
        n_baskets = len(baskets)
        min_supp = self._model_config.min_support
        min_count = min_supp * n_baskets

        # Step 1: Candidate 1-itemsets (C1)
        item_counts: Dict[str, int] = {}
        for b in baskets:
            for item in b:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Frequent 1-itemsets (L1)
        l1: Dict[Tuple[str, ...], float] = {
            (item,): count / n_baskets
            for item, count in item_counts.items()
            if count >= min_count
        }

        self._logger.info("Found %d frequent 1-itemsets.", len(l1))
        all_frequent: Dict[Tuple[str, ...], float] = dict(l1)
        current_l = l1

        # Step 2..k: Candidate k-itemsets (Ck) & Pruning
        k = 2
        while current_l and k <= self._model_config.max_itemset_length:
            candidates = self._generate_candidates(list(current_l.keys()), k)
            if not candidates:
                break

            # Count support for candidates
            candidate_counts: Dict[Tuple[str, ...], int] = {cand: 0 for cand in candidates}
            for b in baskets:
                for cand in candidates:
                    if set(cand).issubset(b):
                        candidate_counts[cand] += 1

            # Filter frequent k-itemsets (Lk)
            lk: Dict[Tuple[str, ...], float] = {
                cand: count / n_baskets
                for cand, count in candidate_counts.items()
                if count >= min_count
            }

            self._logger.info("Found %d frequent %d-itemsets.", len(lk), k)
            if not lk:
                break

            all_frequent.update(lk)
            current_l = lk
            k += 1

        return all_frequent

    def _generate_candidates(
        self, frequent_itemsets: List[Tuple[str, ...]], k: int
    ) -> Set[Tuple[str, ...]]:
        """Generate candidate k-itemsets (Ck) from L_{k-1} with anti-monotonicity pruning.

        Args:
            frequent_itemsets: List of frequent (k-1)-itemsets.
            k: Target itemset size.

        Returns:
            Set of pruned candidate k-itemsets.
        """
        candidates: Set[Tuple[str, ...]] = set()
        n = len(frequent_itemsets)
        frequent_set = set(frequent_itemsets)

        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = frequent_itemsets[i]
                itemset2 = frequent_itemsets[j]

                # Join step: merge if first (k-2) items match
                if itemset1[: k - 2] == itemset2[: k - 2]:
                    candidate = tuple(sorted(set(itemset1).union(set(itemset2))))
                    if len(candidate) == k:
                        # Prune step: check if all (k-1) subsets are frequent
                        subsets_frequent = True
                        for sub in itertools.combinations(candidate, k - 1):
                            sub_tuple = tuple(sorted(sub))
                            if sub_tuple not in frequent_set:
                                subsets_frequent = False
                                break
                        if subsets_frequent:
                            candidates.add(candidate)

        return candidates

    # -- Private Association Rule Engine --------------------------------------

    def _generate_association_rules(
        self, frequent_itemsets: Dict[Tuple[str, ...], float], n_baskets: int
    ) -> pd.DataFrame:
        """Mine association rules A -> B and calculate metrics.

        Args:
            frequent_itemsets: Dict mapping itemsets to support.
            n_baskets: Total transaction count.

        Returns:
            DataFrame of valid association rules.
        """
        rules = []

        for itemset, supp_AB in frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            # Generate all non-empty proper subsets A of itemset
            for r in range(1, len(itemset)):
                for A_tuple in itertools.combinations(itemset, r):
                    A = tuple(sorted(A_tuple))
                    B = tuple(sorted(set(itemset) - set(A)))

                    supp_A = frequent_itemsets.get(A, 0.0)
                    supp_B = frequent_itemsets.get(B, 0.0)

                    if supp_A <= 0 or supp_B <= 0:
                        continue

                    confidence = supp_AB / supp_A
                    lift = confidence / supp_B
                    leverage = supp_AB - (supp_A * supp_B)

                    # Conviction formula: (1 - supp_B) / (1 - confidence)
                    if confidence >= 1.0:
                        conviction = np.inf
                    else:
                        conviction = (1.0 - supp_B) / (1.0 - confidence)

                    if (
                        confidence >= self._model_config.min_confidence
                        and lift >= self._model_config.min_lift
                    ):
                        rules.append({
                            "antecedents": ", ".join(A),
                            "consequents": ", ".join(B),
                            "antecedent_support": supp_A,
                            "consequent_support": supp_B,
                            "support": supp_AB,
                            "confidence": confidence,
                            "lift": lift,
                            "leverage": leverage,
                            "conviction": conviction,
                        })

        if not rules:
            return pd.DataFrame(columns=[
                "antecedents", "consequents", "antecedent_support",
                "consequent_support", "support", "confidence", "lift",
                "leverage", "conviction"
            ])

        df_rules = pd.DataFrame(rules).sort_values(
            by="lift", ascending=False
        ).reset_index(drop=True)

        return df_rules

    # -- Helpers & Outputs ---------------------------------------------------

    def _log_hyperparameters(self) -> None:
        """Log pipeline hyperparameters."""
        self._logger.info("Apriori Hyperparameters:")
        self._logger.info("  min_support       : %.3f", self._model_config.min_support)
        self._logger.info("  min_confidence    : %.2f", self._model_config.min_confidence)
        self._logger.info("  min_lift          : %.2f", self._model_config.min_lift)
        self._logger.info("  max_itemset_length: %d", self._model_config.max_itemset_length)

    def _log_top_rules(self, df_rules: pd.DataFrame) -> None:
        """Log top mined rules to console."""
        if df_rules.empty:
            self._logger.info("No association rules met the criteria.")
            return

        self._logger.info("-" * 70)
        self._logger.info("TOP MINED ASSOCIATION RULES (Ranked by Lift):")
        self._logger.info("-" * 70)
        top_n = min(self._model_config.top_n_rules, len(df_rules))

        for idx in range(top_n):
            row = df_rules.iloc[idx]
            self._logger.info(
                "Rule #%-2d: {%s} -> {%s}",
                idx + 1, row["antecedents"], row["consequents"]
            )
            self._logger.info(
                "        Supp: %.4f | Conf: %.4f | Lift: %.4f | Lev: %.4f",
                row["support"], row["confidence"], row["lift"], row["leverage"]
            )
        self._logger.info("=" * 70)

    def _save_results(
        self, df_itemsets: pd.DataFrame, df_rules: pd.DataFrame, n_baskets: int
    ) -> None:
        """Save results summary to mba_results.txt."""
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        results_path = os.path.join(self._path_config.output_dir, "mba_results.txt")

        with open(results_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("MARKET BASKET ANALYSIS & APRIORI RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS & DATASET INFO\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Total Baskets Analyzed : {n_baskets}\n")
            fh.write(f"  min_support            : {self._model_config.min_support}\n")
            fh.write(f"  min_confidence         : {self._model_config.min_confidence}\n")
            fh.write(f"  min_lift               : {self._model_config.min_lift}\n")
            fh.write(f"  max_itemset_length     : {self._model_config.max_itemset_length}\n\n")

            fh.write("FREQUENT ITEMSETS SUMMARY\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Total Frequent Itemsets Discovered: {len(df_itemsets)}\n\n")
            fh.write("Top 15 Frequent Itemsets:\n")
            top_itemsets = df_itemsets.head(15)
            for idx, row in top_itemsets.iterrows():
                items_str = ", ".join(row["itemset"])
                fh.write(f"  {idx+1:2d}. {{{items_str}}} (Support: {row['support']:.4f})\n")
            fh.write("\n")

            fh.write("TOP ASSOCIATION RULES (RANKED BY LIFT)\n")
            fh.write("-" * 70 + "\n")
            fh.write(f"  Total Rules Mined: {len(df_rules)}\n\n")

            if not df_rules.empty:
                top_rules = df_rules.head(self._model_config.top_n_rules)
                for idx, row in top_rules.iterrows():
                    fh.write(f"Rule #{idx+1:2d}:\n")
                    fh.write(f"  Antecedent (A) : {{{row['antecedents']}}}\n")
                    fh.write(f"  Consequent (B) : {{{row['consequents']}}}\n")
                    fh.write(f"  Support (A->B) : {row['support']:.4f}\n")
                    fh.write(f"  Confidence     : {row['confidence']:.4f} ({row['confidence']*100:.1f}%)\n")
                    fh.write(f"  Lift           : {row['lift']:.4f}\n")
                    fh.write(f"  Leverage       : {row['leverage']:.4f}\n")
                    fh.write(f"  Conviction     : {row['conviction']:.4f}\n\n")
            fh.write("=" * 70 + "\n")

        self._logger.info("MBA results saved to: %s", results_path)

    def _generate_plots(
        self, df_itemsets: pd.DataFrame, df_rules: pd.DataFrame
    ) -> None:
        """Generate evaluation plots: top itemsets, rule scatter plot, top lift rules."""
        self._logger.info("Generating MBA visualization plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # Plot 1: Top 20 Frequent Itemsets Bar Chart
        if not df_itemsets.empty:
            plt.figure(figsize=(12, 7))
            top_20 = df_itemsets.head(20).copy()
            top_20["itemset_label"] = top_20["itemset"].apply(
                lambda x: "\n".join(x) if len(x) > 1 else x[0]
            )

            sns.barplot(
                x="support",
                y="itemset_label",
                data=top_20,
                hue="itemset_label",
                palette="mako",
                legend=False
            )
            plt.title("Top 20 Frequent Itemsets by Support", fontsize=12, fontweight="bold")
            plt.xlabel("Support (Fraction of Total Transactions)")
            plt.ylabel("Itemset")
            plt.tight_layout()
            fi_path = os.path.join(self._path_config.output_dir, "frequent_itemsets_top20.png")
            plt.savefig(fi_path, dpi=300)
            plt.close()

        # Plot 2: Rules Scatter Plot (Support vs Confidence colored by Lift)
        if not df_rules.empty:
            plt.figure(figsize=(9, 6))
            scatter = plt.scatter(
                df_rules["support"],
                df_rules["confidence"],
                c=df_rules["lift"],
                cmap="viridis",
                alpha=0.8,
                edgecolors="w",
                s=80
            )
            plt.colorbar(scatter, label="Lift")
            plt.title("Association Rules -- Support vs Confidence (Color = Lift)", fontsize=12, fontweight="bold")
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            scatter_path = os.path.join(self._path_config.output_dir, "rules_scatter_plot.png")
            plt.savefig(scatter_path, dpi=300)
            plt.close()

            # Plot 3: Top Rules Ranked by Lift
            plt.figure(figsize=(12, 7))
            top_lift_rules = df_rules.head(15).copy()
            top_lift_rules["rule_name"] = (
                top_lift_rules["antecedents"] + " -> " + top_lift_rules["consequents"]
            )
            # Truncate long rule names for readability
            top_lift_rules["rule_name_short"] = top_lift_rules["rule_name"].apply(
                lambda x: x[:45] + "..." if len(x) > 45 else x
            )

            sns.barplot(
                x="lift",
                y="rule_name_short",
                data=top_lift_rules,
                hue="rule_name_short",
                palette="flare",
                legend=False
            )
            plt.title("Top Association Rules Ranked by Lift Metric", fontsize=12, fontweight="bold")
            plt.xlabel("Lift Strength Ratio")
            plt.ylabel("Association Rule (Antecedent -> Consequent)")
            plt.tight_layout()
            top_rules_path = os.path.join(self._path_config.output_dir, "top_rules_lift.png")
            plt.savefig(top_rules_path, dpi=300)
            plt.close()

        self._logger.info("Evaluation plots saved successfully.")

    def _save_analysis(
        self, df_itemsets: pd.DataFrame, df_rules: pd.DataFrame, n_baskets: int
    ) -> None:
        """Write technical markdown explanation report mba_analysis.md."""
        report_path = os.path.join(self._path_config.output_dir, "mba_analysis.md")

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Market Basket Analysis & Apriori Technical Report\n\n")

            fh.write("## 1. Executive Summary\n")
            fh.write(
                "This report details the Market Basket Analysis (MBA) performed on Online Retail transaction data "
                "using the Apriori algorithm. "
            )
            fh.write(
                f"From **{n_baskets}** transaction baskets, the algorithm extracted **{len(df_itemsets)}** frequent itemsets "
                f"and derived **{len(df_rules)}** actionable association rules (min_support={self._model_config.min_support}, "
                f"min_confidence={self._model_config.min_confidence}, min_lift={self._model_config.min_lift}).\n\n"
            )

            fh.write("## 2. Model Configuration\n\n")
            fh.write("| Hyperparameter | Value |\n")
            fh.write("|----------------|-------|\n")
            fh.write(f"| `min_support` | `{self._model_config.min_support}` |\n")
            fh.write(f"| `min_confidence` | `{self._model_config.min_confidence}` |\n")
            fh.write(f"| `min_lift` | `{self._model_config.min_lift}` |\n")
            fh.write(f"| `max_itemset_length` | `{self._model_config.max_itemset_length}` |\n\n")

            fh.write("## 3. Top Discovered Association Rules\n\n")
            if not df_rules.empty:
                fh.write("| Rank | Antecedent (A) | Consequent (B) | Support | Confidence | Lift | Leverage |\n")
                fh.write("|------|----------------|----------------|---------|------------|------|----------|\n")
                top_show = min(10, len(df_rules))
                for idx in range(top_show):
                    row = df_rules.iloc[idx]
                    fh.write(
                        f"| {idx+1} | `{row['antecedents']}` | `{row['consequents']}` | "
                        f"{row['support']:.4f} | {row['confidence']:.4f} | "
                        f"{row['lift']:.4f} | {row['leverage']:.4f} |\n"
                    )
                fh.write("\n")

            fh.write("### Theoretical Interpretation of MBA Metrics\n")
            fh.write(
                "- **Support**: Measures the proportion of total transactions containing the itemset. High support indicates popular item combinations.\n"
                "- **Confidence ($P(B|A)$)**: Probability of purchasing consequent $B$ given antecedent $A$ was purchased. Measures rule reliability.\n"
                "- **Lift**: Ratio of observed rule support to expected support assuming $A$ and $B$ are independent. "
                "A lift $> 1.0$ indicates a strong positive association (co-purchasing pattern).\n"
                "- **Leverage**: Difference between observed frequency of $A \cap B$ and expected frequency under independence.\n"
                "- **Conviction**: Quantifies rule dependency based on incorrect predictions.\n\n"
            )

            fh.write("## 4. Retail Business & Cross-Selling Recommendations\n\n")
            if not df_rules.empty:
                top_rule = df_rules.iloc[0]
                fh.write(
                    f"1. **Product Bundling**: Combine `{top_rule['antecedents']}` and `{top_rule['consequents']}` "
                    f"into a promotional bundle due to a strong Lift of **{top_rule['lift']:.2f}**.\n"
                )
                fh.write(
                    "2. **Store Layout & E-Commerce Placement**: Place antecedent and consequent items near each other "
                    "or recommend consequent items automatically at checkout when antecedents are added to cart.\n"
                )
                fh.write(
                    "3. **Targeted Pricing & Discounts**: Discount high-support antecedent products to drive cross-purchasing "
                    "of high-margin consequent products.\n\n"
                )

            fh.write("## 5. Output Artifacts Summary\n\n")
            fh.write("| File | Description |\n")
            fh.write("|------|-------------|\n")
            fh.write("| `mba_results.txt` | Text summary of frequent itemsets and mined association rules |\n")
            fh.write("| `mba_analysis.md` | Technical markdown report on Market Basket Analysis |\n")
            fh.write("| `frequent_itemsets_top20.png` | Bar chart of top 20 frequent itemsets by support |\n")
            fh.write("| `rules_scatter_plot.png` | Scatter plot of Support vs Confidence colored by Lift |\n")
            fh.write("| `top_rules_lift.png` | Bar chart of top association rules ranked by Lift metric |\n")

        self._logger.info("MBA technical report saved to: %s", report_path)
