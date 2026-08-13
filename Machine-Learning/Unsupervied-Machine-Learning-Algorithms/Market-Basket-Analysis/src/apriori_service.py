# ============================================================================
# Apriori Service and Association Rules Mining Module
# ============================================================================
# Core algorithm service implementing the Apriori algorithm from scratch,
# association rule extraction (Support, Confidence, Lift, Leverage, Conviction),
# visualization chart generation, and comprehensive analytical report generation.
# ============================================================================

import itertools
import logging
import os
import time
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import AprioriConfig, PathConfig


class AprioriService:
    """Service encapsulating Apriori frequent itemset mining and association rule generation."""

    def __init__(
        self,
        apriori_config: AprioriConfig,
        path_config: PathConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize AprioriService.

        Args:
            apriori_config: Algorithm hyperparameter settings.
            path_config:    Path settings.
            logger:         Logger instance.
        """
        self._apriori_config = apriori_config
        self._path_config = path_config
        self._logger = logger

    def fit_and_evaluate_all(
        self, basket_df: pd.DataFrame, transactions: List[set]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full Apriori mining pipeline and export visualizations & markdown report.

        Args:
            basket_df:    Binarized transaction DataFrame (BillNo x Itemname).
            transactions: List of item sets per transaction.

        Returns:
            Tuple of (frequent_itemsets_df, association_rules_df).
        """
        self._logger.info("=" * 70)
        self._logger.info("APRIORI FREQUENT ITEMSET & ASSOCIATION RULES MINING")
        self._logger.info("=" * 70)

        n_transactions = len(transactions)
        self._logger.info(
            "Mining parameters: min_support=%.4f, min_confidence=%.4f, min_lift=%.4f, max_len=%d",
            self._apriori_config.min_support,
            self._apriori_config.min_confidence,
            self._apriori_config.min_lift,
            self._apriori_config.max_len,
        )
        self._logger.info("Total transactions evaluated: %d", n_transactions)

        t0 = time.perf_counter()

        # Step 1: Custom Apriori algorithm from scratch
        frequent_itemsets = self._mine_frequent_itemsets(transactions, n_transactions)
        t_itemsets = time.perf_counter() - t0
        self._logger.info(
            "Found %d frequent itemsets in %.4f seconds.",
            len(frequent_itemsets),
            t_itemsets,
        )

        if not frequent_itemsets:
            self._logger.warning("No frequent itemsets found matching min_support criteria.")
            return pd.DataFrame(), pd.DataFrame()

        # Format frequent itemsets into DataFrame
        itemset_rows = []
        itemset_support_map: Dict[frozenset, float] = {}
        for itemset, supp in frequent_itemsets.items():
            itemset_support_map[itemset] = supp
            itemset_rows.append(
                {
                    "itemset": set(itemset),
                    "itemset_str": ", ".join(sorted(list(itemset))),
                    "length": len(itemset),
                    "support": supp,
                }
            )

        frequent_df = (
            pd.DataFrame(itemset_rows)
            .sort_values(by=["support", "length"], ascending=[False, True])
            .reset_index(drop=True)
        )

        # Step 2: Generate Association Rules from Scratch
        t0_rules = time.perf_counter()
        rules = self._generate_association_rules(
            frequent_itemsets, itemset_support_map, n_transactions
        )
        t_rules = time.perf_counter() - t0_rules
        self._logger.info(
            "Generated %d association rules in %.4f seconds.",
            len(rules),
            t_rules,
        )

        rules_df = pd.DataFrame(rules)
        if not rules_df.empty:
            rules_df = rules_df.sort_values(
                by=["lift", "confidence", "support"], ascending=[False, False, False]
            ).reset_index(drop=True)

        # Step 3: Render Visualizations
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        self._plot_frequent_itemsets(frequent_df)
        self._plot_rules_scatter(rules_df)
        self._plot_rules_network(rules_df)

        # Step 4: Save Markdown Analytical Report
        self._generate_report(frequent_df, rules_df, n_transactions)

        return frequent_df, rules_df

    def _mine_frequent_itemsets(
        self, transactions: List[set], n_transactions: int
    ) -> Dict[frozenset, float]:
        """Mine frequent itemsets using fast set-intersection Apriori algorithm from scratch.

        Args:
            transactions:   List of sets, each containing item strings.
            n_transactions: Total count of transactions.

        Returns:
            Dictionary mapping frozenset itemsets to their support value.
        """
        frequent_itemsets: Dict[frozenset, float] = {}
        min_supp = self._apriori_config.min_support

        # Build transaction index mapping each item to the set of transaction IDs containing it
        item_trans_map: Dict[str, Set[int]] = {}
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                if item not in item_trans_map:
                    item_trans_map[item] = set()
                item_trans_map[item].add(tid)

        # ---------------------------------------------------------------------
        # Iteration 1: 1-itemset mining
        # ---------------------------------------------------------------------
        l_current: Dict[frozenset, float] = {}
        item_sets_map: Dict[str, Set[int]] = {}

        for item, t_ids in item_trans_map.items():
            supp = len(t_ids) / n_transactions
            if supp >= min_supp:
                fs = frozenset([item])
                l_current[fs] = supp
                frequent_itemsets[fs] = supp
                item_sets_map[item] = t_ids

        self._logger.info("Iteration k=1: Found %d frequent 1-itemsets.", len(l_current))

        # ---------------------------------------------------------------------
        # Iterations k=2..max_len
        # ---------------------------------------------------------------------
        k = 2
        while l_current and k <= self._apriori_config.max_len:
            # Candidate generation: join L_{k-1} with L_{k-1}
            candidates = self._generate_candidates(list(l_current.keys()), k)
            if not candidates:
                break

            # Fast support counting using set intersection of item transaction IDs
            l_next: Dict[frozenset, float] = {}
            for candidate in candidates:
                # Intersect transaction sets for all items in candidate
                candidate_t_ids = set.intersection(*(item_sets_map[item] for item in candidate))
                supp = len(candidate_t_ids) / n_transactions
                if supp >= min_supp:
                    l_next[candidate] = supp
                    frequent_itemsets[candidate] = supp

            self._logger.info("Iteration k=%d: Found %d frequent %d-itemsets.", k, len(l_next), k)
            l_current = l_next
            k += 1

        return frequent_itemsets

    def _generate_candidates(
        self, prev_frequent_itemsets: List[frozenset], k: int
    ) -> Set[frozenset]:
        """Generate candidate k-itemsets by joining frequent (k-1)-itemsets and pruning invalid subsets.

        Args:
            prev_frequent_itemsets: List of frequent itemsets from iteration k-1.
            k:                       Target candidate length.

        Returns:
            Set of candidate frozenset itemsets of length k.
        """
        candidates: Set[frozenset] = set()
        prev_set = set(prev_frequent_itemsets)

        n = len(prev_frequent_itemsets)
        for i in range(n):
            for j in range(i + 1, n):
                i1 = prev_frequent_itemsets[i]
                i2 = prev_frequent_itemsets[j]

                union_set = i1.union(i2)
                if len(union_set) == k:
                    # Apriori Pruning property: all k-1 subsets must be frequent
                    is_valid = True
                    for sub in itertools.combinations(union_set, k - 1):
                        if frozenset(sub) not in prev_set:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.add(union_set)

        return candidates

    def _generate_association_rules(
        self,
        frequent_itemsets: Dict[frozenset, float],
        itemset_support_map: Dict[frozenset, float],
        n_transactions: int,
    ) -> List[Dict[str, object]]:
        """Extract association rules A -> B from frequent itemsets matching confidence and lift criteria.

        Args:
            frequent_itemsets:   Dictionary of frequent itemsets and their support.
            itemset_support_map: Support lookup map for itemsets.
            n_transactions:      Total transaction count.

        Returns:
            List of dictionaries containing rule metrics.
        """
        rules: List[Dict[str, object]] = []
        min_conf = self._apriori_config.min_confidence
        min_lift = self._apriori_config.min_lift

        for itemset, support_ab in frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            # Generate all proper non-empty subsets of itemset
            items = list(itemset)
            for r in range(1, len(items)):
                for antecedent_tuple in itertools.combinations(items, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset.difference(antecedent)

                    support_a = itemset_support_map.get(antecedent, 0.0)
                    support_b = itemset_support_map.get(consequent, 0.0)

                    if support_a == 0.0 or support_b == 0.0:
                        continue

                    # Confidence = Support(A U B) / Support(A)
                    confidence = support_ab / support_a

                    if confidence >= min_conf:
                        # Lift = Confidence(A -> B) / Support(B)
                        lift = confidence / support_b

                        if lift >= min_lift:
                            # Leverage = Support(A U B) - (Support(A) * Support(B))
                            leverage = support_ab - (support_a * support_b)

                            # Conviction = (1 - Support(B)) / (1 - Confidence(A -> B))
                            conviction = (
                                (1.0 - support_b) / (1.0 - confidence)
                                if confidence < 1.0
                                else float("inf")
                            )

                            ant_str = ", ".join(sorted(list(antecedent)))
                            cons_str = ", ".join(sorted(list(consequent)))

                            rules.append(
                                {
                                    "antecedent": set(antecedent),
                                    "consequent": set(consequent),
                                    "antecedent_str": ant_str,
                                    "consequent_str": cons_str,
                                    "rule_str": f"{ant_str} -> {cons_str}",
                                    "support": support_ab,
                                    "confidence": confidence,
                                    "lift": lift,
                                    "leverage": leverage,
                                    "conviction": conviction,
                                }
                            )

        return rules

    def _plot_frequent_itemsets(self, frequent_df: pd.DataFrame) -> None:
        """Render bar chart of top 20 frequent itemsets by support.

        Args:
            frequent_df: DataFrame of frequent itemsets sorted by support.
        """
        top_20 = frequent_df.head(20).copy()
        if top_20.empty:
            return

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            data=top_20,
            x="support",
            y="itemset_str",
            palette="Blues_r",
            hue="itemset_str",
            legend=False,
        )

        plt.title("Top 20 Frequent Itemsets by Support (Apriori)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Support (Proportion of Transactions)", fontsize=12)
        plt.ylabel("Itemset", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        # Add support percentage labels
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(
                f"{width:.3f}",
                (width, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
            )

        plt.tight_layout()
        save_path = os.path.join(self._path_config.output_dir, "frequent_itemsets_top20.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        self._logger.info("Saved top 20 frequent itemsets plot to: %s", save_path)

    def _plot_rules_scatter(self, rules_df: pd.DataFrame) -> None:
        """Render scatter plot of Support vs Confidence colored by Lift.

        Args:
            rules_df: DataFrame of association rules.
        """
        if rules_df.empty:
            return

        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(
            rules_df["support"],
            rules_df["confidence"],
            c=rules_df["lift"],
            cmap="viridis",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
            s=90,
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label("Lift Metric", fontsize=11, fontweight="bold")

        plt.title(
            "Association Rules: Support vs Confidence (Color = Lift)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Support (Proportion of Transactions)", fontsize=12)
        plt.ylabel("Confidence P(B|A)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self._path_config.output_dir, "rules_scatter_plot.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        self._logger.info("Saved association rules scatter plot to: %s", save_path)

    def _plot_rules_network(self, rules_df: pd.DataFrame) -> None:
        """Render clear, non-overlapping visual analysis chart of top 15 association rules by lift.

        Args:
            rules_df: DataFrame of association rules.
        """
        import textwrap

        top_rules = rules_df.head(15).copy()
        if top_rules.empty:
            return

        # Format rule label with line wrapping for maximum readability
        def format_rule_label(row: pd.Series) -> str:
            ant_wrapped = textwrap.fill(row["antecedent_str"], width=30)
            cons_wrapped = textwrap.fill(row["consequent_str"], width=30)
            return f"{ant_wrapped}\n  => {cons_wrapped}"

        top_rules["wrapped_rule"] = top_rules.apply(format_rule_label, axis=1)

        plt.figure(figsize=(15, 10))
        ax = sns.barplot(
            data=top_rules,
            x="lift",
            y="wrapped_rule",
            hue="confidence",
            palette="magma",
            dodge=False,
        )
        # Remove unwanted duplicate legend created by hue
        if ax.get_legend():
            ax.get_legend().remove()

        plt.title(
            "Top 15 Association Rules Ranked by Lift (Color = Confidence)",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Lift Metric (Strength of Association)", fontsize=12, fontweight="bold")
        plt.ylabel("Association Rule (Antecedent => Consequent)", fontsize=12, fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        # Annotate each bar with exact Lift, Confidence, and Support values
        for idx, row in top_rules.reset_index(drop=True).iterrows():
            lift_val = row["lift"]
            conf_val = row["confidence"] * 100
            supp_val = row["support"] * 100
            label_text = f" Lift: {lift_val:.2f} | Conf: {conf_val:.1f}% | Supp: {supp_val:.1f}%"
            ax.text(
                lift_val + 0.2,
                idx,
                label_text,
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#1a1a1a",
            )

        # Expand x-limit slightly to accommodate bar text labels cleanly
        max_lift = top_rules["lift"].max()
        plt.xlim(0, max_lift * 1.45)
        plt.yticks(fontsize=9.5)

        plt.tight_layout()
        save_path = os.path.join(self._path_config.output_dir, "association_rules_network.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info("Saved formatted association rules network plot to: %s", save_path)

    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Format a pandas DataFrame as a Markdown table without external tabulate dependency.

        Args:
            df: Input DataFrame to format.

        Returns:
            Markdown table formatted string.
        """
        if df.empty:
            return "No data available."

        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for _, row in df.iterrows():
            row_str = " | ".join(str(row[c]) for c in df.columns)
            lines.append(f"| {row_str} |")

        return "\n".join(lines)

    def _generate_report(
        self,
        frequent_df: pd.DataFrame,
        rules_df: pd.DataFrame,
        n_transactions: int,
    ) -> None:
        """Generate comprehensive markdown analytical report in output directory.

        Args:
            frequent_df:    DataFrame of frequent itemsets.
            rules_df:       DataFrame of association rules.
            n_transactions: Total number of transactions analyzed.
        """
        report_path = os.path.join(
            self._path_config.output_dir, "market_basket_analysis_report.md"
        )

        top_frequent_md = ""
        if not frequent_df.empty:
            top_20 = frequent_df.head(10)[["itemset_str", "length", "support"]].copy()
            top_20["support_pct"] = (top_20["support"] * 100).round(2).astype(str) + "%"
            top_frequent_md = self._dataframe_to_markdown(top_20)
        else:
            top_frequent_md = "No frequent itemsets found matching criteria."

        top_rules_md = ""
        if not rules_df.empty:
            top_15_rules = rules_df.head(15)[
                ["rule_str", "support", "confidence", "lift", "leverage", "conviction"]
            ].copy()
            top_15_rules["support"] = top_15_rules["support"].round(4)
            top_15_rules["confidence"] = top_15_rules["confidence"].round(4)
            top_15_rules["lift"] = top_15_rules["lift"].round(4)
            top_15_rules["leverage"] = top_15_rules["leverage"].round(4)
            top_15_rules["conviction"] = top_15_rules["conviction"].round(4)
            top_rules_md = self._dataframe_to_markdown(top_15_rules)
        else:
            top_rules_md = "No association rules found matching criteria."

        content = f"""# Market Basket Analysis (Apriori Algorithm) Analytical Report

## Executive Summary
This report summarizes the results of Market Basket Analysis performed using the Apriori algorithm from scratch on the transaction dataset. Market Basket Analysis identifies products frequently purchased together to uncover cross-selling and product recommendation opportunities.

---

## Mining Configuration & Pipeline Metadata
- **Total Transactions Analyzed**: {n_transactions}
- **Minimum Support Threshold**: {self._apriori_config.min_support} ({(self._apriori_config.min_support * 100):.1f}%)
- **Minimum Confidence Threshold**: {self._apriori_config.min_confidence} ({(self._apriori_config.min_confidence * 100):.1f}%)
- **Minimum Lift Threshold**: {self._apriori_config.min_lift}
- **Maximum Itemset Size**: {self._apriori_config.max_len}
- **Total Frequent Itemsets Discovered**: {len(frequent_df)}
- **Total Association Rules Discovered**: {len(rules_df)}

---

## Top Frequent Itemsets (by Support)
{top_frequent_md}

---

## Top Association Rules (by Lift)
{top_rules_md}

---

## Strategic Retail Recommendations
1. **Product Bundling**: Items appearing in association rules with high Lift (> 1.5) and Confidence (> 30%) should be offered in promotional bundles or displayed together in digital catalogs.
2. **Layout & Placement Optimization**: High-confidence item pairs can be placed in adjacent shelf locations or as smart checkout recommendations to maximize impulse purchases.
3. **Cross-Selling Promotions**: Target promotional discounts on antecedent items to drive sales of high-margin consequent items.

---

## Visualizations Generated
- `frequent_itemsets_top20.png`: Bar chart highlighting top frequent itemsets.
- `rules_scatter_plot.png`: Scatter plot mapping Support vs Confidence colored by Lift.
- `association_rules_network.png`: Heatmap matrix of antecedent and consequent item relationships.
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._logger.info("Saved Market Basket Analysis report to: %s", report_path)
