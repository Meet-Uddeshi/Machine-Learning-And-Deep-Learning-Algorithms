# Unsupervised Machine Learning - Market Basket Analysis (MBA) & Apriori Algorithm

> Unsupervised Machine Learning | Association Rule Mining & Pattern Discovery Algorithm

---

## Table of Contents

1. [What is Market Basket Analysis & Apriori?](#1-what-is-market-basket-analysis--apriori)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Apriori Sum (Step-by-Step)](#5-worked-apriori-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Market Basket Analysis & Apriori?

**Market Basket Analysis (MBA)** is an unsupervised data mining technique that discovers co-occurrence relationships and buying patterns among products in large transactional databases. 

The **Apriori Algorithm** is the foundational algorithm used to perform Market Basket Analysis. It efficiently identifies **frequent itemsets** (groups of items that appear together in at least a minimum fraction of transactions) and generates IF-THEN **association rules** (e.g. $\text{Bread} \implies \text{Butter}$).

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Unsupervised Learning / Pattern Mining                               |
| Output type        | Frequent Itemsets ($L_k$) & Association Rules ($A \implies B$)       |
| Primary Objectives | Customer basket profiling, cross-selling, product recommendation      |
| Key Metrics        | Support, Confidence, Lift, Leverage, Conviction                       |
| Optimization       | Apriori Property (Anti-Monotonicity Pruning)                         |

---

## 2. Theoretical Explanation

### How the Apriori Algorithm Works

The Apriori algorithm operates in a level-wise candidate generation search ($k = 1, 2, \dots, K$):

```
Transaction Baskets ---> Candidate 1-Itemsets (C1) ---> Prune by Min Support ---> Frequent 1-Itemsets (L1)
                                                                                         |
                                                                                         v
        Association Rules (A -> B) <--- Prune Non-Frequent Subsets <--- Join & Candidate Gen (Ck)
```

1. **Frequent 1-Itemsets ($L_1$)**: Scan the database to count single item frequencies. Retain items meeting $\text{min\_support}$.
2. **Candidate Generation ($C_k$)**: Join frequent $(k-1)$-itemsets ($L_{k-1}$) to form candidate $k$-itemsets ($C_k$).
3. **Apriori Property (Anti-Monotonicity Pruning)**:
   > "If an itemset is frequent, all of its subsets must also be frequent. Conversely, if an itemset is infrequent, all of its supersets are automatically infrequent."
   Before scanning transactions, any candidate $c \in C_k$ that contains an infrequent $(k-1)$-subset is immediately pruned.
4. **Database Scan & Support Filtering**: Scan transactions to count support for candidate $k$-itemsets and filter out those below $\text{min\_support}$ to obtain $L_k$.
5. **Association Rule Generation**: For each frequent itemset $I$, generate rules $A \implies B$ where $A \subset I$ and $B = I \setminus A$. Filter rules based on $\text{min\_confidence}$ and $\text{min\_lift}$.

---

## 3. Mathematical Operations

### 1. Support

**Itemset Support** measures how frequently an itemset $A$ appears in the database of $N$ transaction baskets:

$$\text{Support}(A) = \frac{\text{Count}(A)}{N} = P(A)$$

**Rule Support** for $A \implies B$ measures the proportion of transactions containing both antecedent $A$ and consequent $B$:

$$\text{Support}(A \implies B) = \text{Support}(A \cup B) = P(A \cap B)$$

---

### 2. Confidence

Confidence measures the conditional probability that a customer buys consequent $B$, given that they purchased antecedent $A$:

$$\text{Confidence}(A \implies B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B \mid A)$$

- Range: $[0, 1]$ ($0\%$ to $100\%$). Higher confidence indicates a stronger conditional rule.

---

### 3. Lift

Lift measures how much more often $A$ and $B$ occur together than expected if they were statistically independent:

$$\text{Lift}(A \implies B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \times \text{Support}(B)} = \frac{P(B \mid A)}{P(B)}$$

- $\text{Lift} = 1.0$: $A$ and $B$ are independent (no association).
- $\text{Lift} > 1.0$: Positive association ($A$ promotes buying $B$).
- $\text{Lift} < 1.0$: Negative association ($A$ discourages buying $B$).

---

### 4. Leverage

Leverage computes the difference between the observed frequency of $A \cap B$ and the expected frequency under independence:

$$\text{Leverage}(A \implies B) = \text{Support}(A \cup B) - (\text{Support}(A) \times \text{Support}(B))$$

- Range: $[-0.25, 0.25]$. $\text{Leverage} = 0$ indicates independence.

---

### 5. Conviction

Conviction quantifies the expected frequency that the rule makes an incorrect prediction:

$$\text{Conviction}(A \implies B) = \frac{1 - \text{Support}(B)}{1 - \text{Confidence}(A \implies B)} = \frac{P(A) P(\neg B)}{P(A \cap \neg B)}$$

- $\text{Conviction} = 1.0$: Items are unassociated.
- Infinite ($\infty$): Perfect rule ($\text{Confidence} = 1.0$).

---

## 4. Real-World Example

### E-Commerce Online Retail Cross-Selling
The pipeline processes line-item transaction invoices from the Online Retail dataset (`archive (8).zip`).

- **Dataset**: `Assignment-1_Data.csv` inside `archive (8).zip`
- **Target Task**: Unsupervised Association Rule Mining on Transaction Baskets
- **Key Columns**:
  - `BillNo`: Transaction invoice identifier
  - `Itemname`: Product description
  - `Quantity`: Number of units purchased
  - `Price`: Unit price
  - `Country`: Customer location (e.g. `United Kingdom`)

### Applications
- **Cross-Selling & Recommendations**: "Customers who bought HERB MARKER PARSLEY also bought HERB MARKER THYME."
- **Promotional Bundling**: Packaging complimentary high-lift products together.
- **Shelf Layout Optimization**: Placing frequently co-purchased items in close physical proximity.

---

## 5. Worked Apriori Sum (Step-by-Step)

Let us trace Apriori manually on a toy database with $N=5$ transaction baskets:
- $T_1 = \{\text{Milk}, \text{Bread}, \text{Diaper}\}$
- $T_2 = \{\text{Milk}, \text{Diaper}, \text{Beer}\}$
- $T_3 = \{\text{Milk}, \text{Bread}, \text{Diaper}, \text{Beer}\}$
- $T_4 = \{\text{Bread}, \text{Diaper}, \text{Beer}\}$
- $T_5 = \{\text{Milk}, \text{Bread}, \text{Diaper}, \text{Beer}\}$

Set thresholds: $\text{min\_support} = 0.60$ (must appear in $\ge 3$ transactions), $\text{min\_confidence} = 0.75$.

---

### Step 1: Candidate 1-Itemsets ($C_1$) & Frequent 1-Itemsets ($L_1$)

Count support for single items:
- $\{\text{Milk}\}$: 4 / 5 = 0.80 ($\ge 0.60$) -> **Keep**
- $\{\text{Bread}\}$: 4 / 5 = 0.80 ($\ge 0.60$) -> **Keep**
- $\{\text{Diaper}\}$: 5 / 5 = 1.00 ($\ge 0.60$) -> **Keep**
- $\{\text{Beer}\}$: 4 / 5 = 0.80 ($\ge 0.60$) -> **Keep**

$$L_1 = \{\{\text{Milk}\}, \{\text{Bread}\}, \{\text{Diaper}\}, \{\text{Beer}\}\}$$

---

### Step 2: Candidate 2-Itemsets ($C_2$) & Frequent 2-Itemsets ($L_2$)

Join $L_1$ items to form candidate pairs and count support:
- $\{\text{Milk}, \text{Bread}\}$: $T_1, T_3, T_5 \implies 3 / 5 = 0.60$ ($\ge 0.60$) -> **Keep**
- $\{\text{Milk}, \text{Diaper}\}$: $T_1, T_2, T_3, T_5 \implies 4 / 5 = 0.80$ ($\ge 0.60$) -> **Keep**
- $\{\text{Milk}, \text{Beer}\}$: $T_2, T_3, T_5 \implies 3 / 5 = 0.60$ ($\ge 0.60$) -> **Keep**
- $\{\text{Bread}, \text{Diaper}\}$: $T_1, T_3, T_4, T_5 \implies 4 / 5 = 0.80$ ($\ge 0.60$) -> **Keep**
- $\{\text{Bread}, \text{Beer}\}$: $T_3, T_4, T_5 \implies 3 / 5 = 0.60$ ($\ge 0.60$) -> **Keep**
- $\{\text{Diaper}, \text{Beer}\}$: $T_2, T_3, T_4, T_5 \implies 4 / 5 = 0.80$ ($\ge 0.60$) -> **Keep**

All 6 pair candidates pass! $L_2$ contains all 6 pairs.

---

### Step 3: Candidate 3-Itemsets ($C_3$) & Frequent 3-Itemsets ($L_3$)

Join $L_2$ to form 3-itemsets and check support:
- $\{\text{Milk}, \text{Bread}, \text{Diaper}\}$: $T_1, T_3, T_5 \implies 3 / 5 = 0.60$ ($\ge 0.60$) -> **Keep**
- $\{\text{Milk}, \text{Diaper}, \text{Beer}\}$: $T_2, T_3, T_5 \implies 3 / 5 = 0.60$ ($\ge 0.60$) -> **Keep**
- $\{\text{Bread}, \text{Diaper}, \text{Beer}\}$: $T_3, T_4, T_5 \implies 3 / 5 = 0.60$ ($\ge 0.60$) -> **Keep**
- $\{\text{Milk}, \text{Bread}, \text{Beer}\}$: $T_3, T_5 \implies 2 / 5 = 0.40$ ($< 0.60$) -> **Prune**

$$L_3 = \{\{\text{Milk}, \text{Bread}, \text{Diaper}\}, \{\text{Milk}, \text{Diaper}, \text{Beer}\}, \{\text{Bread}, \text{Diaper}, \text{Beer}\}\}$$

---

### Step 4: Mine Rule $\{\text{Milk}, \text{Bread}\} \implies \{\text{Diaper}\}$

- $\text{Support}(\text{Milk}, \text{Bread}, \text{Diaper}) = 0.60$
- $\text{Support}(\text{Milk}, \text{Bread}) = 0.60$
- $\text{Support}(\text{Diaper}) = 1.00$

1. **Confidence**:
   $$\text{Confidence} = \frac{0.60}{0.60} = 1.00 \quad (100\% \ge 75\%)$$
2. **Lift**:
   $$\text{Lift} = \frac{0.60}{0.60 \times 1.00} = 1.00$$

---

## 6. Program Flowchart

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig   (zip and dataset paths)             |
|  - DataConfig   (United Kingdom, min_quantity=1)    |
|  - ModelConfig  (min_support=0.015, confidence=0.2)|
|  - LoggingConfig (console-only logs)                |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 2: Initialize Console Logger                  |
|  - Output formatted to sys.stdout (No local files)  |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
+-----------------------------------------------------+
                           |
           +---------------+-------------------+
           |                                   |
           v                                   v
+------------------+               +-------------------+
|  _load_data()    |               |  File not found?  |
|  Read CSV from   |  -- Error --> |  Raise            |
|  zip archive     |               |  FileNotFoundError|
+------------------+               +-------------------+
           |
           v
+------------------+
|  _clean_data()   |
|  - Parse Price & Quantity to numeric               |
|  - Remove cancellations (BillNo starting 'C')     |
|  - Filter by Country & minimum price/quantity     |
+------------------+
           |
           v
+------------------+
|  _build_baskets()|
|  Group items by  |
|  BillNo into sets|
+------------------+
           |
           v
+-----------------------------------------------------+
|  Step 4: AprioriMBAService.run_pipeline()           |
+-----------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  _find_frequent_itemsets() (Apriori Algorithm)   |
|  - Generate L1 single frequent items             |
|  - For k=2..3: Join L_{k-1} & candidate gen (Ck) |
|  - Apply Anti-Monotonicity Pruning               |
|  - Filter by min_support to build Lk             |
+--------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  _generate_association_rules()                   |
|  - For frequent itemsets |I| >= 2:               |
|  - Compute Support, Confidence, Lift, Leverage,  |
|    and Conviction for A -> B                     |
|  - Filter by min_confidence and min_lift         |
+--------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  Save Output Artifacts to output/                |
|  - Write mba_results.txt                         |
|  - Write mba_analysis.md report                  |
|  - Generate frequent_itemsets_top20.png          |
|  - Generate rules_scatter_plot.png               |
|  - Generate top_rules_lift.png                   |
+--------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|               END: Pipeline Complete               |
+-----------------------------------------------------+
```

---

## 7. Module Responsibility Map

```
main.py
  |
  +-- config.py           (PipelineConfig, PathConfig, DataConfig,
  |                        ModelConfig, LoggingConfig)
  |
  +-- logger.py           (LoggerFactory - console stream log setup)
  |
  +-- data_loader.py      (DataLoaderService - read zip, clean transactions,
  |                        group item baskets)
  |
  +-- apriori_service.py  (AprioriMBAService - candidate generation, anti-monotonicity
                           pruning, rule mining, scatter/bar plots, reporting)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter            | Location      | Default              | Description                                      |
|----------------------|---------------|----------------------|--------------------------------------------------|
| `min_support`        | `ModelConfig` | `0.015`              | Minimum support threshold fraction               |
| `min_confidence`     | `ModelConfig` | `0.20`               | Minimum confidence threshold fraction            |
| `min_lift`           | `ModelConfig` | `1.0`                | Minimum lift ratio threshold                     |
| `max_itemset_length` | `ModelConfig` | `3`                  | Maximum size of frequent itemsets to search      |
| `country_filter`     | `DataConfig`  | `'United Kingdom'`   | Filter transactions by country                   |
| `min_quantity`       | `DataConfig`  | `1`                  | Minimum quantity per line item                   |
| `min_price`          | `DataConfig`  | `0.01`               | Minimum unit price per item                      |
