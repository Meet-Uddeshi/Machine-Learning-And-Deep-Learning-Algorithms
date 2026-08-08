# Supervised Machine Learning - Decision Tree Classifier

> Supervised Machine Learning | Non-Parametric Hierarchical Tree-Based Classification

---

## Table of Contents

1. [What is Decision Tree Classification?](#1-what-is-decision-tree-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Decision Tree Calculation (Step-by-Step)](#5-worked-decision-tree-calculation-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Decision Tree Classification?

Decision Tree Classification is a **non-parametric supervised learning algorithm** that recursively partitions feature space into hierarchical axis-aligned decision boundaries. The resulting structure resembles an inverted tree with internal decision nodes, branches representing feature thresholds, and leaf nodes representing final class labels.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Classification & Regression)                    |
| Splitting Criteria | Gini Impurity, Information Gain (Entropy), Log Loss                  |
| Model Architecture | Hierarchical Binary or Multi-way Decision Tree                       |
| Interpretability   | Extremely high (direct human-readable decision rules)                |
| Key Hyperparameters| `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`    |

---

## 2. Theoretical Explanation

### How Decision Trees Work

Decision Trees operate via recursive binary splitting to maximize feature node purity.

```
Full Training Set ---> Select Best Feature & Threshold ---> Split Node into Sub-nodes
                                                                 |
                                                                 v
            Final Class Label Predictions <--- Check Stopping Criterion (Depth / Purity)
```

1. **Feature Evaluation**: For each feature, compute potential split thresholds.
2. **Impurity Reduction**: Select the feature and threshold that yield the maximum reduction in impurity (Gini or Entropy).
3. **Recursive Partitioning**: Divide child samples and repeat the process on sub-nodes.
4. **Pruning & Termination**: Stop when reaching max depth, minimum node samples, or pure leaves to prevent overfitting.

---

## 3. Mathematical Operations

### 1. Gini Impurity

For a dataset node $D$ containing $C$ classes where $p_i$ is the probability of class $i$:

$$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$

### 2. Entropy & Information Gain

$$\text{Entropy}(D) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

Information Gain for a split on feature $A$ dividing $D$ into subsets $D_1$ and $D_2$:

$$\text{GAIN}(D, A) = \text{Entropy}(D) - \sum_{k=1}^{2} \frac{|D_k|}{|D|} \text{Entropy}(D_k)$$

---

## 4. Real-World Example

### Medical Diagnosis Prediction (`heart.csv`)

- **Dataset**: `heart.csv` (Heart Disease Dataset)
- **Target Variable**: `target` (`0` = Healthy, `1` = Heart Disease)
- **Features Used**: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.

---

## 5. Worked Decision Tree Calculation (Step-by-Step)

Consider a toy dataset of $N=4$ patients evaluated on Blood Pressure ($X$) for Heart Disease ($Y$):
- $P_1: X = 110 \implies Y = 0$
- $P_2: X = 120 \implies Y = 0$
- $P_3: X = 140 \implies Y = 1$
- $P_4: X = 150 \implies Y = 1$

### Step 1: Compute Root Node Gini Impurity
Class probabilities: $p_0 = 2/4 = 0.5$, $p_1 = 2/4 = 0.5$.

$$\text{Gini}(D_{\text{root}}) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 0.50$$

### Step 2: Evaluate Split Threshold at $X \le 130$
- **Left Node ($D_L$)**: $P_1, P_2 \implies (2 \text{ Class } 0, 0 \text{ Class } 1)$
  $$\text{Gini}(D_L) = 1 - (1.0^2 + 0.0^2) = 0.0$$
- **Right Node ($D_R$)**: $P_3, P_4 \implies (0 \text{ Class } 0, 2 \text{ Class } 1)$
  $$\text{Gini}(D_R) = 1 - (0.0^2 + 1.0^2) = 0.0$$

### Step 3: Weighted Impurity of Split
$$\text{Gini}_{\text{split}} = \frac{2}{4}(0.0) + \frac{2}{4}(0.0) = 0.0$$

Gini Reduction = $0.50 - 0.0 = 0.50$ (Perfect separation achieved in one split!).

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
|  - PathConfig   (dataset and output paths)          |
|  - DataConfig   (target_column='target', test_size) |
|  - ModelConfig  (criterion='gini', max_depth=5)     |
|  - LoggingConfig (console-only logs)                |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 2: Initialize Console Logger                  |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
|  - Load heart.csv, drop duplicates, split train/test|
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: DecisionTreeClassifierService              |
|  - Fit DecisionTreeClassifier                       |
|  - Evaluate Accuracy, Precision, Recall, F1, ROC-AUC|
|  - Generate confusion_matrix.png & feature_imp.png  |
|  - Export decision_tree_analysis.md report          |
+-----------------------------------------------------+
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
  +-- config.py                   (PipelineConfig, PathConfig, DataConfig, ModelConfig)
  +-- logger.py                   (LoggerFactory - stdout stream logging)
  +-- data_loader.py              (DataLoaderService - loading, scaling, splitting)
  +-- decision_tree_classifier.py (DecisionTreeClassifierService - training, tree plot, metrics)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter            | Location      | Default   | Description                                      |
|----------------------|---------------|-----------|--------------------------------------------------|
| `criterion`          | `ModelConfig` | `'gini'`  | Splitting function (`'gini'` or `'entropy'`)     |
| `max_depth`          | `ModelConfig` | `5`       | Maximum tree depth constraint                    |
| `min_samples_split`  | `ModelConfig` | `2`       | Minimum samples required to split a node         |
| `test_size`          | `DataConfig`  | `0.20`    | Partition percentage reserved for test set       |
| `target_column`      | `DataConfig`  | `'target'`| Target classification outcome variable           |
