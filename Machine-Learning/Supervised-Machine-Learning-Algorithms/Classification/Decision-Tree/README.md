# Classification - Decision Tree

> Supervised Machine Learning | Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Decision Tree Sum (Step-by-Step)](#5-worked-decision-tree-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where the goal is to predict a discrete categorical class label for an input data point based on learned features.

The algorithm builds a rule-based decision structure from a **labelled training dataset**, enabling it to classify new transactions as legitimate or fraudulent.

---

## 2. Theoretical Explanation

### How Decision Trees Work

A Decision Tree splits data into subset branches based on feature values, forming an inverted tree hierarchy:

1. **Root Node**: The top node representing the entire dataset, split using the feature offering maximum information gain.
2. **Internal Nodes**: Decision rules checking whether a feature value meets a threshold condition.
3. **Leaf Nodes**: Terminal nodes outputting the final class prediction.

### Key Concepts

- **Recursive Binary Splitting**: At each node, every feature and possible numerical threshold is tested to find the split minimizing child node impurity.
- **Overfitting & Pruning**: Deep trees can memorize training noise. Parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` restrict growth and improve generalization.

---

## 3. Mathematical Operations

### Gini Impurity

Gini Impurity measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the subset.

For a dataset node with `K` classes and class probabilities `p_i`:

```
Gini(Node) = 1 - sum(p_i ^ 2) for i = 1 to K
```

- `Gini = 0`: Pure node (all samples belong to one class).
- `Gini = 0.5`: Equal binary distribution (maximum uncertainty).

### Information Gain

Information Gain measures the reduction in impurity after splitting a parent node `P` into left (`L`) and right (`R`) children:

```
Information Gain = Gini(P) - [ (N_L / N_P) * Gini(L) + (N_R / N_P) * Gini(R) ]
```

---

## 4. Real-World Example

### Financial Fraud Detection

This project predicts whether an online financial transaction is **Fraudulent (`isFraud = 1`)** or **Legitimate (`isFraud = 0`)**.

**Dataset:** `onlinefraud.csv`  
**Target Variable:** `isFraud`  
**Features Used:** `step`, `type` (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN), `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.

**Excluded Columns:** `nameOrig`, `nameDest`, `isFlaggedFraud`.

---

## 5. Worked Decision Tree Sum (Step-by-Step)

Consider a dataset of 10 financial transactions (6 Legitimate, 4 Fraudulent).

### Initial Parent Node (P)

```
N = 10 (Legitimate = 6, Fraudulent = 4)
p_Legit = 6/10 = 0.6
p_Fraud = 4/10 = 0.4

Gini(Parent) = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48
```

### Test Split on Feature: `Amount > 200,000`

- **Left Child (L): `Amount <= 200,000`** (N_L = 7, 6 Legit, 1 Fraud)
  ```
  p_Legit = 6/7,  p_Fraud = 1/7
  Gini(L) = 1 - [ (6/7)^2 + (1/7)^2 ] = 1 - [ 36/49 + 1/49 ] = 1 - (37/49) = 0.2449
  ```

- **Right Child (R): `Amount > 200,000`** (N_R = 3, 0 Legit, 3 Fraud)
  ```
  p_Legit = 0/3 = 0,  p_Fraud = 3/3 = 1
  Gini(R) = 1 - (0^2 + 1^2) = 0.0000  (Pure Node!)
  ```

### Compute Information Gain

```
Weighted Child Gini = (7/10) * 0.2449 + (3/10) * 0.0000 = 0.1714
Information Gain    = Gini(Parent) - Weighted Child Gini
                     = 0.4800 - 0.1714 = 0.3086
```

Because Information Gain > 0, the decision tree selects `Amount > 200,000` as an optimal split point.

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
|  - PathConfig (dataset path, output path)           |
|  - DataConfig (target='isFraud', max_samples=50000) |
|  - ModelConfig (criterion='gini', max_depth=10)     |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
|  - Ingest CSV -> Validate schema -> Downsample      |
|  - Impute nulls & encode categorical variables      |
|  - Standardize features & perform Stratified Split  |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 4: DecisionTreeClassifierService.train()      |
|  - Fit decision tree estimator on training set      |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 5: DecisionTreeClassifierService.evaluate()   |
|  - Evaluate metrics (Accuracy, Precision, Recall)   |
|  - Plot Confusion Matrix & Feature Importances      |
|  - Write results text file & markdown analysis      |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|               END: Pipeline Complete               |
+-----------------------------------------------------+
```

### Module Responsibility Map

```
main.py
  |
  +-- config.py                   (PipelineConfig, PathConfig, DataConfig, ModelConfig, LoggingConfig)
  +-- logger.py                   (LoggerFactory)
  +-- data_loader.py              (DataLoaderService)
  +-- decision_tree_classifier.py (DecisionTreeClassifierService)
```
