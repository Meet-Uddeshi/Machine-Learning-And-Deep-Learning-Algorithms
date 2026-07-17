# Ensemble Methods - Random Forest

> Supervised Machine Learning | Ensemble Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Random Forest Sum (Step-by-Step)](#5-worked-random-forest-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where the goal is to assign an input data point to one of a fixed set of **discrete categories (classes)**.

The algorithm learns from a **labelled training dataset** - data where the correct class is already known. After learning, it applies that knowledge to predict the class of **new, unseen data points**.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning                                                  |
| Output type        | Discrete class label (e.g., "Furniture", "Food", "Clothing", etc.)   |
| Training data      | Labelled (input + known correct class)                               |
| Prediction         | Assigns one class label from a predefined set                        |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, OOB Score   |

---

## 2. Theoretical Explanation

### How Random Forests Work

A Random Forest is a meta-estimator that fits a number of **Decision Tree Classifiers** on various sub-samples of the dataset and uses averaging (majority voting) to improve predictive accuracy and control over-fitting. Its core idea is:

> "Build a collection of diverse, uncorrelated decision trees through bagging and feature subspace sampling, and aggregate their predictions to make a robust ensemble prediction."

A Random Forest is an **eager learning** ensemble algorithm. Instead of relying on a single complex classifier, it combines predictions from $B$ individual tree estimators.

### Key Ensemble Strategies

```
Bootstrap Aggregation (Bagging) -- Training each tree on a separate random sample of the training set, selected with replacement (bootstrap sample).
Feature Subspace Sampling      -- Selecting a random subset of features at each node split. This decorrelates the trees, preventing dominant features from making all trees identical.
Out-of-Bag (OOB) Error         -- Evaluating trees on the samples (~36.8%) that were NOT selected in their bootstrap training set. This provides a built-in cross-validation score.
Ensemble Voting                -- Aggregating predictions. Each tree casts a vote; the class with the majority of votes is the forest's final prediction.
```

### Why No Feature Scaling is Needed

Random Forests partition features using threshold inequalities at split nodes (e.g. $x_i \le \theta$). The criteria for splitting nodes depend entirely on the **rank order** of values inside the subset, not their absolute scale. Multiplicative or additive scaling does not alter the relative ordering of values. Therefore, feature scaling is **unnecessary** for Random Forest and is disabled by default.

---

## 3. Mathematical Operations

### Bootstrap Selection Probability

For a training set of size $N$, the probability of a specific sample **not** being selected in $N$ bootstrap draws with replacement is:

$$P(\text{not selected}) = \left( 1 - \frac{1}{N} \right)^N$$

As $N \to \infty$:

$$\lim_{N \to \infty} \left( 1 - \frac{1}{N} \right)^N = \frac{1}{e} \approx 0.368 \quad (36.8\%)$$

- **In-bag samples**: $\approx 63.2\%$ of training rows are used to train each tree.
- **Out-of-bag (OOB) samples**: $\approx 36.8\%$ of training rows are held out.

### Splitting Criterion (Gini Impurity)

Like individual Decision Trees, Random Forest estimators evaluate splits using Gini Impurity:

$$\text{Gini}(node) = 1 - \sum_{c=1}^C p_c^2$$

where $p_c$ is the proportion of samples belonging to class $c$ in that node.

### Gini Feature Importance (Mean Decrease in Impurity)

The importance of feature $j$ is calculated as the sum of Gini impurity decreases across all nodes $k$ where feature $j$ was selected to split, weighted by the fraction of samples reaching node $k$, averaged across all $B$ trees:

$$\text{Importance}(feature_j) = \frac{1}{B} \sum_{b=1}^B \sum_{k \in \text{splits on } j \text{ in tree } b} \left( \frac{N_k}{N_{\text{total}}} \times \Delta\text{Gini}(k) \right)$$

These values are normalized to sum to $1.0$.

### Ensemble Voting Scheme

For a new sample $x$, each tree $t_b$ outputs a predicted class $\hat{y}_b$. The ensemble prediction is the argmax of class frequencies (majority vote):

$$\hat{y} = \text{mode} \{ \hat{y}_1, \hat{y}_2, \dots, \hat{y}_B \}$$

---

## 4. Real-Word Example

### Transaction Product Category Prediction

This project uses Random Forest to classify transaction records into their **Product Category** based on rep, region, amounts, quantities, and prices.

**Dataset:** `sales_data.csv`

**Target Variable:** `Product_Category`
Possible classes: `Furniture`, `Food`, `Clothing`, `Electronics`

**Features Used:**

| Feature | Description |
|---------|-------------|
| `Sales_Rep` | Name of sales representative (categorical) |
| `Region` | Geographic region of sale (categorical) |
| `Sales_Amount` | Total transaction sale amount |
| `Quantity_Sold` | Total number of units sold |
| `Unit_Cost` | Unit cost of product |
| `Unit_Price` | Unit retail price of product |
| `Customer_Type` | Customer segment (`New` or `Returning`) |
| `Discount` | Applied discount percentage |
| `Payment_Method` | Transaction payment channel (Cash, Credit Card, etc.) |
| `Sales_Channel` | Mode of sale (`Online` or `Retail`) |

**Columns excluded from features:**
- `Product_ID` - unique transaction transaction ID (non-predictive).
- `Sale_Date` - timestamp string (non-predictive for categorization).
- `Region_and_Sales_Rep` - combined feature (collinear redundancy).

**Why Random Forest is a good fit here:**
- The dataset contains complex mixed categorical and numerical features. Random Forest handles heterogeneous features natively after label encoding.
- Ensembling prevents model overfitting, maintaining generalization error OOB bounds.
- MDI Feature Importances reveal which aspects of sales (e.g. amount, discount, rep) are most predictive of category.

---

## 5. Worked Random Forest Sum (Step-by-Step)

Let's trace how an ensemble of $B=3$ simple decision trees classifies a new transaction:

### The Query Transaction

- `Quantity_Sold` = 45
- `Sales_Amount` = 1200
- `Customer_Type` = `Returning`

### Tree 1 ($T_1$) Decision Route

- **Split 1 (Root)**: `Quantity_Sold` <= 30?
  - Since `Quantity_Sold` = 45, the condition is **False** (go Right).
- **Leaf (Right)**: Predict **`Furniture`**

**Result $T_1(x) = \text{Furniture}$**

---

### Tree 2 ($T_2$) Decision Route

- **Split 1 (Root)**: `Customer_Type` == `New`?
  - Since `Customer_Type` = `Returning`, the condition is **False** (go Right).
- **Split 2**: `Sales_Amount` <= 1500?
  - Since `Sales_Amount` = 1200, the condition is **True** (go Left).
- **Leaf (Left)**: Predict **`Food`**

**Result $T_2(x) = \text{Food}$**

---

### Tree 3 ($T_3$) Decision Route

- **Split 1 (Root)**: `Sales_Amount` <= 800?
  - Since `Sales_Amount` = 1200, the condition is **False** (go Right).
- **Split 2**: `Quantity_Sold` <= 40?
  - Since `Quantity_Sold` = 45, the condition is **False** (go Right).
- **Leaf (Right)**: Predict **`Furniture`**

**Result $T_3(x) = \text{Furniture}$**

---

### Aggregate Ensemble Voting

| Class | Votes | Estimators |
|-------|-------|------------|
| **Furniture** | 2 | $T_1$, $T_3$ |
| **Food** | 1 | $T_2$ |
| **Clothing** | 0 | - |
| **Electronics** | 0 | - |

**Final Ensemble Prediction: `Furniture`** (Winner by majority vote, 2 against 1).

---

## 6. Program Flowchart

The flowchart below describes the complete execution flow of the Random Forest pipeline.

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig   (dataset path, output path)         |
|  - DataConfig   (target, splits, scale_features=F)  |
|  - ModelConfig  (n_estimators, max_depth, bootstrap,|
|                  oob_score, criterion, min_samples) |
|  - LoggingConfig (level=DEBUG, log_to_file=True)    |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler  (random_forest_sales.log in output/)|
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
|  _load_csv()     |               |  File not found?  |
|  Read CSV from   |  -- Error --> |  Raise            |
|  data/ folder    |               |  FileNotFoundError|
+------------------+               +-------------------+
          |
          v
+------------------+
|  _validate_schema|
|  Verify target   |
|  Product_Category|
+------------------+
          |
          v
+------------------+
|  _log_data_      |
|  summary()       |
|  Log rows/cols,  |
|  class balance   |
+------------------+
          |
          v
+--------------------------------------------------+
|  _preprocess()                                   |
|  - Separate target ('Product_Category')          |
|  - Drop Product_ID, Sale_Date, Region_Rep        |
|  - Impute nulls (median/mode)                    |
|  - LabelEncode Product_Category (Food=0, etc.)   |
|  - LabelEncode categorical features              |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  _split()                                        |
|  - Split into 80% Train, 20% Test (stratified)   |
|  - Scale features is False (no scaling needed)   |
+--------------------------------------------------+
          |
          v
+-----------------------------------------------------+
|  Step 4: RandomForestClassifierService.train()      |
|  - Build RandomForestClassifier with config params  |
|  - For each tree b = 1 to B:                        |
|    - Draw bootstrap sample from training set        |
|    - Split nodes recursively using random features  |
|  - Call model.fit(x_train, y_train)                 |
|  - Log train duration, OOB score, estimator stats   |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 5: RandomForestClassifierService.evaluate()   |
+-----------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Generate Predictions                            |
|  - Call model.predict(x_test)                    |
|  - Aggregate majority vote across B estimators   |
|  - Calculate Accuracy, Precision, Recall, F1      |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  _log_evaluation()                               |
|  - Write key metrics to stdout and logs          |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  _save_results() & _save_forest_details()        |
|  - Save classification_results.txt               |
|  - Save random_forest_details.txt (tree metrics) |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  _generate_plots()                               |
|  1. confusion_matrix.png                         |
|  2. classification_report.png                    |
|  3. feature_importance.png                       |
|  4. random_forest_tree_structure.png (Tree [0])  |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  _save_analysis()                                |
|  - Save classification_analysis.md report        |
+--------------------------------------------------+
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
  +-- config.py                   (PipelineConfig, PathConfig, DataConfig,
  |                                ModelConfig, LoggingConfig)
  |
  +-- logger.py                   (LoggerFactory - console + file handlers)
  |
  +-- data_loader.py              (DataLoaderService - load, preprocess, split;
  |                                scale features is scale-invariant no-op)
  |
  +-- random_forest_classifier.py (RandomForestClassifierService - train,
                                   predict, evaluate, forest details,
                                   tree structure plotting, saving results)
```

---

### Configuration

All tunable parameters are located in `src/config.py`. Edit the dataclass defaults to adjust:

| Parameter            | Location      | Default      | Description                                      |
|----------------------|---------------|--------------|--------------------------------------------------|
| `n_estimators`       | `ModelConfig` | `100`        | Number of estimators (trees) in ensemble         |
| `criterion`          | `ModelConfig` | `'gini'`     | Split metric (`'gini'` or `'entropy'`)           |
| `max_depth`          | `ModelConfig` | `None`       | Maximum tree depth limit                         |
| `min_samples_split`  | `ModelConfig` | `2`          | Min samples to split an internal node            |
| `min_samples_leaf`   | `ModelConfig` | `1`          | Min samples required at a leaf node              |
| `max_features`       | `ModelConfig` | `'sqrt'`     | Features considered per node split               |
| `bootstrap`          | `ModelConfig` | `True`       | Enable bootstrap aggregation sample draws        |
| `oob_score`          | `ModelConfig` | `True`       | Out-of-bag validation metrics evaluation         |
| `test_size`          | `DataConfig`  | `0.20`       | Fraction of data used for testing                |
| `target_column`      | `DataConfig`  | `'Product_Category'` | Target label column in dataset           |
