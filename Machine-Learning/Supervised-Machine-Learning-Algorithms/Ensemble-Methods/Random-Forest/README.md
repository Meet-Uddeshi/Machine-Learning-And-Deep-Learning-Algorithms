# Supervised Machine Learning - Random Forest Classifier

> Supervised Machine Learning | Ensemble Bagging & Subspace Randomization Algorithm

---

## Table of Contents

1. [What is Random Forest?](#1-what-is-random-forest)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Random Forest Calculation (Step-by-Step)](#5-worked-random-forest-calculation-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Random Forest?

Random Forest is a highly effective **ensemble learning algorithm** that combines Bagging (Bootstrap Aggregating) with Random Subspace Selection to train a large collection of de-correlated decision trees. By averaging predictions across hundreds of individual decision trees, Random Forest significantly reduces model variance without increasing bias.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Classification & Regression)                    |
| Ensemble Technique | Bagging + Feature Subspace Sampling ($m = \sqrt{D}$)                 |
| Base Estimators    | Fully grown unpruned decision trees                                  |
| Out-Of-Bag (OOB)   | Inherent internal cross-validation error estimation                 |
| Key Hyperparameters| `n_estimators`, `max_features`, `max_depth`, `bootstrap`             |

---

## 2. Theoretical Explanation

### How Random Forest Works

Random Forest decorrelates decision trees by injecting randomness during both data sampling and feature node selection.

```
Training Dataset ---> Generate M Bootstrap Samples ---> Train M Decision Trees (Random Feature Subspaces)
                                                                                  |
                                                                                  v
                Final Class Prediction <--- Majority Vote / Probability Average <--- Aggregate Tree Outputs
```

1. **Bootstrap Sampling**: Draw $M$ independent bootstrap samples with replacement from training dataset $D$.
2. **Random Subspace Selection**: At each node split, consider only a random subset $m = \sqrt{D}$ features instead of all $D$ features.
3. **Aggregated Voting**: Combine predictions across all trees via majority voting.

---

## 3. Mathematical Operations

### 1. Ensemble Prediction Aggregation

For classification with $M$ decision trees $h_m(x)$:

$$\hat{y} = \arg\max_{c} \frac{1}{M} \sum_{m=1}^{M} I(h_m(x) = c)$$

### 2. Feature Importance (Mean Decrease Gini)

Total decrease in Gini impurity brought by feature $X_j$ averaged across all trees:

$$\text{Importance}(X_j) = \frac{1}{M} \sum_{m=1}^{M} \sum_{t \in T_m: v(t)=j} \Delta \text{Gini}(t)$$

---

## 4. Real-World Example

### Heart Disease Medical Classification (`heart.csv`)

- **Dataset**: `heart.csv` (Heart Disease Dataset)
- **Target Variable**: `target` (`0` = Healthy, `1` = Heart Disease)
- **Features Used**: `age`, `cp`, `trestbps`, `chol`, `thalach`, `oldpeak`, `slope`, `ca`, `thal`.

---

## 5. Worked Random Forest Calculation (Step-by-Step)

Consider an ensemble of $M=3$ decision trees evaluating a test instance $x$:

### Step 1: Obtain Individual Tree Predictions
- Tree 1 Output: $h_1(x) = 1$ (Confidence: 0.85)
- Tree 2 Output: $h_2(x) = 1$ (Confidence: 0.90)
- Tree 3 Output: $h_3(x) = 0$ (Confidence: 0.60)

### Step 2: Majority Vote Aggregation
- Class 1 Votes: 2 ($h_1, h_2$)
- Class 0 Votes: 1 ($h_3$)

$$\hat{y}_{\text{ensemble}} = \text{Class 1} \quad (2 / 3 = 66.7\% \text{ Vote Share})$$

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
|  - ModelConfig  (n_estimators=100, max_features='sqrt')|
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
|  - Load heart.csv, scale with StandardScaler        |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: RandomForestClassifierService              |
|  - Fit RandomForestClassifier                       |
|  - Evaluate Accuracy, Precision, Recall, F1, ROC-AUC|
|  - Plot confusion_matrix.png & feature_importance.png|
|  - Export random_forest_analysis.md report          |
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
  +-- random_forest_classifier.py (RandomForestClassifierService - fit, feature importance, metrics)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter      | Location      | Default  | Description                                      |
|----------------|---------------|----------|--------------------------------------------------|
| `n_estimators` | `ModelConfig` | `100`    | Total number of decision trees in ensemble       |
| `max_features` | `ModelConfig` | `'sqrt'` | Number of features to consider per split         |
| `max_depth`    | `ModelConfig` | `None`   | Maximum depth per tree (None grows until pure)   |
| `test_size`    | `DataConfig`  | `0.20`   | Partition percentage reserved for test set       |
| `target_column`| `DataConfig`  | `'target'`| Target classification outcome variable          |
