# Supervised Machine Learning - K-Nearest Neighbors (K-NN) Classifier

> Supervised Machine Learning | Instance-Based Lazy Learning Classification Algorithm

---

## Table of Contents

1. [What is K-Nearest Neighbors?](#1-what-is-k-nearest-neighbors)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked K-NN Calculation (Step-by-Step)](#5-worked-k-nn-calculation-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is K-Nearest Neighbors?

K-Nearest Neighbors (K-NN) is a non-parametric, instance-based **lazy learning algorithm**. Unlike eager learning models, K-NN does not construct an explicit internal model during training; instead, it stores all training observations and computes distance metrics during query time to classify new query points based on majority voting of their $k$ nearest neighbors.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Classification & Regression)                    |
| Learning Type      | Lazy Learning (No explicit training phase)                           |
| Distance Metrics   | Euclidean, Manhattan, Minkowski distance                             |
| Decision Rule      | Majority voting (Classification) or Mean averaging (Regression)       |
| Key Hyperparameters| `n_neighbors` ($k$), `metric`, `weights` (`'uniform'`, `'distance'`) |

---

## 2. Theoretical Explanation

### How K-NN Works

K-NN classifies unlabeled query vectors based on spatial proximity in standardized feature space.

```
Query Data Point ---> Compute Distance to All Training Points ---> Select K Smallest Distances
                                                                             |
                                                                             v
               Assign Query Point to Majority Class <--- Aggregate Class Votes of K Neighbors
```

1. **Distance Computation**: Calculate distance between unlabelled test instance $x$ and all stored training samples.
2. **Neighbor Selection**: Identify the $k$ training samples with minimum distance to $x$.
3. **Majority Voting**: Count class occurrences among the $k$ neighbors and assign $x$ to the majority class.

---

## 3. Mathematical Operations

### 1. Distance Metrics

For vectors $x = (x_1, \dots, x_D)$ and $y = (y_1, \dots, y_D)$:

- **Euclidean Distance ($L_2$ norm)**:
  $$d(x, y) = \sqrt{\sum_{j=1}^{D} (x_j - y_j)^2}$$

- **Manhattan Distance ($L_1$ norm)**:
  $$d(x, y) = \sum_{j=1}^{D} |x_j - y_j|$$

- **Minkowski Distance ($L_p$ norm)**:
  $$d(x, y) = \left( \sum_{j=1}^{D} |x_j - y_j|^p \right)^{\frac{1}{p}}$$

### 2. Weighted Voting Rule

$$\hat{y} = \arg\max_{c} \sum_{i \in N_k(x)} w_i \cdot I(y_i = c)$$

where $w_i = \frac{1}{d(x, x_i)}$ for distance-weighted classification.

---

## 4. Real-World Example

### Medical Heart Disease Diagnosis (`heart.csv`)

- **Dataset**: `heart.csv` (Heart Disease Dataset)
- **Target Variable**: `target` (`0` = Healthy, `1` = Heart Disease)
- **Features Used**: Standardized clinical measures (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, etc.).

---

## 5. Worked K-NN Calculation (Step-by-Step)

Consider $N=4$ training points in 2D space ($X_1, X_2$) with binary class $Y$:
- $A = (1, 2) \implies Y = 0$
- $B = (2, 3) \implies Y = 0$
- $C = (6, 5) \implies Y = 1$
- $D = (7, 7) \implies Y = 1$

Query point: $Q = (3, 3)$. Let $k=3$, using Euclidean distance.

### Step 1: Compute Distances to $Q(3, 3)$
- $d(Q, A) = \sqrt{(3-1)^2 + (3-2)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.236$
- $d(Q, B) = \sqrt{(3-2)^2 + (3-3)^2} = \sqrt{1 + 0} = \sqrt{1} = 1.000$
- $d(Q, C) = \sqrt{(3-6)^2 + (3-5)^2} = \sqrt{9 + 4} = \sqrt{13} \approx 3.606$
- $d(Q, D) = \sqrt{(3-7)^2 + (3-7)^2} = \sqrt{16 + 16} = \sqrt{32} \approx 5.657$

### Step 2: Select $k=3$ Nearest Neighbors
Smallest 3 distances:
1. Point $B$ ($d=1.000$, Class 0)
2. Point $A$ ($d=2.236$, Class 0)
3. Point $C$ ($d=3.606$, Class 1)

### Step 3: Majority Vote
- Class 0: 2 votes ($B, A$)
- Class 1: 1 vote ($C$)

Prediction for $Q$: **Class 0**.

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
|  - ModelConfig  (n_neighbors=5, metric='minkowski') |
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
|  Step 4: KNNClassifierService                       |
|  - Store training instances                         |
|  - Predict query instances                          |
|  - Evaluate Accuracy, Precision, Recall, F1, ROC-AUC|
|  - Plot confusion matrix & k-value accuracy curve   |
|  - Export knn_analysis.md report                    |
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
  +-- config.py         (PipelineConfig, PathConfig, DataConfig, ModelConfig)
  +-- logger.py         (LoggerFactory - stdout stream logging)
  +-- data_loader.py    (DataLoaderService - loading, scaling, splitting)
  +-- knn_classifier.py (KNNClassifierService - fit, k-curve plot, metrics)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter     | Location      | Default       | Description                                      |
|---------------|---------------|---------------|--------------------------------------------------|
| `n_neighbors` | `ModelConfig` | `5`           | Number of nearest neighbors ($k$)                |
| `weights`     | `ModelConfig` | `'uniform'`   | Weight function (`'uniform'` or `'distance'`)    |
| `metric`      | `ModelConfig` | `'minkowski'` | Distance metric algorithm                        |
| `p`           | `ModelConfig` | `2`           | Power parameter for Minkowski metric ($p=2 \implies L_2$)|
| `test_size`   | `DataConfig`  | `0.20`        | Partition percentage reserved for test evaluation|