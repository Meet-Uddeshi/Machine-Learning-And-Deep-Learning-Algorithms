# Classification - Support Vector Machine (SVM)

> Supervised Machine Learning | Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked SVM Sum (Step-by-Step)](#5-worked-svm-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where the goal is to assign an input data point to one of a fixed set of **discrete categories (classes)**.

The algorithm learns from a **labelled training dataset** - data where the correct class is already known. After learning, it applies that knowledge to predict the class of **new, unseen data points**.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning                                                  |
| Output type        | Discrete class label (e.g., "setosa", "versicolor", "virginica")     |
| Training data      | Labelled (input + known correct class)                               |
| Prediction         | Assigns one class label from a predefined set                        |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix              |

---

## 2. Theoretical Explanation

### How Support Vector Machines Work

A Support Vector Machine (SVM) is a powerful, versatile classification algorithm. Its core idea is:

> "Find the optimal separating hyperplane that maximizes the geometric margin between different classes in a high-dimensional space."

An SVM is an **eager learning** algorithm - it constructs a decision boundary (hyperplane) during training. At prediction time, a new data point is classified depending on which side of the decision boundary it falls.

### Key Concepts

```
Hyperplane      -- The decision boundary that separates the classes. In 2D, it is a line; in 3D, a plane; and in higher dimensions, a hyperplane.
Support Vectors -- The training data points that lie closest to the decision boundary. They are the critical elements that define the hyperplane.
Margin          -- The distance between the separating hyperplane and the closest data points (Support Vectors). Maximizing the margin provides generalization robustness.
Kernel Trick    -- A mathematical function that projects low-dimensional, non-linearly separable data into a higher-dimensional space where it becomes linearly separable.
```

### Why Feature Scaling is Essential for SVM

Unlike Decision Trees which are scale-invariant (they split features based on thresholds), SVM is highly **scale-sensitive**. 

Because SVM computes Euclidean distances between data points to define the margin and hyperplane, features with larger scales (e.g., house prices in thousands vs. number of rooms) will dominate the distance metric. This distorts the boundary and ignores smaller-scale but highly predictive features. Therefore, standardizing features to zero mean and unit variance ($\mu=0, \sigma=1$) is a mandatory preprocessing step.

---

## 3. Mathematical Operations

### The Decision Boundary (Hyperplane)

For a dataset of $N$ points $(x_i, y_i)$ where $x_i \in \mathbb{R}^D$ and $y_i \in \{-1, +1\}$, the separating hyperplane is defined as:

$$w^T x + b = 0$$

where:
- $w$ is the normal vector to the hyperplane.
- $b$ is the bias term.

The classifier predicts the class label for a new point $x$ as:

$$\hat{y} = \text{sign}(w^T x + b)$$

### Primal Optimization Problem (Hard Margin)

To maximize the margin width $2/\|w\|$, we minimize $\|w\|^2 / 2$ subject to no training misclassifications:

$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i (w^T x_i + b) \ge 1 \quad \forall i=1, \dots, N$$

### Soft Margin Formulation (C Parameter)

To handle noisy, overlapping, or non-linearly separable data, slack variables $\xi_i \ge 0$ are introduced to allow some margin violations. The regularization parameter $C$ controls the trade-off:

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- **Large $C$**: Penalizes errors heavily. Focuses on minimizing misclassifications on the training set, leading to a narrower margin (potential overfitting).
- **Small $C$**: Tolerates more errors. Focuses on finding a wider margin, improving generalization (potential underfitting).

### The Dual Formulation and Kernels

Using Lagrange multipliers $\alpha_i$, the optimization is transformed into its dual representation:

$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

subject to:

$$\sum_{i=1}^N \alpha_i y_i = 0 \quad \text{and} \quad 0 \le \alpha_i \le C$$

The dot product $x_i^T x_j$ is replaced by a **Kernel Function** $K(x_i, x_j) = \Phi(x_i)^T \Phi(x_j)$ to implicitly compute similarity in higher dimensions:

| Kernel | Formula | Description |
|--------|---------|-------------|
| **Linear** | $K(x_i, x_j) = x_i^T x_j$ | Used when classes are already linearly separable. |
| **Polynomial** | $K(x_i, x_j) = (x_i^T x_j + 1)^d$ | Non-linear boundary; degree $d$ determines complexity. |
| **Radial Basis Function (RBF)** | $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ | Default kernel; uses radial distance for localized boundaries. |
| **Sigmoid** | $K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$ | Emulates neural network activation. |

---

## 4. Real-World Example

### Iris Flower Species Classification

This project implements SVM to classify **Iris Flower Species** based on physical measurements of their sepals and petals.

**Dataset:** `iris.csv`

**Target Variable:** `species`
Possible classes: `setosa`, `versicolor`, `virginica`

**Features Used:**

| Feature | Description |
|---------|-------------|
| `sepal_length` | Length of the sepal (cm) |
| `sepal_width`  | Width of the sepal (cm)  |
| `petal_length` | Length of the petal (cm) |
| `petal_width`  | Width of the petal (cm)  |

**Why SVM is a good fit here:**
- SVM handles small, high-quality datasets (like Iris) exceptionally well.
- The multi-class classification is resolved using the standard One-Vs-One (OVO) approach internally.
- Features are highly correlated, and standardizing them allows SVM to find clear separating boundaries.
- Decision boundary visualization projected using PCA shows how support vectors act as boundary anchors.

---

## 5. Worked SVM Sum (Step-by-Step)

Let's solve a simple binary classification problem by hand. We have 4 training points in 2D space.

### The Training Data

- **Class +1 ($y_i = 1$)**: 
  - $x_1 = (1, 1)$
  - $x_2 = (2, 0)$
- **Class -1 ($y_i = -1$)**:
  - $x_3 = (-1, -1)$
  - $x_4 = (0, -2)$

```
   y
   ^
   |     * x2 (2,0)
   |   * x1 (1,1)
---+------------------> x
   | * x3 (-1,-1)
   | * x4 (0,-2)
   |
```

### Step 1: Find the Hyperplane Equation

By looking at the grid, the classes are separated by the line:

$$x^{(1)} + x^{(2)} = 0$$

which can be written as:

$$w_1 x^{(1)} + w_2 x^{(2)} + b = 0 \implies w = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, b = 0$$

### Step 2: Scale $w$ to satisfy the Support Vector condition

The SVM hard margin formulation requires that for the closest points (Support Vectors):

$$y_i (w^T x_i + b) = 1$$

Let's calculate $w^T x_i + b$ for our points with the initial candidate $w = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$:
- For $x_1 = (1, 1)$: $y_1 (w^T x_1 + b) = 1 \times (1 \times 1 + 1 \times 1 + 0) = 2$
- For $x_2 = (2, 0)$: $y_2 (w^T x_2 + b) = 1 \times (1 \times 2 + 1 \times 0 + 0) = 2$
- For $x_3 = (-1, -1)$: $y_3 (w^T x_3 + b) = -1 \times (1 \times (-1) + 1 \times (-1) + 0) = 2$
- For $x_4 = (0, -2)$: $y_4 (w^T x_4 + b) = -1 \times (1 \times 0 + 1 \times (-2) + 0) = 2$

All values equal $2$. To force them to equal $1$ at the margin boundary, we scale $w$ and $b$ by $1/2$:

$$w = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}, \quad b = 0$$

Let's re-verify the support vector constraint:
- For $x_1$: $1 \times (0.5 \times 1 + 0.5 \times 1) = 1.0$ (Strict equality, SV!)
- For $x_2$: $1 \times (0.5 \times 2 + 0.5 \times 0) = 1.0$ (Strict equality, SV!)
- For $x_3$: $-1 \times (0.5 \times (-1) + 0.5 \times (-1)) = 1.0$ (Strict equality, SV!)
- For $x_4$: $-1 \times (0.5 \times 0 + 0.5 \times (-2)) = 1.0$ (Strict equality, SV!)

All four points lie exactly on the margin boundaries, making all of them **Support Vectors**.

### Step 3: Compute the Margin Width

The margin width is given by:

$$\text{Margin} = \frac{2}{\|w\|}$$

Let's calculate the norm of $w$:

$$\|w\| = \sqrt{w_1^2 + w_2^2} = \sqrt{0.5^2 + 0.5^2} = \sqrt{0.25 + 0.25} = \sqrt{0.5} = \frac{1}{\sqrt{2}}$$

Now compute the margin:

$$\text{Margin} = \frac{2}{1/\sqrt{2}} = 2\sqrt{2} \approx 2.8284$$

### Step 4: Classify a New Point

Let's classify a new point $x_{new} = (1.5, 1.5)$:

$$\hat{y} = \text{sign}(w^T x_{new} + b) = \text{sign}(0.5 \times 1.5 + 0.5 \times 1.5 + 0) = \text{sign}(1.5) = +1$$

The new point is classified correctly into Class +1.

---

## 6. Program Flowchart

The flowchart below describes the complete execution flow of the SVM Classification Pipeline.

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig   (dataset path, output path)         |
|  - DataConfig   (target column, splits, scaling)    |
|  - ModelConfig  (C, kernel, degree, gamma, prob)    |
|  - LoggingConfig (level=DEBUG, log_to_file=True)    |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler  (svm_iris.log in output/)          |
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
|  column species  |
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
|  - Separate target ('species') from features     |
|  - LabelEncode species (setosa=0, etc.)          |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  _split()                                        |
|  - Split into 80% Train, 20% Test (stratified)   |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  _scale_features()                               |
|  - Fit StandardScaler on training features       |
|  - Scale train features (mean=0, std=1)          |
|  - Transform test features                       |
+--------------------------------------------------+
          |
          v
+-----------------------------------------------------+
|  Step 4: SVMClassifierService.train()               |
|  - Build SVC with configuration parameters          |
|  - Fit the model using model.fit(x_train, y_train)  |
|  - Log train duration and support vector counts     |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 5: SVMClassifierService.evaluate()            |
+-----------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Generate Predictions                            |
|  - Call model.predict(x_test)                    |
|  - Calculate Accuracy, Precision, Recall, F1      |
|  - Calculate Permutation Feature Importance      |
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
|  _save_results() & _save_model_details()          |
|  - Save classification_results.txt               |
|  - Save svm_model_details.txt (coefs/intercepts) |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  _generate_plots()                               |
|  1. confusion_matrix.png                         |
|  2. classification_report.png                    |
|  3. feature_importance.png                       |
|  4. svm_decision_boundary.png (2D PCA Contour)   |
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
  +-- data_loader.py              (DataLoaderService - load, preprocess, split,
  |                                and StandardScaler application)
  |
  +-- svm_classifier.py           (SVMClassifierService - train, predict,
                                   evaluate, permutation importance,
                                   PCA boundary plotting, saving results)
```

---

### Configuration

All tunable parameters are located in `src/config.py`. Edit the dataclass defaults to adjust:

| Parameter            | Location      | Default      | Description                                      |
|----------------------|---------------|--------------|--------------------------------------------------|
| `C`                  | `ModelConfig` | `1.0`        | Regularization penalty term                      |
| `kernel`             | `ModelConfig` | `'rbf'`      | Kernel function (`'linear'`, `'poly'`, `'rbf'`, etc.)|
| `degree`             | `ModelConfig` | `3`          | Polynomial degree (ignored by RBF/linear)        |
| `gamma`              | `ModelConfig` | `'scale'`    | RBF/poly kernel coefficient                      |
| `probability`        | `ModelConfig` | `True`       | Enable probability estimations                   |
| `test_size`          | `DataConfig`  | `0.20`       | Fraction of data used for testing                |
| `scale_features`     | `DataConfig`  | `True`       | Enable StandardScaler on features                |
| `target_column`      | `DataConfig`  | `'species'`  | Target label column in dataset                   |
