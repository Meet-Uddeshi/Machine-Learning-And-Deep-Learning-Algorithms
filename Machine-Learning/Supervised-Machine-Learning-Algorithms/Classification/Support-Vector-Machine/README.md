# Supervised Machine Learning - Support Vector Machine (SVM) Classifier

> Supervised Machine Learning | Maximum Margin Separating Hyperplane & Kernel Classification Algorithm

---

## Table of Contents

1. [What is Support Vector Machine?](#1-what-is-support-vector-machine)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked SVM Calculation (Step-by-Step)](#5-worked-svm-calculation-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Support Vector Machine?

Support Vector Machine (SVM) is a robust **supervised learning algorithm** used for linear and non-linear classification and regression. SVM constructs an optimal high-dimensional separating hyperplane that maximizes the margin (distance) between the decision boundary and the closest training points of any class, known as **support vectors**.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Classification & Regression)                    |
| Decision Boundary  | Optimal Maximum-Margin Hyperplane                                    |
| Support Vectors    | Critical data points defining the boundary margins                   |
| Kernel Trick       | Linear, Polynomial, RBF (Radial Basis Function), Sigmoid kernels     |
| Key Hyperparameters| `C` (Regularization), `kernel`, `gamma`, `degree`                    |

---

## 2. Theoretical Explanation

### How SVM Works

SVM seeks the optimal linear hyperplane $w^T x + b = 0$ that maximizes geometric margin $\frac{2}{\|w\|}$.

```
Standardized Features ---> Compute Kernel Matrix K(x, y) ---> Solve Convex Optimization (Quadratic Programming)
                                                                                  |
                                                                                  v
                 Class Prediction <--- Sum Support Vector Coefficients <--- Extract Support Vectors & Weights
```

1. **Margin Maximization**: Compute optimal weight vector $w$ and bias $b$ to maximize margin distance.
2. **Soft Margin Penalty ($C$)**: Balance margin maximization against misclassification errors via slack variables $\xi_i$.
3. **Kernel Trick**: Map non-linearly separable inputs into higher-dimensional feature space $\Phi(x)$ where linear separation is possible using kernel functions $K(x, y) = \langle \Phi(x), \Phi(y) \rangle$.

---

## 3. Mathematical Operations

### 1. Primal Optimization Problem (Soft-Margin SVM)

$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \xi_i$$

subject to constraints:
$$y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i$$

### 2. Dual Formulation with Kernel Trick

$$\max_{\alpha} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

subject to $\sum_{i=1}^{N} \alpha_i y_i = 0$ and $0 \le \alpha_i \le C$.

### 3. Common Kernel Functions

- **Linear Kernel**: $K(x, y) = x^T y$
- **Polynomial Kernel**: $K(x, y) = (x^T y + c)^d$
- **Radial Basis Function (RBF) Kernel**:
  $$K(x, y) = \exp\left( -\gamma \|x - y\|^2 \right)$$

---

## 4. Real-World Example

### Heart Disease Medical Classification (`heart.csv`)

- **Dataset**: `heart.csv` (Heart Disease Dataset)
- **Target Variable**: `target` (`0` = Healthy, `1` = Heart Disease)
- **Features Used**: Standardized clinical metrics (`age`, `chol`, `thalach`, `oldpeak`, etc.).

---

## 5. Worked SVM Calculation (Step-by-Step)

Consider a toy 1D dataset with $N=3$ samples and binary target $y \in \{-1, +1\}$:
- $x_1 = 1 \implies y_1 = -1$
- $x_2 = 2 \implies y_2 = -1$
- $x_3 = 4 \implies y_3 = +1$

Find linear decision boundary $w \cdot x + b = 0$.

### Step 1: Identify Support Vectors
Support vectors are the boundary points closest to the split: $x_2 = 2$ ($y_2 = -1$) and $x_3 = 4$ ($y_3 = +1$).

### Step 2: Set Boundary Equations
- For $x_3 = 4$: $w(4) + b = +1$
- For $x_2 = 2$: $w(2) + b = -1$

### Step 3: Solve Linear System
Subtracting equations:
$$w(4 - 2) = 1 - (-1) \implies 2w = 2 \implies w = 1.0$$

Substitute $w = 1.0$:
$$1(2) + b = -1 \implies b = -3.0$$

Decision Boundary: $1.0 x - 3.0 = 0 \implies x = 3.0$.
Margin Width = $\frac{2}{\|w\|} = \frac{2}{1} = 2.0$.

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
|  - ModelConfig  (kernel='rbf', C=1.0, gamma='scale')|
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
|  - Load heart.csv, scale features with StandardScaler|
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: SVMClassifierService                       |
|  - Fit SVC model                                    |
|  - Evaluate Accuracy, Precision, Recall, F1, ROC-AUC|
|  - Plot confusion matrix & decision boundary        |
|  - Export svm_analysis.md report                    |
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
  +-- svm_classifier.py (SVMClassifierService - fit, decision boundary, metrics)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter     | Location      | Default    | Description                                      |
|---------------|---------------|------------|--------------------------------------------------|
| `C`           | `ModelConfig` | `1.0`      | Regularization parameter ($C > 0$)               |
| `kernel`      | `ModelConfig` | `'rbf'`    | Kernel type (`'linear'`, `'rbf'`, `'poly'`)      |
| `gamma`       | `ModelConfig` | `'scale'`  | Kernel coefficient for RBF/Poly                  |
| `test_size`   | `DataConfig`  | `0.20`     | Partition percentage reserved for test evaluation|
| `target_column`| `DataConfig` | `'target'` | Target classification outcome variable           |
