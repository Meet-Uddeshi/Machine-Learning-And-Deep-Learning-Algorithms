# Supervised Machine Learning - Logistic Regression Classifier

> Supervised Machine Learning | Parametric Probabilistic Classification Algorithm

---

## Table of Contents

1. [What is Logistic Regression?](#1-what-is-logistic-regression)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Logistic Regression Calculation (Step-by-Step)](#5-worked-logistic-regression-calculation-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Logistic Regression?

Logistic Regression is a **parametric supervised learning classification algorithm**. Despite its name containing "Regression", it is used for binary and multiclass classification by passing a linear combination of input features through the non-linear Sigmoid (logistic) function to output posterior class probabilities $P(Y=1 | X) \in (0, 1)$.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Binary & Multiclass Classification)             |
| Output Type        | Calibrated Class Probabilities $P(Y=1 \mid X)$                       |
| Loss Function      | Binary Cross-Entropy / Log Loss                                      |
| Decision Boundary  | Linear Hyperplane ($w^T x + b = 0$)                                  |
| Key Hyperparameters| `C` (Inverse Regularization Strength), `penalty` (`'l1'`, `'l2'`)   |

---

## 2. Theoretical Explanation

### How Logistic Regression Works

Logistic Regression models log-odds (logit) as a linear function of input features.

```
Standardized Features ---> Compute Logit z = w^T x + b ---> Map via Sigmoid sigma(z) = 1 / (1 + e^-z)
                                                                                  |
                                                                                  v
           Class Output (y=1 if P>=0.5 else 0) <--- Evaluate Log Loss & Gradient Optimization
```

1. **Logit Score**: Calculate linear score $z = w^T x + b$.
2. **Sigmoid Activation**: Map real-valued score $z$ into probability bounds $(0, 1)$ via $\sigma(z) = \frac{1}{1 + e^{-z}}$.
3. **Log Loss Optimization**: Minimize negative log-likelihood via iterative optimization solvers (e.g. L-BFGS, SGD).

---

## 3. Mathematical Operations

### 1. Sigmoid Function & Probability Model

$$P(Y=1 | X) = \sigma(z) = \frac{1}{1 + e^{-(w^T X + b)}}$$

### 2. Log-Odds (Logit Transformation)

$$\text{logit}(P) = \ln\left( \frac{P}{1 - P} \right) = w^T X + b$$

### 3. Log Loss (Binary Cross-Entropy Loss)

$$L(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

---

## 4. Real-World Example

### Customer Churn Classification (`churn.csv`)

- **Dataset**: `churn.csv` (Telecom Customer Churn Dataset)
- **Target Variable**: `Churn` (`0` = Retained, `1` = Churned)
- **Features Used**: Standardized account metrics (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`).

---

## 5. Worked Logistic Regression Calculation (Step-by-Step)

Consider a single feature $x=2.0$ with trained weight $w=1.5$ and intercept $b=-1.0$.

### Step 1: Calculate Logit Score ($z$)
$$z = w \cdot x + b = (1.5)(2.0) + (-1.0) = 3.0 - 1.0 = 2.0$$

### Step 2: Compute Sigmoid Probability $\sigma(z)$
$$P(Y=1 | x=2.0) = \frac{1}{1 + e^{-2.0}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} \approx 0.8808$$

### Step 3: Decision Rule Threshold ($\tau = 0.5$)
Since $P(Y=1 | x) = 0.8808 \ge 0.5$, predicted label is **Class 1** (High probability of churn!).

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
|  - DataConfig   (target_column='Churn', test_size)  |
|  - ModelConfig  (C=1.0, penalty='l2', solver='lbfgs')|
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
|  - Load churn.csv, scale features with StandardScaler|
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: LogisticRegressorService                   |
|  - Fit Logistic Regression model                    |
|  - Evaluate Accuracy, Precision, Recall, F1, ROC-AUC|
|  - Plot confusion_matrix.png & roc_curve.png        |
|  - Export logistic_regression_analysis.md report    |
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
  +-- config.py             (PipelineConfig, PathConfig, DataConfig, ModelConfig)
  +-- logger.py             (LoggerFactory - stdout stream logging)
  +-- data_loader.py        (DataLoaderService - loading, scaling, splitting)
  +-- logistic_regressor.py (LogisticRegressorService - fit, ROC plot, metrics)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter     | Location      | Default    | Description                                      |
|---------------|---------------|------------|--------------------------------------------------|
| `C`           | `ModelConfig` | `1.0`      | Inverse regularization strength ($C > 0$)        |
| `penalty`     | `ModelConfig` | `'l2'`     | Norm used in penalization (`'l1'`, `'l2'`)       |
| `solver`      | `ModelConfig` | `'lbfgs'`  | Optimization solver algorithm                    |
| `test_size`   | `DataConfig`  | `0.20`     | Partition percentage reserved for test evaluation|
| `target_column`| `DataConfig` | `'Churn'`  | Target binary classification outcome variable    |
