# Supervised Machine Learning - Ordinary Least Squares (OLS) Linear Regression

> Supervised Machine Learning | Parametric Continuous Target Regression Algorithm

---

## Table of Contents

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Linear Regression Calculation (Step-by-Step)](#5-worked-linear-regression-calculation-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Linear Regression?

Ordinary Least Squares (OLS) Linear Regression is a fundamental **parametric supervised learning algorithm** that models the linear relationship between continuous input features $X$ and a continuous outcome target $y$ by minimizing the sum of squared residual errors.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Continuous Regression)                          |
| Model Formulation  | $y = \beta_0 + \sum_{j=1}^{D} \beta_j X_j + \varepsilon$             |
| Loss Function      | Mean Squared Error (MSE) / Residual Sum of Squares (RSS)             |
| Key Metrics        | MAE, MSE, RMSE, Coefficient of Determination ($R^2$)                |
| Key Hyperparameters| `fit_intercept`, `n_jobs`                                           |

---

## 2. Theoretical Explanation

### How Linear Regression Works

OLS Linear Regression calculates optimal weight parameters $\beta$ that minimize the sum of squared vertical distances between observed targets and the regression hyperplane.

```
Standardized Features ---> Compute Design Matrix X ---> Solve Closed-Form Normal Equation / Gradient Descent
                                                                                   |
                                                                                   v
                 Evaluate MAE, RMSE, R^2 <--- Compute Predictions & Residuals <--- Extract Coefficients beta
```

1. **Closed-Form Normal Equation**: Directly computes optimal coefficients $\beta = (X^T X)^{-1} X^T y$.
2. **Residual Diagnostics**: Evaluates residual errors ($y - \hat{y}$) for zero mean, homoscedasticity, and independence assumptions.

---

## 3. Mathematical Operations

### 1. Matrix Formulation & Closed-Form Solution

$$y = X \beta + \varepsilon$$

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

### 2. Simple Linear Regression Estimators

For single feature $x$:

$$\beta_1 = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

### 3. Evaluation Metrics

- **Mean Squared Error (MSE)**: $\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE)**: $\text{RMSE} = \sqrt{\text{MSE}}$
- **Mean Absolute Error (MAE)**: $\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$
- **Coefficient of Determination ($R^2$)**: $R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## 4. Real-World Example

### Walmart Weekly Sales Prediction (`Walmart_Sales.csv`)

- **Dataset**: `Walmart_Sales.csv` (Retail Sales Dataset)
- **Target Variable**: `Weekly_Sales` (Continuous Dollar Sales Target)
- **Features Used**: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `Holiday_Flag`, `Month`.

---

## 5. Worked Linear Regression Calculation (Step-by-Step)

Consider $N=3$ observations of advertising spend ($x$) vs sales ($y$):
- $P_1 = (1, 2)$
- $P_2 = (2, 3)$
- $P_3 = (3, 7)$

### Step 1: Calculate Means
$$\bar{x} = \frac{1+2+3}{3} = 2.0, \quad \bar{y} = \frac{2+3+7}{3} = 4.0$$

### Step 2: Compute Covariance and Variance Terms
- $(x_1-\bar{x})(y_1-\bar{y}) = (-1)(-2) = 2$
- $(x_2-\bar{x})(y_2-\bar{y}) = (0)(-1) = 0$
- $(x_3-\bar{x})(y_3-\bar{y}) = (1)(3) = 3$
$$\text{Numerator} = 2 + 0 + 3 = 5.0$$

- $(x_1-\bar{x})^2 = (-1)^2 = 1$
- $(x_2-\bar{x})^2 = (0)^2 = 0$
- $(x_3-\bar{x})^2 = (1)^2 = 1$
$$\text{Denominator} = 1 + 0 + 1 = 2.0$$

### Step 3: Compute Slope $\beta_1$ and Intercept $\beta_0$
$$\beta_1 = \frac{5.0}{2.0} = 2.5$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x} = 4.0 - (2.5)(2.0) = 4.0 - 5.0 = -1.0$$

Fitted Regression Line: $\hat{y} = -1.0 + 2.5 x$.

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
|  - DataConfig   (target_column='Weekly_Sales')      |
|  - ModelConfig  (fit_intercept=True, n_jobs=1)     |
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
|  - Load Walmart_Sales.csv, scale continuous features|
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: LinearRegressorService                     |
|  - Fit OLS Linear Regression model                  |
|  - Evaluate MAE, MSE, RMSE, R^2                     |
|  - Plot actual_vs_predicted.png & residuals_plot.png|
|  - Export regression_analysis.md report             |
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
  +-- config.py           (PipelineConfig, PathConfig, DataConfig, ModelConfig)
  +-- logger.py           (LoggerFactory - stdout stream logging)
  +-- data_loader.py      (DataLoaderService - loading, scaling, splitting)
  +-- linear_regressor.py (LinearRegressorService - OLS fit, residual plots, metrics)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter       | Location      | Default          | Description                                      |
|-----------------|---------------|------------------|--------------------------------------------------|
| `fit_intercept` | `ModelConfig` | `True`           | Whether to calculate intercept $\beta_0$         |
| `n_jobs`        | `ModelConfig` | `1`              | Number of CPU computation threads                |
| `test_size`     | `DataConfig`  | `0.20`           | Partition percentage reserved for test set       |
| `target_column` | `DataConfig`  | `'Weekly_Sales'` | Continuous outcome target column                 |
