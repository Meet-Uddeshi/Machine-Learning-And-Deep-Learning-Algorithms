# Regression - Ordinary Least Squares (OLS) Linear Regression

> Supervised Machine Learning | Regression Algorithm

---

## Table of Contents

1. [What is Regression?](#1-what-is-regression)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Linear Regression Sum (Step-by-Step)](#5-worked-linear-regression-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Regression?

Regression is a type of **supervised machine learning** where the goal is to predict a **continuous, numerical value** (e.g., price, temperature, sales) based on one or more input features.

The algorithm learns a mapping function from the **labelled training dataset** (where correct target values are known) and fits a line or hyperplane that best captures the trend of the data to estimate targets for **new, unseen data points**.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Regression)                                     |
| Output type        | Continuous numerical value (e.g., $1,500,000, 23.5°C)                 |
| Training data      | Labelled (input features + continuous target value)                  |
| Prediction         | Estimates a real number along a continuous scale                     |
| Evaluation metrics | MAE, MSE, RMSE, R-Squared ($R^2$)                                    |

### Simple vs. Multiple Regression

- **Simple Linear Regression** - uses exactly one feature to predict the target (e.g., predicting sales using temperature alone).
- **Multiple Linear Regression** - uses two or more features to predict the target (e.g., predicting sales using temperature, fuel price, CPI, and unemployment).

---

## 2. Theoretical Explanation

### How Ordinary Least Squares (OLS) Works

Linear Regression assumes a linear relationship between the input features ($X$) and the target variable ($y$). The core idea is:

> "Draw a straight line (or hyperplane) through the data points that minimizes the sum of the squared distances between the actual points and the line."

OLS is an analytical, non-iterative method (though it can be solved iteratively via Gradient Descent) that finds the optimal parameters by solving the system of equations directly.

### Step-by-Step Logic

```
Step 1: Ingest input features and target values.

Step 2: Parse and engineer time features (extract Month and Year from Date).

Step 3: Standardize the input features (StandardScaler) to prevent features with 
        larger scales from dominating coefficient determination.

Step 4: Train the model by computing coefficients (weights) and intercept (bias) 
        using OLS matrix operations (or singular value decomposition).

Step 5: Apply the fitted equation to predict values for the test set.

Step 6: Compute residuals (Actual - Predicted) for validation.

Step 7: Calculate regression metrics (MAE, MSE, RMSE, R2) and output diagnostic plots.
```

### Core Assumptions of Linear Regression

For Linear Regression to be reliable and mathematically sound, the data should satisfy:

- **Linearity**: The relationship between features and target must be linear.
- **Independence**: Residuals must be independent of each other (no autocorrelation).
- **Normality**: Residuals should be normally distributed around zero.
- **Homoscedasticity**: Residuals must have constant variance across all levels of the features (no funnel patterns).

---

## 3. Mathematical Operations

### The Linear Regression Equation

For a target $y$ with $n$ input features $x_1, x_2, \dots, x_n$:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

Where:
- $\hat{y}$ is the predicted value.
- $\beta_0$ is the intercept term (value of $y$ when all inputs are 0).
- $\beta_1, \beta_2, \dots, \beta_n$ are the coefficients (slopes) representing feature weights.

### Cost Function (Mean Squared Error)

To find the best-fitting line, OLS minimizes the sum of squared residuals. The cost function $J(\beta)$ is:

$$J(\beta) = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2$$

### Closed-Form Solution: The Normal Equation

The optimal parameters $\beta$ can be solved analytically using matrix calculus:

$$\beta = (X^T X)^{-1} X^T y$$

Where $X$ is the feature matrix (with a column of 1s added for the intercept term) and $y$ is the target vector.

### Feature Standardization

Standardizing features ensures coefficients represent standard deviation impacts:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean of the feature and $\sigma$ is its standard deviation.

### Evaluation Metrics

**Mean Absolute Error (MAE)** - Average absolute difference between actual and predicted:

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|$$

**Mean Squared Error (MSE)** - Average squared difference (penalizes outliers):

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

**Root Mean Squared Error (RMSE)** - Square root of MSE (in target units):

$$\text{RMSE} = \sqrt{\text{MSE}}$$

**Coefficient of Determination ($R^2$)** - Proportion of target variance explained:

$$R^2 = 1 - \frac{\sum (y^{(i)} - \hat{y}^{(i)})^2}{\sum (y^{(i)} - \bar{y})^2}$$

---

## 4. Real-World Example

### Walmart Weekly Sales Prediction

This project predicts the **Weekly Sales** of Walmart stores based on environmental, economic, and seasonal indicators.

**Dataset:** `Walmart_Sales.csv`

**Target Variable:** `Weekly_Sales` (continuous currency value)

**Features Used:**

| Feature       | Description                                  |
|---------------|----------------------------------------------|
| `Holiday_Flag`| Binary flag (1 if holiday week, 0 otherwise) |
| `Temperature` | Average temperature in the region            |
| `Fuel_Price`  | Cost of fuel in the region                   |
| `CPI`         | Consumer Price Index                         |
| `Unemployment`| Unemployment rate in the region              |
| `Month`       | Extracted month (1 to 12) from Date          |
| `Year`        | Extracted year from Date                     |

**Columns Excluded:**
- `Store` - Treated as a categorical identifier and dropped by default.
- `Date` - Raw string dropped after extracting `Month` and `Year`.

**Why Linear Regression is a good fit:**
- Macroeconomic conditions (CPI, fuel, unemployment) show linear trends with retail consumption.
- Coefficients directly explain which economic indicators positively or negatively influence sales volume.

---

## 5. Worked Linear Regression Sum (Step-by-Step)

Let's solve a simple linear regression problem ($y = \beta_0 + \beta_1 x$) using 3 data points.

### The Training Data

| Sample | Feature $X$ (Temperature) | Target $y$ (Weekly Sales in $10k) |
|--------|---------------------------|-----------------------------------|
| 1      | 2                         | 4                                 |
| 2      | 4                         | 5                                 |
| 3      | 6                         | 9                                 |

### Step 1: Calculate Means of $X$ and $y$

$$\bar{X} = \frac{2 + 4 + 6}{3} = 4.0$$

$$\bar{y} = \frac{4 + 5 + 9}{3} = 6.0$$

### Step 2: Calculate Slope Coefficient ($\beta_1$)

The slope formula is:

$$\beta_1 = \frac{\sum (X_i - \bar{X})(y_i - \bar{y})}{\sum (X_i - \bar{X})^2}$$

Let's compute the terms:

| $X_i$ | $y_i$ | $X_i - \bar{X}$ | $y_i - \bar{y}$ | $(X_i - \bar{X})(y_i - \bar{y})$ | $(X_i - \bar{X})^2$ |
|-------|-------|-----------------|-----------------|-----------------------------------|----------------------|
| 2     | 4     | -2              | -2              | 4                                 | 4                    |
| 4     | 5     | 0               | -1              | 0                                 | 0                    |
| 6     | 9     | 2               | 3               | 6                                 | 4                    |
| **Sum**|      |                 |                 | **10**                            | **8**                |

$$\beta_1 = \frac{10}{8} = 1.25$$

### Step 3: Calculate Intercept ($\beta_0$)

$$\beta_0 = \bar{y} - \beta_1 \bar{X}$$

$$\beta_0 = 6.0 - (1.25 \times 4.0) = 6.0 - 5.0 = 1.0$$

### Step 4: The Final Model Equation

$$y = 1.0 + 1.25 X$$

### Step 5: Predict for a New Point ($X = 5$)

$$\hat{y} = 1.0 + 1.25(5) = 1.0 + 6.25 = 7.25\text{ (which represents } \$72,500\text{)}$$

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
|  - DataConfig (target column, drop columns, split)  |
|  - ModelConfig (fit intercept, jobs)                |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler (linear_regression.log)             |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
+-----------------------------------------------------+
                          |
         +----------------+----------------+
         |                                 |
         v                                 v
+------------------+             +-------------------+
|  _load_csv()     |             |  File not found?  |
|  Read CSV from   | --Error-->  |  Raise            |
|  data/ folder    |             |  FileNotFoundError |
+------------------+             +-------------------+
         |
         v
+------------------+
|  _validate_schema|
|  Check target    |
|  Check drop cols |
+------------------+
         |
         v
+------------------+
|  _log_data_      |
|  summary()       |
|  Log target metrics|
+------------------+
         |
         v
+--------------------------------------------------+
|  _preprocess()                                   |
|  1. Parse Date and extract Month & Year features |
|  2. Drop target column, Store and raw Date       |
|  3. Impute nulls (median for numeric)            |
|  4. StandardScaler: scale features to            |
|     mean=0, std=1 (Target left unscaled)         |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|  _split()                                        |
|  Random Train/Test split                         |
|  - 80% Training data                             |
|  - 20% Test data                                 |
|  - Random state = 42                             |
+--------------------------------------------------+
         |
         v
+-----------------------------------------------------+
|  Step 4: LinearRegressorService.train()             |
|  - Build LinearRegression estimator                 |
|  - Call model.fit(X_train, y_train)                 |
|  - Log fit performance time                         |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 5: LinearRegressorService.evaluate()          |
+-----------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Generate predictions and compute metrics:       |
|  - MAE, MSE, RMSE, R-Squared                     |
|  - Log intercepts and feature weights table      |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Generate Plots:                                 |
|  - actual_vs_predicted.png                       |
|  - residuals_plot.png                            |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Save Outputs:                                   |
|  - Write regression_results.txt                  |
|  - Write regression_analysis.md                  |
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
  +-- config.py             (PipelineConfig, PathConfig, DataConfig,
  |                           ModelConfig, LoggingConfig)
  |
  +-- logger.py             (LoggerFactory - console & file)
  |
  +-- data_loader.py        (DataLoaderService - load, validate,
  |                           preprocess, split, feature engineering)
  |
  +-- linear_regressor.py   (LinearRegressorService - train, predict,
                              evaluate, plot, write analytical report)
```

---

## 8. Configuration

All tunable pipeline settings are centralized in `src/config.py`.

| Parameter        | Location      | Default            | Description                                  |
|------------------|---------------|--------------------|----------------------------------------------|
| `fit_intercept`  | `ModelConfig` | `True`             | Calculate intercept bias term                |
| `test_size`      | `DataConfig`  | `0.20`             | Split fraction reserved for validation       |
| `target_column`  | `DataConfig`  | `Weekly_Sales`     | Variable predicted                           |
| `drop_columns`   | `DataConfig`  | `['Store', 'Date']`| Columns removed prior to regression fitting  |
