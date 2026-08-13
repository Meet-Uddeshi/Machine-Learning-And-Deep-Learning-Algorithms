# Classification - Logistic Regression

> Supervised Machine Learning | Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Logistic Regression Sum (Step-by-Step)](#5-worked-logistic-regression-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where the goal is to assign an input data point to one of a fixed set of **discrete categories (classes)**.

Despite the word "regression" in its name, **Logistic Regression** is used primarily for classification tasks, most commonly **binary classification** (predicting one of two possible classes, such as "Disruption" vs. "No Disruption").

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning (Classification)                                 |
| Output type        | Probability score mapped to a discrete class label (e.g., 0 or 1)    |
| Training data      | Labelled (input features + discrete target category)                  |
| Prediction         | Estimates the probability of a data point belonging to the default class |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix     |

---

## 2. Theoretical Explanation

### How Logistic Regression Works

Logistic Regression model maps a linear combination of features to a value between 0 and 1 using the **Sigmoid (logistic) function**. The output is interpreted as the probability of the target event occurring.

> "A data point is classified as class 1 if its predicted probability of belonging to class 1 is greater than or equal to a threshold (typically 0.5)."

Unlike K-NN, which is instance-based and relies on local distance metrics, Logistic Regression is a **parametric model** that learns global weights (coefficients) for each feature. This results in a linear decision boundary.

### Step-by-Step Logic

```
Step 1: Ingest input features and target labels (0 or 1).

Step 2: Parse raw date strings and extract numerical features (Month, Year).

Step 3: Impute missing values with medians (numeric) or modes (categorical).

Step 4: Encode categorical columns (e.g., ports, transport modes) to numeric categories.

Step 5: Standardize features (StandardScaler) to stabilize the optimization solver.

Step 6: Train the model by minimizing binary cross-entropy loss (Log-Loss) using 
        optimization algorithms (like L-BFGS) to find optimal coefficients.

Step 7: Predict probability scores on the test set, and threshold them to assign class labels.

Step 8: Calculate classification metrics (Accuracy, Precision, Recall, F1, AUC) 
        and output heatmaps/ROC curves.
```

### Regularization ($L_1$ and $L_2$)

To prevent overfitting, Logistic Regression applies regularization:
- **$L_2$ Regularization (Ridge)**: Adds a penalty proportional to the *square* of the coefficients. It shrinks coefficients uniformly (default in this project).
- **$L_1$ Regularization (Lasso)**: Adds a penalty proportional to the *absolute value* of the coefficients. It can shrink coefficients to exactly zero, performing feature selection.

---

## 3. Mathematical Operations

### The Sigmoid Function

The sigmoid function $\sigma(z)$ maps any real value $z$ to a probability in the range $[0, 1]$:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of inputs:

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

### Log-Odds and Logit Transformation

We can express the relationship in terms of **odds** (probability of event occurring divided by probability of it not occurring):

$$\text{Odds} = \frac{p}{1 - p} = e^{\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n}$$

Taking the natural logarithm yields the **logit** transformation (log-odds), which is linear:

$$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

### Cost Function (Binary Cross-Entropy / Log-Loss)

The model parameters $\beta$ are learned by minimizing Log-Loss, which penalizes confident incorrect predictions:

$$J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{p}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{p}^{(i)}) \right]$$

### Evaluation Metrics

**Accuracy** - Fraction of correct predictions:

$$\text{Accuracy} = \frac{\text{True Positives (TP)} + \text{True Negatives (TN)}}{\text{Total Samples}}$$

**Precision** - Proportion of predicted positives that are actually positive:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{False Positives (FP)}}$$

**Recall (Sensitivity)** - Proportion of actual positives correctly identified:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{False Negatives (FN)}}$$

**F1-Score** - Balanced measure combining precision and recall:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**ROC AUC** - Area Under the Receiver Operating Characteristic curve; plots True Positive Rate (Recall) vs False Positive Rate ($FP / (TN + FP)$) across all thresholds.

---

## 4. Real-World Example

### Supply Chain Disruption Prediction

This project predicts **Disruption_Occurred** (a supply chain delay or cancellation) based on shipment properties, route details, and environmental risk factors.

**Dataset:** `global_supply_chain_risk_2026.csv`

**Target Variable:** `Disruption_Occurred`  
Possible classes: `0` (No Disruption), `1` (Disruption occurred)

**Features Used:**

| Feature                    | Description                                         |
|----------------------------|-----------------------------------------------------|
| `Origin_Port`              | Port from which the shipment departs (Encoded)       |
| `Destination_Port`         | Port where the shipment arrives (Encoded)            |
| `Transport_Mode`           | Mode of transport: Air, Rail, Sea, Road (Encoded)    |
| `Product_Category`         | Category of products being shipped (Encoded)        |
| `Distance_km`              | Total distance of the shipment route in kilometers   |
| `Weight_MT`                | Cargo weight in Metric Tons                         |
| `Fuel_Price_Index`         | Indexed fuel cost during transit                    |
| `Geopolitical_Risk_Score`  | Safety rating of shipping lanes (0 to 10 scale)      |
| `Weather_Condition`        | Weather during transit: Clear, Storm, etc. (Encoded)|
| `Carrier_Reliability_Score`| Reliability rating of the logistics carrier (0 to 1) |
| `Month`                    | Month extracted from Date string                    |
| `Year`                     | Year extracted from Date string                     |

**Columns Excluded:**
- `Shipment_ID` - Unique transaction identifier (no predictive value).
- `Date` - Raw string dropped after feature engineering.

**Why Logistic Regression is a good fit:**
- Direct probability output helps supply chain managers assess the percentage risk of disruption before booking a shipment.
- Exponentiating coefficients yields the **Odds Ratio ($e^\beta$)**, showing exactly how risk increases with distance or geopolitical tension.

---

## 5. Worked Logistic Regression Sum (Step-by-Step)

Let's trace how the model calculates the probability of a disruption for a single shipment.

### The Fitted Parameters (Learned during training)

Suppose our model learned the following weights for a single feature $X_1$ (standardized Geopolitical Risk Score):
- Intercept ($\beta_0$) = $-1.5$
- Coefficient ($\beta_1$) = $+0.5$

### The New Shipment

Let's evaluate a shipment that has a standardized Geopolitical Risk Score of **$X_1 = 3.0$**.

### Step 1: Calculate the Linear Combination ($z$)

$$z = \beta_0 + \beta_1 X_1$$

$$z = -1.5 + (0.5 \times 3.0) = -1.5 + 1.5 = 0.0$$

### Step 2: Apply the Sigmoid Function to Find Probability ($p$)

$$p = \frac{1}{1 + e^{-z}}$$

$$p = \frac{1}{1 + e^{-0.0}} = \frac{1}{1 + 1.0} = 0.50\text{ (or } 50\%\text{ probability)}$$

### Step 3: Threshold and Classify

Using a standard classification threshold of $0.5$:
- If $p \ge 0.5$, predict Class 1 (Disruption).
- If $p < 0.5$, predict Class 0 (No Disruption).

Since $p = 0.50$, the predicted class is **1 (Disruption occurred)**.

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
|  - ModelConfig (penalty=l2, C=1.0, solver=lbfgs)    |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler (logistic_regression.log)           |
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
|  Log target class|
|  distribution    |
+------------------+
         |
         v
+--------------------------------------------------+
|  _preprocess()                                   |
|  1. Parse Date and extract Month & Year features |
|  2. Drop Shipment_ID and raw Date                |
|  3. Impute nulls (median for numeric, mode for   |
|     categorical features)                        |
|  4. LabelEncode categorical feature columns      |
|  5. LabelEncode binary target variable           |
|  6. StandardScaler: scale all features to        |
|     mean=0, std=1                                |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|  _split()                                        |
|  Stratified Train/Test split                     |
|  - 80% Training data                             |
|  - 20% Test data                                 |
|  - Random state = 42                             |
+--------------------------------------------------+
         |
         v
+-----------------------------------------------------+
|  Step 4: LogisticRegressorService.train()           |
|  - Build LogisticRegression classifier              |
|  - Call model.fit(X_train, y_train)                 |
|  - Log fit performance time                         |
+-----------------------------------------------------+
                          |
                          v
+-----------------------------------------------------+
|  Step 5: LogisticRegressorService.evaluate()        |
+-----------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Generate predictions and compute metrics:       |
|  - Accuracy, Precision, Recall, F1, ROC AUC      |
|  - Log Log-Odds and Odds Ratios weights table    |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Generate Plots:                                 |
|  - confusion_matrix.png (Purples heatmap)        |
|  - roc_curve.png (Receiver Operating Curve)      |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|  Save Outputs:                                   |
|  - Write classification_results.txt              |
|  - Write classification_analysis.md              |
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
  |                           preprocess, split, feature engineering, encoding)
  |
  +-- logistic_regressor.py (LogisticRegressorService - train, predict,
                              evaluate, plot, write analytical report)
```

---

## 8. Configuration

All tunable pipeline settings are centralized in `src/config.py`.

| Parameter       | Location      | Default                     | Description                                   |
|-----------------|---------------|-----------------------------|-----------------------------------------------|
| `penalty`       | `ModelConfig` | `'l2'`                      | Regularization norm selection                 |
| `C`             | `ModelConfig` | `1.0`                       | Inverse regularization strength              |
| `solver`        | `ModelConfig` | `'lbfgs'`                   | Optimization solver selection                 |
| `max_iter`      | `ModelConfig` | `1000`                      | Solver iteration limit                        |
| `test_size`     | `DataConfig`  | `0.20`                      | Split fraction reserved for validation        |
| `target_column` | `DataConfig`  | `Disruption_Occurred`       | Variable predicted                            |
| `drop_columns`  | `DataConfig`  | `['Shipment_ID', 'Date']`   | Columns removed prior to regression fitting   |
