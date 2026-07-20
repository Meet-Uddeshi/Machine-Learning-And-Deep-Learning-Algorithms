# Ensemble Methods - Boosting

> Supervised Machine Learning | Ensemble Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Boosting Sum (Step-by-Step)](#5-worked-boosting-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Classification?

Classification is a subfield of **supervised machine learning** where the objective is to predict discrete, categorical target labels for given input data points. The model learns decision boundaries from a **labelled training dataset** and generalizes this capability to classify new, unseen observations.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning                                                  |
| Output type        | Discrete class label (e.g., Diabetic [1] vs Non-Diabetic [0])        |
| Training data      | Labelled (input features mapped to known correct class labels)       |
| Prediction         | Assigns class membership probabilities or hard class labels         |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Stage Loss  |

---

## 2. Theoretical Explanation

### How Boosting Works

Unlike Bagging (e.g., Random Forest) where multiple classifiers are trained in parallel, **Boosting** is a sequential ensemble technique. It combines a series of weak learners (typically shallow decision trees or decision stumps) to form a strong classifier. The core intuition is:

> "Train base models sequentially, where each subsequent model focuses heavily on correcting the classification errors made by the cumulative ensemble preceding it."

Boosting is an **eager learning** ensemble algorithm. Instead of using a single complex model that might overfit, it builds $M$ successive estimators, updates data weights or target residuals at each stage, and computes a weighted consensus for predictions.

```
Initial Weights / Residuals ---> Train Weak Learner ---> Compute Stage Weight / Error
                                                                    |
                                                                    v
                  Compute Final Prediction <--- Update Sample Weights / Residuals
```

### Boosting Paradigms

This project supports the two major boosting implementations:

#### 1. Adaptive Boosting (AdaBoost)
- **Error Correction**: Updates sample weights. Data points misclassified in stage $m-1$ receive higher weights in stage $m$, forcing the next base estimator to focus on hard samples.
- **Aggregation**: Base learners are weighted based on their individual training error rates. A highly accurate learner gets a larger say in the final vote.

#### 2. Gradient Boosting Machine (GBM)
- **Error Correction**: Optimises a loss function (e.g. log-loss for classification) by fitting each new base learner to the negative gradient of the loss function (pseudo-residuals) with respect to the current ensemble's predictions.
- **Aggregation**: Adds base learners sequentially with a constant learning rate (shrinkage factor) $\nu$ to prevent overfitting.

### Invariance to Feature Scale
Because the base learners in the boosting ensemble are regression/decision trees, splits are based on ordinal inequalities (e.g., $x_i \le \theta$). This makes the feature evaluations scale-invariant. Consequently, feature scaling (like StandardScaler or MinMaxScaler) is mathematically redundant and is disabled by default.

---

## 3. Mathematical Operations

### AdaBoost Formulations (Binary Classification)

Given a dataset $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$ where $y_i \in \{-1, 1\}$, and initial weights $w_i^{(1)} = \frac{1}{N}$:

1. **Base Learner Error Rate ($e_m$)**:
   For the $m$-th weak learner $h_m(x)$:
   $$e_m = \sum_{i=1}^N w_i^{(m)} I(y_i \neq h_m(x_i))$$

2. **Estimator Voting Weight ($\alpha_m$)**:
   $$\alpha_m = \frac{1}{2} \ln\left(\frac{1 - e_m}{e_m}\right)$$

3. **Sample Weight Update**:
   $$w_i^{(m+1)} = \frac{w_i^{(m)} \exp(-\alpha_m y_i h_m(x_i))}{Z_m}$$
   where $Z_m$ is a normalization constant ensuring $\sum_{i=1}^N w_i^{(m+1)} = 1$.

4. **Ensemble Consensus**:
   $$H(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m h_m(x)\right)$$

---

### Gradient Boosting Formulations (Binary Classification)

Given $y_i \in \{0, 1\}$ and a differentiable loss function $L(y, F(x)) = - [y \ln(p) + (1-y)\ln(1-p)]$ where $p = \frac{1}{1 + e^{-F(x)}}$:

1. **Initialisation**:
   Initialize $F_0(x)$ with a constant value:
   $$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \gamma) = \ln\left(\frac{\sum y_i}{N - \sum y_i}\right) \quad (\text{log-odds of positive class})$$

2. **Sequential Iteration (for $m = 1 \dots M$)**:
   - Compute pseudo-residuals (negative gradient of loss):
     $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)} = y_i - p_i \quad \text{where } p_i = \frac{1}{1 + e^{-F_{m-1}(x_i)}}$$
   - Fit a regression tree base learner $h_m(x)$ to predict pseudo-residuals $r_{im}$, yielding terminal regions $R_{jm}$ for leaf $j$.
   - Compute optimal leaf values $\gamma_{jm}$:
     $$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma) \approx \frac{\sum_{x_i \in R_{jm}} r_{im}}{\sum_{x_i \in R_{jm}} p_i(1 - p_i)}$$
   - Update ensemble predictions with learning rate $\nu$:
     $$F_m(x) = F_{m-1}(x) + \nu \sum_{j} \gamma_{jm} I(x \in R_{jm})$$

3. **Final Class Output**:
   $$p(x) = \frac{1}{1 + e^{-F_M(x)}}, \quad \hat{y}(x) = I(p(x) \ge 0.5)$$

---

## 4. Real-World Example

### Diabetes Onset Prediction
The pipeline applies Boosting algorithms to predict whether a patient has diabetes based on clinical and physiological measurements.

- **Dataset**: `diabetes.csv` (Pima Indians Diabetes Dataset)
- **Target Variable**: `Outcome` (0: Non-Diabetic, 1: Diabetic)
- **Features Used**:

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg/(height in m)^2) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (genetic risk score) |
| `Age` | Age in years |

---

## 5. Worked Boosting Sum (Step-by-Step)

Let us trace a manual step of **AdaBoost** on a toy dataset of $N=4$ samples to understand how weights adapt.

### Initial Setup
Suppose our training set is:
- $x_1$: `Glucose` = 150 (Label $y_1 = 1$)
- $x_2$: `Glucose` = 130 (Label $y_2 = 1$)
- $x_3$: `Glucose` = 90 (Label $y_3 = -1$)
- $x_4$: `Glucose` = 80 (Label $y_4 = -1$)

Initial sample weights are $w_i^{(1)} = 0.25$ for all $i$.

### Iteration 1 ($m=1$)
1. **Train Base Classifier ($h_1$)**:
   Suppose the optimal decision stump splits at `Glucose` > 100:
   - For $x_1, x_2$: predict $1$.
   - For $x_3, x_4$: predict $-1$.
   All predictions are correct! The error rate $e_1 = 0$.
   
   To show weight updates, let's assume a slightly sub-optimal stump splits at `Glucose` > 140:
   - For $x_1$: predicts $1$ (Correct)
   - For $x_2$: predicts $-1$ (Incorrect!)
   - For $x_3$: predicts $-1$ (Correct)
   - For $x_4$: predicts $-1$ (Correct)

2. **Compute Error Rate ($e_1$)**:
   $$e_1 = \sum w_i^{(1)} I(y_i \neq h_1(x_i)) = w_2^{(1)} = 0.25$$

3. **Compute Estimator Weight ($\alpha_1$)**:
   $$\alpha_1 = \frac{1}{2} \ln\left(\frac{1 - 0.25}{0.25}\right) = \frac{1}{2} \ln(3) \approx 0.5493$$

4. **Update Sample Weights**:
   For correct classifications ($x_1, x_3, x_4$):
   $$w_i^{(2)\text{ unnormalized}} = 0.25 \times e^{-0.5493} \approx 0.25 \times 0.5774 \approx 0.1443$$
   For incorrect classification ($x_2$):
   $$w_2^{(2)\text{ unnormalized}} = 0.25 \times e^{0.5493} \approx 0.25 \times 1.7320 \approx 0.4330$$

5. **Normalize Weights**:
   Sum of unnormalized weights: $Z_1 = 0.1443 \times 3 + 0.4330 = 0.8659$.
   - $w_1^{(2)} = \frac{0.1443}{0.8659} \approx 0.1666$
   - $w_2^{(2)} = \frac{0.4330}{0.8659} \approx 0.5000$ (Weight doubled due to error!)
   - $w_3^{(2)} = w_4^{(2)} \approx 0.1666$

In the next iteration, the weak learner will prioritize fitting $x_2$ correctly since it constitutes $50\%$ of the total sample weight.

---

## 6. Program Flowchart

The flowchart below describes the execution sequence of the pipeline.

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig   (dataset and output paths)          |
|  - DataConfig   (Outcome, test size, splits)        |
|  - ModelConfig  (boosting type, depth, estimators)  |
|  - LoggingConfig (console-only logs)                |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 2: Initialize Console Logger                  |
|  - Output formatted to sys.stdout (No local files)  |
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
|  Check Outcome   |
|  column presence |
+------------------+
           |
           v
+------------------+
|  _log_data_      |
|  summary()       |
|  Exploratory stats|
+------------------+
           |
           v
+--------------------------------------------------+
|  _preprocess()                                   |
|  - Separate target ('Outcome')                   |
|  - Fill null values with median/mode             |
|  - Encode target class to standard categories    |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  _split()                                        |
|  - Partition: 80% Train, 20% Test (stratified)   |
|  - Scale features is False                       |
+--------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: BoostingClassifierService.train()          |
+-----------------------------------------------------+
                           |
           +---------------+-------------------+
           |                                   |
           v                                   v
+--------------------------------+   +--------------------------------+
|  If gradient_boosting:         |   |  If adaboost:                  |
|  Fit Gradient Boosting         |   |  Fit AdaBoost with Decision    |
|  Classifier on train set       |   |  Trees as base estimators      |
+--------------------------------+   +--------------------------------+
           |                                   |
           +-----------------+-----------------+
                             |
                             v
+-----------------------------------------------------+
|  Step 5: BoostingClassifierService.evaluate()       |
+-----------------------------------------------------+
                             |
                             v
+--------------------------------------------------+
|  Evaluate Ensemble Performance                   |
|  - Compute staging error rates/deviance logs     |
|  - Calculate Accuracy, Precision, Recall, F1      |
+--------------------------------------------------+
                             |
                             v
+--------------------------------------------------+
|  Save Output Artifacts to output/                |
|  - Write classification_results.txt              |
|  - Write classification_analysis.md report       |
|  - Generate confusion_matrix.png                 |
|  - Generate classification_report.png            |
|  - Generate feature_importance.png               |
|  - Generate boosting_stage_loss.png              |
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
  +-- config.py              (PipelineConfig, PathConfig, DataConfig,
  |                           ModelConfig, LoggingConfig)
  |
  +-- logger.py              (LoggerFactory - console stream log setup)
  |
  +-- data_loader.py         (DataLoaderService - data ingestion, validation, splits)
  |
  +-- boosting_classifier.py (BoostingClassifierService - train, evaluate,
                              plot generation, text/markdown reports)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter            | Location      | Default      | Description                                      |
|----------------------|---------------|--------------|--------------------------------------------------|
| `boosting_type`      | `ModelConfig` | `'gradient_boosting'` | Boosting algorithm style (`'gradient_boosting'` or `'adaboost'`) |
| `n_estimators`       | `ModelConfig` | `100`        | Number of iterations / weak learners in the model |
| `learning_rate`      | `ModelConfig` | `0.1`        | Shrinkage rate for contribution of base trees    |
| `max_depth`          | `ModelConfig` | `3`          | Maximum depth of base tree estimators            |
| `min_samples_split`  | `ModelConfig` | `2`          | Min samples required to split an internal node   |
| `min_samples_leaf`   | `ModelConfig` | `1`          | Min samples required to form a leaf node         |
| `subsample`          | `ModelConfig` | `1.0`        | Subsampling ratio for stochastic gradient steps  |
| `test_size`          | `DataConfig`  | `0.20`       | Partition percentage reserved for test data      |
| `target_column`      | `DataConfig`  | `'Outcome'`  | Name of target outcome class column              |
