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

The algorithm learns from a **labelled training dataset** where the ground-truth class is known. After training, it applies the learned decision boundary to predict class labels for new, unseen data points.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning                                                  |
| Output type        | Discrete class label (e.g., "Malignant", "Benign")                   |
| Training data      | Labelled (input features + ground-truth class)                       |
| Prediction         | Assigns class label based on maximum margin separation               |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix              |

---

## 2. Theoretical Explanation

### How Support Vector Machines (SVM) Work

Support Vector Machine (SVM) is a powerful supervised algorithm used for classification and regression. The fundamental principle of SVM is:

> "Find the optimal decision boundary (hyperplane) that maximizes the margin between different classes in the feature space."

### Key Concepts

1. **Hyperplane**: The decision boundary that separates classes. In an N-dimensional space, a hyperplane is an (N-1)-dimensional subspace.
2. **Support Vectors**: The training data points closest to the hyperplane. These points dictate the orientation and position of the hyperplane.
3. **Margin**: The distance between the hyperplane and the nearest data point of any class. SVM maximizes this margin.
4. **Soft Margin & C Parameter**: Real-world data is rarely linearly separable. The hyperparameter C controls the trade-off between maximizing the margin and minimizing classification errors.
5. **Kernel Trick**: Transforms low-dimensional non-linearly separable input data into a higher-dimensional space where a linear hyperplane can separate the classes.

---

## 3. Mathematical Operations

### Hyperplane Equation

A decision boundary in N-dimensional space is defined as:

```
w^T * x + b = 0
```

Where:
- `w` is the weight vector perpendicular to the hyperplane.
- `x` is the feature vector.
- `b` is the bias scalar offset.

### Functional & Geometric Margin

For a training sample `(x_i, y_i)` where `y_i in {-1, +1}`:

```
Functional Margin  = y_i * (w^T * x_i + b)
Geometric Margin   = y_i * (w^T * x_i + b) / ||w||
```

### Optimization Objective (Soft Margin SVM)

```
minimize:  (1/2) * ||w||^2 + C * sum(xi_i)
subject to: y_i * (w^T * x_i + b) >= 1 - xi_i,  xi_i >= 0
```

Where `xi_i` are slack variables accounting for misclassified points or points inside the margin.

### Common Kernel Functions

1. **Linear**: `K(x, z) = x^T * z`
2. **Polynomial**: `K(x, z) = (gamma * x^T * z + coef0)^degree`
3. **Radial Basis Function (RBF)**: `K(x, z) = exp(-gamma * ||x - z||^2)`
4. **Sigmoid**: `K(x, z) = tanh(gamma * x^T * z + coef0)`

---

## 4. Real-World Example

### Breast Cancer Wisconsin Diagnosis

This project predicts whether a breast mass tumor sample is **Malignant (M)** or **Benign (B)** based on cell nucleus characteristics.

**Dataset:** `breast_cancer_wisconsin.csv`  
**Target Variable:** `diagnosis` (`M` = Malignant, `B` = Benign)  
**Features Used:** Nucleus radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

**Excluded Column:** `id` (non-predictive identifier).

---

## 5. Worked SVM Sum (Step-by-Step)

Consider a 2D dataset with two classes (`y = +1` and `y = -1`).

### Training Points

- `P1 (2, 2)` with label `y_1 = +1`
- `P2 (4, 4)` with label `y_2 = +1`
- `P3 (1, 3)` with label `y_3 = -1`
- `P4 (3, 5)` with label `y_4 = -1`

Assume the maximum margin line equation is `w1*x1 + w2*x2 + b = 0` with `w = [-1, 1]` and `b = -1`.

### Step 1: Evaluate Line Equation for Support Vector

For point `P1 (2, 2)`:
```
w^T * x + b = (-1)*(2) + (1)*(2) - 1 = -1
y_1 * (w^T * x + b) = (+1) * (-1) = -1  (on margin boundary)
```

For point `P4 (3, 5)`:
```
w^T * x + b = (-1)*(3) + (1)*(5) - 1 = +1
y_4 * (w^T * x + b) = (-1) * (+1) = -1  (on margin boundary)
```

### Step 2: Compute Margin Width

```
Margin = 2 / ||w||
||w||  = sqrt((-1)^2 + (1)^2) = sqrt(2)
Margin = 2 / sqrt(2) = sqrt(2) approx 1.414
```

### Step 3: Classify Unseen Query Point Q(4, 2)

```
w^T * Q + b = (-1)*(4) + (1)*(2) - 1 = -3
Sign(w^T * Q + b) = Sign(-3) = -1
Prediction = Class -1
```

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
|  - DataConfig (target='diagnosis', drop=['id'])      |
|  - ModelConfig (kernel='rbf', C=1.0, gamma='scale') |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler (svm_classification.log)            |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
|  - Load CSV -> Validate Schema                      |
|  - Log exploratory data summary                     |
|  - Impute missing values                            |
|  - Encode target & categorical columns              |
|  - Standardize features via StandardScaler          |
|  - Stratified Train/Test split (80/20)              |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 4: SVMClassifierService.train()               |
|  - Build SVC estimator (cuML GPU / sklearn CPU)     |
|  - Fit model on X_train, y_train                    |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 5: SVMClassifierService.evaluate()            |
|  - Predict X_test labels                            |
|  - Compute Accuracy, Precision, Recall, F1-Score    |
|  - Write classification_results.txt                 |
|  - Save confusion_matrix.png & report bar chart     |
|  - Save technical markdown explanation report       |
+-----------------------------------------------------+
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
  +-- config.py          (PipelineConfig, PathConfig, DataConfig, ModelConfig, LoggingConfig)
  +-- logger.py          (LoggerFactory)
  +-- data_loader.py     (DataLoaderService - load, validate, preprocess, scale, split)
  +-- svm_classifier.py  (SVMClassifierService - build, train, evaluate, plot, report)
```
