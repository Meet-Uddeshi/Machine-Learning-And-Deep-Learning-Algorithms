# Classification - Random Forest

> Supervised Machine Learning | Ensemble Learning Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Random Forest Sum (Step-by-Step)](#5-worked-random-forest-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where an algorithm learns to assign input feature vectors to discrete target classes based on patterns in a labelled training set.

Random Forest extends single decision tree classification by constructing an ensemble of decorrelated decision trees, aggregating their votes to produce robust and highly accurate predictions.

---

## 2. Theoretical Explanation

### How Random Forest Works

Random Forest is a **bagging (bootstrap aggregating)** ensemble method that builds `N` decision trees in parallel.

1. **Bootstrap Sampling**: Each tree is trained on a random sample of rows drawn with replacement from the original dataset.
2. **Random Feature Selection**: At each node split, only a random subset `m = sqrt(M)` of total `M` features is considered. This decorrelates individual trees.
3. **Majority Voting**: For a new test sample, each tree casts a vote, and the class receiving the majority vote becomes the final prediction.

### Out-of-Bag (OOB) Score

Because bootstrap sampling draws samples with replacement, approximately 36.8% of training data is left out of any individual tree's training set. These "Out-of-Bag" (OOB) samples act as an internal validation set, allowing OOB generalization error estimation without a separate validation split.

---

## 3. Mathematical Operations

### Bootstrap Sample Inclusion Probability

The probability that a specific observation is **NOT** selected in a bootstrap sample of size `N` is:

```
P(Not Selected) = (1 - 1/N)^N
```

As `N -> infinity`, this probability approaches:

```
lim_{N -> inf} (1 - 1/N)^N = 1 / e approx 0.368 (36.8%)
```

Thus, ~63.2% of distinct samples are included in each tree, while ~36.8% form the OOB evaluation set.

### Ensemble Prediction Formula

For a classification task with `B` trees where tree `b` outputs predicted class `C_b(x)`:

```
Final Prediction(x) = argmax_k [ sum_{b=1}^{B} I(C_b(x) == k) ]
```

Where `I(...)` is the indicator function returning 1 if true and 0 otherwise.

---

## 4. Real-World Example

### Indian Railway Maintenance Detection

This project predicts whether a train locomotive requires immediate **Maintenance (`maintenance_required = 1`)** or is **Operational (`maintenance_required = 0`)** based on sensor readings and operational conditions.

**Dataset:** `indian_railway_failure_detection_maintenance_v2.csv`  
**Target Variable:** `maintenance_required`  
**Features Used:** `region`, `season`, `train_type`, `train_age_years`, `average_speed_kmph`, `distance_travelled_km`, `ambient_temperature_c`, `humidity_percent`, `rainfall_mm`, `wheel_wear_percent`, `track_vibration_level`, `rail_wear_mm`, `bearing_temperature_c`, `axle_temperature_c`, `brake_pad_wear_percent`, `brake_pressure_psi`, `battery_voltage`, `last_maintenance_days`, `sensor_health_index`, `inspection_score`, `delay_minutes`.

**Excluded Columns:** `train_id`, `failure_type`, `failure_severity`, `risk_score`.

---

## 5. Worked Random Forest Sum (Step-by-Step)

Consider an ensemble of **B = 3 Trees** predicting whether maintenance is required for a locomotive.

### Individual Tree Predictions for Query Sample X

- **Tree 1**: Evaluates `bearing_temperature_c > 90.0` and `track_vibration_level > 5.0`  
  `Prediction_1 = 1 (Maintenance Required)`
- **Tree 2**: Evaluates `wheel_wear_percent > 60.0` and `brake_pressure_psi < 80.0`  
  `Prediction_2 = 1 (Maintenance Required)`
- **Tree 3**: Evaluates `battery_voltage < 22.0`  
  `Prediction_3 = 0 (No Maintenance Required)`

### Step 1: Vote Aggregation

| Class Label | Tree 1 | Tree 2 | Tree 3 | Total Votes |
|-------------|--------|--------|--------|-------------|
| Class 0     | 0      | 0      | 1      | 1           |
| Class 1     | 1      | 1      | 0      | 2           |

### Step 2: Majority Decision

```
Class 1 (Maintenance Required) = 2 votes
Class 0 (No Maintenance)        = 1 vote

Ensemble Output = Class 1 (Maintenance Required)
```

The ensemble correctly overrides the single noisy vote of Tree 3, demonstrating variance reduction through bagging.

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
|  - DataConfig (target='maintenance_required',       |
|                max_samples=50000)                   |
|  - ModelConfig (n_estimators=100, max_depth=15,    |
|                oob_score=True)                      |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
|  - Ingest CSV -> Validate schema -> Downsample      |
|  - Impute nulls & encode categorical variables      |
|  - Standardize features & perform Stratified Split  |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 4: RandomForestClassifierService.train()      |
|  - Fit 100 decision trees via bagging & random      |
|    feature sampling                                 |
|  - Extract OOB generalization score                 |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 5: RandomForestClassifierService.evaluate()  |
|  - Compute test accuracy, precision, recall, F1     |
|  - Plot Confusion Matrix & Feature Importances      |
|  - Save classification_results.txt & analysis.md    |
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
  +-- config.py                    (PipelineConfig, PathConfig, DataConfig, ModelConfig, LoggingConfig)
  +-- logger.py                    (LoggerFactory)
  +-- data_loader.py               (DataLoaderService)
  +-- random_forest_classifier.py  (RandomForestClassifierService)
```
