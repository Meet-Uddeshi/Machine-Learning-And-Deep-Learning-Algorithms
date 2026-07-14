# Classification - Decision Tree

> Supervised Machine Learning | Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Decision Tree Sum (Step-by-Step)](#5-worked-decision-tree-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where the goal is to assign an input data point to one of a fixed set of **discrete categories (classes)**.

The algorithm learns from a **labelled training dataset** - data where the correct class is already known. After learning, it applies that knowledge to predict the class of **new, unseen data points**.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning                                                  |
| Output type        | Discrete class label (e.g., "High Risk", "Low Risk", "Moderate")    |
| Training data      | Labelled (input + known correct class)                               |
| Prediction         | Assigns one class label from a predefined set                        |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix              |

### Binary vs. Multi-Class Classification

- **Binary Classification** - only two possible classes (e.g., spam / not spam, pass / fail).
- **Multi-Class Classification** - three or more possible classes (e.g., Low Risk / Medium Risk / High Risk).

---

## 2. Theoretical Explanation

### How Decision Trees Work

A Decision Tree is a **greedy, recursive partitioning** algorithm. Its core idea is:

> "Repeatedly split the data on the feature and threshold that best separates the classes, until the subsets are pure or a stopping condition is reached."

A Decision Tree is an **eager learning** algorithm - it builds an explicit model (a tree of if-then rules) during training. At prediction time, a new data point simply traverses the tree from root to leaf, following the decision rules at each node.

### Anatomy of a Decision Tree

```
Root Node    -- the first split; uses the feature with the highest information gain
Internal Node -- an intermediate split condition: "feature <= threshold?"
Leaf Node    -- a terminal node; outputs the majority class of samples reaching it
Branch       -- the connection between a parent node and its children
```

### Step-by-Step Logic

```
Step 1: Start with the entire training dataset at the root node.

Step 2: For every feature, evaluate every possible threshold t:
        - Split the dataset into:
            Left  child: all rows where feature <= t
            Right child: all rows where feature >  t
        - Compute the impurity of both children (Gini or Entropy).
        - Compute the weighted impurity reduction (Information Gain).

Step 3: Select the (feature, threshold) pair with the maximum
        Information Gain as the current split.

Step 4: Recursively repeat Steps 2-3 on each child subset.

Step 5: Stop recursion when:
        - All samples at a node belong to the same class (pure node), OR
        - max_depth has been reached, OR
        - A node has fewer than min_samples_split samples, OR
        - A node has fewer than min_samples_leaf samples in a child.

Step 6: Assign the majority class of training samples at a leaf as
        the prediction label for that leaf.

Step 7: To classify a new point:
        - Start at the root.
        - At each node: if feature <= threshold, go left; else go right.
        - Return the class label of the leaf reached.
```

### Why No Feature Scaling is Needed

Decision Trees compare feature values against a threshold at each node. The split criterion depends only on the **rank order** of values (which samples fall left vs. right of the threshold), not on the absolute magnitude. Multiplying a feature by any positive constant does not change which split minimises impurity. Feature scaling is therefore **unnecessary** and is deliberately omitted in this implementation.

---

## 3. Mathematical Operations

### Splitting Criterion 1: Gini Impurity

Gini impurity measures the probability that a randomly chosen sample from a node would be incorrectly classified if labelled according to the class distribution of that node.

For a node with `C` classes where `p_i` is the fraction of class `i` samples:

```
Gini(node)  =  1  -  sum( p_i^2 )   for i = 1 to C
```

A **pure node** (only one class) has Gini = 0.
A **maximally impure node** (all classes equally likely) has Gini = 1 - 1/C.

### Splitting Criterion 2: Entropy (Information Gain)

Entropy measures the disorder (uncertainty) in a node's class distribution.

```
Entropy(node)  =  -  sum( p_i  x  log2(p_i) )   for i = 1 to C
```

A **pure node** has Entropy = 0.
A **maximally disordered node** has Entropy = log2(C).

### Information Gain

Information Gain quantifies the reduction in impurity obtained by splitting a parent node into two children.

Let `N` = total samples in the parent, `N_L` = samples in left child, `N_R` = samples in right child.

```
Information Gain  =  Impurity(parent)
                     -  (N_L / N) x Impurity(left child)
                     -  (N_R / N) x Impurity(right child)
```

The algorithm selects the split that **maximises** this value.

### Feature Importance

After training, each feature receives an importance score equal to the total (weighted) impurity reduction it produces across all nodes that split on it, normalized to sum to 1.

```
Importance(feature_j)  =
    sum over all nodes k that split on j:
        (N_k / N_total)  x  Impurity(k)
        -  (N_k_left  / N_total)  x  Impurity(left child of k)
        -  (N_k_right / N_total)  x  Impurity(right child of k)
```

Then all values are normalized so they sum to 1.0.

### Evaluation Metrics

**Accuracy** - What fraction of all predictions were correct?

```
Accuracy  =  Number of Correct Predictions  /  Total Number of Predictions
```

**Precision** - Of all points predicted as a class, how many actually belong to it?

```
Precision  =  True Positives  /  (True Positives  +  False Positives)
```

**Recall** - Of all actual points in a class, how many were correctly predicted?

```
Recall  =  True Positives  /  (True Positives  +  False Negatives)
```

**F1-Score** - Harmonic mean of Precision and Recall (best when classes are imbalanced):

```
F1-Score  =  2  x  (Precision  x  Recall)  /  (Precision  +  Recall)
```

---

## 4. Real-World Example

### Student Burnout Risk Prediction

This project predicts the **Burnout Risk Level** of university students based on their academic habits, AI tool usage, and well-being indicators.

**Dataset:** `ai_student_impact_dataset (1).csv`

**Target Variable:** `Burnout_Risk_Level`
Possible classes: `Low`, `Moderate`, `High`

**Features Used (examples):**

| Feature                        | Description                                   |
|--------------------------------|-----------------------------------------------|
| `Study_Hours_Per_Day`          | Average daily study hours                     |
| `AI_Tool_Usage_Hours`          | Daily hours spent using AI tools              |
| `Sleep_Hours_Per_Night`        | Average nightly sleep hours                   |
| `Social_Media_Hours_Per_Day`   | Daily social media usage                      |
| `Assignment_Completion_Rate`   | Percentage of assignments completed           |
| `Mental_Health_Score`          | Self-reported mental health rating            |
| `Stress_Level`                 | Self-reported stress rating                   |

**Columns excluded from features:**

- `Student_ID` - a unique identifier with no predictive value.
- `Post_Semester_GPA` - a post-outcome measurement (data leakage risk).
- `Skill_Retention_Score` - a post-outcome measurement (data leakage risk).

**Why Decision Tree is a good fit here:**
- The resulting tree produces explicit if-then rules (e.g., "If Stress_Level > 7 AND Sleep_Hours_Per_Night <= 5, predict High Risk") that are directly interpretable by educators and counsellors -- no ML expertise required to act on the output.
- Decision Trees naturally handle multi-class problems without requiring one-vs-rest wrappers.
- Feature importances directly reveal which student habits are the strongest predictors of burnout, enabling targeted interventions.

---

## 5. Worked Decision Tree Sum (Step-by-Step)

The following is a simplified, hand-calculated demonstration of how a Decision Tree classifies a new student using **2 features** and **Gini impurity** as the criterion.

### The Training Data (6 known students, 2 features for simplicity)

| Student | Stress_Level | Sleep_Hours | Burnout Risk |
|---------|-------------|-------------|--------------|
| S1      | 3           | 7           | Low          |
| S2      | 5           | 6           | Low          |
| S3      | 6           | 5           | Moderate     |
| S4      | 7           | 4           | High         |
| S5      | 8           | 3           | High         |
| S6      | 9           | 3           | High         |

### Step 1: Compute Gini at Root (all 6 samples)

Class distribution: Low=2, Moderate=1, High=3. Total N=6.

```
p_Low      = 2/6 = 0.333
p_Moderate = 1/6 = 0.167
p_High     = 3/6 = 0.500

Gini(root) = 1 - (0.333^2 + 0.167^2 + 0.500^2)
           = 1 - (0.111  + 0.028  + 0.250)
           = 1 - 0.389
           = 0.611
```

### Step 2: Evaluate Candidate Splits

**Candidate: Stress_Level <= 6**

```
Left  (Stress_Level <= 6): S1, S2, S3  ->  Low=2, Moderate=1, High=0  (N_L=3)
Right (Stress_Level >  6): S4, S5, S6  ->  Low=0, Moderate=0, High=3  (N_R=3)

Gini(Left)  = 1 - ((2/3)^2 + (1/3)^2 + 0^2) = 1 - (0.444 + 0.111) = 0.444
Gini(Right) = 1 - (0 + 0 + (3/3)^2)          = 1 - 1.000           = 0.000

Information Gain = 0.611 - (3/6) x 0.444 - (3/6) x 0.000
                 = 0.611 - 0.222 - 0.000
                 = 0.389
```

**Candidate: Sleep_Hours <= 5**

```
Left  (Sleep_Hours <= 5): S3, S4, S5, S6  ->  Moderate=1, High=3  (N_L=4)
Right (Sleep_Hours >  5): S1, S2           ->  Low=2               (N_R=2)

Gini(Left)  = 1 - ((1/4)^2 + (3/4)^2) = 1 - (0.0625 + 0.5625) = 0.375
Gini(Right) = 1 - ((2/2)^2)            = 1 - 1.000              = 0.000

Information Gain = 0.611 - (4/6) x 0.375 - (2/6) x 0.000
                 = 0.611 - 0.250 - 0.000
                 = 0.361
```

### Step 3: Select Best Split

```
Stress_Level <= 6  -> Information Gain = 0.389  <-- WINNER
Sleep_Hours  <= 5  -> Information Gain = 0.361
```

**Root split: Stress_Level <= 6**

### Step 4: Recurse on Left Child (S1, S2, S3)

```
Left child: Low=2, Moderate=1, High=0  (still impure)

Candidate: Sleep_Hours <= 5
  Left  (Sleep <= 5): S3  ->  Moderate=1  (N_L=1)  -- pure leaf
  Right (Sleep >  5): S1, S2  ->  Low=2  (N_R=2)   -- pure leaf
  Gain = 0.444 - (1/3) x 0 - (2/3) x 0 = 0.444
```

Left child splits on **Sleep_Hours <= 5**.

### Step 5: Right Child is Pure

```
Right child (Stress_Level > 6): High=3  ->  already pure (Gini=0)
Prediction: HIGH
```

### Final Decision Tree

```
                    Stress_Level <= 6?
                   /                  \
                YES                    NO
         Sleep_Hours <= 5?           Predict: HIGH
        /               \
      YES                NO
  Predict: MODERATE   Predict: LOW
```

### Classify a New Student

New student: Stress_Level = 8, Sleep_Hours = 4

```
Step 1: Stress_Level <= 6?  --> 8 <= 6 is False --> go RIGHT
Step 2: Right child is a leaf --> Predict: HIGH
```

This makes intuitive sense: a student with high stress (8) and low sleep (4) is predicted as high burnout risk.

---

## 6. Program Flowchart

The following flowchart describes the complete execution flow of the Decision Tree Classification Pipeline as implemented in this project.

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig  (dataset path, output path)          |
|  - DataConfig  (target column, drop columns, split) |
|  - ModelConfig (criterion=gini, max_depth=None,     |
|                 min_samples_split=2,                |
|                 min_samples_leaf=1,                 |
|                 class_weight=balanced,              |
|                 ccp_alpha=0.0)                      |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler  (decision_tree_classification.log) |
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
|  Check target    |
|  column exists   |
|  Check drop cols |
|  exist in CSV    |
+------------------+
         |
         v
+------------------+
|  _log_data_      |
|  summary()       |
|  Log shape,      |
|  dtypes, nulls,  |
|  class balance   |
+------------------+
         |
         v
+--------------------------------------------------+
|  _preprocess()                                   |
|  1. Separate target from features                |
|  2. Drop Student_ID, Post_Semester_GPA,          |
|     Skill_Retention_Score                        |
|  3. Impute nulls (median for numeric,            |
|     mode for categorical)                        |
|  4. LabelEncode target (Low=0, Moderate=1,       |
|     High=2)                                      |
|  5. LabelEncode each categorical feature column  |
|  NOTE: No StandardScaler -- Decision Trees       |
|  are scale-invariant; scaling is a no-op.        |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|  _split()                                        |
|  Stratified Train/Test split                     |
|  - 80% Training data                             |
|  - 20% Test data                                 |
|  - Random state = 42 (reproducible)              |
+--------------------------------------------------+
         |
         v
+-----------------------------------------------------+
|  Step 4: DecisionTreeClassifierService.train()      |
|  - Build DecisionTreeClassifier with config params  |
|  - For each node, greedily select the (feature,     |
|    threshold) pair that maximises Information Gain  |
|    (minimises weighted Gini / Entropy of children)  |
|  - Recursively split until stopping criteria met    |
|  - Call model.fit(X_train, y_train)                 |
|  - Log training time, tree depth, leaf count        |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 5: DecisionTreeClassifierService.evaluate()   |
+-----------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  For each test point in X_test:                  |
|  - Start at root node                            |
|  - At each node: feature <= threshold?           |
|    YES --> traverse left child                   |
|    NO  --> traverse right child                  |
|  - Assign the majority class of the leaf reached |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  Compute Evaluation Metrics                      |
|  - Accuracy Score                                |
|  - Precision (macro average)                     |
|  - Recall    (macro average)                     |
|  - F1-Score  (macro average)                     |
|  - Confusion Matrix (3x3 for 3 classes)          |
|  - Per-class Classification Report               |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _log_evaluation()                               |
|  Print all metrics to console and log file       |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _save_results()                                 |
|  Write classification_results.txt to output/    |
|  Includes feature importances ranked by impact   |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _save_tree_text()                               |
|  Export decision_tree_rules.txt                  |
|  Human-readable if-then rule set                 |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _generate_plots()                               |
|  1. confusion_matrix.png (heatmap)               |
|  2. classification_report.png (grouped bars)     |
|  3. feature_importance.png (ranked bar chart)    |
|  4. decision_tree_structure.png (tree diagram)   |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _save_analysis()                                |
|  Write classification_analysis.md               |
+--------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|               END: Pipeline Complete               |
|  Total elapsed time is logged                      |
+-----------------------------------------------------+
```

### Module Responsibility Map

```
main.py
  |
  +-- config.py                   (PipelineConfig, PathConfig, DataConfig,
  |                                 ModelConfig, LoggingConfig)
  |
  +-- logger.py                   (LoggerFactory - console + file)
  |
  +-- data_loader.py              (DataLoaderService - load, validate,
  |                                 preprocess, split; no scaling)
  |
  +-- decision_tree_classifier.py (DecisionTreeClassifierService - train,
                                    predict, evaluate, save results,
                                    export tree text, generate 4 plots)
```

---

### Configuration

All tunable parameters are located in `src/config.py`. No changes to business logic are needed. Edit the dataclass defaults to adjust:

| Parameter            | Location      | Default      | Description                                      |
|----------------------|---------------|--------------|--------------------------------------------------|
| `criterion`          | `ModelConfig` | `'gini'`     | Split quality metric (`'gini'` or `'entropy'`)   |
| `max_depth`          | `ModelConfig` | `None`       | Maximum tree depth (`None` = grow until pure)    |
| `min_samples_split`  | `ModelConfig` | `2`          | Min samples to split an internal node            |
| `min_samples_leaf`   | `ModelConfig` | `1`          | Min samples required at a leaf node              |
| `max_features`       | `ModelConfig` | `None`       | Features considered per split (`None` = all)     |
| `class_weight`       | `ModelConfig` | `'balanced'` | Class weighting strategy                         |
| `ccp_alpha`          | `ModelConfig` | `0.0`        | Pruning complexity parameter (0 = no pruning)    |
| `test_size`          | `DataConfig`  | `0.20`       | Fraction of data used for testing                |
| `target_column`      | `DataConfig`  | `Burnout_Risk_Level` | Column being predicted              |
