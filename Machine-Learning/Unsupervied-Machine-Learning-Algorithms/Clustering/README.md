# Unsupervised Machine Learning - K-Means Clustering

> Unsupervised Machine Learning | Clustering Algorithm

---

## Table of Contents

1. [What is Clustering?](#1-what-is-clustering)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked Clustering Sum (Step-by-Step)](#5-worked-clustering-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is Clustering?

Clustering is a core method in **unsupervised machine learning** where the objective is to partition a dataset into groups (clusters) such that data points in the same cluster are more similar to each other than to those in other clusters. Unlike supervised learning, the data contains **no target labels** during the learning phase. The algorithm discovers the underlying structure of the feature space entirely on its own.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Unsupervised Learning                                                |
| Output type        | Discrete cluster assignments (e.g. Cluster 0, Cluster 1)             |
| Training data      | Unlabelled (only input features are used for cluster updates)        |
| Objective          | Group similar points together and separate dissimilar ones          |
| Evaluation metrics | Inertia (WCSS), Silhouette Coefficient, Adjusted Rand Index (ARI)*, NMI* |

*\*Requires ground-truth labels for post-training comparison (optional).*

---

## 2. Theoretical Explanation

### How K-Means Works

K-Means is a centroid-based clustering algorithm. It aims to partition $N$ observations into $K$ clusters in which each observation belongs to the cluster with the nearest mean (centroid). The centroid serves as the prototype of the cluster.

The algorithm runs in an iterative, two-step process:

1. **Assignment Phase**: Assign each data point to its closest centroid based on Euclidean distance.
2. **Update Phase**: Recompute the centroids of the clusters by taking the mean of all data points assigned to each cluster.

These steps repeat until the centroids no longer move significantly or the maximum number of iterations is reached.

### Why Feature Scaling is Essential
K-Means relies heavily on the **Euclidean distance** metric:
$$d(x, y) = \sqrt{\sum_{j=1}^D (x_j - y_j)^2}$$

If features have different units or ranges (e.g. `Insulin` ranging from 0 to 800, and `Pregnancies` ranging from 0 to 17), the distance will be dominated by features with larger absolute scales. Standardizing features using a Z-score scaler ensures that each feature has a mean of 0 and a standard deviation of 1, allowing all features to contribute equally to the distance metrics.

---

## 3. Mathematical Operations

### Objective Function (Within-Cluster Sum of Squares)

K-Means minimizes the sum of squared distances between each point $x_i$ and its assigned cluster centroid $\mu_k$. This metric is also called **Inertia** or **WCSS (Within-Cluster Sum of Squares)**:

$$J = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|x_i - \mu_k\|^2$$

where $r_{ik}$ is a binary indicator variable:
$$r_{ik} = \begin{cases} 
      1 & \text{if } x_i \text{ is assigned to cluster } k \\
      0 & \text{otherwise}
   \end{cases}$$

---

### Centroid Update Equations

During the Update Phase, centroids are calculated as the mean of all data points belonging to cluster $k$:

$$\mu_k = \frac{\sum_{i=1}^N r_{ik} x_i}{\sum_{i=1}^N r_{ik}}$$

---

### Intrinsic Evaluation: Silhouette Coefficient

The Silhouette Coefficient measures how close a point is to its own cluster compared to other clusters. For a single sample $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$ is the mean distance between sample $i$ and all other points in the same cluster.
- $b(i)$ is the mean distance between sample $i$ and all points in the nearest cluster that $i$ is not part of.

The average Silhouette Coefficient across all points ranges from $-1$ to $+1$.

---

## 4. Real-World Example

### Credit Card Fraud Transaction Clustering
This project uses K-Means, DBSCAN, and Agglomerative Clustering to identify natural patterns and groupings in credit card transactions without using target labels.

- **Dataset**: `creditcard.csv`
- **Validation outcome**: `Class` (0: Genuine, 1: Fraud - dropped during training, used only as ground-truth metadata)
- **Features Used**:
  - `V1` to `V28`: Principal components obtained from PCA projection of raw transaction features.
  - `Amount`: The monetary value of the transaction.

---

## 5. Worked Clustering Sum (Step-by-Step)

Let us trace K-Means clustering ($K=2$) on a toy 2D dataset with $N=4$ samples:
- $P_1 = (1, 1)$
- $P_2 = (2, 1)$
- $P_3 = (4, 3)$
- $P_4 = (5, 4)$

### Initial Centroids
Suppose we initialize our centroids randomly as:
- $\mu_1 = (1, 1)$
- $\mu_2 = (4, 3)$

---

### Iteration 1

#### 1. Assignment Phase
Compute Euclidean distance $d(P_i, \mu_k)$ from each point to centroids:

- **Point $P_1(1, 1)$**:
  - $d(P_1, \mu_1) = \sqrt{(1-1)^2 + (1-1)^2} = 0$
  - $d(P_1, \mu_2) = \sqrt{(1-4)^2 + (1-3)^2} = \sqrt{9+4} \approx 3.61$
  - *Assign to Cluster 0*

- **Point $P_2(2, 1)$**:
  - $d(P_2, \mu_1) = \sqrt{(2-1)^2 + (1-1)^2} = 1.0$
  - $d(P_2, \mu_2) = \sqrt{(2-4)^2 + (1-3)^2} = \sqrt{4+4} \approx 2.83$
  - *Assign to Cluster 0*

- **Point $P_3(4, 3)$**:
  - $d(P_3, \mu_1) = \sqrt{(4-1)^2 + (3-1)^2} = \sqrt{9+4} \approx 3.61$
  - $d(P_3, \mu_2) = \sqrt{(4-4)^2 + (3-3)^2} = 0$
  - *Assign to Cluster 1*

- **Point $P_4(5, 4)$**:
  - $d(P_4, \mu_1) = \sqrt{(5-1)^2 + (4-1)^2} = \sqrt{16+9} = 5.0$
  - $d(P_4, \mu_2) = \sqrt{(5-4)^2 + (4-3)^2} = \sqrt{1+1} \approx 1.41$
  - *Assign to Cluster 1*

#### 2. Centroid Update Phase
Recompute centroids:
- **New $\mu_1$** (mean of $P_1, P_2$):
  $$\mu_1 = \left( \frac{1+2}{2}, \frac{1+1}{2} \right) = (1.5, 1.0)$$
- **New $\mu_2$** (mean of $P_3, P_4$):
  $$\mu_2 = \left( \frac{4+5}{2}, \frac{3+4}{2} \right) = (4.5, 3.5)$$

---

### Iteration 2

#### 1. Re-assignment Phase
- **Point $P_1(1, 1)$**:
  - $d(P_1, \mu_1) = \sqrt{(1-1.5)^2 + (1-1)^2} = 0.5$
  - $d(P_1, \mu_2) = \sqrt{(1-4.5)^2 + (1-3.5)^2} = \sqrt{12.25+6.25} \approx 4.30$
  - *Assign to Cluster 0*

- **Point $P_2(2, 1)$**:
  - $d(P_2, \mu_1) = \sqrt{(2-1.5)^2 + (1-1)^2} = 0.5$
  - $d(P_2, \mu_2) = \sqrt{(2-4.5)^2 + (1-3.5)^2} = \sqrt{6.25+6.25} \approx 3.54$
  - *Assign to Cluster 0*

- **Point $P_3(4, 3)$**:
  - $d(P_3, \mu_1) = \sqrt{(4-1.5)^2 + (3-1)^2} = \sqrt{6.25+4} \approx 3.20$
  - $d(P_3, \mu_2) = \sqrt{(4-4.5)^2 + (3-3.5)^2} = \sqrt{0.25+0.25} \approx 0.71$
  - *Assign to Cluster 1*

- **Point $P_4(5, 4)$**:
  - $d(P_4, \mu_1) = \sqrt{(5-1.5)^2 + (4-1)^2} = \sqrt{12.25+9} \approx 4.61$
  - $d(P_4, \mu_2) = \sqrt{(5-4.5)^2 + (4-3.5)^2} = \sqrt{0.25+0.25} \approx 0.71$
  - *Assign to Cluster 1*

**Assignments have not changed.** The centroids remain constant and the algorithm has converged.

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
|  - DataConfig   (Outcome, test size, splits)        |
|  - ModelConfig  (n_clusters, n_init, max_iter)      |
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
|  presence        |
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
|  - Separate validation outcome ('Outcome')       |
|  - Fill null values with median/mode             |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  _split() & standardise features                 |
|  - Split into 80% Train, 20% Test (stratified)   |
|  - Fit and transform StandardScaler on train set |
|  - Transform test set                            |
+--------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: KMeansClusteringService.run_analysis()    |
|  - Calculate WCSS & Silhouette across K=[2..10]    |
|  - Save elbow_method.png & silhouette_analysis.png  |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 5: KMeansClusteringService.train()            |
|  - Instantiate KMeans(n_clusters=2, init='k-means++')|
|  - Call model.fit(x_train_scaled)                   |
|  - Log convergence iterations and final Inertia     |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 6: KMeansClusteringService.evaluate()         |
+-----------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  Calculate Clustering Quality                    |
|  - Compute train/test silhouette scores          |
|  - Compute ARI & NMI compared to Outcome labels   |
|  - Compute cluster contingency cross-tab table   |
+--------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  Save Output Artifacts to output/                |
|  - Write clustering_results.txt                  |
|  - Write clustering_analysis.md report           |
|  - Project data to 2D using PCA and save plot    |
|    as cluster_visualization_2d.png               |
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
  +-- data_loader.py         (DataLoaderService - load, scale features, splits)
  |
  +-- clustering_service.py   (ClusteringService - train, evaluate,
                               PCA projection, parameter searches, reporting)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter            | Location      | Default      | Description                                      |
|----------------------|---------------|--------------|--------------------------------------------------|
| `n_clusters`         | `ModelConfig` | `2`          | Number of centroids / partitions to construct    |
| `init`               | `ModelConfig` | `'k-means++'`| Centroid initialization method                   |
| `max_iter`           | `ModelConfig` | `300`        | Maximum iterations allowed for convergence        |
| `n_init`             | `ModelConfig` | `10`         | Number of random initial seed runs to execute    |
| `scale_features`     | `DataConfig`  | `True`       | Apply StandardScaler prior to centroid fittings  |
| `test_size`          | `DataConfig`  | `0.20`       | Partition percentage reserved for test evaluation|
