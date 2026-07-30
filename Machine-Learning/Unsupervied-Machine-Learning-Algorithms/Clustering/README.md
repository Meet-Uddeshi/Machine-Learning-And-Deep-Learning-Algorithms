# Unsupervised Clustering Module

An end-to-end Object-Oriented machine learning pipeline implementing K-Means Clustering, DBSCAN (Density-Based Spatial Clustering of Applications with Noise), and Agglomerative Hierarchical Clustering on transaction feature dataset (`creditcard.csv`).

---

## Architecture and Directory Structure

The project follows a clean Service-Oriented Architecture (SOA) where data loading, preprocessing, feature scaling, model fitting, and validation metric evaluations are isolated into dedicated service classes inside `src/`. Thin orchestration scripts manage pipeline execution, while immutable dataclasses handle configuration settings.

```text
Clustering/
│
├── README.md                          # Comprehensive module guide & documentation
│
├── data/                              # Input dataset folder
│   └── creditcard.csv                 # Credit Card transaction features dataset
│
├── output/                            # Generated evaluation outputs and figures
│   ├── kmeans_elbow_curve.png
│   ├── cluster_scatter_k_means.png
│   ├── cluster_scatter_agglomerative.png
│   ├── cluster_scatter_dbscan.png
│   └── clustering_analysis_report.md
│
└── src/                               # Python source codebase
    ├── config.py                      # Immutable configuration dataclasses
    ├── logger.py                      # Centralized stdout logger factory
    ├── data_loader.py                 # Data sampling & feature scaling service
    ├── clustering_service.py          # Clustering algorithms & validation service
    └── main.py                        # Main pipeline orchestration entry point
```

---

## Key Algorithms & Mathematical Formulations

### 1. K-Means Clustering
- **Concept**: Partitioning method that divides data into $k$ distinct non-overlapping clusters by minimizing the within-cluster sum of squares (WCSS / Inertia).
- **Formulation**: Minimizes objective function:
  $$J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - \mu_j \|^2$$
  where $\mu_j$ is the centroid of cluster $C_j$.

### 2. Agglomerative Hierarchical Clustering
- **Concept**: Bottom-up hierarchical clustering strategy. Starts with each observation in its own individual cluster and iteratively merges the closest pair of clusters based on distance metric (Euclidean distance with Ward linkage).
- **Formulation**: Minimizes variance increase when merging clusters $A$ and $B$:
  $$\Delta \text{ESS} = \frac{n_A n_B}{n_A + n_B} \| \mu_A - \mu_B \|^2$$

### 3. DBSCAN (Density-Based Spatial Clustering)
- **Concept**: Discovers clusters of arbitrary shape based on spatial density. Groups core points having at least $\text{MinPts}$ within neighborhood radius $\varepsilon$ and flags sparse points as noise (class -1).

---

## Cluster Validation Metrics

Since unsupervised clustering lacks true ground truth labels, model quality is evaluated using intrinsic validation metrics:

1. **Silhouette Coefficient**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance. Ranges from -1 to +1 (higher is better).

2. **Calinski-Harabasz Index (Variance Ratio Criterion)**:
   $$CH = \frac{\text{Trace}(B_k) / (k - 1)}{\text{Trace}(W_k) / (n - k)}$$
   where $B_k$ is between-cluster dispersion and $W_k$ is within-cluster dispersion (higher is better).

3. **Davies-Bouldin Index**:
   $$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)} \right)$$
   where $\sigma_i$ is average distance of points in cluster $i$ to centroid $\mu_i$ (lower is better).

---

## Execution Instructions

Execute the main orchestration script from your terminal:

```bash
python Machine-Learning/Unsupervied-Machine-Learning-Algorithms/Clustering/src/main.py
```
